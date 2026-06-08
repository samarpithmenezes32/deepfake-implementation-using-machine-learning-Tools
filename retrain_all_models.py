"""
Retrain All Deepfake Detection Models
Trains Hybrid, CNN, LSTM, Transformer, and Spectral models on the curated dataset
in input/real and input/fake, and saves the best weights to saved_models/.
"""

import os
import time
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from deepfake_unified import (
    HybridDeepfakeDetector,
    CNNDeepfakeDetector,
    TemporalLSTMDetector,
    TransformerDeepfakeDetector,
    SpectralAnalysisDetector
)

IMAGENET_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

class DeepfakeImageDataset(Dataset):
    def __init__(self, root: str, img_size: int = 224, max_items: int = 500):
        self.samples = []
        img_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        for label, cls in enumerate(["real", "fake"]):
            class_root = os.path.join(root, cls)
            if not os.path.exists(class_root):
                continue
            for dirpath, _, filenames in os.walk(class_root):
                for f in filenames:
                    if f.lower().endswith(img_exts):
                        self.samples.append((os.path.join(dirpath, f), label))
        
        # Cap items per class
        capped = {0: 0, 1: 0}
        new_samples = []
        for p, l in self.samples:
            if capped[l] < max_items:
                new_samples.append((p, l))
                capped[l] += 1
        self.samples = new_samples
        
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            IMAGENET_NORM
        ])
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        p, l = self.samples[idx]
        im = cv2.imread(p)
        if im is None:
            im = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return self.tf(im), torch.tensor(l, dtype=torch.long)


def retrain_model(model_type: str, epochs: int, batch_size: int, device: str):
    print(f"\n{'='*60}\nRetraining Model: {model_type.upper()}\n{'='*60}")
    
    os.makedirs('saved_models', exist_ok=True)
    save_path = os.path.join('saved_models', f'{model_type}.pth')
    
    # Load dataset
    dataset = DeepfakeImageDataset(root='input', max_items=500)
    if len(dataset) == 0:
        print("[Error] No dataset found in input/real and input/fake.")
        return 0.0
        
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=(train_size > batch_size))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Instantiate model
    has_weights = os.path.exists(save_path)
    pretrained_flag = not has_weights
    
    if model_type == 'hybrid':
        model = HybridDeepfakeDetector(pretrained=pretrained_flag)
        if has_weights:
            print(f"Loading existing weights from {save_path} for fine-tuning...")
            model.load_state_dict(torch.load(save_path, map_location=device))
        for param in model.backbone.parameters():
            param.requires_grad = False
    elif model_type == 'cnn':
        model = CNNDeepfakeDetector(pretrained=pretrained_flag)
        if has_weights:
            print(f"Loading existing weights from {save_path} for fine-tuning...")
            model.load_state_dict(torch.load(save_path, map_location=device))
        for param in model.backbone.parameters():
            param.requires_grad = False
    elif model_type == 'lstm':
        model = TemporalLSTMDetector(pretrained=pretrained_flag, sequence_length=16)
        if has_weights:
            print(f"Loading existing weights from {save_path} for fine-tuning...")
            model.load_state_dict(torch.load(save_path, map_location=device))
        for param in model.cnn_backbone.parameters():
            param.requires_grad = False
        for param in model.cnn_backbone.fc.parameters():
            param.requires_grad = True
    elif model_type == 'transformer':
        model = TransformerDeepfakeDetector()
        if has_weights:
            print(f"Loading existing weights from {save_path} for fine-tuning...")
            model.load_state_dict(torch.load(save_path, map_location=device))
        for param in model.conv_stem.parameters():
            param.requires_grad = False
    elif model_type == 'spectral':
        model = SpectralAnalysisDetector(pretrained=pretrained_flag)
        if has_weights:
            print(f"Loading existing weights from {save_path} for fine-tuning...")
            model.load_state_dict(torch.load(save_path, map_location=device))
        for param in model.spatial_branch.parameters():
            param.requires_grad = False
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            if model_type == 'lstm':
                seq_len = getattr(model, 'sequence_length', 16)
                inputs = inputs.unsqueeze(1).repeat(1, seq_len, 1, 1, 1)
                
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)
            
        train_acc = correct_train / max(1, total_train)
        
        # Validation
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                if model_type == 'lstm':
                    seq_len = getattr(model, 'sequence_length', 16)
                    inputs = inputs.unsqueeze(1).repeat(1, seq_len, 1, 1, 1)
                    
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                preds = torch.argmax(outputs, dim=1)
                correct_val += (preds == targets).sum().item()
                total_val += targets.size(0)
                
        val_acc = correct_val / max(1, total_val)
        
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss/max(1, total_train):.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Best Val Acc: {best_acc:.4f}")
        
    elapsed = time.time() - start_time
    print(f"[Success] Finished retraining {model_type.upper()} in {elapsed:.1f}s. Best Weights saved to {save_path}")
    return best_acc


def main():
    parser = argparse.ArgumentParser(description="Retrain Deepfake Models")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs per model")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--model', type=str, default='all', choices=['all', 'hybrid', 'cnn', 'lstm', 'transformer', 'spectral'])
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    models_to_train = ['hybrid', 'cnn', 'lstm', 'transformer', 'spectral'] if args.model == 'all' else [args.model]
    
    results = {}
    for m in models_to_train:
        acc = retrain_model(m, epochs=args.epochs, batch_size=args.batch_size, device=device)
        results[m] = acc
        
    print(f"\n{'='*60}\nRETRAINING SUMMARY\n{'='*60}")
    for m, acc in results.items():
        print(f"Model: {m.ljust(12)} | Best Validation Accuracy: {acc*100:.1f}%")
    print('='*60)

if __name__ == '__main__':
    main()
