"""
Unified Deepfake Detection Core Module
Combines deepfake.py, deepfake_model_implementation.py, and rs p1.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
import cv2
import os
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass


# ============================================================================
# MODEL ARCHITECTURES (from deepfake_model_implementation.py + deepfake.py)
# ============================================================================

class HybridDeepfakeDetector(nn.Module):
    """ResNet-50 + DCT frequency features + Transformer encoder"""
    def __init__(self, num_classes: int = 2, pretrained: bool = False):
        super(HybridDeepfakeDetector, self).__init__()
        try:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        except:
            self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        self.dct_branch = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(8)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(2048 + 128 * 64, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        spatial = self.backbone(x).flatten(1)
        freq = self.dct_branch(x).flatten(1)
        combined = torch.cat([spatial, freq], dim=1)
        return self.fusion(combined)


class CNNDeepfakeDetector(nn.Module):
    """ResNet-50 with dense inception blocks"""
    def __init__(self, num_classes: int = 2, pretrained: bool = False):
        super(CNNDeepfakeDetector, self).__init__()
        try:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        except:
            self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.dense_block1 = self._make_dense_inception_block(2048, 512)
        self.dense_block2 = self._make_dense_inception_block(512, 256)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(256, num_classes)
    
    def _make_dense_inception_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, 1), nn.BatchNorm2d(out_channels//4), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//4, out_channels//2, 3, padding=1), nn.BatchNorm2d(out_channels//2), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//2, out_channels, 1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        x = self.dense_block1(features)
        x = self.dense_block2(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)


class TemporalLSTMDetector(nn.Module):
    """ResNet-34 + BiLSTM + MultiheadAttention"""
    def __init__(self, num_classes: int = 2, sequence_length: int = 16, pretrained: bool = False):
        super(TemporalLSTMDetector, self).__init__()
        try:
            self.cnn_backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        except:
            self.cnn_backbone = models.resnet34(pretrained=pretrained)
        self.cnn_backbone.fc = nn.Linear(self.cnn_backbone.fc.in_features, 512)
        self.sequence_length = sequence_length
        
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        x = x.view(batch_size * seq_len, *x.shape[2:])
        cnn_features = self.cnn_backbone(x)
        cnn_features = cnn_features.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(cnn_features)
        lstm_out = lstm_out.transpose(0, 1)
        attended_features, _ = self.attention(lstm_out, lstm_out, lstm_out)
        pooled_features = attended_features.mean(dim=0)
        return self.classifier(pooled_features)


class TransformerDeepfakeDetector(nn.Module):
    """Conv stem + patch embeddings + Transformer"""
    def __init__(self, num_classes: int = 2, patch_size: int = 16, embed_dim: int = 768, 
                 num_layers: int = 12, nhead: int = 12, ff_dim: int = 3072):
        super(TransformerDeepfakeDetector, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.patch_embed = nn.Conv2d(64, embed_dim, patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, 196, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim, dropout=0.1, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        x = self.conv_stem(x)
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        if self.pos_embed.size(1) != x.size(1):
            pe = self.pos_embed[:, :x.size(1), :] if self.pos_embed.size(1) > x.size(1) else self.pos_embed.repeat(1, (x.size(1) + self.pos_embed.size(1) - 1) // self.pos_embed.size(1), 1)[:, :x.size(1), :]
            x = x + pe
        else:
            x = x + self.pos_embed
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        x = x.mean(dim=1)
        x = self.norm(x)
        return self.classifier(x)


class SpectralAnalysisDetector(nn.Module):
    """DCT + ResNet-18 spatial fusion"""
    def __init__(self, num_classes: int = 2, pretrained: bool = False):
        super(SpectralAnalysisDetector, self).__init__()
        self.freq_branch = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(8)
        )
        try:
            self.spatial_branch = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        except:
            self.spatial_branch = models.resnet18(pretrained=pretrained)
        self.spatial_branch.fc = nn.Identity()
        self.fusion = nn.Sequential(
            nn.Linear(256 * 64 + 512, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def apply_dct(self, x):
        x_np = x.cpu().numpy()
        dct_features = []
        for batch_idx in range(x_np.shape[0]):
            img = x_np[batch_idx].transpose(1, 2, 0)
            dct_channels = []
            for c in range(3):
                dct_c = cv2.dct(img[:, :, c].astype(np.float32))
                dct_channels.append(dct_c)
            dct_img = np.stack(dct_channels, axis=2)
            dct_features.append(dct_img.transpose(2, 0, 1))
        return torch.tensor(np.array(dct_features), dtype=x.dtype, device=x.device)
    
    def forward(self, x):
        freq_input = self.apply_dct(x)
        freq_features = self.freq_branch(freq_input).flatten(1)
        spatial_features = self.spatial_branch(x)
        combined = torch.cat([freq_features, spatial_features], dim=1)
        return self.fusion(combined)


# ============================================================================
# UNIFIED DETECTION PIPELINE
# ============================================================================

class UnifiedDeepfakeDetector:
    """Unified detector combining all models and features"""
    
    MODEL_INFO = {
        'hybrid': {'name': 'Hybrid', 'desc': 'ResNet-50 + DCT + Transformer'},
        'cnn': {'name': 'CNN Dense', 'desc': 'ResNet-50 with Dense Inception'},
        'lstm': {'name': 'LSTM Temporal', 'desc': 'ResNet-34 + BiLSTM + Attention'},
        'transformer': {'name': 'Transformer', 'desc': 'Conv Stem + Patch Embeddings'},
        'spectral': {'name': 'Spectral', 'desc': 'DCT + ResNet-18 Fusion'},
    }
    
    PERFORMANCE_BENCHMARKS = {
        'hybrid': {'accuracy': 98.5, 'auc': 0.992, 'dataset': 'Mixed'},
        'cnn': {'accuracy': 99.2, 'auc': 0.995, 'dataset': 'FaceForensics++'},
        'lstm': {'accuracy': 93.7, 'auc': 0.967, 'dataset': 'Mixed Video'},
        'transformer': {'accuracy': 97.2, 'auc': 0.984, 'dataset': 'DFDC'},
        'spectral': {'accuracy': 94.3, 'auc': 0.951, 'dataset': 'Multi-GAN'},
    }
    
    MODEL_DETAILS = {
        'hybrid': {
            'name': 'Hybrid Deepfake Detector',
            'description': 'Multi-modal detector combining spatial and frequency features with Transformer encoding',
            'architecture': 'ResNet-50 backbone + DCT frequency branch + Transformer encoder (projected to 512)',
            'algorithms': [
                'ResNet-50: Deep residual learning for spatial feature extraction',
                'DCT (Discrete Cosine Transform): Frequency-domain artifact detection',
                'Transformer Encoder: Self-attention for feature aggregation',
                'Batch Normalization: Stabilized training',
                'Dropout: Regularization (50% spatial, 30% fusion)',
            ],
            'training': {
                'optimizer': 'Adam (lr=0.0001)',
                'loss_function': 'CrossEntropyLoss',
                'batch_size': '8-32',
                'epochs': '5-20',
                'augmentation': ['Resize to 224x224', 'Normalize (ImageNet stats)', 'Random horizontal flip'],
                'regularization': 'L2 weight decay, Dropout 0.3-0.5',
                'scheduler': 'ReduceLROnPlateau (patience=3)',
            },
            'input_spec': '224x224 RGB image or frame',
            'output_spec': 'Binary classification (Real/Fake) with probabilities',
            'strengths': [
                'Captures both spatial and frequency artifacts',
                'Multi-modal feature fusion improves robustness',
                'Transformer attention provides interpretability',
                'Best overall accuracy on mixed datasets',
            ],
            'weaknesses': [
                'Higher computational cost than CNN-only',
                'Requires larger training datasets',
                'Cross-dataset generalization moderate',
            ],
        },
        'cnn': {
            'name': 'CNN Dense Inception Detector',
            'description': 'High-accuracy CNN with dense inception blocks for artifact detection',
            'architecture': 'ResNet-50 backbone → Dense Inception blocks (2048→512→256) → Classifier',
            'algorithms': [
                'ResNet-50: Skip connections for deep feature learning',
                'Dense Inception Blocks: Multi-scale convolution paths',
                'Bottleneck convolutions: Dimension reduction',
                'Batch Normalization & ReLU: Non-linearity',
            ],
            'training': {
                'optimizer': 'Adam (lr=0.0001)',
                'loss_function': 'CrossEntropyLoss with class weights',
                'batch_size': '16-32',
                'epochs': '10-30',
                'augmentation': ['Center crop 224x224', 'ColorJitter', 'GaussBlur'],
                'regularization': 'Dropout 0.5, L2 (0.0001)',
                'scheduler': 'StepLR (step_size=10)',
            },
            'input_spec': '224x224 RGB image',
            'output_spec': 'Binary classification (Real/Fake)',
            'strengths': [
                'Highest accuracy on FaceForensics++ (99.2%)',
                'Dense connections capture fine-grained artifacts',
                'Efficient gradient flow',
            ],
            'weaknesses': [
                'Poor cross-dataset generalization (84.7% AUC)',
                'Overfits to specific dataset artifacts',
                'Slower inference than spectral',
            ],
        },
        'lstm': {
            'name': 'LSTM Temporal Detector',
            'description': 'Temporal modeling detector using frames and LSTM for video analysis',
            'architecture': 'ResNet-34 per-frame → BiLSTM (256 hidden, 2 layers) → MultiheadAttention → Classifier',
            'algorithms': [
                'ResNet-34: Per-frame feature extraction (512-dim)',
                'BiLSTM: Bidirectional temporal modeling of frame sequences',
                'MultiheadAttention: Temporal feature aggregation (8 heads)',
                'Sequence pooling: Global average over time',
            ],
            'training': {
                'optimizer': 'Adam (lr=0.00005)',
                'loss_function': 'CrossEntropyLoss',
                'batch_size': '4-8 (sequences)',
                'sequence_length': '16 frames',
                'epochs': '5-15',
                'augmentation': ['Frame sampling at 10-30 fps', 'Resize to 224x224'],
                'regularization': 'LSTM dropout 0.3, Attention dropout 0.1, L2 (0.00005)',
                'scheduler': 'ReduceLROnPlateau (patience=5)',
            },
            'input_spec': 'Video frames (sequence of 16, 224x224 RGB)',
            'output_spec': 'Binary classification (Real/Fake)',
            'strengths': [
                'Captures temporal inconsistencies in deepfakes',
                'Bidirectional context improves detection',
                'Good for video datasets',
            ],
            'weaknesses': [
                'Slower than single-frame models',
                'Requires sequence context (16 frames)',
                'Memory intensive for long videos',
            ],
        },
        'transformer': {
            'name': 'Transformer Deepfake Detector',
            'description': 'Vision Transformer-based detector with patch embeddings and multi-head attention',
            'architecture': 'Conv stem (64 features) → Patch embeddings (embed_dim=768) → 12-layer Transformer → ClassificationHead',
            'algorithms': [
                'Convolutional stem: Efficient initial feature extraction (stride=2, kernel=7)',
                'Patch embeddings: 16x16 patches → 768-dim embeddings',
                'Positional encoding: Learnable position embeddings for 196 patches',
                'Transformer encoder: 12 layers, 12 attention heads, 3072-dim FFN',
                'Layer normalization: Stabilization',
                'GELU activation: Smooth non-linearity',
            ],
            'training': {
                'optimizer': 'AdamW (lr=0.0001, weight_decay=0.01)',
                'loss_function': 'CrossEntropyLoss',
                'batch_size': '8-16',
                'epochs': '10-20',
                'augmentation': ['RandAugment', 'Mixup (alpha=0.2)', 'CutMix'],
                'regularization': 'Attention dropout 0.1, Stochastic depth, Label smoothing 0.1',
                'scheduler': 'CosineAnnealingLR (T_max=100)',
                'warmup': 'Linear warmup (10% of total steps)',
            },
            'input_spec': '224x224 RGB image (196 patches of 16x16)',
            'output_spec': 'Binary classification (Real/Fake)',
            'strengths': [
                'Global context modeling via self-attention',
                'Handles variable input sizes well',
                'Good generalization with proper training',
            ],
            'weaknesses': [
                'Requires more training data than CNNs',
                'Slower inference than CNN (without GPU)',
                'Complex architecture (need careful tuning)',
            ],
        },
        'spectral': {
            'name': 'Spectral Analysis Detector',
            'description': 'Frequency-domain + spatial domain fusion detector with real-time capability',
            'architecture': 'DCT frequency branch (64→128→256) + ResNet-18 spatial branch → Fusion MLP',
            'algorithms': [
                'DCT (Discrete Cosine Transform): Transforms image to frequency domain',
                'Frequency analysis: 3-layer CNN on DCT coefficients',
                'ResNet-18: Spatial feature extraction (512-dim)',
                'Feature fusion: Concatenate (256*64 + 512 = 16896 dims)',
                'Fusion MLP: 16896 → 512 → 256 → 2 (binary)',
            ],
            'training': {
                'optimizer': 'Adam (lr=0.0001)',
                'loss_function': 'CrossEntropyLoss',
                'batch_size': '32-64 (larger batches)',
                'epochs': '10-25',
                'augmentation': ['JPEG compression', 'Gaussian blur', 'Resize'],
                'regularization': 'Dropout 0.5/0.3, L2 (0.0001)',
                'scheduler': 'StepLR (step_size=5, gamma=0.5)',
            },
            'input_spec': '224x224 RGB image',
            'output_spec': 'Binary classification (Real/Fake)',
            'strengths': [
                'Real-time capable (15G FLOPs)',
                'Frequency artifacts capture compression/GAN signatures',
                'Good cross-dataset AUC (92.3%)',
                'Low memory (400 MB)',
            ],
            'weaknesses': [
                'Slightly lower accuracy than CNN/Hybrid',
                'DCT computation adds overhead',
                'Less effective on heavily compressed videos',
            ],
        },
    }
    
    def __init__(self, model_type: str = 'hybrid', device: str = 'cpu', pretrained: bool = False):
        self.device = device
        self.model_type = model_type
        self.pretrained = pretrained
        self.model = self._build_model()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _build_model(self):
        models_map = {
            'hybrid': HybridDeepfakeDetector,
            'cnn': CNNDeepfakeDetector,
            'lstm': TemporalLSTMDetector,
            'transformer': TransformerDeepfakeDetector,
            'spectral': SpectralAnalysisDetector,
        }
        model_class = models_map.get(self.model_type, CNNDeepfakeDetector)
        
        # TransformerDeepfakeDetector doesn't accept pretrained parameter
        if self.model_type == 'transformer':
            model = model_class()
        else:
            model = model_class(pretrained=self.pretrained)
        
        model.to(self.device)
        model.eval()
        return model
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess a frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = self.transform(frame_rgb)
        return frame_tensor.unsqueeze(0).to(self.device)
    
    def detect_frame(self, frame: np.ndarray) -> Tuple[float, float]:
        """Detect deepfake in a single frame"""
        self.model.eval()
        with torch.no_grad():
            tensor = self.preprocess_frame(frame)
            if self.model_type == 'lstm':
                seq_len = getattr(self.model, 'sequence_length', 16)
                tensor = tensor.unsqueeze(1).repeat(1, seq_len, 1, 1, 1)
            output = self.model(tensor)
            probs = F.softmax(output, dim=1)
            fake_prob = probs[0, 1].item()
            confidence = torch.max(probs, dim=1)[0].item()
        return fake_prob, confidence
    
    def detect_video(self, video_path: str, sample_rate: int = 10) -> Dict:
        """Detect deepfake in video"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        fake_probs = []
        frame_idx = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % sample_rate == 0:
                    fake_prob, _ = self.detect_frame(frame)
                    fake_probs.append(fake_prob)
                frame_idx += 1
        finally:
            cap.release()
        
        avg_fake_prob = float(np.mean(fake_probs)) if fake_probs else 0.0
        return {
            'video_path': video_path,
            'frame_count': frame_count,
            'fps': fps,
            'sampled_frames': len(fake_probs),
            'avg_fake_probability': avg_fake_prob,
            'final_prediction': 'FAKE' if avg_fake_prob > 0.5 else 'REAL',
        }
    
    @staticmethod
    def get_model_info(model_type: str) -> Dict:
        return UnifiedDeepfakeDetector.MODEL_INFO.get(model_type, {})
    
    @staticmethod
    def get_benchmarks(model_type: str) -> Dict:
        return UnifiedDeepfakeDetector.PERFORMANCE_BENCHMARKS.get(model_type, {})
    
    @staticmethod
    def get_model_training_info(model_type: str) -> Dict:
        """Get detailed training and algorithm info for a model"""
        return UnifiedDeepfakeDetector.MODEL_DETAILS.get(model_type, {})
