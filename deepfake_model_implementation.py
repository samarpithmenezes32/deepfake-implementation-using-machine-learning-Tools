"""
Deepfake Detection Model Implementation Examples
Based on comprehensive literature review findings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
from typing import Tuple, List, Dict
import cv2

class CNNDeepfakeDetector(nn.Module):
    """
    CNN-based deepfake detector following Alharbi et al. (2025) approach
    Dense Inception Network architecture for spatial artifact detection
    """
    def __init__(self, num_classes: int = 2, pretrained: bool = False):
        super(CNNDeepfakeDetector, self).__init__()
        
        # Use ResNet-50 as backbone (balance of accuracy and efficiency)
        try:
            # Newer torchvision API
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        except Exception:
            self.backbone = models.resnet50(pretrained=pretrained)
        
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Dense inception blocks for fine-grained feature extraction
        self.dense_block1 = self._make_dense_inception_block(2048, 512)
        self.dense_block2 = self._make_dense_inception_block(512, 256)
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(256, num_classes)
        
    def _make_dense_inception_block(self, in_channels: int, out_channels: int):
        """Create dense inception block for artifact detection"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, 1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//4, out_channels//2, 3, padding=1),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract backbone features
        features = self.backbone(x)
        
        # Apply dense inception blocks
        x = self.dense_block1(features)
        x = self.dense_block2(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x

class TemporalLSTMDetector(nn.Module):
    """
    LSTM-based temporal deepfake detector following Muruganandham et al. (2025)
    Combines CNN spatial features with LSTM temporal modeling
    """
    def __init__(self, num_classes: int = 2, sequence_length: int = 16, pretrained: bool = False):
        super(TemporalLSTMDetector, self).__init__()
        
        self.sequence_length = sequence_length
        
        # CNN feature extractor for each frame
        try:
            self.cnn_backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        except Exception:
            self.cnn_backbone = models.resnet34(pretrained=pretrained)
        self.cnn_backbone.fc = nn.Linear(self.cnn_backbone.fc.in_features, 512)
        
        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Attention mechanism for sequence aggregation
        self.attention = nn.MultiheadAttention(
            embed_dim=512,  # bidirectional LSTM output
            num_heads=8,
            dropout=0.1
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Reshape for CNN processing
        x = x.view(batch_size * seq_len, *x.shape[2:])
        
        # Extract CNN features for each frame
        cnn_features = self.cnn_backbone(x)  # (batch_size * seq_len, 512)
        
        # Reshape back to sequence format
        cnn_features = cnn_features.view(batch_size, seq_len, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(cnn_features)  # (batch_size, seq_len, 512)
        
        # Attention-based aggregation
        lstm_out = lstm_out.transpose(0, 1)  # (seq_len, batch_size, 512)
        attended_features, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling over sequence dimension
        pooled_features = attended_features.mean(dim=0)  # (batch_size, 512)
        
        # Final classification
        output = self.classifier(pooled_features)
        
        return output

class TransformerDeepfakeDetector(nn.Module):
    """
    Transformer-based detector following Alattas et al. (2025) CoAtNet approach
    Combines convolutional efficiency with transformer expressiveness
    """
    def __init__(self, num_classes: int = 2, patch_size: int = 16, embed_dim: int = 768):
        super(TransformerDeepfakeDetector, self).__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Convolutional stem for efficient low-level feature extraction
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Patch embedding layer
        self.patch_embed = nn.Conv2d(64, embed_dim, patch_size, stride=patch_size)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, 196, embed_dim))  # For 224x224 input
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=12,
            dim_feedforward=3072,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=12)
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional stem
        x = self.conv_stem(x)  # (B, 64, H/4, W/4)
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        
        # Flatten spatial dimensions
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Transformer processing
        x = x.transpose(0, 1)  # (num_patches, B, embed_dim)
        x = self.transformer(x)
        x = x.transpose(0, 1)  # (B, num_patches, embed_dim)
        
        # Global average pooling
        x = x.mean(dim=1)  # (B, embed_dim)
        
        # Final classification
        x = self.norm(x)
        x = self.classifier(x)
        
        return x

class SpectralAnalysisDetector(nn.Module):
    """
    Spectral analysis detector following Huang et al. (2022) approach
    Exploits frequency-domain artifacts for robust detection
    """
    def __init__(self, num_classes: int = 2, pretrained: bool = False):
        super(SpectralAnalysisDetector, self).__init__()
        
        # Frequency analysis branch
        self.freq_branch = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(8)
        )
        
        # Spatial analysis branch
        try:
            self.spatial_branch = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        except Exception:
            self.spatial_branch = models.resnet18(pretrained=pretrained)
        self.spatial_branch.fc = nn.Identity()
        
        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(256 * 64 + 512, 512),  # freq features + spatial features
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def apply_dct(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DCT transform for frequency analysis"""
        # Convert to numpy for DCT processing
        x_np = x.cpu().numpy()
        dct_features = []
        
        for batch_idx in range(x_np.shape[0]):
            img = x_np[batch_idx].transpose(1, 2, 0)  # CHW to HWC
            
            # Apply DCT to each channel
            dct_channels = []
            for c in range(3):
                dct_c = cv2.dct(img[:, :, c].astype(np.float32))
                dct_channels.append(dct_c)
            
            dct_img = np.stack(dct_channels, axis=2)
            dct_features.append(dct_img.transpose(2, 0, 1))  # HWC to CHW
        
        return torch.tensor(np.array(dct_features), dtype=x.dtype, device=x.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frequency domain analysis
        freq_input = self.apply_dct(x)
        freq_features = self.freq_branch(freq_input)
        freq_features = torch.flatten(freq_features, 1)
        
        # Spatial domain analysis
        spatial_features = self.spatial_branch(x)
        
        # Feature fusion and classification
        combined_features = torch.cat([freq_features, spatial_features], dim=1)
        output = self.fusion(combined_features)
        
        return output

class DeepfakeDetectionPipeline:
    """
    Complete deepfake detection pipeline with preprocessing and evaluation
    """
    def __init__(self, model_type: str = 'cnn', device: str = 'auto'):
        # Auto-select device if not provided or set to auto
        if device is None or device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device == 'cuda' and not torch.cuda.is_available():
            print("[Info] CUDA not available. Falling back to CPU.")
            device = 'cpu'
        self.device = device
        self.model_type = model_type
        
        # Initialize model based on type
        if model_type == 'cnn':
            self.model = CNNDeepfakeDetector(pretrained=False)
        elif model_type == 'lstm':
            self.model = TemporalLSTMDetector(pretrained=False)
        elif model_type == 'transformer':
            self.model = TransformerDeepfakeDetector()
        elif model_type == 'spectral':
            self.model = SpectralAnalysisDetector(pretrained=False)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.to(device)
        
        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess a single frame for detection"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        frame_tensor = self.transform(frame_rgb)
        
        return frame_tensor.unsqueeze(0)  # Add batch dimension
    
    def detect_frame(self, frame: np.ndarray) -> Tuple[float, float]:
        """
        Detect deepfake in a single frame
        Returns: (fake_probability, confidence_score)
        """
        self.model.eval()
        
        with torch.no_grad():
            frame_tensor = self.preprocess_frame(frame).to(self.device)
            
            # Get model predictions
            outputs = self.model(frame_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            fake_prob = probabilities[0, 1].item()  # Probability of fake
            confidence = torch.max(probabilities, dim=1)[0].item()
            
        return fake_prob, confidence
    
    def detect_video(self, video_path: str, sample_rate: int = 10) -> Dict:
        """
        Detect deepfake in video by sampling frames
        Args:
            video_path: Path to video file
            sample_rate: Sample every N frames
        Returns:
            Dictionary with detection results
        """
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        fake_probs = []
        confidences = []
        frame_indices = []
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                fake_prob, confidence = self.detect_frame(frame)
                fake_probs.append(fake_prob)
                confidences.append(confidence)
                frame_indices.append(frame_idx)
            
            frame_idx += 1
        
        cap.release()
        
        # Aggregate results
        avg_fake_prob = np.mean(fake_probs)
        avg_confidence = np.mean(confidences)
        max_fake_prob = np.max(fake_probs)
        
        # Temporal smoothing
        smoothed_probs = self._temporal_smoothing(fake_probs)
        final_prediction = np.mean(smoothed_probs) > 0.5
        
        return {
            'video_path': video_path,
            'model_type': self.model_type,
            'frame_count': frame_count,
            'fps': fps,
            'sampled_frames': len(fake_probs),
            'avg_fake_probability': avg_fake_prob,
            'max_fake_probability': max_fake_prob,
            'avg_confidence': avg_confidence,
            'final_prediction': 'FAKE' if final_prediction else 'REAL',
            'frame_predictions': list(zip(frame_indices, fake_probs, confidences))
        }
    
    def _temporal_smoothing(self, probs: List[float], window_size: int = 5) -> List[float]:
        """Apply temporal smoothing to frame predictions"""
        if len(probs) <= window_size:
            return probs
        
        smoothed = []
        for i in range(len(probs)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(probs), i + window_size // 2 + 1)
            smoothed.append(np.mean(probs[start_idx:end_idx]))
        
        return smoothed

def evaluate_model_performance():
    """
    Performance comparison based on literature review findings
    """
    performance_data = {
        'CNN (Dense Inception)': {
            'dataset': 'FaceForensics++',
            'accuracy': 99.2,
            'auc': 0.995,
            'cross_dataset_auc': 0.847,
            'real_time': False,
            'flops_per_frame': '50G',
            'memory_mb': 800
        },
        'Spectral Analysis': {
            'dataset': 'Multi-GAN',
            'accuracy': 94.3,
            'auc': 0.951,
            'cross_dataset_auc': 0.923,
            'real_time': True,
            'flops_per_frame': '15G',
            'memory_mb': 400
        },
        'Self-Consistency CNN': {
            'dataset': 'DFDC',
            'accuracy': 95.8,
            'auc': 0.981,
            'cross_dataset_auc': 0.922,
            'real_time': True,
            'flops_per_frame': '20G',
            'memory_mb': 500
        },
        'CoAtNet Transformer': {
            'dataset': 'DFDC',
            'accuracy': 97.2,
            'auc': 0.984,
            'cross_dataset_auc': 0.889,
            'real_time': False,
            'flops_per_frame': '80G',
            'memory_mb': 1200
        },
        'Multiscale Features': {
            'dataset': 'Celeb-DF',
            'accuracy': 96.1,
            'auc': 0.976,
            'cross_dataset_auc': 0.901,
            'real_time': True,
            'flops_per_frame': '25G',
            'memory_mb': 600
        },
        'LSTM Temporal': {
            'dataset': 'Mixed Video',
            'accuracy': 93.7,
            'auc': 0.967,
            'cross_dataset_auc': 0.885,
            'real_time': False,
            'flops_per_frame': '35G',
            'memory_mb': 700
        }
    }
    
    return performance_data

# Example usage
if __name__ == "__main__":
    # Initialize detection pipeline (auto-select device, avoid big downloads by default)
    detector = DeepfakeDetectionPipeline(model_type='cnn', device='auto')
    
    # Example frame detection
    # frame = cv2.imread('sample_face.jpg')
    # fake_prob, confidence = detector.detect_frame(frame)
    # print(f"Fake probability: {fake_prob:.3f}, Confidence: {confidence:.3f}")
    
    # Example video detection
    # results = detector.detect_video('sample_video.mp4')
    # print(f"Video prediction: {results['final_prediction']}")
    # print(f"Average fake probability: {results['avg_fake_probability']:.3f}")
    
    # Display performance comparison
    performance = evaluate_model_performance()
    print("\nModel Performance Comparison:")
    print("-" * 80)
    for model, metrics in performance.items():
        print(f"{model}:")
        print(f"  Dataset: {metrics['dataset']}")
        print(f"  Accuracy: {metrics['accuracy']:.1f}%")
        print(f"  AUC: {metrics['auc']:.3f}")
        print(f"  Cross-dataset AUC: {metrics['cross_dataset_auc']:.3f}")
        print(f"  Real-time: {metrics['real_time']}")
        print(f"  FLOPs/frame: {metrics['flops_per_frame']}")
        print(f"  Memory: {metrics['memory_mb']} MB")
        print()