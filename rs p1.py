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
import argparse
import os

class CNNDeepfakeDetector(nn.Module):
    """
    CNN-based deepfake detector following Alharbi et al. (2025) approach
    Dense Inception Network architecture for spatial artifact detection
    """
    def __init__(self, num_classes: int = 2, pretrained: bool = False):
        
        super(CNNDeepfakeDetector, self).__init__()
        
        # Use ResNet-50 as backbone (balance of accuracy and efficiency)
        # Torchvision API compatibility (weights vs pretrained)
        try:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        except Exception:
            # Fallback for older torchvision
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
    def __init__(self, num_classes: int = 2, patch_size: int = 16, embed_dim: int = 768, num_layers: int = 12, nhead: int = 12, ff_dim: int = 3072):
        super(TransformerDeepfakeDetector, self).__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.nhead = nhead
        self.ff_dim = ff_dim

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
        # For 224x224 input; will be sliced/expanded if patch number differs
        self.pos_embed = nn.Parameter(torch.randn(1, 196, embed_dim))

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

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

        # Add positional encoding (handle variable number of patches)
        if self.pos_embed.size(1) != x.size(1):
            # Interpolate positional encodings if count differs
            if self.pos_embed.size(1) > x.size(1):
                pe = self.pos_embed[:, :x.size(1), :]
            else:
                # Repeat to match (simple fallback)
                repeat_times = (x.size(1) + self.pos_embed.size(1) - 1) // self.pos_embed.size(1)
                pe = self.pos_embed.repeat(1, repeat_times, 1)[:, :x.size(1), :]
            x = x + pe
        else:
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
    def __init__(self, model_type: str = 'cnn', device: str | None = None, pretrained: bool = False, sequence_length: int = 16, face_crop: bool = False, tta: bool = False):
        # Auto-select device if not provided
        if device is None or device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device == 'cuda' and not torch.cuda.is_available():
            print("[Info] CUDA not available. Falling back to CPU.")
            device = 'cpu'

        self.device = device
        self.model_type = model_type
        self.face_crop = face_crop
        self.tta = tta
        # Initialize face detector if needed
        self._face_cascade = None
        if self.face_crop:
            try:
                self._face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            except Exception:
                self._face_cascade = None
        
        # Initialize model based on type
        if model_type == 'cnn':
            self.model = CNNDeepfakeDetector(pretrained=pretrained)
        elif model_type == 'lstm':
            self.model = TemporalLSTMDetector(sequence_length=sequence_length, pretrained=pretrained)
        elif model_type == 'transformer':
            # TransformerDeepfakeDetector doesn't use pretrained weights
            self.model = TransformerDeepfakeDetector()
        elif model_type == 'spectral':
            self.model = SpectralAnalysisDetector(pretrained=pretrained)
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
        # Optional face crop
        if self.face_crop and self._face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))
            if len(faces) > 0:
                # Choose the largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                pad = int(0.2 * max(w, h))
                x0 = max(0, x - pad)
                y0 = max(0, y - pad)
                x1 = min(frame.shape[1], x + w + pad)
                y1 = min(frame.shape[0], y + h + pad)
                frame = frame[y0:y1, x0:x1]

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
            def infer_one(img: np.ndarray) -> Tuple[float, float]:
                ft = self.preprocess_frame(img).to(self.device)
                if self.model_type == 'lstm':
                    seq_len = getattr(self.model, 'sequence_length', 16)
                    seq_tensor = ft.unsqueeze(1).repeat(1, seq_len, 1, 1, 1)
                    out = self.model(seq_tensor)
                else:
                    out = self.model(ft)
                probs = F.softmax(out, dim=1)
                return probs[0, 1].item(), torch.max(probs, dim=1)[0].item()

            if not self.tta:
                fake_prob, confidence = infer_one(frame)
            else:
                # Simple TTA: original + horizontal flip + slight JPEG re-encode
                variants = [frame]
                # hflip
                variants.append(cv2.flip(frame, 1))
                # jpeg re-encode at lower quality to simulate compression
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
                ok, enc = cv2.imencode('.jpg', frame, encode_param)
                if ok:
                    comp = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                    if comp is not None:
                        variants.append(comp)
                probs, confs = [], []
                for v in variants:
                    p, c = infer_one(v)
                    probs.append(p)
                    confs.append(c)
                fake_prob = float(np.mean(probs))
                confidence = float(np.mean(confs))
            
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
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fake_probs: List[float] = []
        confidences: List[float] = []
        frame_indices: List[int] = []

        frame_idx = 0
        try:
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
        finally:
            cap.release()

        # Aggregate results
        avg_fake_prob = float(np.mean(fake_probs)) if fake_probs else 0.0
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        max_fake_prob = float(np.max(fake_probs)) if fake_probs else 0.0

        # Temporal smoothing
        smoothed_probs = self._temporal_smoothing(fake_probs) if fake_probs else []
        final_prediction = (np.mean(smoothed_probs) > 0.5) if smoothed_probs else False

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

def _print_performance_table():
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake detection demo")
    parser.add_argument("--model-type", choices=["cnn", "lstm", "transformer", "spectral"], default="cnn")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--image", type=str, help="Path to an image for single-frame detection")
    parser.add_argument("--video", type=str, help="Path to a video for detection")
    parser.add_argument("--sample-rate", type=int, default=10, help="Sample every N frames for video")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights (will download if not cached)")
    parser.add_argument("--sequence-length", type=int, default=16, help="Sequence length for LSTM model")
    parser.add_argument("--face-crop", action="store_true", help="Detect and crop face region before inference")
    parser.add_argument("--tta", action="store_true", help="Enable simple test-time augmentation ensemble")
    parser.add_argument("--all-models", action="store_true", help="Run all model types sequentially on the provided inputs")
    # Transformer tuning and speed options
    parser.add_argument("--tr-embed-dim", type=int, default=768, help="Transformer embed dimension")
    parser.add_argument("--tr-layers", type=int, default=12, help="Transformer encoder layers")
    parser.add_argument("--tr-heads", type=int, default=12, help="Transformer attention heads")
    parser.add_argument("--tr-ff", type=int, default=3072, help="Transformer feed-forward dimension")
    parser.add_argument("--fast", action="store_true", help="Use lighter transformer and higher sample-rate for speed")
    args = parser.parse_args()

    def run_once(selected_model: str) -> bool:
        """Instantiate a detector for selected_model and run on provided inputs. Returns True if anything ran."""
        # Derive fast settings
        tr_embed = getattr(args, 'tr_embed_dim', 768)
        tr_layers = getattr(args, 'tr_layers', 12)
        tr_heads = getattr(args, 'tr_heads', 12)
        tr_ff = getattr(args, 'tr_ff', 3072)
        eff_sample_rate = args.sample_rate
        if getattr(args, 'fast', False):
            tr_embed = min(tr_embed, 384)
            tr_layers = min(tr_layers, 4)
            tr_heads = min(tr_heads, 6)
            tr_ff = min(tr_ff, 1024)
            eff_sample_rate = max(eff_sample_rate, 30)

        detector = DeepfakeDetectionPipeline(
            model_type=selected_model,
            device=args.device,
            pretrained=args.pretrained,
            sequence_length=args.sequence_length,
            face_crop=args.face_crop,
            tta=args.tta,
        )

        # If transformer, rebuild with lighter config
        if selected_model == 'transformer':
            detector.model = TransformerDeepfakeDetector(
                num_classes=2,
                patch_size=16,
                embed_dim=tr_embed,
                num_layers=tr_layers,
                nhead=tr_heads,
                ff_dim=tr_ff,
            ).to(detector.device)

        ran = False
        if args.image:
            if not os.path.exists(args.image):
                print(f"[Error] Image not found: {args.image}")
            else:
                img = cv2.imread(args.image)
                fake_prob, confidence = detector.detect_frame(img)
                print(f"\n[IMAGE] {args.image}")
                print(f"  Model: {selected_model}")
                print(f"  Device: {detector.device}")
                print(f"  Fake probability: {fake_prob:.3f}")
                print(f"  Confidence: {confidence:.3f}")
                ran = True

        if args.video:
            try:
                results = detector.detect_video(args.video, sample_rate=eff_sample_rate)
                print(f"\n[VIDEO] {args.video}")
                print(f"  Model: {selected_model}")
                print(f"  Device: {detector.device}")
                print(f"  Frames: {results['frame_count']} (sampled: {results['sampled_frames']}) at {results['fps']:.2f} fps")
                print(f"  Final prediction: {results['final_prediction']}")
                print(f"  Avg fake probability: {results['avg_fake_probability']:.3f}")
                print(f"  Max fake probability: {results['max_fake_probability']:.3f}")
                print(f"  Avg confidence: {results['avg_confidence']:.3f}")
                ran = True
            except Exception as e:
                print(f"[Error] Video detection failed for model {selected_model}: {e}")

        return ran

    ran_any = False
    if args.all_models:
        models_to_run = ["cnn", "lstm", "transformer", "spectral"]
        print("Running all models:\n  - " + "\n  - ".join(models_to_run))
        for m in models_to_run:
            ran_any = run_once(m) or ran_any
    else:
        ran_any = run_once(args.model_type)

    if not ran_any:
        # Default: print performance table only
        _print_performance_table()