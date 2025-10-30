"""
Deepfake Detection - Unified CLI (Hybrid, CNN, LSTM, Transformer, Spectral)
- No webcam; dataset-only in input/real and input/fake.
- Optional: download Kaggle dataset nanduncs/1000-videos-split into input/.
"""

import os
import argparse
from glob import glob
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models

# --------------- Models ---------------
class HybridDeepfakeDetector(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.cnn_backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.cnn_backbone.fc = nn.Identity()
        self.freq_branch = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((7,7))
        )
        self.proj = nn.Linear(2048 + 128*7*7, 512)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, activation="gelu")
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.fc = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(512, num_classes))

    @torch.no_grad()
    def apply_dct(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().numpy()
        dct_features = []
        for img in x_np:
            img = img.transpose(1, 2, 0)
            dct_channels = [cv2.dct(img[:, :, c].astype(np.float32)) for c in range(3)]
            dct_img = np.stack(dct_channels, axis=2)
            dct_features.append(dct_img.transpose(2, 0, 1))
        return torch.tensor(np.array(dct_features), dtype=x.dtype, device=x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_features = self.cnn_backbone(x)
        freq_input = self.apply_dct(x)
        freq_features = self.freq_branch(freq_input)
        freq_features = torch.flatten(freq_features, 1)
        combined = torch.cat([spatial_features, freq_features], dim=1)
        combined = self.proj(combined)
        seq = combined.unsqueeze(1).repeat(1, 5, 1).transpose(0, 1)
        attn_out = self.transformer(seq).mean(dim=0)
        return self.fc(attn_out)

class CNNDeepfakeDetector(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = False):
        super().__init__()
        try:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        except Exception:
            self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.d1 = self._block(2048, 512)
        self.d2 = self._block(512, 256)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.5)
        self.cls = nn.Linear(256, num_classes)
    def _block(self, c1, c2):
        return nn.Sequential(
            nn.Conv2d(c1, c2//4, 1), nn.BatchNorm2d(c2//4), nn.ReLU(inplace=True),
            nn.Conv2d(c2//4, c2//2, 3, padding=1), nn.BatchNorm2d(c2//2), nn.ReLU(inplace=True),
            nn.Conv2d(c2//2, c2, 1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True)
        )
    def forward(self, x):
        f = self.backbone(x)
        x = self.d1(f); x = self.d2(x)
        x = self.gap(x); x = torch.flatten(x, 1); x = self.drop(x); return self.cls(x)

class TemporalLSTMDetector(nn.Module):
    def __init__(self, num_classes: int = 2, sequence_length: int = 8, pretrained: bool = False):
        super().__init__()
        try:
            self.cnn = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        except Exception:
            self.cnn = models.resnet34(pretrained=pretrained)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 512)
        self.lstm = nn.LSTM(512, 256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.1)
        self.cls = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes))
    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t = x.size(0), x.size(1)
        x = x.view(b*t, *x.shape[2:])
        f = self.cnn(x)  # (b*t, 512)
        f = f.view(b, t, -1)
        y, _ = self.lstm(f)
        y = y.transpose(0, 1)  # (T, B, 512)
        y, _ = self.attn(y, y, y)
        y = y.mean(dim=0)  # (B, 512)
        return self.cls(y)

class TransformerDeepfakeDetector(nn.Module):
    def __init__(self, num_classes: int = 2, patch_size: int = 16, embed_dim: int = 384):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.patch = nn.Conv2d(64, embed_dim, patch_size, stride=patch_size)
        self.pos = nn.Parameter(torch.randn(1, 196, embed_dim))
        enc = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=1536, dropout=0.1, activation='gelu')
        self.tr = nn.TransformerEncoder(enc, num_layers=8)
        self.norm = nn.LayerNorm(embed_dim)
        self.cls = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        x = self.conv(x); x = self.patch(x)
        b, c, h, w = x.shape; x = x.flatten(2).transpose(1, 2)
        pe = self.pos
        if pe.size(1) != x.size(1):
            if pe.size(1) > x.size(1): pe = pe[:, :x.size(1), :]
            else:
                reps = (x.size(1) + pe.size(1) - 1)//pe.size(1); pe = pe.repeat(1, reps, 1)[:, :x.size(1), :]
        x = x + pe; x = x.transpose(0, 1); x = self.tr(x); x = x.transpose(0, 1)
        x = x.mean(dim=1); x = self.norm(x); return self.cls(x)

class SpectralAnalysisDetector(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = False):
        super().__init__()
        self.freq = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(8)
        )
        try:
            self.spatial = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        except Exception:
            self.spatial = models.resnet18(pretrained=pretrained)
        self.spatial.fc = nn.Identity()
        self.fuse = nn.Sequential(
            nn.Linear(256*64 + 512, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )
    def apply_dct(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.cpu().numpy(); out=[]
        for img in x_np:
            img = img.transpose(1,2,0)
            d = [cv2.dct(img[:,:,c].astype(np.float32)) for c in range(3)]
            out.append(np.stack(d,axis=2).transpose(2,0,1))
        return torch.tensor(np.array(out), dtype=x.dtype, device=x.device)
    def forward(self, x):
        fi = self.apply_dct(x); ff = self.freq(fi); ff = torch.flatten(ff,1)
        sf = self.spatial(x)
        z = torch.cat([ff, sf], 1)
        return self.fuse(z)

# --------------- Data ---------------
IMAGENET_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, img_size: int = 224, max_items: int = 100):
        self.samples: List[Tuple[str,int]] = []
        img_exts = (".jpg",".jpeg",".png",".bmp",".webp")
        for label, cls in enumerate(["real","fake"]):
            class_root = os.path.join(root, cls)
            for dirpath, dirnames, filenames in os.walk(class_root):
                for f in filenames:
                    if f.lower().endswith(img_exts):
                        self.samples.append((os.path.join(dirpath, f), label))
        # cap to max_items per class
        capped = {0:0,1:0}; new=[]
        for p,l in self.samples:
            if capped[l] < max_items:
                new.append((p,l)); capped[l]+=1
        self.samples=new
        self.tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize((img_size,img_size)), transforms.ToTensor(), IMAGENET_NORM])
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p,l = self.samples[idx]
        im = cv2.imread(p); im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return self.tf(im), torch.tensor(l,dtype=torch.long)

class VideoDataset(torch.utils.data.Dataset):
    exts = (".mp4",".mov",".avi",".mkv")
    def __init__(self, root: str, img_size: int = 224, frames_per_video: int = 8, max_items: int = 100):
        # entries: list of (path, label, kind) where kind in {"video","dir"}
        entries: List[Tuple[str,int,str]] = []
        for label, cls in enumerate(["real","fake"]):
            class_root = os.path.join(root, cls)
            # collect video files recursively
            vids: List[str] = []
            for ext in self.exts:
                vids += glob(os.path.join(class_root, "**", "*"+ext), recursive=True)
            entries += [(p, label, "video") for p in vids]
            # collect frame directories (folders containing images)
            for dirpath, dirnames, filenames in os.walk(class_root):
                if any(f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp")) for f in filenames):
                    entries.append((dirpath, label, "dir"))
        # Cap per class while preferring videos first then dirs
        capped = {0:0, 1:0}
        videos_first = [e for e in entries if e[2] == "video"] + [e for e in entries if e[2] == "dir"]
        selected: List[Tuple[str,int,str]] = []
        seen_paths = set()
        for p, l, k in videos_first:
            if (p,l) in seen_paths: continue
            if capped[l] < max_items:
                selected.append((p,l,k)); capped[l] += 1; seen_paths.add((p,l))
        self.entries = selected
        self.tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize((img_size,img_size)), transforms.ToTensor(), IMAGENET_NORM])
        self.frames_per_video = frames_per_video
        self.img_size = img_size
    def __len__(self):
        return len(self.entries)
    def _read_video(self, vpath: str):
        cap = cv2.VideoCapture(vpath)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        frames=[]
        for i in range(self.frames_per_video):
            pos = int((i+1)*(total/(self.frames_per_video+1)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0,pos))
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self.tf(frame))
        cap.release()
        return frames
    def _read_dir(self, dpath: str):
        imgs = []
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
            imgs += glob(os.path.join(dpath, ext))
        imgs = sorted(imgs)
        total = len(imgs) or 1
        frames=[]
        for i in range(self.frames_per_video):
            pos = int((i+1)*(total/(self.frames_per_video+1)))
            pos = min(max(pos,0), total-1)
            impath = imgs[pos] if imgs else None
            if impath and os.path.isfile(impath):
                im = cv2.imread(impath)
                if im is None: continue
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                frames.append(self.tf(im))
        return frames
    def __getitem__(self, idx):
        path, label, kind = self.entries[idx]
        if kind == "video":
            frames = self._read_video(path)
        else:
            frames = self._read_dir(path)
        if not frames:
            frames = [torch.zeros(3, self.img_size, self.img_size)]
        seq = torch.stack(frames, dim=0)
        return seq, torch.tensor(label, dtype=torch.long)

# --------------- Train/Eval ---------------

def train_one(model: nn.Module, loader, device: str, is_seq: bool):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=3e-4); crit = nn.CrossEntropyLoss()
    for x,y in loader:
        if is_seq:
            if isinstance(model, TemporalLSTMDetector):
                x = x.to(device)
            else:
                x = x.mean(dim=1).to(device)
        else:
            # image mode; if LSTM, add a singleton time dimension
            if isinstance(model, TemporalLSTMDetector):
                x = x.unsqueeze(1).to(device)
            else:
                x = x.to(device)
        y = y.to(device)
        opt.zero_grad(); logits = model(x); loss = crit(logits,y); loss.backward(); opt.step()

def eval_one(model: nn.Module, loader, device: str, is_seq: bool) -> float:
    model.eval(); correct=0; total=0
    with torch.no_grad():
        for x,y in loader:
            if is_seq:
                if isinstance(model, TemporalLSTMDetector):
                    x = x.to(device)
                else:
                    x = x.mean(dim=1).to(device)
            else:
                if isinstance(model, TemporalLSTMDetector):
                    x = x.unsqueeze(1).to(device)
                else:
                    x = x.to(device)
            y = y.to(device)
            logits = model(x); preds = torch.argmax(logits,1)
            correct += (preds==y).sum().item(); total += y.size(0)
    return (correct/total) if total>0 else 0.0

# --------------- Kaggle download helper ---------------

def maybe_download_kaggle() -> str:
    """Download nanduncs/1000-videos-split using kagglehub if available; return base path or ''."""
    try:
        import kagglehub  # type: ignore
    except Exception:
        return ""
    try:
        path = kagglehub.dataset_download("nanduncs/1000-videos-split")
        return path
    except Exception:
        return ""

# --------------- CLI ---------------

def run_single(args) -> float:
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    # Build dataset/loaders
    if args.use_videos:
        ds = VideoDataset(args.input, frames_per_video=args.frames_per_video, max_items=args.max_items)
        is_seq = True
    else:
        ds = ImageDataset(args.input, max_items=args.max_items)
        is_seq = False
    if len(ds) < 2:
        print("Not enough data; please populate input/real and input/fake.")
        return 0.0
    n_val = max(1, int(0.2*len(ds))); n_train = len(ds)-n_val
    tr, va = torch.utils.data.random_split(ds, [n_train, n_val])
    print(f"Dataset size: total={len(ds)} | train={n_train} | val={n_val}")
    # Safe batch sizes for tiny datasets
    train_bs = min(args.batch_size, max(1, len(tr)))
    val_bs = min(args.batch_size, max(1, len(va)))
    drop_last = train_bs > 1
    tr_loader = None
    if n_train >= 2:
        tr_loader = torch.utils.data.DataLoader(tr, batch_size=max(2, train_bs), shuffle=True, drop_last=True)
    else:
        print("Too few training samples (<2); skipping training and evaluating only.")
    va_loader = torch.utils.data.DataLoader(va, batch_size=val_bs, shuffle=False)

    # Model selection
    if args.model == "hybrid":
        model = HybridDeepfakeDetector(num_classes=2, pretrained=args.pretrained)
    elif args.model == "cnn":
        model = CNNDeepfakeDetector(num_classes=2, pretrained=args.pretrained)
    elif args.model == "lstm":
        model = TemporalLSTMDetector(num_classes=2, sequence_length=args.frames_per_video, pretrained=args.pretrained)
    elif args.model == "transformer":
        model = TransformerDeepfakeDetector(num_classes=2)
    elif args.model == "spectral":
        model = SpectralAnalysisDetector(num_classes=2, pretrained=args.pretrained)
    else:
        raise ValueError("Unknown model")

    model.to(device)
    best = 0.0
    for epoch in range(args.epochs):
        if tr_loader is not None:
            train_one(model, tr_loader, device, is_seq)
        acc = eval_one(model, va_loader, device, is_seq)
        best = max(best, acc)
        print(f"[{args.model}] Epoch {epoch+1}/{args.epochs} | Val Acc: {acc:.4f} (best {best:.4f})")
    print(f"[{args.model}] Final Best Val Acc: {best:.4f}")
    return best

@torch.no_grad()
def infer_paths(args):
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    # build model
    if args.model == "hybrid":
        model = HybridDeepfakeDetector(num_classes=2, pretrained=args.pretrained)
    elif args.model == "cnn":
        model = CNNDeepfakeDetector(num_classes=2, pretrained=args.pretrained)
    elif args.model == "lstm":
        model = TemporalLSTMDetector(num_classes=2, sequence_length=args.frames_per_video, pretrained=args.pretrained)
    elif args.model == "transformer":
        model = TransformerDeepfakeDetector(num_classes=2)
    elif args.model == "spectral":
        model = SpectralAnalysisDetector(num_classes=2, pretrained=args.pretrained)
    else:
        raise ValueError("Unknown model")
    model.to(device).eval()
    tf_img = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        IMAGENET_NORM
    ])
    def predict_image(path: str):
        im = cv2.imread(path)
        if im is None:
            return None
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        x = tf_img(im).unsqueeze(0).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        return probs[1].item(), probs[0].item()
    def predict_video(path: str):
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        frame_indices = np.linspace(0, total-1, num=min(args.frames_per_video, total)).astype(int)
        preds=[]
        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            x = tf_img(frame).unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            preds.append(probs[1].item())
        cap.release()
        if not preds:
            return None
        avg_fake = float(np.mean(preds))
        return avg_fake, 1.0-avg_fake
    results = []
    for p in args.infer:
        if not os.path.isfile(p):
            print(f"[skip] {p} not found")
            continue
        ext = os.path.splitext(p)[1].lower()
        if ext in ('.jpg','.jpeg','.png','.bmp','.webp'):
            r = predict_image(p)
            if r is None:
                print(f"[err] could not read image {p}")
                continue
            fake, real = r
            results.append((p, fake))
            print(f"IMAGE {p} -> fake_prob={fake:.4f} real_prob={real:.4f}")
        elif ext in ('.mp4','.mov','.avi','.mkv'):
            r = predict_video(p)
            if r is None:
                print(f"[err] could not read video {p}")
                continue
            fake, real = r
            results.append((p, fake))
            print(f"VIDEO {p} -> avg_fake_prob={fake:.4f} avg_real_prob={real:.4f}")
        else:
            print(f"[skip] unsupported extension {p}")
    return results

def main():
    ap = argparse.ArgumentParser(description="Deepfake Detection - Unified")
    ap.add_argument("--input", type=str, default="input", help="Folder with real/fake subfolders")
    ap.add_argument("--kaggle", action="store_true", help="Download nanduncs/1000-videos-split and set input accordingly")
    ap.add_argument("--kaggle_split", choices=["train","validation","test"], default="validation", help="Which split to use from Kaggle dataset")
    ap.add_argument("--bench", action="store_true", help="Run all models")
    ap.add_argument("--model", choices=["hybrid","cnn","lstm","transformer","spectral"], default="hybrid")
    ap.add_argument("--use_videos", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_items", type=int, default=100)
    ap.add_argument("--frames_per_video", type=int, default=8)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--pretrained", action="store_true", help="Use pretrained weights for backbones (may download)")
    ap.add_argument("--infer", nargs='+', help="List of image/video files to run inference on (skips training)")
    args = ap.parse_args()

    if args.kaggle:
        base = maybe_download_kaggle()
        if base:
            # Helper to quickly count available sequence items (videos + frame dirs)
            def quick_count(root: str) -> int:
                total = 0
                for label, cls in enumerate(["real","fake"]):
                    class_root = os.path.join(root, cls)
                    if not os.path.isdir(class_root):
                        continue
                    # videos
                    for ext in (".mp4",".mov",".avi",".mkv"):
                        total += len(glob(os.path.join(class_root, "**", "*"+ext), recursive=True))
                    # frame directories
                    for dirpath, dirnames, filenames in os.walk(class_root):
                        if any(f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp")) for f in filenames):
                            total += 1
                return total

            # choose split
            splits = [args.kaggle_split]
            if args.kaggle_split != "train":
                splits.append("train")
            chosen = None
            best_count = -1
            for sp in splits:
                cand = os.path.join(base, "1000_videos", sp)
                if os.path.isdir(os.path.join(cand, "real")) and os.path.isdir(os.path.join(cand, "fake")):
                    cnt = quick_count(cand)
                    if cnt > best_count:
                        best_count = cnt
                        chosen = cand
            if chosen:
                args.input = chosen
                print(f"Using Kaggle {os.path.basename(chosen)} split as input:", args.input, "(sequence items:", best_count, ")")
            else:
                print("Downloaded Kaggle dataset, but expected folders not found. Set --input manually.")
        else:
            print("Kaggle download unavailable; ensure --input has real/fake subfolders.")

    if args.infer:
        infer_paths(args)
        return
    if args.bench:
        results = {}
        for m in ["hybrid","cnn","lstm","transformer","spectral"]:
            args.model = m
            results[m] = run_single(args)
        print("Benchmark results:")
        for k,v in results.items():
            print(f"  {k}: {v:.4f}")
    else:
        run_single(args)

if __name__ == "__main__":
    main()