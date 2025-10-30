"""
Create a minimal, deduplicated deepfake dataset (Celeb-DF-like) using publicly available sources.

Features:
- Optionally downloads a small Kaggle frames dataset (nanduncs/1000-videos-split) via kagglehub
- Discovers images and frame directories under real/ and fake/ recursively
- Deduplicates images using perceptual hashing (aHash) and MD5 exact-match
- Deduplicates videos/frame-dirs by sampling frames and comparing hash sets (Jaccard)
- Writes a compact dataset to target directory: target/{real,fake}/images and target/{real,fake}/videos

Note: This does NOT download restricted Celeb-DF data; it curates a small, similar-structure dataset
from public sources. For actual Celeb-DF, follow their official access procedure and set --input to its path.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def md5_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ahash_image(img_bgr: np.ndarray, hash_size: int = 8) -> int:
    """Compute average-hash on BGR image; returns 64-bit integer."""
    if img_bgr is None:
        raise ValueError("Invalid image for aHash")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    avg = float(small.mean())
    bits = (small >= avg).astype(np.uint8).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def ahash_path(path: str, hash_size: int = 8) -> Optional[int]:
    img = cv2.imread(path)
    if img is None:
        return None
    return ahash_image(img, hash_size)


def hamming(a: int, b: int) -> int:
    return int(bin(a ^ b).count("1"))


def jaccard(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def discover_images(root: str) -> List[str]:
    out: List[str] = []
    for r, _, files in os.walk(root):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMG_EXTS:
                out.append(os.path.join(r, f))
    return out


def discover_videos_or_framedirs(root: str) -> List[str]:
    paths: List[str] = []
    # video files
    for r, _, files in os.walk(root):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in VID_EXTS:
                paths.append(os.path.join(r, f))
    # frame directories (heuristic: contains many image frames)
    for r, dirs, files in os.walk(root):
        img_count = sum(1 for f in files if os.path.splitext(f)[1].lower() in IMG_EXTS)
        if img_count >= 10:  # likely a frame directory
            paths.append(r)
    # de-dup paths
    paths = sorted(set(paths))
    return paths


def sample_frames_from_video(path: str, max_frames: int = 12) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return frames
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if n <= 0:
        # attempt to read first N frames sequentially
        i = 0
        while i < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            i += 1
    else:
        idxs = np.linspace(0, max(0, n - 1), num=max_frames, dtype=int)
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
    cap.release()
    return frames


def sample_frames_from_dir(path: str, max_frames: int = 12) -> List[np.ndarray]:
    files = [os.path.join(path, f) for f in os.listdir(path)
             if os.path.splitext(f)[1].lower() in IMG_EXTS]
    files.sort()
    if not files:
        return []
    idxs = np.linspace(0, len(files) - 1, num=min(max_frames, len(files)), dtype=int)
    frames: List[np.ndarray] = []
    for i in idxs:
        img = cv2.imread(files[int(i)])
        if img is not None:
            frames.append(img)
    return frames


def video_signature(path: str, max_frames: int = 12) -> Set[int]:
    hashes: Set[int] = set()
    if os.path.isdir(path):
        frames = sample_frames_from_dir(path, max_frames=max_frames)
    else:
        frames = sample_frames_from_video(path, max_frames=max_frames)
    for fr in frames:
        try:
            hashes.add(ahash_image(fr))
        except Exception:
            continue
    return hashes


@dataclass
class CurateConfig:
    target: str
    input_root: Optional[str]
    download_kaggle: bool
    kaggle_split: str
    max_images_per_class: int
    max_videos_per_class: int
    frames_per_video_sig: int
    jaccard_thresh: float


def maybe_download_kaggle(split: str) -> Optional[str]:
    try:
        import kagglehub
    except Exception:
        print("[Info] kagglehub not available; skipping download.")
        return None
    ds = "nanduncs/1000-videos-split"
    try:
        base = kagglehub.dataset_download(ds)
        cand = [
            os.path.join(base, "1000_videos", split),
            os.path.join(base, split),
        ]
        for c in cand:
            if os.path.isdir(c):
                print(f"[OK] Kaggle split found: {c}")
                return c
        print(f"[Warn] Could not locate expected split in {base}; using base path")
        return base
    except Exception as e:
        print(f"[Error] Kaggle download failed: {e}")
        return None


def ensure_dirs(target: str) -> Dict[str, str]:
    sub = {}
    for cls in ("real", "fake"):
        imgs = os.path.join(target, cls, "images")
        vids = os.path.join(target, cls, "videos")
        os.makedirs(imgs, exist_ok=True)
        os.makedirs(vids, exist_ok=True)
        sub[f"{cls}_images"] = imgs
        sub[f"{cls}_videos"] = vids
    return sub


def copy_unique_images(src_paths: List[str], dst_dir: str, max_items: int) -> int:
    seen_md5: Set[str] = set()
    seen_hash: List[int] = []
    kept = 0
    for p in src_paths:
        if kept >= max_items:
            break
        h_md5 = md5_file(p)
        if h_md5 in seen_md5:
            continue
        h_ah = ahash_path(p)
        if h_ah is None:
            continue
        # near-dup check (loose)
        dup = False
        for q in seen_hash:
            if hamming(h_ah, q) <= 4:
                dup = True
                break
        if dup:
            continue
        fname = os.path.basename(p)
        base, ext = os.path.splitext(fname)
        out = os.path.join(dst_dir, f"{base}_{kept}{ext}")
        shutil.copy2(p, out)
        seen_md5.add(h_md5)
        seen_hash.append(h_ah)
        kept += 1
    return kept


def copy_unique_videos(src_paths: List[str], dst_dir: str, max_items: int, sig_frames: int, jaccard_thresh: float) -> int:
    signatures: List[Set[int]] = []
    kept = 0
    for p in src_paths:
        if kept >= max_items:
            break
        sig = video_signature(p, max_frames=sig_frames)
        if not sig:
            continue
        is_dup = False
        for s in signatures:
            if jaccard(sig, s) >= jaccard_thresh:
                is_dup = True
                break
        if is_dup:
            continue
        fname = os.path.basename(p.rstrip(os.sep))
        base, ext = os.path.splitext(fname)
        if os.path.isdir(p):
            # copy frame directory
            dst = os.path.join(dst_dir, f"{base}_{kept}")
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(p, dst)
        else:
            dst = os.path.join(dst_dir, f"{base}_{kept}{ext}")
            shutil.copy2(p, dst)
        signatures.append(sig)
        kept += 1
    return kept


def curate_dataset(cfg: CurateConfig) -> Dict[str, int]:
    if cfg.download_kaggle and not cfg.input_root:
        dl = maybe_download_kaggle(cfg.kaggle_split)
        if dl:
            input_root = dl
        else:
            raise RuntimeError("Kaggle download failed and no --input provided.")
    else:
        if not cfg.input_root:
            raise RuntimeError("Either --input must be provided or --kaggle must be used.")
        input_root = cfg.input_root

    # Expect structure with real/ and fake/ somewhere under input_root; search recursively
    def find_class_dir(name: str) -> Optional[str]:
        for r, dirs, _ in os.walk(input_root):
            for d in dirs:
                if d.lower() == name:
                    return os.path.join(r, d)
        return None

    real_root = find_class_dir("real")
    fake_root = find_class_dir("fake")
    if not real_root or not fake_root:
        raise RuntimeError(f"Could not find 'real'/'fake' subfolders under {input_root}")

    layout = ensure_dirs(cfg.target)

    # Images
    real_images = discover_images(real_root)
    fake_images = discover_images(fake_root)
    real_images.sort(); fake_images.sort()
    kept_real_img = copy_unique_images(real_images, layout["real_images"], cfg.max_images_per_class)
    kept_fake_img = copy_unique_images(fake_images, layout["fake_images"], cfg.max_images_per_class)

    # Videos or frame directories
    real_vids = discover_videos_or_framedirs(real_root)
    fake_vids = discover_videos_or_framedirs(fake_root)
    kept_real_vid = copy_unique_videos(real_vids, layout["real_videos"], cfg.max_videos_per_class, cfg.frames_per_video_sig, cfg.jaccard_thresh)
    kept_fake_vid = copy_unique_videos(fake_vids, layout["fake_videos"], cfg.max_videos_per_class, cfg.frames_per_video_sig, cfg.jaccard_thresh)

    return {
        "kept_real_images": kept_real_img,
        "kept_fake_images": kept_fake_img,
        "kept_real_videos": kept_real_vid,
        "kept_fake_videos": kept_fake_vid,
    }


def main():
    ap = argparse.ArgumentParser(description="Curate a minimal, deduplicated deepfake dataset (Celeb-DF-like)")
    ap.add_argument("--target", default=os.path.join("inputs", "cdf_min"), help="Output dataset root")
    ap.add_argument("--input", help="Existing dataset root (must contain real/ and fake/). If omitted, use --kaggle")
    ap.add_argument("--kaggle", action="store_true", help="Download public Kaggle frames dataset as source")
    ap.add_argument("--kaggle-split", choices=["train", "validation", "test"], default="validation")
    ap.add_argument("--max-images", type=int, default=200, help="Max images per class")
    ap.add_argument("--max-videos", type=int, default=20, help="Max videos/frame-dirs per class")
    ap.add_argument("--sig-frames", type=int, default=12, help="Frames sampled per video for signature")
    ap.add_argument("--jaccard", type=float, default=0.7, help="Jaccard threshold for video near-duplicate filtering")
    args = ap.parse_args()

    cfg = CurateConfig(
        target=args.target,
        input_root=args.input,
        download_kaggle=bool(args.kaggle),
        kaggle_split=args.kaggle_split,
        max_images_per_class=args.max_images,
        max_videos_per_class=args.max_videos,
        frames_per_video_sig=args.sig_frames,
        jaccard_thresh=args.jaccard,
    )

    os.makedirs(cfg.target, exist_ok=True)
    stats = curate_dataset(cfg)
    print("\n[Summary]")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\n[Done] Minimal dataset created at: {cfg.target}")


if __name__ == "__main__":
    main()
