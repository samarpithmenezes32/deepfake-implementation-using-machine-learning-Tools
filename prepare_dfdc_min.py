"""
Prepare a minimal, deduplicated DFDC sample from a DFDC dataset folder.

Requirements:
- --input points to a DFDC dataset root containing metadata.json and video files

Output structure:
  target/
    real/
      videos/  (subset of real videos)
      images/  (sampled frames extracted from those videos)
    fake/
      videos/
      images/

This does NOT download DFDC. If you need to download, retrieve the DFDC preview set from Kaggle
and pass its extracted folder via --input.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def ensure_layout(root: str) -> Dict[str, str]:
    sub = {}
    for cls in ("real", "fake"):
        vids = os.path.join(root, cls, "videos")
        imgs = os.path.join(root, cls, "images")
        os.makedirs(vids, exist_ok=True)
        os.makedirs(imgs, exist_ok=True)
        sub[f"{cls}_videos"] = vids
        sub[f"{cls}_images"] = imgs
    return sub


def ahash_image(img_bgr, hash_size: int = 8) -> int:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    avg = float(small.mean())
    bits = (small >= avg).astype(np.uint8).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def jaccard(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def sample_frames_from_video(path: str, max_frames: int = 8) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return frames
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    idxs = np.linspace(0, max(0, n - 1), num=max_frames, dtype=int)
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)
    cap.release()
    return frames


def video_signature(path: str, max_frames: int = 8) -> Set[int]:
    sig: Set[int] = set()
    for fr in sample_frames_from_video(path, max_frames=max_frames):
        try:
            sig.add(ahash_image(fr))
        except Exception:
            continue
    return sig


def collect_videos_with_labels(root: str, metadata_file: Optional[str] = None) -> Dict[str, str]:
    # metadata maps filename -> {label: REAL/FAKE}
    if metadata_file is None:
        metadata_file = os.path.join(root, "metadata.json")
    if not os.path.isfile(metadata_file):
        raise FileNotFoundError(f"metadata.json not found at {metadata_file}")
    with open(metadata_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    # Walk and index videos
    mapping: Dict[str, str] = {}
    videos: List[str] = []
    for r, _, files in os.walk(root):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in VID_EXTS:
                videos.append(os.path.join(r, f))
    for v in videos:
        name = os.path.basename(v)
        if name in meta and isinstance(meta[name], dict):
            label = meta[name].get("label") or meta[name].get("class")
            if label is None and isinstance(meta[name], str):
                label = meta[name]  # some variants map to a string directly
            if label is None:
                continue
            lab = str(label).strip().lower()
            if lab in ("real", "fake"):
                mapping[v] = lab
            elif lab in ("0", "1"):
                mapping[v] = "fake" if lab == "1" else "real"
            elif lab in ("real", "fake", "REAL", "FAKE"):
                mapping[v] = lab.lower()
    return mapping


def curate_from_dfdc(input_root: str, target_root: str, max_per_class: int, frames_per_video: int, jaccard_thresh: float) -> Dict[str, int]:
    layout = ensure_layout(target_root)
    mapping = collect_videos_with_labels(input_root)
    # group by class
    cls_to_videos: Dict[str, List[str]] = {"real": [], "fake": []}
    for v, lab in mapping.items():
        if lab in cls_to_videos:
            cls_to_videos[lab].append(v)
    stats = {"kept_real_videos": 0, "kept_fake_videos": 0, "extracted_real_images": 0, "extracted_fake_images": 0}
    for cls in ("real", "fake"):
        picked = 0
        signatures: List[Set[int]] = []
        out_v = layout[f"{cls}_videos"]
        out_i = layout[f"{cls}_images"]
        for vpath in sorted(cls_to_videos.get(cls, [])):
            if picked >= max_per_class:
                break
            sig = video_signature(vpath, max_frames=max(4, frames_per_video // 2))
            if not sig:
                continue
            dup = any(jaccard(sig, s) >= jaccard_thresh for s in signatures)
            if dup:
                continue
            # copy video
            base = os.path.splitext(os.path.basename(vpath))[0]
            dst_v = os.path.join(out_v, f"{base}.mp4")
            try:
                shutil.copy2(vpath, dst_v)
            except Exception:
                continue
            # extract frames
            frames = sample_frames_from_video(dst_v, max_frames=frames_per_video)
            saved = 0
            for idx, fr in enumerate(frames):
                fp = os.path.join(out_i, f"{base}_{idx:02d}.jpg")
                ok = cv2.imwrite(fp, fr)
                if ok:
                    saved += 1
            signatures.append(sig)
            picked += 1
            stats[f"kept_{cls}_videos"] += 1
            stats[f"extracted_{cls}_images"] += saved
    return stats


def main():
    ap = argparse.ArgumentParser(description="Prepare a minimal DFDC sample from a DFDC dataset folder")
    ap.add_argument("--input", required=True, help="DFDC dataset root (must contain metadata.json)")
    ap.add_argument("--target", default=os.path.join("inputs", "dfdc_min"), help="Output dataset root")
    ap.add_argument("--max-per-class", type=int, default=10, help="Max number of videos per class")
    ap.add_argument("--frames-per-video", type=int, default=8, help="Frames extracted per video")
    ap.add_argument("--jaccard", type=float, default=0.75, help="Near-duplicate threshold for videos")
    args = ap.parse_args()

    os.makedirs(args.target, exist_ok=True)
    stats = curate_from_dfdc(args.input, args.target, args.max_per_class, args.frames_per_video, args.jaccard)
    print("\n[DFDC minimal sample created]")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  target: {args.target}")


if __name__ == "__main__":
    main()
