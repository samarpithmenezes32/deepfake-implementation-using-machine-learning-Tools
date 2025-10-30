import os
import sys
import random
import argparse
import cv2

IMG_EXTS = (".jpg",".jpeg",".png",".bmp",".webp")

def gather_images(root):
    real, fake = [], []
    for cls, bucket in [("real", real), ("fake", fake)]:
        class_root = os.path.join(root, cls)
        if not os.path.isdir(class_root):
            continue
        for dirpath, _, filenames in os.walk(class_root):
            for f in filenames:
                if f.lower().endswith(IMG_EXTS):
                    bucket.append(os.path.join(dirpath, f))
    return real, fake

def copy_resize(paths, dst_dir, cls, size, quality):
    os.makedirs(os.path.join(dst_dir, cls), exist_ok=True)
    n = 0
    for i, p in enumerate(paths):
        im = cv2.imread(p)
        if im is None:
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
        out_path = os.path.join(dst_dir, cls, f"{cls}_{i:05d}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        n += 1
    return n

def main():
    ap = argparse.ArgumentParser(description="Prepare a minimal-size deepfake dataset subset")
    ap.add_argument("--src", required=True, help="Source folder with real/ and fake/")
    ap.add_argument("--out", default="input/lite", help="Output folder for lite dataset")
    ap.add_argument("--items", type=int, default=100, help="Images per class")
    ap.add_argument("--size", type=int, default=256, help="Resize to this square size")
    ap.add_argument("--quality", type=int, default=85, help="JPEG quality (1-100)")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        print("ERROR: --src folder not found:", args.src, file=sys.stderr)
        sys.exit(1)

    real, fake = gather_images(args.src)
    if not real or not fake:
        print("ERROR: did not find images under real/ and fake/ in", args.src, file=sys.stderr)
        sys.exit(2)

    random.shuffle(real); random.shuffle(fake)
    real = real[:args.items]; fake = fake[:args.items]

    n_r = copy_resize(real, args.out, "real", args.size, args.quality)
    n_f = copy_resize(fake, args.out, "fake", args.size, args.quality)

    print(f"Lite dataset prepared at {args.out} | real={n_r} fake={n_f} | size={args.size}px | quality={args.quality}")

if __name__ == "__main__":
    main()
