import os, sys, argparse
from glob import glob

def count_split(root: str):
    stats = {}
    for cls in ["real","fake"]:
        class_root = os.path.join(root, cls)
        vids = []
        for ext in ("*.mp4","*.mov","*.avi","*.mkv"):
            vids += glob(os.path.join(class_root, "**", ext), recursive=True)
        frame_dirs = []
        img_files = 0
        for dirpath, dirnames, filenames in os.walk(class_root):
            if any(f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp")) for f in filenames):
                frame_dirs.append(dirpath)
                img_files += sum(1 for f in filenames if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp")))
        stats[cls] = {
            "videos": len(vids),
            "frame_dirs": len(frame_dirs),
            "images": img_files,
            "example_video": vids[0] if vids else None,
            "example_dir": frame_dirs[0] if frame_dirs else None,
        }
    return stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base path to 1000_videos/{split}")
    args = ap.parse_args()
    print("Inspecting:", args.base)
    if not os.path.isdir(args.base):
        print("Base does not exist")
        sys.exit(1)
    for sp in ["train","validation","test"]:
        p = os.path.join(args.base) if any(x in args.base.lower() for x in ["train","validation","test"]) else os.path.join(args.base, sp)
        if not os.path.isdir(p):
            continue
        print("--- Split:", os.path.basename(p))
        for cls in ["real","fake"]:
            cp = os.path.join(p, cls)
            if not os.path.isdir(cp):
                print(f"  {cls}: MISSING")
                continue
            st = count_split(p)[cls]
            print(f"  {cls}: videos={st['videos']} frame_dirs={st['frame_dirs']} images={st['images']}")
            if st['example_video']:
                print("    eg video:", st['example_video'])
            if st['example_dir']:
                print("    eg dir:", st['example_dir'])

if __name__ == "__main__":
    main()
