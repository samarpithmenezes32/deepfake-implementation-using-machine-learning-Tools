import os
import sys

try:
    import kagglehub
except ImportError:
    print("ERROR: kagglehub not installed. Install with: pip install kagglehub", file=sys.stderr)
    sys.exit(1)

# Downloads a small deepfake dataset from Kaggle (images organized under real/ and fake/)
# Dataset: nanduncs/1000-videos-split
# This is relatively small and good for quick experiments.
print("Downloading Kaggle dataset: nanduncs/1000-videos-split ...")
path = kagglehub.dataset_download("nanduncs/1000-videos-split")
print("Downloaded to:", path)

# Point to the 'validation' split which is small and has real/fake
val = os.path.join(path, "1000_videos", "validation")
if os.path.isdir(os.path.join(val, "real")) and os.path.isdir(os.path.join(val, "fake")):
    print("Suggested source split:", val)
else:
    print("WARNING: Expected validation/real and validation/fake folders not found.")
    print("Please inspect:", path)
