import sys

try:
    import kagglehub
except ImportError:
    print("kagglehub is not installed. Install it with: pip install kagglehub")
    sys.exit(1)

path = kagglehub.dataset_download("sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset")
print("Path to dataset files:", path)
print("Next: run your program with --input pointing to this path, or use deepfake.py --dfd to automate.")
