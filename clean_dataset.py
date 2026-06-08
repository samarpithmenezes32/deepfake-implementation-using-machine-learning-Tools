import os

def clean_dataset():
    cleaned_counts = {'real': 0, 'fake': 0}
    remaining_counts = {'real': 0, 'fake': 0}
    
    for cls in ['real', 'fake']:
        dirpath = os.path.join('input', cls)
        if not os.path.exists(dirpath):
            print(f"Directory {dirpath} does not exist.")
            continue
            
        for f in os.listdir(dirpath):
            fpath = os.path.join(dirpath, f)
            if os.path.isdir(fpath):
                continue
                
            # If the file does not start with 'shahzaib', delete it
            if not f.lower().startswith('shahzaib'):
                try:
                    os.remove(fpath)
                    cleaned_counts[cls] += 1
                except Exception as e:
                    print(f"Error deleting {fpath}: {e}")
            else:
                remaining_counts[cls] += 1
                
    print("\n" + "="*50)
    print("DATASET CLEANING COMPLETE")
    print("="*50)
    print(f"Real folder: Deleted {cleaned_counts['real']} non-face images. Remaining: {remaining_counts['real']}")
    print(f"Fake folder: Deleted {cleaned_counts['fake']} non-face images. Remaining: {remaining_counts['fake']}")
    print("="*50 + "\n")

if __name__ == '__main__':
    clean_dataset()
