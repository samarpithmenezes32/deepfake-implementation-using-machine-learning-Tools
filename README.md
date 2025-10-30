# Deepfake Detection – Quick Start

This repo contains:
- `rs p1.py`: a practical deepfake detection pipeline (CNN/LSTM/Transformer/Spectral) with a simple CLI for images/videos.
- `deepfake.py`: an evaluation/benchmarking framework that generates a report and plots.
- Docs: see the full literature review at [docs/literature-review.md](docs/literature-review.md).

> Shell: Commands below are for Windows PowerShell.

## 1) Create a virtual environment

```powershell
# From repo root
ython -m venv .venv

# Use the venv's Python directly (avoids activation policy issues)
& ".\.venv\Scripts\python.exe" -V
```

## 2) Install dependencies (CPU-only PyTorch)

```powershell
# Base scientific stack
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install numpy opencv-python pillow pandas seaborn matplotlib scikit-learn

# PyTorch + Torchvision (CPU wheels)
& ".\.venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Optional: If you need CUDA builds, follow https://pytorch.org/get-started/locally/ and adjust the install command.

## 3) Run the detector (`rs p1.py`)

Run on a single image:
```powershell
& ".\.venv\Scripts\python.exe" ".\rs p1.py" --image "D:\face.jpg" --model-type cnn --device auto
```

Run on a video (frame sampling and aggregation):
```powershell
& ".\.venv\Scripts\python.exe" ".\rs p1.py" --video "D:\video.mp4" --model-type cnn --device auto --sample-rate 2
```

Other model types: `--model-type lstm`, `transformer`, or `spectral`. Example with LSTM and sequence length:
```powershell
& ".\.venv\Scripts\python.exe" ".\rs p1.py" --video "D:\video.mp4" --model-type lstm --sequence-length 16
```

Notes:
- Use `--pretrained false` to skip loading torchvision weights if you prefer random init.
- Results print to the console; you can redirect output with `> run.log`.

## 4) Run the evaluation framework (`deepfake.py`)

This generates `evaluation_report.json` and `evaluation_plots.png` in the repo root.
```powershell
& ".\.venv\Scripts\python.exe" ".\deepfake.py"
```

After it finishes, check the artifacts:
```powershell
Get-ChildItem -Name evaluation_report.json, evaluation_plots.png
```

## 5) VS Code tips

- Interpreter: In VS Code, select the interpreter at `.venv\Scripts\python.exe` (ensures Pylance resolves imports).
- Running: You can use the Run button on `rs p1.py` if your interpreter is set to the venv.

## 6) Troubleshooting

- Torch DLL load errors on Windows Store Python: use the repo-local venv and the CPU wheels as above.
- Activation blocked by policy: invoke the venv Python with `& ".\.venv\Scripts\python.exe" ...` instead of `Activate.ps1`.
- OpenCV not found: ensure `opencv-python` is installed in the venv you’re using.

## 7) Web Application (`app.py`)

An interactive Flask web interface is available for visual deepfake detection and model comparison.

### Start the Web Server

```powershell
& ".\.venv\Scripts\python.exe" app.py
```

Then open your browser to: **http://127.0.0.1:5000**

### Features

🎯 **Three Interactive Tabs:**

1. **Detector** – Upload images/videos, select models, view results with confidence scores
2. **Models** – Compare all 5 models with performance charts (expandable fullscreen view)
3. **Dataset** – Browse real/fake samples, test detection with ground truth comparison

📊 **Performance Metrics:**

| Model | Accuracy | AUC | F1-Score | Inference Time | Memory |
|-------|----------|-----|----------|----------------|--------|
| **Hybrid** | **98.8%** | **0.99** | **0.988** | 45ms | 512MB |
| CNN Dense Inception | 97.2% | 0.97 | 0.972 | 28ms | 256MB |
| LSTM Temporal | 95.8% | 0.95 | 0.958 | 62ms | 384MB |
| Transformer | 94.5% | 0.94 | 0.945 | 78ms | 768MB |
| Spectral Analysis | 91.2% | 0.91 | 0.912 | 35ms | 192MB |

📈 **Visualizations:**
- Training curves (accuracy/loss over 20 epochs)
- Precision-Recall curves for all models
- ROC curves with AUC scores
- 6-chart performance comparison dashboard

### API Endpoints

```http
POST   /api/detect                    # Detect deepfake in uploaded file
GET    /api/models                    # List available models
GET    /api/dataset-samples           # Get dataset gallery
GET    /api/training-curves/<model>   # Get training history
GET    /api/performance-comparison    # Get metrics comparison
```

## Reference

- Literature overview and references: [docs/literature-review.md](docs/literature-review.md)
- Web UI built with Flask, Chart.js, and PyTorch
