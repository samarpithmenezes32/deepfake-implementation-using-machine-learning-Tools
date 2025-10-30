"""
Flask Web UI for Unified Deepfake Detection System
Combines deepfake.py, deepfake_model_implementation.py, and rs p1.py
"""

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import json
import cv2
import numpy as np
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import torch
from datetime import datetime

from deepfake_unified import UnifiedDeepfakeDetector

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global detector instance
detector = None

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov', 'mkv'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_dataset_samples():
    """Get sample images and videos from dataset"""
    samples = {'real': {'images': [], 'videos': []}, 'fake': {'images': [], 'videos': []}}
    
    input_dir = 'input'
    if not os.path.exists(input_dir):
        return samples
    
    for cls in ('real', 'fake'):
        cls_dir = os.path.join(input_dir, cls)
        if os.path.exists(cls_dir):
            # Images
            for f in os.listdir(cls_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(cls_dir, f)
                    samples[cls]['images'].append({'name': f, 'path': path})
            
            # Videos
            videos_dir = os.path.join(cls_dir, 'videos')
            if os.path.exists(videos_dir):
                for f in os.listdir(videos_dir):
                    if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        path = os.path.join(videos_dir, f)
                        samples[cls]['videos'].append({'name': f, 'path': path})
                    elif os.path.isdir(os.path.join(videos_dir, f)):
                        samples[cls]['videos'].append({'name': f, 'path': os.path.join(videos_dir, f), 'is_dir': True})
    
    return samples


def img_to_base64(img_path):
    """Convert image to base64 for HTML display"""
    try:
        with open(img_path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None


@app.route('/')
def index():
    """Home/Dashboard"""
    return render_template('index.html')


@app.route('/api/models')
def api_models():
    """Get available models and info"""
    models_list = []
    for name in ['hybrid', 'cnn', 'lstm', 'transformer', 'spectral']:
        info = UnifiedDeepfakeDetector.get_model_info(name)
        bench = UnifiedDeepfakeDetector.get_benchmarks(name)
        models_list.append({
            'id': name,
            'name': info.get('name'),
            'description': info.get('desc'),
            'accuracy': bench.get('accuracy'),
            'auc': bench.get('auc'),
            'dataset': bench.get('dataset'),
        })
    return jsonify(models_list)


@app.route('/api/dataset-samples')
def api_dataset_samples():
    """Get dataset sample images and videos"""
    samples = get_dataset_samples()
    result = {'real': {'images': [], 'videos': []}, 'fake': {'images': [], 'videos': []}}
    
    for cls in ('real', 'fake'):
        # Images
        for img_info in samples[cls]['images'][:5]:  # Limit to 5 per class
            b64 = img_to_base64(img_info['path'])
            if b64:
                result[cls]['images'].append({
                    'name': img_info['name'],
                    'data': b64,
                    'path': img_info['path'],
                    'label': cls
                })
        
        # Videos - show thumbnail from first frame
        for vid_info in samples[cls]['videos'][:3]:  # Limit to 3 per class
            try:
                if vid_info.get('is_dir'):
                    # Get first image from frame directory
                    vid_path = vid_info['path']
                    frames = [f for f in os.listdir(vid_path) if f.lower().endswith(('.jpg', '.png'))]
                    if frames:
                        img_path = os.path.join(vid_path, frames[0])
                        b64 = img_to_base64(img_path)
                        if b64:
                            result[cls]['videos'].append({
                                'name': vid_info['name'],
                                'thumbnail': b64,
                                'type': 'frames',
                                'path': vid_path,
                                'label': cls
                            })
                else:
                    # Video file - extract first frame
                    cap = cv2.VideoCapture(vid_info['path'])
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        _, buffer = cv2.imencode('.jpg', frame)
                        b64 = base64.b64encode(buffer).decode()
                        result[cls]['videos'].append({
                            'name': vid_info['name'],
                            'thumbnail': b64,
                            'type': 'video',
                            'path': vid_info['path'],
                            'label': cls
                        })
            except:
                continue
    
    return jsonify(result)


@app.route('/api/detect', methods=['POST'])
def api_detect():
    """Run detection on uploaded file or dataset sample"""
    model_type = request.form.get('model', 'hybrid')
    filepath = None
    filename = None
    
    # Check if it's a dataset sample path or uploaded file
    if 'filepath' in request.form:
        filepath = request.form.get('filepath')
        filename = os.path.basename(filepath)
        if not os.path.exists(filepath):
            return jsonify({'error': f'File not found: {filepath}'}), 400
    elif 'file' in request.files:
        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
    else:
        return jsonify({'error': 'No file provided'}), 400
    
    try:
        global detector
        if detector is None or detector.model_type != model_type:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            detector = UnifiedDeepfakeDetector(model_type=model_type, device=device, pretrained=True)
        
        # Run detection
        ext = filename.rsplit('.', 1)[1].lower()
        if ext in {'jpg', 'jpeg', 'png'}:
            img = cv2.imread(filepath)
            if img is None:
                return jsonify({'error': 'Could not read image file'}), 400
            fake_prob, confidence = detector.detect_frame(img)
            result = {
                'type': 'image',
                'filename': filename,
                'model': model_type,
                'fake_probability': round(float(fake_prob), 4),
                'confidence': round(float(confidence), 4),
                'prediction': 'FAKE' if fake_prob > 0.5 else 'REAL',
                'timestamp': datetime.now().isoformat(),
            }
        else:  # Video
            result_dict = detector.detect_video(filepath, sample_rate=10)
            result = {
                'type': 'video',
                'filename': filename,
                'model': model_type,
                'avg_fake_probability': round(result_dict['avg_fake_probability'], 4),
                'frame_count': result_dict['frame_count'],
                'fps': round(result_dict['fps'], 2),
                'sampled_frames': result_dict['sampled_frames'],
                'prediction': result_dict['final_prediction'],
                'timestamp': datetime.now().isoformat(),
            }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-details/<model_type>')
def api_model_details(model_type):
    """Get detailed info about a model"""
    info = UnifiedDeepfakeDetector.get_model_info(model_type)
    bench = UnifiedDeepfakeDetector.get_benchmarks(model_type)
    training_info = UnifiedDeepfakeDetector.get_model_training_info(model_type)
    
    details = {
        'name': info.get('name'),
        'description': info.get('desc'),
        'accuracy': bench.get('accuracy'),
        'auc': bench.get('auc'),
        'dataset': bench.get('dataset'),
        'model_type': model_type,
    }
    
    details.update(training_info)
    return jsonify(details)


@app.route('/api/training-status')
def api_training_status():
    """Get training status (placeholder for future training UI)"""
    return jsonify({
        'status': 'ready',
        'message': 'System ready for training. Use /api/train to start.',
        'models': ['hybrid', 'cnn', 'lstm', 'transformer', 'spectral'],
    })


@app.route('/api/performance-comparison')
def api_performance_comparison():
    """Get performance comparison data for all models"""
    models_data = []
    for model_type in ['hybrid', 'cnn', 'lstm', 'transformer', 'spectral']:
        bench = UnifiedDeepfakeDetector.get_benchmarks(model_type)
        info = UnifiedDeepfakeDetector.get_model_info(model_type)
        models_data.append({
            'model': model_type,
            'name': info.get('name'),
            'accuracy': bench.get('accuracy'),
            'auc': bench.get('auc'),
            'dataset': bench.get('dataset'),
        })
    return jsonify(models_data)


@app.route('/api/training-curves/<model_type>')
def api_training_curves(model_type):
    """Get simulated training curves for visualization"""
    # Simulated training data (in production, this would come from saved training logs)
    import numpy as np
    epochs = list(range(1, 21))
    
    # Simulated curves based on typical training patterns
    curves = {
        'hybrid': {
            'train_loss': [2.5 - 0.08*e + np.random.normal(0, 0.1) for e in epochs],
            'val_loss': [2.4 - 0.07*e + np.random.normal(0, 0.15) for e in epochs],
            'train_acc': [0.4 + 0.03*e + np.random.normal(0, 0.02) for e in epochs],
            'val_acc': [0.4 + 0.028*e + np.random.normal(0, 0.03) for e in epochs],
        },
        'cnn': {
            'train_loss': [2.3 - 0.09*e + np.random.normal(0, 0.1) for e in epochs],
            'val_loss': [2.2 - 0.07*e + np.random.normal(0, 0.15) for e in epochs],
            'train_acc': [0.45 + 0.035*e + np.random.normal(0, 0.02) for e in epochs],
            'val_acc': [0.42 + 0.03*e + np.random.normal(0, 0.03) for e in epochs],
        },
        'lstm': {
            'train_loss': [2.6 - 0.07*e + np.random.normal(0, 0.12) for e in epochs],
            'val_loss': [2.55 - 0.065*e + np.random.normal(0, 0.16) for e in epochs],
            'train_acc': [0.38 + 0.025*e + np.random.normal(0, 0.02) for e in epochs],
            'val_acc': [0.37 + 0.022*e + np.random.normal(0, 0.035) for e in epochs],
        },
        'transformer': {
            'train_loss': [2.7 - 0.06*e + np.random.normal(0, 0.11) for e in epochs],
            'val_loss': [2.65 - 0.055*e + np.random.normal(0, 0.17) for e in epochs],
            'train_acc': [0.35 + 0.032*e + np.random.normal(0, 0.025) for e in epochs],
            'val_acc': [0.34 + 0.03*e + np.random.normal(0, 0.035) for e in epochs],
        },
        'spectral': {
            'train_loss': [2.4 - 0.08*e + np.random.normal(0, 0.1) for e in epochs],
            'val_loss': [2.35 - 0.075*e + np.random.normal(0, 0.14) for e in epochs],
            'train_acc': [0.42 + 0.03*e + np.random.normal(0, 0.02) for e in epochs],
            'val_acc': [0.4 + 0.027*e + np.random.normal(0, 0.032) for e in epochs],
        },
    }
    
    data = curves.get(model_type, curves['hybrid'])
    return jsonify({
        'epochs': epochs,
        'train_loss': [max(0.1, x) for x in data['train_loss']],
        'val_loss': [max(0.1, x) for x in data['val_loss']],
        'train_acc': [min(1.0, max(0.0, x)) for x in data['train_acc']],
        'val_acc': [min(1.0, max(0.0, x)) for x in data['val_acc']],
    })


@app.template_filter('b64_image')
def b64_image(data):
    return f"data:image/jpeg;base64,{data}" if data else ""


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
