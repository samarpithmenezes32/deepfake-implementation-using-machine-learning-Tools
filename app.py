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

from deepfake_unified import UnifiedDeepfakeDetector, ForensicAnalyzer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global detectors cache
detectors_cache = {}


def get_cached_detector(model_type, device):
    global detectors_cache
    if model_type not in detectors_cache:
        detectors_cache[model_type] = UnifiedDeepfakeDetector(model_type=model_type, device=device, pretrained=True)
    return detectors_cache[model_type]

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
            images_dir = os.path.join(cls_dir, 'images')
            target_img_dir = images_dir if os.path.exists(images_dir) else cls_dir
            for f in os.listdir(target_img_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(target_img_dir, f)
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
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        det = get_cached_detector(model_type, device)
        
        # Run detection
        ext = filename.rsplit('.', 1)[1].lower()
        if ext in {'jpg', 'jpeg', 'png'}:
            img = cv2.imread(filepath)
            if img is None:
                return jsonify({'error': 'Could not read image file'}), 400
            fake_prob, confidence = det.detect_frame(img)
            
            # Run forensic analysis
            forensic = ForensicAnalyzer.analyze_image(img)
            
            result = {
                'type': 'image',
                'filename': filename,
                'model': model_type,
                'fake_probability': round(float(fake_prob), 4),
                'confidence': round(float(confidence), 4),
                'prediction': 'FAKE' if fake_prob > 0.5 else 'REAL',
                'forensic_analysis': forensic,
                'timestamp': datetime.now().isoformat(),
            }
        else:  # Video
            result_dict = det.detect_video(filepath, sample_rate=10)
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


@app.route('/api/detect-all', methods=['POST'])
def api_detect_all():
    """Run ALL 5 models on the same file — production multi-model ensemble pipeline"""
    import time

    filepath = None
    filename = None

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

    ext = filename.rsplit('.', 1)[1].lower()
    is_image = ext in {'jpg', 'jpeg', 'png'}

    if is_image:
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Could not read image file'}), 400
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_order = ['spectral', 'cnn', 'transformer', 'hybrid', 'lstm']
    per_model = []
    total_start = time.time()

    for mtype in model_order:
        try:
            t0 = time.time()
            det = get_cached_detector(mtype, device)
            
            if is_image:
                fake_prob, confidence = det.detect_frame(img)
                pred = 'FAKE' if fake_prob > 0.5 else 'REAL'
            else:
                vr = det.detect_video(filepath, sample_rate=15)
                fake_prob = vr['avg_fake_probability']
                confidence = max(fake_prob, 1 - fake_prob)
                pred = vr['final_prediction']

            elapsed = round((time.time() - t0) * 1000)
            info = UnifiedDeepfakeDetector.get_model_info(mtype)
            bench = UnifiedDeepfakeDetector.get_benchmarks(mtype)

            per_model.append({
                'model': mtype,
                'name': info.get('name', mtype),
                'description': info.get('desc', ''),
                'fake_probability': round(float(fake_prob), 4),
                'confidence': round(float(confidence), 4),
                'prediction': pred,
                'latency_ms': elapsed,
                'benchmark_accuracy': bench.get('accuracy', 0),
                'benchmark_auc': bench.get('auc', 0),
            })
        except Exception as e:
            per_model.append({
                'model': mtype,
                'name': mtype,
                'error': str(e),
                'prediction': 'ERROR',
                'fake_probability': 0,
                'confidence': 0,
                'latency_ms': 0,
            })

    total_elapsed = round((time.time() - total_start) * 1000)

    # Weighted ensemble verdict (weight by benchmark accuracy)
    valid = [m for m in per_model if m.get('prediction') not in ('ERROR',)]
    if valid:
        weights = [m.get('benchmark_accuracy', 95) for m in valid]
        w_sum = sum(weights)
        ensemble_prob = sum(m['fake_probability'] * w for m, w in zip(valid, weights)) / w_sum
        ensemble_conf = sum(m['confidence'] * w for m, w in zip(valid, weights)) / w_sum
        vote_fake = sum(1 for m in valid if m['prediction'] == 'FAKE')
        vote_real = len(valid) - vote_fake
    else:
        ensemble_prob = 0.5
        ensemble_conf = 0.5
        vote_fake = 0
        vote_real = 0

    result = {
        'type': 'image' if is_image else 'video',
        'filename': filename,
        'models': per_model,
        'ensemble': {
            'weighted_fake_probability': round(float(ensemble_prob), 4),
            'weighted_confidence': round(float(ensemble_conf), 4),
            'prediction': 'FAKE' if ensemble_prob > 0.5 else 'REAL',
            'votes_fake': vote_fake,
            'votes_real': vote_real,
            'total_models': len(valid),
        },
        'total_latency_ms': total_elapsed,
        'timestamp': datetime.now().isoformat(),
    }

    if is_image:
        result['forensic_analysis'] = ForensicAnalyzer.analyze_image(img)

    return jsonify(result)


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


@app.route('/api/reload-models')
def api_reload_models():
    """Clear cached models to force reloading from disk weights"""
    global detectors_cache
    detectors_cache.clear()
    return jsonify({
        'status': 'success',
        'message': 'All models successfully flushed from memory cache. Next inference will reload weights from disk.'
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
