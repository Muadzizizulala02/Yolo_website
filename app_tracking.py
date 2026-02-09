# """
# Advanced YOLO Web Application with Progress Tracking
# This is an enhanced version with real-time progress updates
# """

# from flask import Flask, request, jsonify, send_file, render_template
# from werkzeug.utils import secure_filename
# import os
# import json
# import uuid
# from pathlib import Path
# import torch
# from ultralytics import YOLO
# from datetime import datetime
# import cv2
# from threading import Thread

# app = Flask(__name__)

# # Configuration
# UPLOAD_FOLDER = 'uploads'
# RESULTS_FOLDER = 'results'
# ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
# MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(RESULTS_FOLDER, exist_ok=True)

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# # Load your fine-tuned YOLO model
# MODEL_PATH = 'yolo26n.pt'
# model = None

# # Store processing progress
# processing_status = {}

# def load_model():
#     global model
#     try:
#         # Check if CUDA is available
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         model = YOLO(MODEL_PATH)
#         model.to(device)
#         print(f"Model loaded successfully on {device}")
#         print(f"Model classes: {model.names}")
#     except Exception as e:
#         print(f"Warning: Could not load model from {MODEL_PATH}")
#         print(f"Error: {e}")

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def get_video_info(video_path):
#     """Extract video metadata"""
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     duration = frame_count / fps if fps > 0 else 0
#     cap.release()
    
#     return {
#         'fps': fps,
#         'frame_count': frame_count,
#         'width': width,
#         'height': height,
#         'duration': duration
#     }

# def process_video_async(job_id, upload_path, conf_threshold=0.25):
#     """Process video in background thread"""
#     try:
#         processing_status[job_id] = {
#             'status': 'processing',
#             'progress': 0,
#             'message': 'Analyzing video...'
#         }
        
#         # Get video info
#         video_info = get_video_info(upload_path)
#         total_frames = video_info['frame_count']
        
#         # Run YOLO inference with progress tracking
#         results = []
#         detections_data = []
        
#         # Process video
#         cap = cv2.VideoCapture(upload_path)
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         output_video_path = os.path.join(
#             app.config['RESULTS_FOLDER'], 
#             f"{job_id}_output.mp4"
#         )
#         out = None
        
#         frame_idx = 0
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             # Update progress
#             progress = int((frame_idx / total_frames) * 100)
#             processing_status[job_id]['progress'] = progress
            
#             # Run inference on frame
#             result = model(frame, conf=conf_threshold, verbose=False)[0]
            
#             # Draw detections on frame
#             annotated_frame = result.plot()
            
#             # Initialize video writer
#             if out is None:
#                 height, width = annotated_frame.shape[:2]
#                 out = cv2.VideoWriter(
#                     output_video_path,
#                     fourcc,
#                     video_info['fps'],
#                     (width, height)
#                 )
            
#             out.write(annotated_frame)
            
#             # Extract detection information
#             frame_detections = []
#             boxes = result.boxes
#             for box in boxes:
#                 detection = {
#                     'class': result.names[int(box.cls[0])],
#                     'confidence': float(box.conf[0]),
#                     'bbox': box.xyxy[0].tolist()
#                 }
#                 frame_detections.append(detection)
            
#             detections_data.append({
#                 'frame': frame_idx,
#                 'timestamp': frame_idx / video_info['fps'],
#                 'detections': frame_detections
#             })
            
#             frame_idx += 1
        
#         cap.release()
#         if out:
#             out.release()
        
#         # Save JSON results
#         json_path = os.path.join(
#             app.config['RESULTS_FOLDER'], 
#             f"{job_id}_detections.json"
#         )
        
#         full_results = {
#             'video_info': video_info,
#             'model_info': {
#                 'model_name': MODEL_PATH,
#                 'confidence_threshold': conf_threshold,
#                 'classes': model.names
#             },
#             'detections': detections_data
#         }
        
#         with open(json_path, 'w') as f:
#             json.dump(full_results, f, indent=2)
        
#         # Generate summary
#         summary = generate_summary(detections_data)
#         summary['video_info'] = video_info
        
#         # Update status to completed
#         processing_status[job_id] = {
#             'status': 'completed',
#             'progress': 100,
#             'message': 'Processing complete!',
#             'summary': summary,
#             'json_file': f"/api/download/json/{job_id}",
#             'video_file': f"/api/download/video/{job_id}"
#         }
        
#     except Exception as e:
#         processing_status[job_id] = {
#             'status': 'error',
#             'progress': 0,
#             'message': str(e)
#         }

# def generate_summary(detections_data):
#     """Generate a comprehensive summary"""
#     summary = {
#         'total_frames': len(detections_data),
#         'total_detections': 0,
#         'object_counts': {},
#         'avg_detections_per_frame': 0,
#         'frames_with_detections': 0,
#         'confidence_stats': {
#             'min': 1.0,
#             'max': 0.0,
#             'avg': 0.0
#         }
#     }
    
#     all_confidences = []
    
#     for frame_data in detections_data:
#         frame_detections = frame_data.get('detections', [])
#         num_detections = len(frame_detections)
        
#         summary['total_detections'] += num_detections
        
#         if num_detections > 0:
#             summary['frames_with_detections'] += 1
        
#         for detection in frame_detections:
#             obj_class = detection['class']
#             confidence = detection['confidence']
            
#             summary['object_counts'][obj_class] = \
#                 summary['object_counts'].get(obj_class, 0) + 1
            
#             all_confidences.append(confidence)
    
#     if summary['total_frames'] > 0:
#         summary['avg_detections_per_frame'] = \
#             summary['total_detections'] / summary['total_frames']
    
#     if all_confidences:
#         summary['confidence_stats']['min'] = min(all_confidences)
#         summary['confidence_stats']['max'] = max(all_confidences)
#         summary['confidence_stats']['avg'] = \
#             sum(all_confidences) / len(all_confidences)
    
#     return summary

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/api/upload', methods=['POST'])
# def upload_video():
#     if model is None:
#         return jsonify({
#             'error': 'Model not loaded. Please check MODEL_PATH in app.py'
#         }), 500
    
#     if 'video' not in request.files:
#         return jsonify({'error': 'No video file provided'}), 400
    
#     file = request.files['video']
    
#     if file.filename == '':
#         return jsonify({'error': 'No file selected'}), 400
    
#     if not allowed_file(file.filename):
#         return jsonify({
#             'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv'
#         }), 400
    
#     try:
#         # Generate unique job ID
#         job_id = str(uuid.uuid4())
        
#         # Save uploaded file
#         filename = secure_filename(file.filename)
#         upload_path = os.path.join(
#             app.config['UPLOAD_FOLDER'], 
#             f"{job_id}_{filename}"
#         )
#         file.save(upload_path)
        
#         # Get confidence threshold from request
#         conf_threshold = float(request.form.get('confidence', 0.25))
        
#         # Start processing in background
#         thread = Thread(
#             target=process_video_async,
#             args=(job_id, upload_path, conf_threshold)
#         )
#         thread.start()
        
#         return jsonify({
#             'success': True,
#             'job_id': job_id,
#             'message': 'Processing started'
#         })
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/status/<job_id>')
# def get_status(job_id):
#     """Get processing status"""
#     if job_id not in processing_status:
#         return jsonify({'error': 'Job not found'}), 404
    
#     return jsonify(processing_status[job_id])

# @app.route('/api/download/json/<job_id>')
# def download_json(job_id):
#     json_path = os.path.join(
#         app.config['RESULTS_FOLDER'], 
#         f"{job_id}_detections.json"
#     )
    
#     if not os.path.exists(json_path):
#         return jsonify({'error': 'JSON file not found'}), 404
    
#     return send_file(
#         json_path, 
#         as_attachment=True, 
#         download_name=f"detections_{job_id}.json"
#     )

# @app.route('/api/download/video/<job_id>')
# def download_video(job_id):
#     video_path = os.path.join(
#         app.config['RESULTS_FOLDER'], 
#         f"{job_id}_output.mp4"
#     )
    
#     if not os.path.exists(video_path):
#         return jsonify({'error': 'Video file not found'}), 404
    
#     return send_file(
#         video_path, 
#         as_attachment=True, 
#         download_name=f"detected_{job_id}.mp4"
#     )

# @app.route('/api/stream/video/<job_id>')
# def stream_video(job_id):
#     """Stream video for preview"""
#     video_path = os.path.join(
#         app.config['RESULTS_FOLDER'], 
#         f"{job_id}_output.mp4"
#     )
    
#     if not os.path.exists(video_path):
#         return jsonify({'error': 'Video file not found'}), 404
    
#     return send_file(video_path, mimetype='video/mp4')

# if __name__ == '__main__':
#     load_model()
#     app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)



# from flask import Flask, request, jsonify, send_file, render_template
# from werkzeug.utils import secure_filename
# import os
# import json
# import uuid
# from pathlib import Path
# import torch
# from ultralytics import YOLO
# from datetime import datetime

# app = Flask(__name__)

# # Configuration
# UPLOAD_FOLDER = 'uploads'
# RESULTS_FOLDER = 'results'
# ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
# MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# # Create necessary directories
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(RESULTS_FOLDER, exist_ok=True)

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# # Load your fine-tuned YOLO model
# # Replace 'path/to/your/model.pt' with your actual model path
# MODEL_PATH = 'yolo26n.pt'  # Update this with your model path
# model = None

# def load_model():
#     global model
#     try:
#         model = YOLO(MODEL_PATH)
#         print(f"Model loaded successfully from {MODEL_PATH}")
#     except Exception as e:
#         print(f"Warning: Could not load model from {MODEL_PATH}")
#         print(f"Error: {e}")
#         print("Please update MODEL_PATH in app.py with your model file path")

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def generate_summary(detections_data):
#     """Generate a human-readable summary from detection results"""
#     summary = {
#         'total_frames': 0,
#         'total_detections': 0,
#         'object_counts': {},
#         'detection_timeline': []
#     }
    
#     for frame_data in detections_data:
#         summary['total_frames'] += 1
#         frame_detections = len(frame_data.get('detections', []))
#         summary['total_detections'] += frame_detections
        
#         for detection in frame_data.get('detections', []):
#             obj_class = detection['class']
#             summary['object_counts'][obj_class] = summary['object_counts'].get(obj_class, 0) + 1
    
#     # Create timeline summary (every 10th frame or significant changes)
#     for i, frame_data in enumerate(detections_data):
#         if i % 10 == 0 or len(frame_data.get('detections', [])) > 0:
#             summary['detection_timeline'].append({
#                 'frame': frame_data['frame'],
#                 'timestamp': frame_data.get('timestamp', 0),
#                 'objects_detected': len(frame_data.get('detections', []))
#             })
    
#     return summary

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/api/upload', methods=['POST'])
# def upload_video():
#     if model is None:
#         return jsonify({'error': 'Model not loaded. Please check MODEL_PATH in app.py'}), 500
    
#     # Check if file is in request
#     if 'video' not in request.files:
#         return jsonify({'error': 'No video file provided'}), 400
    
#     file = request.files['video']
    
#     if file.filename == '':
#         return jsonify({'error': 'No file selected'}), 400
    
#     if not allowed_file(file.filename):
#         return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv'}), 400
    
#     try:
#         # Generate unique ID for this processing job
#         job_id = str(uuid.uuid4())
        
#         # Save uploaded file
#         filename = secure_filename(file.filename)
#         upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
#         file.save(upload_path)
        
#         # Run YOLO inference
#         results = model(upload_path, save=True, project=app.config['RESULTS_FOLDER'], name=job_id)
        
#         # Process results and create JSON
#         detections_data = []
#         for frame_idx, result in enumerate(results):
#             frame_detections = []
            
#             # Extract detection information
#             boxes = result.boxes
#             for box in boxes:
#                 detection = {
#                     'class': result.names[int(box.cls[0])],
#                     'confidence': float(box.conf[0]),
#                     'bbox': box.xyxy[0].tolist()  # [x1, y1, x2, y2]
#                 }
#                 frame_detections.append(detection)
            
#             detections_data.append({
#                 'frame': frame_idx,
#                 'timestamp': frame_idx / 30.0,  # Assuming 30 fps, adjust as needed
#                 'detections': frame_detections
#             })
        
#         # Save JSON results
#         json_path = os.path.join(app.config['RESULTS_FOLDER'], f"{job_id}_detections.json")
#         with open(json_path, 'w') as f:
#             json.dump(detections_data, f, indent=2)
        
#         # Find the output video
#         output_video_dir = os.path.join(app.config['RESULTS_FOLDER'], job_id)
#         output_video_path = None
        
#         if os.path.exists(output_video_dir):
#             for file in os.listdir(output_video_dir):
#                 if file.endswith(('.mp4', '.avi')):
#                     output_video_path = os.path.join(output_video_dir, file)
#                     break
        
#         # Generate summary
#         summary = generate_summary(detections_data)
        
#         return jsonify({
#             'success': True,
#             'job_id': job_id,
#             'summary': summary,
#             'json_file': f"/api/download/json/{job_id}",
#             'video_file': f"/api/download/video/{job_id}" if output_video_path else None
#         })
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/download/json/<job_id>')
# def download_json(job_id):
#     json_path = os.path.join(app.config['RESULTS_FOLDER'], f"{job_id}_detections.json")
    
#     if not os.path.exists(json_path):
#         return jsonify({'error': 'JSON file not found'}), 404
    
#     return send_file(json_path, as_attachment=True, download_name=f"detections_{job_id}.json")

# @app.route('/api/download/video/<job_id>')
# def download_video(job_id):
#     output_video_dir = os.path.join(app.config['RESULTS_FOLDER'], job_id)
    
#     if not os.path.exists(output_video_dir):
#         return jsonify({'error': 'Video not found'}), 404
    
#     # Find video file
#     for file in os.listdir(output_video_dir):
#         if file.endswith(('.mp4', '.avi')):
#             video_path = os.path.join(output_video_dir, file)
#             return send_file(video_path, as_attachment=True, download_name=f"detected_{job_id}.mp4")
    
#     return jsonify({'error': 'Video file not found'}), 404

# @app.route('/api/stream/video/<job_id>')
# def stream_video(job_id):
#     """Stream video for preview in browser"""
#     output_video_dir = os.path.join(app.config['RESULTS_FOLDER'], job_id)
    
#     if not os.path.exists(output_video_dir):
#         return jsonify({'error': 'Video not found'}), 404
    
#     for file in os.listdir(output_video_dir):
#         if file.endswith(('.mp4', '.avi')):
#             video_path = os.path.join(output_video_dir, file)
#             return send_file(video_path, mimetype='video/mp4')
    
#     return jsonify({'error': 'Video file not found'}), 404

# if __name__ == '__main__':
#     load_model()
#     app.run(debug=True, host='0.0.0.0', port=5000)


# This with video sekali summarize
# from flask import Flask, request, jsonify, send_file, render_template
# from werkzeug.utils import secure_filename
# import os
# import json
# import uuid
# import cv2
# import numpy as np
# from collections import defaultdict
# from ultralytics import YOLO
# from ultralytics.utils.plotting import colors

# app = Flask(__name__)

# # Configuration
# UPLOAD_FOLDER = 'uploads'
# RESULTS_FOLDER = 'results'
# ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
# MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# # Create necessary directories
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(RESULTS_FOLDER, exist_ok=True)

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# # Load your fine-tuned YOLO model
# MODEL_PATH = 'yolo26n.pt'  # Update this with your model path
# model = None

# def load_model():
#     global model
#     try:
#         model = YOLO(MODEL_PATH)
#         print(f"Model loaded successfully from {MODEL_PATH}")
#         print(f"Model classes: {model.names}")
#     except Exception as e:
#         print(f"Warning: Could not load model from {MODEL_PATH}")
#         print(f"Error: {e}")
#         print("Please update MODEL_PATH in app.py with your model file path")

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def format_time(milliseconds):
#     """Converts milliseconds to MM:SS format."""
#     seconds = int(milliseconds / 1000)
#     m, s = divmod(seconds, 60)
#     return f"{m:02d}:{s:02d}"

# def draw_bbox(im0, box, track_id, cls, names, rect_width=2, font=1.0, text_width=2, padding=12, margin=10):
#     """Draw bounding box with label at TOP-LEFT, but TEXT CENTERED in its box."""
#     x1, y1, x2, y2 = map(int, box)
#     color = colors(int(cls), True)

#     # Draw main bounding box
#     cv2.rectangle(im0, (x1, y1), (x2, y2), color, rect_width)

#     # Prepare label
#     label = f"{names[int(cls)]}:{int(track_id)}"

#     # Get text size
#     (tw, th), _ = cv2.getTextSize(
#         label, cv2.FONT_HERSHEY_SIMPLEX, font, text_width
#     )

#     bg_x1 = x1 
#     bg_x2 = bg_x1 + (tw + 2 * padding)
#     bg_y2 = y1 
#     bg_y1 = bg_y2 - (th + 2 * margin)

#     # Draw filled background rectangle
#     cv2.rectangle(im0, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)

#     text_x = bg_x1 + ((bg_x2 - bg_x1) - tw) // 2
#     text_y = bg_y1 + ((bg_y2 - bg_y1) + th) // 2 - 2 

#     cv2.putText(
#         im0, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
#         font, (255, 255, 255), text_width, cv2.LINE_AA,
#     )

# def generate_summary(first_seen_registry):
#     """Generate a human-readable summary from tracking results"""
#     summary = {
#         'total_unique_objects': len(first_seen_registry),
#         'object_counts': {},
#         'objects_by_class': {}
#     }
    
#     # Count objects by class
#     for track_id, data in first_seen_registry.items():
#         obj_class = data['class']
#         summary['object_counts'][obj_class] = summary['object_counts'].get(obj_class, 0) + 1
        
#         if obj_class not in summary['objects_by_class']:
#             summary['objects_by_class'][obj_class] = []
        
#         summary['objects_by_class'][obj_class].append({
#             'track_id': track_id,
#             'first_seen': data['first_seen']
#         })
    
#     return summary

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/api/upload', methods=['POST'])
# def upload_video():
#     if model is None:
#         return jsonify({'error': 'Model not loaded. Please check MODEL_PATH in app.py'}), 500
    
#     # Check if file is in request
#     if 'video' not in request.files:
#         return jsonify({'error': 'No video file provided'}), 400
    
#     file = request.files['video']
    
#     if file.filename == '':
#         return jsonify({'error': 'No file selected'}), 400
    
#     if not allowed_file(file.filename):
#         return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv'}), 400
    
#     try:
#         # Generate unique ID for this processing job
#         job_id = str(uuid.uuid4())
        
#         # Save uploaded file
#         filename = secure_filename(file.filename)
#         upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
#         file.save(upload_path)
        
#         # Define output paths
#         video_output_path = os.path.join(app.config['RESULTS_FOLDER'], f"{job_id}_object-tracking.avi")
#         json_output_path = os.path.join(app.config['RESULTS_FOLDER'], f"{job_id}_tracking_data.json")
        
#         # Video capturing module
#         cap = cv2.VideoCapture(upload_path)
#         assert cap.isOpened(), "Error reading video file"

#         # Video properties
#         w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = cap.get(cv2.CAP_PROP_FPS)
        
#         # Video writing module
#         writer = cv2.VideoWriter(
#             video_output_path,
#             cv2.VideoWriter_fourcc(*"mp4v"),
#             fps, 
#             (w, h)
#         )
        
#         # Track history and first seen registry
#         track_history = defaultdict(lambda: [])
#         first_seen_registry = {}
        
#         # Get model names
#         names = model.names
        
#         # Display settings
#         rect_width = 2
#         font = 1.0
#         text_width = 2
#         padding = 12
#         margin = 10
        
#         print(f"Processing video: {filename}")
#         frame_count = 0
        
#         # Process video with tracking
#         while cap.isOpened():
#             success, im0 = cap.read()
#             if not success:
#                 break
            
#             frame_count += 1
            
#             # Get current timestamp in milliseconds
#             current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            
#             # Run YOLO tracking
#             results = model.track(im0, persist=True, conf=0.7)
            
#             if results and len(results) > 0:
#                 result = results[0]
                
#                 if result.boxes is not None and result.boxes.id is not None:
#                     boxes = result.boxes.xyxy.cpu()
#                     ids = result.boxes.id.cpu()
#                     clss = result.boxes.cls.tolist()
                    
#                     for box, id, cls in zip(boxes, ids.tolist(), clss):
#                         track_id = int(id)
                        
#                         # IF ID NOT SEEN BEFORE, RECORD TIME
#                         if track_id not in first_seen_registry:
#                             formatted_time = format_time(current_ms)
#                             first_seen_registry[track_id] = {
#                                 "class": names[int(cls)],
#                                 "first_seen": formatted_time
#                             }
                        
#                         # Draw bounding box
#                         draw_bbox(im0, box, id, cls, names, rect_width, font, text_width, padding, margin)
                        
#                         # Track lines drawing (optional - can be removed if you don't want trails)
#                         x1, y1, x2, y2 = box
#                         track = track_history[id]
#                         track.append((float((x1+x2)/2), float((y1+y2)/2)))
#                         if len(track) > 50:
#                             track.pop(0)
            
#             # Write frame to output video
#             writer.write(im0)
            
#             # Progress feedback
#             if frame_count % 30 == 0:
#                 print(f"Processed {frame_count} frames...")
        
#         # Cleanup
#         cap.release()
#         writer.release()
        
#         print(f"Video processing complete. Total frames: {frame_count}")
#         print(f"Total unique objects tracked: {len(first_seen_registry)}")
        
#         # Save JSON file with tracking data
#         with open(json_output_path, 'w') as f:
#             json.dump(first_seen_registry, f, indent=4)
        
#         print(f"Tracking data saved to: {json_output_path}")
        
#         # Generate summary
#         summary = generate_summary(first_seen_registry)
        
#         return jsonify({
#             'success': True,
#             'job_id': job_id,
#             'summary': summary,
#             'tracking_data': first_seen_registry,
#             'json_file': f"/api/download/json/{job_id}",
#             'video_file': f"/api/download/video/{job_id}"
#         })
    
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/download/json/<job_id>')
# def download_json(job_id):
#     json_path = os.path.join(app.config['RESULTS_FOLDER'], f"{job_id}_tracking_data.json")
    
#     if not os.path.exists(json_path):
#         return jsonify({'error': 'JSON file not found'}), 404
    
#     return send_file(json_path, as_attachment=True, download_name=f"tracking_data_{job_id}.json")

# @app.route('/api/download/video/<job_id>')
# def download_video(job_id):
#     video_path = os.path.join(app.config['RESULTS_FOLDER'], f"{job_id}_object-tracking.avi")
    
#     if not os.path.exists(video_path):
#         return jsonify({'error': 'Video file not found'}), 404
    
#     return send_file(video_path, as_attachment=True, download_name=f"object-tracking_{job_id}.avi")

# @app.route('/api/stream/video/<job_id>')
# def stream_video(job_id):
#     """Stream video for preview in browser"""
#     video_path = os.path.join(app.config['RESULTS_FOLDER'], f"{job_id}_object-tracking.avi")
    
#     if not os.path.exists(video_path):
#         return jsonify({'error': 'Video file not found'}), 404
    
#     return send_file(video_path, mimetype='video/x-msvideo')

# if __name__ == '__main__':
#     load_model()
#     app.run(debug=True, host='0.0.0.0', port=5000)

from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
import os
import json
import uuid
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Load your fine-tuned YOLO model
MODEL_PATH = 'yolo26n.pt'  # Update this with your model path
model = None

def load_model():
    global model
    try:
        model = YOLO(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
        print(f"Model classes: {model.names}")
    except Exception as e:
        print(f"Warning: Could not load model from {MODEL_PATH}")
        print(f"Error: {e}")
        print("Please update MODEL_PATH in app.py with your model file path")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def format_time(milliseconds):
    """Converts milliseconds to MM:SS format."""
    seconds = int(milliseconds / 1000)
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"

def generate_summary(first_seen_registry):
    """Generate a human-readable summary from tracking results"""
    summary = {
        'total_unique_objects': len(first_seen_registry),
        'object_counts': {},
        'objects_by_class': {}
    }
    
    # Count objects by class
    for track_id, data in first_seen_registry.items():
        obj_class = data['class']
        summary['object_counts'][obj_class] = summary['object_counts'].get(obj_class, 0) + 1
        
        if obj_class not in summary['objects_by_class']:
            summary['objects_by_class'][obj_class] = []
        
        summary['objects_by_class'][obj_class].append({
            'track_id': track_id,
            'first_seen': data['first_seen']
        })
    
    return summary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_video():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check MODEL_PATH in app.py'}), 500
    
    # Check if file is in request
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv'}), 400
    
    try:
        # Generate unique ID for this processing job
        job_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
        file.save(upload_path)
        
        # Define output paths (JSON only, no video)
        json_output_path = os.path.join(app.config['RESULTS_FOLDER'], f"{job_id}_tracking_data.json")
        
        # Video capturing module
        cap = cv2.VideoCapture(upload_path)
        assert cap.isOpened(), "Error reading video file"

        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Track history and first seen registry
        first_seen_registry = {}
        
        # Get model names
        names = model.names
        
        print(f"Processing video: {filename}")
        frame_count = 0
        
        # Process video with tracking (no video output)
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                break
            
            frame_count += 1
            
            # Get current timestamp in milliseconds
            current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            
            # Run YOLO tracking
            results = model.track(im0, persist=True, conf=0.7, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu()
                    ids = result.boxes.id.cpu()
                    clss = result.boxes.cls.tolist()
                    
                    for box, id, cls in zip(boxes, ids.tolist(), clss):
                        track_id = int(id)
                        
                        # IF ID NOT SEEN BEFORE, RECORD TIME
                        if track_id not in first_seen_registry:
                            formatted_time = format_time(current_ms)
                            first_seen_registry[track_id] = {
                                "class": names[int(cls)],
                                "first_seen": formatted_time
                            }
            
            # Progress feedback
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
        
        # Cleanup
        cap.release()
        
        print(f"Video processing complete. Total frames: {frame_count}")
        print(f"Total unique objects tracked: {len(first_seen_registry)}")
        
        # Save JSON file with tracking data
        with open(json_output_path, 'w') as f:
            json.dump(first_seen_registry, f, indent=4)
        
        print(f"Tracking data saved to: {json_output_path}")
        
        # Generate summary
        summary = generate_summary(first_seen_registry)
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'summary': summary,
            'tracking_data': first_seen_registry,
            'json_file': f"/api/download/json/{job_id}"
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/json/<job_id>')
def download_json(job_id):
    json_path = os.path.join(app.config['RESULTS_FOLDER'], f"{job_id}_tracking_data.json")
    
    if not os.path.exists(json_path):
        return jsonify({'error': 'JSON file not found'}), 404
    
    return send_file(json_path, as_attachment=True, download_name=f"tracking_data_{job_id}.json")

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
