from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path
import threading
import queue
import time
import torch
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import json

app = Flask(__name__)
app.secret_key = 'super_secret_key'  # Change this to a secure random key in production
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Database setup
DB_PATH = 'database.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    # Detections history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            timestamp DATETIME NOT NULL,
            detections_json TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT id, username FROM users WHERE id = ?', (user_id,))
    user_data = cursor.fetchone()
    conn.close()
    if user_data:
        return User(user_data[0], user_data[1])
    return None

# Load the trained YOLO model
MODEL_PATH = "best.pt"  # Update this path to your model
# Auto-detect CUDA availability
DEVICE = "0" if torch.cuda.is_available() else "cpu"
print(f"🔧 CUDA available: {torch.cuda.is_available()}")
print(f"📊 Using device: {DEVICE}")

# Initialize variables
model = None
camera = None
detection_queue = queue.Queue(maxsize=1)
frame_lock = threading.Lock()
current_frame = None
is_running = False
detection_history = []  # Temporary in-memory for current session, will save to DB on stop

def load_model():
    """Load the YOLO model"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
            print(f"✓ Model loaded successfully: {MODEL_PATH}")
            return True
        else:
            print(f"✗ Model not found at {MODEL_PATH}")
            return False
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

def process_frame(frame):
    """Process a frame with YOLO and return annotated frame with detections"""
    try:
        # Run inference
        results = model.predict(
            source=frame,
            imgsz=640,
            conf=0.50,
            device=DEVICE,
            verbose=False
        )
        
        # Plot results
        annotated_frame = results[0].plot()
        
        # Extract detections info
        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0]
            
            detections.append({
                'class': class_name,
                'confidence': round(conf, 4),
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            })
        
        return annotated_frame, detections, len(detections) > 0
    
    except Exception as e:
        print(f"Error processing frame: {e}")
        return frame, [], False

def generate_frames():
    """Generate frames from webcam with real-time detection"""
    global camera, current_frame, is_running, detection_history
    
    try:
        camera = cv2.VideoCapture(0)  # Use webcam (0 is default)
        
        if not camera.isOpened():
            print("Error: Cannot open camera")
            is_running = False
            return
        
        # Set camera properties
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        
        is_running = True
        frame_count = 0
        last_save_time = time.time()
        
        while is_running:
            ret, frame = camera.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Process every frame for real-time
            if frame_count % 1 == 0:
                annotated_frame, detections, has_detections = process_frame(frame)
                
                # Store current frame
                with frame_lock:
                    current_frame = annotated_frame
                
                # Store for history (save every 5 seconds if detections)
                current_time = time.time()
                if has_detections and (current_time - last_save_time) > 5:
                    detection_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'detections': detections
                    })
                    last_save_time = current_time
                
                # Try to put detection info in queue (non-blocking)
                try:
                    detection_queue.put_nowait({
                        'count': len(detections),
                        'detections': detections
                    })
                except queue.Full:
                    pass
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', current_frame if current_frame is not None else frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' +
                   frame_bytes + b'\r\n')
    
    except Exception as e:
        print(f"Error in generate_frames: {e}")
    
    finally:
        is_running = False
        if camera:
            camera.release()

# Routes
@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('live_detection'))
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT id, username, password_hash FROM users WHERE username = ?', (username,))
        user_data = cursor.fetchone()
        conn.close()
        if user_data and check_password_hash(user_data[2], password):
            user = User(user_data[0], user_data[1])
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('home'))
        flash('Invalid username or password', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username         = request.form.get('username', '').strip()
        email            = request.form.get('email', '').strip().lower()
        password         = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        errors = []

        # Basic server-side validation
        if not username or len(username) < 4:
            errors.append("Username must be at least 4 characters long.")
        if not email or '@' not in email or '.' not in email.split('@')[-1]:
            errors.append("Please enter a valid email address.")
        if not password or len(password) < 8:
            errors.append("Password must be at least 8 characters long.")
        if password != confirm_password:
            errors.append("Passwords do not match.")

        if errors:
            for err in errors:
                flash(err, 'error')
            return render_template('register.html')

        # Hash the password
        password_hash = generate_password_hash(password)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT INTO users (username, email, password_hash)
                VALUES (?, ?, ?)
            ''', (username, email, password_hash))
            
            conn.commit()
            
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))

        except sqlite3.IntegrityError as e:
            if 'username' in str(e):
                flash('Username is already taken.', 'error')
            elif 'email' in str(e):
                flash('This email is already registered.', 'error')
            else:
                flash('Registration failed. Please try again.', 'error')
        
        finally:
            conn.close()

    return render_template('register.html')

@app.route('/live_detection')
@login_required
def live_detection():
    return render_template('live_detection.html')

@app.route('/video_feed')
@login_required
def video_feed():
    if model is None:
        return "Model not loaded", 500
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/api/detections')
@login_required
def get_detections():
    try:
        detections = detection_queue.get_nowait()
        return jsonify(detections)
    except queue.Empty:
        return jsonify({'count': 0, 'detections': []})

@app.route('/api/status')
@login_required
def get_status():
    return jsonify({
        'model_loaded': model is not None,
        'model_path': MODEL_PATH,
        'streaming': is_running,
        'camera_available': camera is not None and camera.isOpened() if camera else False,
        'cuda_available': torch.cuda.is_available(),
        'device': DEVICE,
        'device_name': f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"
    })

@app.route('/api/start', methods=['POST'])
@login_required
def start_detection():
    global is_running, detection_history
    if model is None:
        return jsonify({'success': False, 'message': 'Model not loaded'}), 500
    detection_history = []  # Reset history for new session
    if not is_running:
        threading.Thread(target=generate_frames, daemon=True).start()  # Start in background thread
        is_running = True
        return jsonify({'success': True, 'message': 'Live detection started'})
    return jsonify({'success': True, 'message': 'Detection already running'})

@app.route('/api/stop', methods=['POST'])
@login_required
def stop_detection():
    global is_running, detection_history
    is_running = False
    # Save history to DB
    if detection_history:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        detections_json = json.dumps(detection_history)
        cursor.execute('INSERT INTO detections (user_id, timestamp, detections_json) VALUES (?, ?, ?)',
                       (current_user.id, datetime.now(), detections_json))
        conn.commit()
        conn.close()
        detection_history = []
    return jsonify({'success': True, 'message': 'Live detection stopped'})

@app.route('/history')
@login_required
def history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT id, timestamp, detections_json FROM detections WHERE user_id = ? ORDER BY timestamp DESC',
                   (current_user.id,))
    history_data = cursor.fetchall()
    conn.close()
    # Process data for display
    processed_history = []
    for item in history_data:
        detections = json.loads(item[2])
        total_detections = sum(len(d['detections']) for d in detections)
        processed_history.append({
            'id': item[0],
            'timestamp': item[1],
            'total_detections': total_detections,
            'detections': detections
        })
    return render_template('history.html', history=processed_history)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully', 'success')
    return redirect(url_for('home'))

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Server error'}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("Solar Panel Fault Detection - YOLO12 Live")
    print("=" * 50)
    print(f"📊 Running on: {DEVICE.upper()}")
    if DEVICE == "cpu":
        print("⚠️ Running on CPU (slower inference). For better performance, install CUDA.")
        print(" See: https://pytorch.org/get-started/locally/")
    
    # Load model on startup
    if load_model():
        print("\nStarting Flask server...")
        print("Open http://localhost:5000 in your browser")
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
    else:
        print("\n✗ Failed to load model. Please check the model path.")
        print(f"Current path: {os.path.abspath(MODEL_PATH)}")