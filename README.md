# Solar Panel Anomaly Detection - Frontend

A Flask-based web application for detecting anomalies in photovoltaic (PV) systems using real-time image processing and deep learning models.

## Features

- **User Authentication**: Secure login and registration system with SQLite database
- **Live Detection**: Real-time anomaly detection using YOLOv8 model (best.pt)
- **Detection History**: Track and view past detections with timestamps
- **Responsive UI**: Bootstrap-based responsive design for desktop and mobile
- **Dashboard**: Home page with statistics and quick access to features

## Project Structure

```
├── app.py                 # Main Flask application
├── best.pt               # YOLOv8 trained model for anomaly detection
├── app.db                # SQLite database (auto-created)
├── requirements.txt      # Python dependencies
├── templates/
│   ├── base.html         # Base template with navigation
│   ├── home.html         # Dashboard home page
│   ├── index.html        # Main page
│   ├── login.html        # Login page
│   ├── register.html     # User registration page
│   ├── live_detection.html # Real-time detection interface
│   └── history.html      # Detection history page
└── README.md             # This file
```

## Installation

### Prerequisites
- Python 3.10.18
- Conda (Anaconda/Miniconda) or pip

### Setup

1. Clone the repository or navigate to the project directory:
```bash
cd "Anomaly detection in PV\CODE\FRONTEND"
```

2. Create and activate the conda environment:
```bash
conda create -n aienv python=3.10.18
conda activate aienv
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to:
```
http://localhost:5000
```

## Database Schema

### Users Table
- `id`: Integer primary key
- `username`: Text, unique, not null
- `email`: Text, unique, not null
- `password_hash`: Text, not null

### Detections Table
- `id`: Integer primary key
- `user_id`: Integer, foreign key to users.id
- `timestamp`: DateTime of detection
- `detections_json`: JSON data of anomalies detected

## Usage

### Registration
1. Click "Register" on the login page
2. Enter a unique username and email
3. Create a secure password
4. Submit to create your account

### Login
1. Enter your username and password
2. Click "Login"

### Live Detection
1. Navigate to "Live Detection" from the dashboard
2. Allow camera/webcam access when prompted
3. The model will process frames in real-time and highlight anomalies
4. Detections are automatically saved to your history

### View History
1. Click "History" to see all your past detections
2. View detection details with timestamps and anomaly information

## Dependencies

- Flask: Web framework
- Flask-Login: User session management
- sqlite3: Database management
- Werkzeug: Password hashing and security
- YOLOv8: Object detection model (via requirements.txt)

See `requirements.txt` for complete list with versions.

## Configuration

The application uses:
- SQLite database: `app.db` (auto-created on first run)
- Flask debug mode: Disabled in production
- Session management: Flask-Login

## Troubleshooting

### Database Error: "table users has no column named email"
If you get this error, delete `app.db` and restart the application:
```bash
Remove-Item app.db -Force
python app.py
```

The database will be recreated with the correct schema.

### Model Loading Issues
Ensure `best.pt` is in the same directory as `app.py`

### Camera/Webcam Not Working
- Check browser permissions for camera access
- Ensure no other application is using the camera
- Try a different browser (Chrome/Firefox recommended)

## Security Notes

- Passwords are hashed using Werkzeug security functions
- Use HTTPS in production environments
- Change Flask SECRET_KEY before deploying to production
- Never commit the database file to version control

## Features

✅ **Live Video Stream** - Real-time webcam feed with object detection
✅ **Real-time Detection** - Instant detection of solar panel faults
✅ **Detection Statistics** - Shows detection count, average confidence, and FPS
✅ **Responsive UI** - Works on desktop and mobile devices
✅ **Start/Stop Controls** - Easy control of detection process

## Setup & Installation

### Prerequisites
- Python 3.8 or higher
- Webcam connected to your computer
- NVIDIA GPU (recommended for real-time performance) or CPU fallback

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Place Your Model

Copy your trained model file (`best.pt`) to the same directory as `app.py`:

```
FRONTEND/
├── app.py
├── best.pt          ← Place your model here
├── requirements.txt
├── templates/
│   └── index.html
└── README.md
```

### 3. Run the Application

```bash
python app.py
```

You should see:
```
==================================================
Solar Panel Fault Detection - YOLO12 Live
==================================================
✓ Model loaded successfully: best.pt

Starting Flask server...
Open http://localhost:5000 in your browser
```

### 4. Open in Browser

Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage

1. **Start Detection**: Click the "▶️ Start Detection" button to begin live detection
2. **View Results**: The video stream will show detected faults with bounding boxes
3. **Monitor Stats**: Track detection count, confidence, and FPS in real-time
4. **Stop Detection**: Click "⏹️ Stop Detection" to stop the detection process

## Configuration

### Model Path
Edit the `MODEL_PATH` variable in `app.py`:
```python
MODEL_PATH = "best.pt"  # Change this to your model path
```

### GPU/CPU
In `app.py`, line in `process_frame()`:
```python
device=0,  # Use 0 for GPU, 'cpu' for CPU
```

### Inference Settings
Modify in `process_frame()` function:
```python
results = model.predict(
    source=frame,
    imgsz=640,        # Image size (increase for higher accuracy)
    conf=0.50,        # Confidence threshold (0-1)
    device=0,         # Device (0=GPU, 'cpu'=CPU)
)
```

### Camera Selection
Change in `generate_frames()`:
```python
camera = cv2.VideoCapture(0)  # 0 = default camera, try 1, 2, etc. for other cameras
```

### Frame Processing
Adjust frame skipping for performance:
```python
if frame_count % 1 == 0:  # Process every 1st frame (change to 2 for every 2nd, etc.)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main page |
| `/video_feed` | GET | Video stream (MJPEG) |
| `/api/detections` | GET | Latest detections (JSON) |
| `/api/status` | GET | System status |
| `/api/start` | POST | Start detection |
| `/api/stop` | POST | Stop detection |

## System Requirements

### Minimum
- 4GB RAM
- CPU: Intel i5 or equivalent
- Webcam with 30 FPS capability

### Recommended (for real-time performance)
- 8GB+ RAM
- NVIDIA GPU with CUDA 11.8+
- Webcam with 30+ FPS capability

## Troubleshooting

### Camera not working
- Ensure webcam permissions are granted
- Try changing camera index: `cv2.VideoCapture(1)` or `cv2.VideoCapture(2)`
- Check if another application is using the camera

### Slow performance
- Reduce `imgsz` from 640 to 416 or 320
- Increase frame skipping: change `% 1` to `% 2` or `% 3`
- Use CPU instead of GPU if having VRAM issues
- Ensure no other heavy processes are running

### Model not loading
- Verify `best.pt` exists in the same directory as `app.py`
- Check file permissions
- Ensure sufficient disk space
- Try downloading model again if corrupted

### High latency
- Check network if accessing remotely
- Reduce frame resolution from 640 to 480 or 320
- Disable other browser tabs/applications

## Performance Tips

1. **GPU Acceleration**: Install CUDA for faster inference
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Optimize Model**: Use smaller YOLOv12n model if available

3. **Reduce Resolution**: Lower camera resolution reduces processing time

4. **Frame Skipping**: Process every 2nd or 3rd frame instead of every frame

## File Structure

```
FRONTEND/
├── app.py                 # Flask backend
├── best.pt               # YOLOv12 trained model
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── templates/
    └── index.html        # Web interface
```

## Advanced Configuration

### Multi-camera support
Modify `generate_frames()` to support multiple cameras:
```python
camera = cv2.VideoCapture(camera_index)  # Pass camera index as parameter
```

### Custom detection confidence per class
In `process_frame()`, filter detections by class:
```python
if conf > 0.5 and class_name == 'your_class':
    detections.append({...})
```

### Export detections to file
Add logging functionality to save detections to JSON/CSV

## License

This project is for Solar Panel Anomaly Detection research.

## Support

For issues or questions, please check:
1. Console output in terminal
2. Browser developer tools (F12)
3. Model file integrity
4. Camera permissions

---
**Last Updated**: January 2026
