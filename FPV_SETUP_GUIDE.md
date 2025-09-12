# FPV Live Feed Setup Guide

This guide explains how to set up and use the FPV (First Person View) live feed functionality in Foresight for real-time object detection and SAR operations.

## 🎯 Overview

The FPV system allows you to:
- Capture live video from webcams, capture cards, or DJI FPV systems
- Apply real-time YOLO object detection
- Display live footage with bounding boxes in a web interface
- Monitor detection statistics and performance

## 🛒 Hardware Requirements

### Option A: DJI O4 Lite / Goggles 3 (Recommended)

**What you need:**
- DJI FPV Goggles 3 or O4 Lite unit
- USB HDMI capture dongle (UVC compatible)
- HDMI cable

**Setup:**
1. Connect Goggles HDMI output to capture dongle
2. Connect capture dongle to computer via USB
3. Device appears as standard webcam in system

**Pros:** Simple setup, low latency, standard UVC device
**Cons:** Requires capture card (~$20-50)

### Option B: DJI SDK Integration

**What you need:**
- DJI drone with SDK support
- DJI Developer account
- SDK setup and configuration

**Setup:**
1. Register for DJI Developer account
2. Download and configure DJI SDK
3. Enable SDK mode on drone
4. Configure network connection

**Pros:** Direct telemetry integration, no capture card needed
**Cons:** More complex setup, firmware dependent

### Option C: Analog FPV (TinyWhoop)

**What you need:**
- Analog FPV transmitter/receiver
- Analog-to-USB capture device
- Composite or HDMI converter

**Setup:**
1. Connect analog receiver to converter
2. Connect converter to USB capture device
3. Connect to computer

**Pros:** Works with any analog FPV system
**Cons:** Lower video quality, additional conversion steps

## 🔧 Software Installation

### 1. Install Dependencies

```bash
# Install FPV-specific requirements
pip install -r requirements-fpv.txt

# Or install core packages manually
pip install opencv-python ultralytics fastapi uvicorn numpy pillow
```

### 2. Download YOLO Models

The system supports multiple YOLO versions:

```bash
# YOLOv8 (recommended)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt

# Place models in models/ directory
mkdir -p models
mv yolov8*.pt models/
```

### 3. Configure Environment

Create or update `.env` file:

```env
# FPV Configuration
FPV_DEFAULT_MODEL=models/yolov8n.pt
FPV_CONFIDENCE_THRESHOLD=0.5
FPV_MAX_FPS=30
FPV_BUFFER_SIZE=5

# Detection Classes (SAR focus)
SAR_CLASSES=person,car,truck,boat,airplane

# Hardware Settings
USE_GPU=true
DEVICE_TIMEOUT=5000
```

## 🚀 Quick Start

### 1. Start the Application

```bash
# Navigate to project directory
cd foresight

# Start the FPV-enabled server
python -m uvicorn src.backend.app:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Access the FPV Interface

Open your browser and navigate to:
- **FPV Live Feed:** http://localhost:8000/fpv
- **Main Interface:** http://localhost:8000/

### 3. Configure Video Source

1. In the FPV interface, click the "Video Source" dropdown
2. Select your capture device (webcam, capture card, etc.)
3. Click "Start FPV Capture"
4. Enable detection if desired

## 🎮 Using the Interface

### Video Controls
- **Select Source:** Choose between screen capture and connected devices
- **Start/Stop FPV:** Control video capture
- **Enable Detection:** Toggle YOLO object detection
- **Toggle SAR Mode:** Enable SAR-specific detection classes

### Real-time Information
- **Live Video:** Main video feed with detection overlays
- **FPS Counter:** Current frame rate
- **Detection Count:** Number of objects detected
- **Statistics:** Total detections, people found, uptime
- **Activity Log:** Real-time system messages

## 🔍 Detection Configuration

### SAR-Specific Classes

The system prioritizes detection of SAR-relevant objects:
- **Person:** Primary target for search and rescue
- **Vehicle:** Cars, trucks, boats for evacuation
- **Aircraft:** Helicopters, planes for coordination
- **Structures:** Buildings, shelters for reference

### Performance Tuning

```python
# In yolo_wrapper.py, adjust these settings:
CONFIDENCE_THRESHOLD = 0.5  # Lower = more detections
NMS_THRESHOLD = 0.4         # Non-max suppression
MAX_DETECTIONS = 100        # Maximum objects per frame
INPUT_SIZE = 640            # Model input resolution
```

## 🛠️ Troubleshooting

### Common Issues

**No video devices found:**
```bash
# List available devices (Linux)
ls /dev/video*

# List devices (Windows)
ffmpeg -list_devices true -f dshow -i dummy

# Test device access
python -c "import cv2; print(cv2.VideoCapture(0).read())"
```

**Low FPS or lag:**
- Reduce video resolution in capture settings
- Use smaller YOLO model (yolov8n vs yolov8s)
- Enable GPU acceleration
- Increase buffer size for smoother playback

**Detection not working:**
- Check model file exists in models/ directory
- Verify CUDA/GPU setup for acceleration
- Lower confidence threshold for more detections
- Check log output for error messages

**Capture card not detected:**
- Ensure UVC (USB Video Class) compatibility
- Try different USB ports (USB 3.0 preferred)
- Check device drivers on Windows
- Test with other applications (VLC, OBS)

### Debug Mode

Enable detailed logging:

```bash
# Set environment variable
export LOG_LEVEL=DEBUG

# Or modify app.py
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Performance Optimization

### Hardware Recommendations

**Minimum:**
- CPU: Intel i5 / AMD Ryzen 5
- RAM: 8GB
- GPU: Integrated graphics
- USB: 2.0 ports

**Recommended:**
- CPU: Intel i7 / AMD Ryzen 7
- RAM: 16GB
- GPU: NVIDIA GTX 1060 / RTX 3060
- USB: 3.0+ ports

**Optimal:**
- CPU: Intel i9 / AMD Ryzen 9
- RAM: 32GB
- GPU: NVIDIA RTX 4070+
- USB: 3.1+ with dedicated controller

### Software Optimization

```python
# GPU acceleration (if available)
import torch
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Multi-threading
threading_enabled = True
max_workers = 4

# Memory management
torch.cuda.empty_cache()  # Clear GPU memory
gc.collect()              # Python garbage collection
```

## 🔗 API Reference

### REST Endpoints

```http
# Get available video sources
GET /api/fpv/sources

# Start FPV capture
POST /api/fpv/start
Content-Type: application/json
{"source_id": "0"}

# Stop FPV capture
POST /api/fpv/stop

# Toggle detection
POST /api/fpv/toggle_detection

# Get statistics
GET /api/fpv/stats

# Video stream
GET /api/video_feed
```

### WebSocket Events

```javascript
// Connect to real-time updates
const ws = new WebSocket('ws://localhost:8000/ws');

// Receive detection updates
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    // data.fps, data.detections, data.mode
};
```

## 📝 Integration Examples

### Custom Detection Pipeline

```python
from src.backend.services.fpv_capture import FPVCaptureService
from src.backend.services.yolo_wrapper import YOLOWrapper

# Initialize services
capture = FPVCaptureService()
yolo = YOLOWrapper(model_path='models/yolov8n.pt')

# Start capture
capture.start_capture(source_id=0)

# Process frames
while True:
    frame = capture.get_frame()
    if frame is not None:
        detections = yolo.detect(frame)
        annotated_frame = yolo.draw_detections(frame, detections)
        # Display or save annotated_frame
```

### Custom Web Interface

```html
<!-- Embed video feed -->
<img src="/api/video_feed" alt="Live Feed">

<!-- Control buttons -->
<button onclick="fetch('/api/fpv/start', {method: 'POST'})">Start</button>
<button onclick="fetch('/api/fpv/stop', {method: 'POST'})">Stop</button>
```

## 🆘 Support

For additional help:
1. Check the activity log in the web interface
2. Review console output for error messages
3. Test hardware with other applications
4. Consult the main project documentation
5. Submit issues with detailed system information

## 📋 Checklist

Before deployment:
- [ ] Hardware connected and detected
- [ ] Dependencies installed
- [ ] YOLO models downloaded
- [ ] Environment configured
- [ ] Video feed working
- [ ] Detection enabled and tested
- [ ] Performance acceptable
- [ ] Web interface accessible