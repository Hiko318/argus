# Edge Device Setup Guide

This guide provides comprehensive instructions for setting up the Foresight drone-based edge computing system on various edge devices, including NVIDIA Jetson platforms, Raspberry Pi, and generic Linux systems.

## 🎯 Overview

The edge device setup includes:
- **DJI SDK Integration**: For drone video streaming and telemetry
- **AI/ML Stack**: PyTorch, OpenCV, YOLOv8 for object detection and tracking
- **CUDA/TensorRT**: GPU acceleration on NVIDIA Jetson devices
- **Offline Mapping**: Pre-downloaded map tiles for offline operation
- **Web Services**: FastAPI backend for real-time communication

## 🔧 Supported Devices

### Primary Targets
- **NVIDIA Jetson Nano** (4GB recommended)
- **NVIDIA Jetson Xavier NX**
- **NVIDIA Jetson AGX Orin**
- **Raspberry Pi 4** (8GB recommended)
- **Generic Linux** (Ubuntu 20.04+)

### Development/Testing
- **Windows 10/11** (limited functionality)
- **macOS** (limited functionality)

## 🚀 Quick Start

### Automated Setup

```bash
# Clone the repository
git clone <repository-url>
cd foresight

# Run automated setup
python scripts/setup_edge_device.py

# Check setup report
cat setup_report.md
```

### Manual Setup

If you prefer manual installation or need to troubleshoot:

```bash
# Install Python dependencies
pip install -r requirements.txt

# Download AI models
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8s.pt')"

# Test the system
python -m src.backend.detection_pipeline yolov8n.pt
```

## 📋 Detailed Setup Instructions

### 1. System Prerequisites

#### For NVIDIA Jetson Devices
```bash
# Install JetPack SDK (includes CUDA, TensorRT, OpenCV)
# Download from: https://developer.nvidia.com/jetpack

# Verify CUDA installation
nvcc --version

# Install additional tools
sudo apt update
sudo apt install -y python3-pip python3-dev build-essential
```

#### For Raspberry Pi
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3-pip python3-dev python3-opencv
sudo apt install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev
sudo apt install -y libatlas-base-dev libjasper-dev libqtgui4 libqt4-test
```

#### For Generic Linux
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3-pip python3-dev python3-opencv
sudo apt install -y build-essential cmake pkg-config

# CentOS/RHEL
sudo yum install -y python3-pip python3-devel opencv-python
sudo yum groupinstall -y "Development Tools"
```

### 2. Python Environment Setup

```bash
# Create virtual environment (recommended)
python3 -m venv foresight-env
source foresight-env/bin/activate  # Linux/macOS
# foresight-env\Scripts\activate     # Windows

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### 3. DJI SDK Integration

#### For DJI Tello (Development/Testing)
```bash
# Already included in requirements.txt
pip install djitellopy av

# Test connection (with Tello powered on)
python -m src.backend.drone_sdk
```

#### For DJI Payload SDK (Production)
```bash
# Download DJI Payload SDK from:
# https://developer.dji.com/payload-sdk/

# Follow DJI's installation guide for your platform
# Update src/backend/drone_sdk.py with Payload SDK integration
```

#### For DJI Mobile SDK (Android)
```bash
# For Android development:
# 1. Download DJI Mobile SDK from DJI Developer Portal
# 2. Integrate with Android Studio project
# 3. Use WebSocket/HTTP API to communicate with edge device
```

### 4. CUDA/TensorRT Optimization (Jetson Only)

```bash
# Verify CUDA installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install TensorRT Python bindings
pip install pycuda

# Test TensorRT optimization
python -m src.backend.cuda_tensorrt

# Convert YOLO model to TensorRT (optional)
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='engine', device=0)  # Creates yolov8n.engine
"
```

### 5. Offline Map Preparation

```bash
# Test offline maps functionality
python -m src.backend.offline_maps

# Download maps for specific area
python -c "
from src.backend.offline_maps import OfflineMapManager
manager = OfflineMapManager('data/maps')

# Define your operation area
bbox = {
    'north': 37.8,   # Replace with your coordinates
    'south': 37.7,
    'east': -122.3,
    'west': -122.5
}

# Download tiles for zoom levels 10-16
stats = manager.download_area(bbox, zoom_levels=[10, 12, 14, 16])
print(f'Downloaded {stats[\"total_tiles\"]} tiles')
"
```

### 6. System Service Setup (Linux)

```bash
# Create systemd service
sudo cp /tmp/foresight-edge.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable foresight-edge
sudo systemctl start foresight-edge

# Check service status
sudo systemctl status foresight-edge

# View logs
sudo journalctl -u foresight-edge -f
```

## 🧪 Testing and Validation

### Component Tests

```bash
# Test AI detection pipeline
python -m src.backend.detector yolov8n.pt

# Test object tracking
python -m src.backend.tracker

# Test complete pipeline
python -m src.backend.detection_pipeline yolov8n.pt

# Test drone integration (with drone connected)
python -m src.backend.drone_sdk

# Test offline maps
python -m src.backend.offline_maps

# Test CUDA/TensorRT (Jetson only)
python -m src.backend.cuda_tensorrt
```

### Performance Benchmarks

```bash
# Run performance tests
python scripts/benchmark_performance.py

# Expected performance (approximate):
# Jetson Nano: 5-10 FPS (YOLOv8n)
# Jetson Xavier NX: 15-25 FPS (YOLOv8s)
# Jetson AGX Orin: 30-50 FPS (YOLOv8m)
# Raspberry Pi 4: 1-3 FPS (YOLOv8n)
```

### Integration Tests

```bash
# Start the complete system
python -m src.backend.main

# In another terminal, test API endpoints
curl http://localhost:8000/health
curl http://localhost:8000/api/detection/status
```

## 📊 Performance Optimization

### For NVIDIA Jetson

```bash
# Enable maximum performance mode
sudo nvpmodel -m 0  # Max performance
sudo jetson_clocks   # Max clock speeds

# Monitor performance
sudo tegrastats

# Use TensorRT optimized models
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='engine', half=True)  # FP16 precision
"
```

### For Raspberry Pi

```bash
# Increase GPU memory split
sudo raspi-config
# Advanced Options > Memory Split > 128

# Enable camera
sudo raspi-config
# Interface Options > Camera > Enable

# Optimize for inference
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
```

### Memory Optimization

```bash
# Use smaller models for resource-constrained devices
# YOLOv8n: ~6MB, fastest
# YOLOv8s: ~22MB, balanced
# YOLOv8m: ~52MB, more accurate

# Enable model quantization
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx', int8=True)  # INT8 quantization
"
```

## 🔧 Troubleshooting

### Common Issues

#### CUDA Not Available
```bash
# Check CUDA installation
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA support
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Memory Issues
```bash
# Monitor memory usage
free -h
nvidia-smi  # For Jetson devices

# Reduce batch size or use smaller models
# Edit detection_pipeline.py to use yolov8n.pt instead of yolov8s.pt
```

#### Drone Connection Issues
```bash
# Check network connectivity
ping 192.168.10.1  # Tello IP

# Verify drone is in AP mode
# Check WiFi networks for "TELLO-XXXXXX"

# Test with DJI Tello app first
```

#### Permission Issues
```bash
# Add user to video group (for camera access)
sudo usermod -a -G video $USER

# Set up udev rules for USB devices
sudo cp configs/99-usb-devices.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
```

### Log Analysis

```bash
# Check setup logs
tail -f setup_edge_device.log

# Check application logs
tail -f logs/foresight.log

# Check system logs
sudo journalctl -u foresight-edge -f
```

## 📁 Directory Structure

```
foresight/
├── src/backend/
│   ├── detector.py          # YOLOv8 detection
│   ├── tracker.py           # Object tracking
│   ├── detection_pipeline.py # Integrated pipeline
│   ├── drone_sdk.py         # DJI integration
│   ├── offline_maps.py      # Map management
│   ├── cuda_tensorrt.py     # GPU optimization
│   └── main.py              # Main application
├── scripts/
│   ├── setup_edge_device.py # Automated setup
│   └── benchmark_performance.py
├── configs/
│   ├── dji_config.json      # Drone settings
│   └── map_config.json      # Map settings
├── data/
│   ├── maps/                # Offline map tiles
│   └── models/              # AI models
├── logs/                    # Application logs
├── requirements.txt         # Python dependencies
└── EDGE_DEVICE_SETUP.md     # This file
```

## 🔒 Security Considerations

- **Network Security**: Use VPN or secure channels for drone communication
- **Data Privacy**: Implement encryption for video streams and telemetry
- **Access Control**: Set up proper user permissions and authentication
- **Updates**: Regularly update dependencies and security patches

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review logs in `setup_edge_device.log` and `logs/foresight.log`
3. Test individual components using the component tests
4. Consult DJI documentation for drone-specific issues

## 🔄 Updates and Maintenance

```bash
# Update Python dependencies
pip install --upgrade -r requirements.txt

# Update AI models
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt', force_reload=True)"

# Update offline maps
python -m src.backend.offline_maps --update

# System updates (Linux)
sudo apt update && sudo apt upgrade
```

---

**Note**: This setup guide assumes basic familiarity with Linux command line, Python development, and drone operations. For production deployments, additional security hardening and monitoring should be implemented.