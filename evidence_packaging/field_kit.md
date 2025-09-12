# Foresight SAR Field Kit Documentation

## Overview

The Foresight SAR (Search and Rescue) Field Kit is a complete hardware and software solution for aerial search and rescue operations. This document outlines the hardware requirements, installation procedures, and acceptance testing protocols for field deployment.

## Hardware Requirements

### Minimum System Requirements

#### Primary Workstation
- **CPU**: Intel i5-8400 / AMD Ryzen 5 3600 or better
- **RAM**: 16GB DDR4 (32GB recommended for large operations)
- **GPU**: NVIDIA GTX 1660 / RTX 3060 or better (CUDA support required)
- **Storage**: 500GB NVMe SSD (1TB recommended)
- **OS**: Windows 10/11 64-bit or Ubuntu 20.04+ LTS

#### Recommended Field Workstation
- **CPU**: Intel i7-10700K / AMD Ryzen 7 5800X
- **RAM**: 32GB DDR4
- **GPU**: NVIDIA RTX 3070 / RTX 4060 or better
- **Storage**: 1TB NVMe SSD + 2TB HDD for data storage
- **Display**: Dual 24" monitors (1920x1080 minimum)

### Capture Hardware

#### Video Capture Card
- **Primary**: Elgato Cam Link 4K or equivalent
- **Alternative**: Blackmagic DeckLink Mini Recorder
- **Requirements**: HDMI input, USB 3.0+ output, 1080p60 minimum

#### Drone Integration
- **Supported**: DJI Mavic series, DJI Air series, DJI Mini series
- **Connection**: HDMI output from drone controller
- **Range**: 2-8km depending on drone model and environment

### Network Equipment

#### Field Communications
- **Primary**: 4G/5G mobile hotspot with unlimited data
- **Backup**: Starlink or similar satellite internet
- **Local**: Gigabit Ethernet switch for multi-station setups

#### Power Management
- **UPS**: 1500VA minimum for 30-minute runtime
- **Portable**: Goal Zero Yeti 1500X or equivalent for remote operations
- **Solar**: 200W solar panel array for extended deployments

## Software Installation

### Pre-Installation Checklist

1. **System Updates**
   ```bash
   # Windows
   Windows Update → Check for updates
   
   # Linux
   sudo apt update && sudo apt upgrade -y
   ```

2. **Driver Installation**
   - NVIDIA GPU drivers (latest stable)
   - Capture card drivers
   - USB 3.0 drivers

3. **Dependencies**
   ```bash
   # Install Python 3.9+
   python --version
   
   # Install Node.js 18+
   node --version
   npm --version
   ```

### Foresight SAR Installation

#### Method 1: Pre-built Installer (Recommended)

1. **Download Release**
   - Visit: https://github.com/Hiko318/foresight/releases/latest
   - Download: `Foresight-SAR-Setup-v0.9.exe` (Windows) or `Foresight-SAR-v0.9.AppImage` (Linux)

2. **Windows Installation**
   ```cmd
   # Run as Administrator
   Foresight-SAR-Setup-v0.9.exe
   
   # Follow installation wizard
   # Default install path: C:\Program Files\Foresight SAR
   ```

3. **Linux Installation**
   ```bash
   # Make executable
   chmod +x Foresight-SAR-v0.9.AppImage
   
   # Run application
   ./Foresight-SAR-v0.9.AppImage
   
   # Optional: Install to system
   sudo cp Foresight-SAR-v0.9.AppImage /usr/local/bin/foresight-sar
   ```

#### Method 2: Source Installation

1. **Clone Repository**
   ```bash
   git clone https://github.com/Hiko318/foresight.git
   cd foresight
   git checkout v0.9
   ```

2. **Install Dependencies**
   ```bash
   # Python dependencies
   pip install -r requirements.txt
   
   # Node.js dependencies (for UI)
   cd foresight-electron
   npm install
   ```

3. **Download Models**
   ```bash
   cd models
   python download_models.py --model yolov8n --model yolov8s
   ```

### Configuration

1. **Environment Setup**
   ```bash
   # Copy example configuration
   cp .env.example .env
   
   # Edit configuration
   nano .env  # Linux
   notepad .env  # Windows
   ```

2. **Key Configuration Parameters**
   ```env
   # Video Input
   VIDEO_SOURCE=0  # Capture card device ID
   VIDEO_RESOLUTION=1920x1080
   VIDEO_FPS=30
   
   # AI Models
   MODEL_PATH=models/yolov8n.pt
   CONFIDENCE_THRESHOLD=0.5
   
   # Network
   API_PORT=8004
   WS_PORT=8005
   
   # Storage
   EVIDENCE_PATH=./evidence
   LOG_LEVEL=INFO
   ```

## Acceptance Testing

### Pre-Deployment Tests

#### 1. System Health Check

```bash
# Run system diagnostics
python scripts/system_check.py

# Expected output:
# ✅ GPU: NVIDIA RTX 3070 (Driver: 531.29)
# ✅ Memory: 32GB available
# ✅ Storage: 850GB free
# ✅ Network: Connected (45ms latency)
# ✅ Capture: Elgato Cam Link detected
```

#### 2. Model Validation

```bash
# Test AI models
python src/backend/detector.py --test

# Expected output:
# ✅ YOLOv8n loaded (6.2MB)
# ✅ Inference test: 15ms average
# ✅ Detection accuracy: 94.2%
```

#### 3. Video Pipeline Test

```bash
# Test video capture and processing
python main.py --test-video

# Expected output:
# ✅ Video source: 1920x1080@30fps
# ✅ Processing latency: <100ms
# ✅ Detection overlay: Active
```

### Field Deployment Tests

#### 1. End-to-End Mission Simulation

1. **Setup**
   - Connect drone controller via HDMI
   - Launch Foresight SAR application
   - Verify video feed from drone

2. **Detection Test**
   - Fly drone over test area with known targets
   - Verify human detection and tracking
   - Check GPS coordinate accuracy

3. **Evidence Packaging**
   ```bash
   # Test evidence packager
   python src/backend/packager.py
   
   # Verify outputs:
   # - metadata.json
   # - manifest.sha256
   # - evidence package with timestamps
   ```

#### 2. Communication Test

1. **Network Connectivity**
   - Test 4G/5G hotspot connection
   - Verify satellite backup (if available)
   - Check local network performance

2. **Data Transmission**
   - Upload test evidence package
   - Verify secure transmission
   - Test real-time streaming (if enabled)

#### 3. Power Management Test

1. **UPS Failover**
   - Disconnect main power
   - Verify UPS takeover
   - Test 30-minute runtime

2. **Portable Power**
   - Test Goal Zero or equivalent
   - Verify 4-hour operation minimum
   - Check solar charging (if applicable)

### Performance Benchmarks

#### Minimum Acceptable Performance

| Metric | Minimum | Target | Excellent |
|--------|---------|--------|-----------|
| Detection Latency | <200ms | <100ms | <50ms |
| Video Frame Rate | 25fps | 30fps | 60fps |
| Detection Accuracy | 85% | 92% | 95% |
| System Uptime | 4 hours | 8 hours | 12+ hours |
| GPS Accuracy | 5m | 3m | 1m |

#### Stress Testing

```bash
# Run 4-hour stress test
python tests/stress_test.py --duration 4h

# Monitor:
# - CPU/GPU temperature
# - Memory usage
# - Detection performance
# - System stability
```

## Troubleshooting

### Common Issues

#### Video Feed Problems

1. **No Video Signal**
   ```bash
   # Check capture device
   python -c "import cv2; print(cv2.VideoCapture(0).read())"
   
   # Verify HDMI connection
   # Check capture card drivers
   ```

2. **Poor Video Quality**
   - Check HDMI cable quality
   - Verify capture card settings
   - Adjust drone video output settings

#### Detection Issues

1. **Low Detection Accuracy**
   - Check lighting conditions
   - Adjust confidence threshold
   - Verify model is appropriate for conditions

2. **High False Positives**
   - Increase confidence threshold
   - Check for environmental interference
   - Consider model retraining

#### Performance Problems

1. **High Latency**
   ```bash
   # Check system resources
   nvidia-smi  # GPU usage
   htop        # CPU/Memory usage
   
   # Optimize settings
   # Reduce video resolution if needed
   # Close unnecessary applications
   ```

### Emergency Procedures

#### System Failure

1. **Primary System Down**
   - Switch to backup laptop
   - Use mobile hotspot for connectivity
   - Continue with reduced capability

2. **Network Failure**
   - Switch to satellite backup
   - Use local storage mode
   - Sync data when connectivity restored

3. **Power Failure**
   - UPS provides 30-minute runtime
   - Switch to portable power
   - Implement power-saving mode

## Maintenance Schedule

### Daily (During Deployment)
- Check system temperatures
- Verify storage space
- Test communication links
- Backup critical data

### Weekly
- Update system logs
- Check for software updates
- Clean equipment
- Test backup systems

### Monthly
- Full system backup
- Performance benchmarking
- Hardware inspection
- Update documentation

## Support Contacts

### Technical Support
- **Primary**: support@foresight-sar.com
- **Emergency**: +1-555-SAR-HELP
- **Documentation**: https://docs.foresight-sar.com

### Hardware Vendors
- **Elgato**: support.elgato.com
- **NVIDIA**: developer.nvidia.com/support
- **DJI**: support.dji.com

---

**Document Version**: 1.0  
**Last Updated**: January 9, 2024  
**Next Review**: February 9, 2024