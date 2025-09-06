# Foresight SAR System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-downloads)

**AI-Powered Search and Rescue System with Real-Time Human Detection**

Foresight is an advanced Search and Rescue (SAR) system that combines computer vision, geolocation services, and real-time processing to assist in locating missing persons during emergency operations. The system processes live video feeds from drones, cameras, or mobile devices to automatically detect humans and provide precise geolocation data to rescue teams.

## 🎯 Project Purpose

- **Real-Time Human Detection**: YOLO-based AI models optimized for aerial and ground-based search scenarios
- **Geolocation Integration**: Precise GPS coordinate mapping with terrain intersection analysis
- **Evidence Management**: Automated capture, annotation, and export of detection evidence
- **Multi-Platform Support**: Desktop application, web interface, and mobile compatibility
- **Field-Ready Deployment**: Designed for use in remote locations with offline capabilities

## 🛠️ Hardware Targets

### Primary Platforms
- **DJI O4 Air Unit**: Direct integration with DJI drone systems
- **NVIDIA Jetson Series**: Edge computing for real-time inference
  - Jetson Orin NX/AGX (recommended)
  - Jetson Xavier NX/AGX
  - Jetson Nano (basic functionality)

### Supported Hardware
- **GPUs**: CUDA-compatible cards (GTX 1060+, RTX series)
- **Cameras**: USB webcams, IP cameras, RTSP streams
- **Mobile Devices**: Android/iOS with camera access
- **Compute**: x86_64, ARM64 architectures

## 🚀 Quick Start

### Prerequisites
```bash
# System requirements
- Python 3.8+
- Node.js 16+
- CUDA 11.0+ (for GPU acceleration)
- 8GB+ RAM
- 50GB+ storage
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Hiko318/foresight.git
   cd foresight
   ```

2. **Set up Python environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Install Electron app dependencies**
   ```bash
   cd foresight-electron
   npm install
   cd ..
   ```

5. **Configure system settings**
   ```bash
   # Configuration files are located in configs/
   cp configs/dji_config.json.example configs/dji_config.json
   # Edit configuration files as needed
   ```

### Running the System

**Option 1: Full System (Recommended)**
```bash
# Windows
.\start_all.ps1

# Linux/macOS
./start_all.sh
```

**Option 2: Individual Components**
```bash
# Terminal 1: Backend service
python src/backend/main.py

# Terminal 2: Desktop application
cd foresight-electron && npm start
```

**Option 3: Development Mode**
```bash
# Run with simulation data
python main.py --simulate
```

## 📁 Repository Structure

```
foresight/
├── configs/           # Configuration files (moved from root)
│   ├── camera_calib.yaml
│   ├── dji_config.json
│   ├── privacy.yaml
│   ├── train_config.yaml
│   ├── pytest.ini
│   └── .pre-commit-config.yaml
├── demos/             # Demo and test files
│   ├── create_test_video.py
│   ├── demo_geolocation_pipeline.py
│   ├── phone_stream_demo.html
│   └── _page.html
├── foresight-electron/ # Desktop application
├── aerial_training/   # Training data and models
├── src/              # Source code
│   ├── backend/      # Backend services
│   └── ui/           # User interface components
├── training/         # Training scripts and utilities
├── ui/               # Web interface files
└── public/           # Public assets and backup files
```

### Access Points
- **Desktop App**: Launches automatically
- **Web Interface**: http://localhost:8004
- **API Documentation**: http://localhost:8004/docs

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input Layer   │    │  Processing Core │    │  Output Layer   │
├─────────────────┤    ├──────────────────┤    ├─────────────────┤
│ • DJI Drones    │───▶│ • YOLO Detection │───▶│ • Evidence DB   │
│ • IP Cameras    │    │ • Geolocation    │    │ • Real-time UI  │
│ • Mobile Feeds  │    │ • Tracking       │    │ • Export Tools  │
│ • File Upload   │    │ • Ray Casting    │    │ • Notifications │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                       ┌──────────────┐
                       │  Services    │
                       ├──────────────┤
                       │ • WebSocket  │
                       │ • Storage    │
                       │ • Telemetry  │
                       │ • Logging    │
                       └──────────────┘
```

### Core Components

- **Detection Pipeline** (`src/backend/detection_pipeline.py`): YOLO-based human detection
- **Geolocation Service** (`src/backend/geolocation_service.py`): GPS coordinate processing
- **SAR Interface** (`foresight-electron/`): Desktop application wrapper
- **Web Dashboard** (`src/frontend/`): Browser-based operator interface
- **Evidence Storage** (`src/backend/storage_service.py`): Automated evidence management

## 📱 Platform-Specific Setup

### NVIDIA Jetson
```bash
# Install Jetson-optimized dependencies
pip install -r requirements-jetson.txt

# Enable TensorRT optimization
export ENABLE_TENSORRT=true
python src/backend/edge_optimizer.py
```

### DJI Integration
```bash
# Configure DJI SDK
cp configs/dji_config.json.example configs/dji_config.json
# Edit with your DJI developer credentials
```

### Mobile Development
```bash
# Start mobile-friendly web interface
cd src/frontend
python -m http.server 3000
# Access via mobile browser at http://[device-ip]:3000
```

## 🔧 Configuration

### Environment Variables
See `.env.example` for complete configuration options:

```bash
# Core settings
SAR_SERVICE_PORT=8004
DETECTION_CONFIDENCE_THRESHOLD=0.5
GEOLOCATION_PRECISION=high

# Hardware optimization
ENABLE_GPU=true
ENABLE_TENSORRT=false
BATCH_SIZE=1

# Integration
DJI_API_KEY=your_dji_api_key
MAPBOX_ACCESS_TOKEN=your_mapbox_token
```

### Model Configuration
```bash
# Download pre-trained models
python models/download_models.py

# Use custom trained model
cp your_model.pt models/
# Update detection_pipeline.py model path
```

## 🧪 Testing & Development

### Running Tests
```bash
# Unit tests
python -m pytest tests/

# Integration tests with simulation
python main.py --simulate --test-mode

# Performance benchmarks
python scripts/benchmark.py
```

### Development Tools
```bash
# Code formatting
black src/
flake8 src/

# Frontend development
cd foresight-electron
npm run dev
```

## 📊 Performance & Optimization

### Benchmarks
- **Detection Latency**: <100ms (RTX 3080)
- **Throughput**: 30+ FPS (1080p)
- **Memory Usage**: <4GB RAM
- **Storage**: ~1GB/hour evidence

### Optimization Tips
- Use TensorRT for Jetson deployment
- Enable GPU acceleration for detection
- Configure batch processing for multiple streams
- Use local storage for evidence during operations

## 🔒 Security & Privacy

- **Data Encryption**: All evidence encrypted at rest
- **Access Control**: Role-based permissions
- **Privacy Compliance**: GDPR/CCPA ready
- **Audit Logging**: Complete operation trails

See `docs/DPIA_template.md` for privacy impact assessment.

## 📚 Documentation

- **[Operational Runbook](docs/operational-runbook.md)**: Field deployment procedures
- **[Contributing Guide](CONTRIBUTING.md)**: Development guidelines
- **[API Documentation](http://localhost:8004/docs)**: REST API reference
- **[Training Guide](HUMAN_DETECTION_PIPELINE.md)**: Model training procedures

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contribution Setup
```bash
# Fork the repository
git clone https://github.com/yourusername/foresight.git

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test
python -m pytest tests/

# Submit pull request
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Disclaimer**: This software is designed for search and rescue operations. Users are responsible for ensuring compliance with local privacy laws and obtaining proper authorization before deployment.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/Hiko318/foresight/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Hiko318/foresight/discussions)
- **Documentation**: [Wiki](https://github.com/Hiko318/foresight/wiki)

## 🏆 Acknowledgments

- YOLO team for object detection models
- DJI for drone integration APIs
- Open source SAR community
- Contributors and testers

---

**⚡ Ready to save lives with AI-powered search and rescue technology!**