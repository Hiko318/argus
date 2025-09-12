# Release Notes Template

## Foresight SAR v[VERSION] - [RELEASE_DATE]

### 🎯 Release Highlights
<!-- Brief summary of the most important changes in this release -->

### ✨ New Features
<!-- List new features and capabilities -->
- 

### 🔧 Improvements
<!-- List enhancements to existing features -->
- 

### 🐛 Bug Fixes
<!-- List resolved issues and bugs -->
- 

### 🔒 Security Updates
<!-- List security-related changes -->
- 

### 📦 Dependencies
<!-- List dependency updates -->
- 

### 🚨 Breaking Changes
<!-- List any breaking changes that require user action -->
- 

### 📋 System Requirements
<!-- Update if requirements have changed -->
- Python 3.8+
- Node.js 16+
- CUDA 11.0+ (for GPU acceleration)
- 8GB+ RAM
- 50GB+ storage

### 🤖 AI Models
<!-- List included or updated AI models -->
- **YOLOv8 Nano** (`yolov8n.pt`) - [SIZE]MB - Fast inference, basic accuracy
- **YOLOv8 Small** (`yolov8s.pt`) - [SIZE]MB - Balanced speed/accuracy
- **YOLOv8 Medium** (`yolov8m.pt`) - [SIZE]MB - Higher accuracy, slower inference

### 📥 Installation

#### Quick Install
```bash
# Download and extract release
wget https://github.com/Hiko318/foresight/releases/download/v[VERSION]/foresight-v[VERSION].zip
unzip foresight-v[VERSION].zip
cd foresight-v[VERSION]

# Install dependencies
pip install -r requirements.txt
cd foresight-electron && npm install && cd ..

# Download AI models
python models/download_models.py

# Run system
python main.py --simulate
```

#### Platform-Specific Installers
- **Windows**: `foresight-v[VERSION]-windows-installer.exe`
- **Linux**: `foresight-v[VERSION]-linux-installer.run`
- **NVIDIA Jetson**: `foresight-v[VERSION]-jetson.tar.gz`

### 🔗 Download Links
- [Source Code (zip)](https://github.com/Hiko318/foresight/archive/v[VERSION].zip)
- [Source Code (tar.gz)](https://github.com/Hiko318/foresight/archive/v[VERSION].tar.gz)
- [Windows Installer](https://github.com/Hiko318/foresight/releases/download/v[VERSION]/foresight-v[VERSION]-windows-installer.exe)
- [Linux Installer](https://github.com/Hiko318/foresight/releases/download/v[VERSION]/foresight-v[VERSION]-linux-installer.run)
- [AI Models Package](https://github.com/Hiko318/foresight/releases/download/v[VERSION]/foresight-models-v[VERSION].zip)

### 📊 Performance Benchmarks
<!-- Include performance metrics if available -->
- **Detection Latency**: <[X]ms (RTX 3080)
- **Throughput**: [X]+ FPS (1080p)
- **Memory Usage**: <[X]GB RAM
- **Model Accuracy**: [X]% mAP@0.5

### 🧪 Testing
<!-- Testing coverage and validation -->
- ✅ Unit tests: [X]% coverage
- ✅ Integration tests: Passed
- ✅ Performance benchmarks: Validated
- ✅ Cross-platform compatibility: Windows/Linux/Jetson

### 📚 Documentation Updates
<!-- List documentation changes -->
- 

### 🤝 Contributors
<!-- Acknowledge contributors -->
Thanks to all contributors who made this release possible!

### 🆘 Support
- **Issues**: [GitHub Issues](https://github.com/Hiko318/foresight/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Hiko318/foresight/discussions)
- **Documentation**: [Wiki](https://github.com/Hiko318/foresight/wiki)

### 🔄 Migration Guide
<!-- If applicable, provide migration instructions -->

#### Upgrading from v[PREVIOUS_VERSION]
1. Backup your current configuration files
2. Download and install the new version
3. Update configuration files as needed
4. Run migration scripts if provided

---

**⚡ Ready to save lives with AI-powered search and rescue technology!**

**Full Changelog**: https://github.com/Hiko318/foresight/compare/v[PREVIOUS_VERSION]...v[VERSION]