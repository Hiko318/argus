# Foresight SAR v0.9 - Field-Ready Prototype

## 🚁 What's New

- **Production-Ready SAR System**: Complete search and rescue drone vision system
- **Real-time Human Detection**: YOLOv8-based detection with <100ms latency
- **Evidence Packaging**: Tamper-evident evidence collection with cryptographic signatures
- **Cross-Platform Support**: Windows and Linux installers available
- **Field Kit Documentation**: Complete deployment guide for field operations

## 📦 Installation

### Windows
1. Download `Foresight-SAR-Setup-v0.9.exe`
2. Run installer as Administrator
3. Follow setup wizard

### Linux
1. Download `Foresight-SAR-v0.9.AppImage`
2. Make executable: `chmod +x Foresight-SAR-v0.9.AppImage`
3. Run: `./Foresight-SAR-v0.9.AppImage`

## 🤖 AI Models

- **YOLOV8N**: 6.25MB - Available via GitHub release


### Model Download

For automatic model download:
```bash
cd models
python download_models.py --model yolov8n --model yolov8s
```

## 🔧 System Requirements

### Minimum
- CPU: Intel i5-8400 / AMD Ryzen 5 3600
- RAM: 16GB DDR4
- GPU: NVIDIA GTX 1660 (CUDA support required)
- Storage: 500GB SSD

### Recommended
- CPU: Intel i7-10700K / AMD Ryzen 7 5800X
- RAM: 32GB DDR4
- GPU: NVIDIA RTX 3070 or better
- Storage: 1TB NVMe SSD

## 📋 Field Kit

See `packaging/field_kit.md` for complete hardware requirements and deployment procedures.

## 🔒 Security Features

- Evidence packaging with SHA256 manifests
- Cryptographic signatures for tamper detection
- Secure data transmission protocols
- Audit trail for all operations

## 🐛 Known Issues

- Electron-builder may fail on some Windows systems (use alternative build script)
- Large model files require external download due to GitHub size limits
- Some capture cards may require manual driver installation

## 📞 Support

- Documentation: See README.md and docs/ folder
- Issues: https://github.com/Hiko318/foresight/issues
- Field Kit Guide: packaging/field_kit.md

---

**Full Changelog**: https://github.com/Hiko318/foresight/compare/v0.8...v0.9
