# Changelog

All notable changes to the ARGUS SAR System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Enhanced SAR model training pipeline with 6-class detection (person, vehicle, structure, debris, dog, cat)
- Advanced data augmentation for covered/partially occluded humans
- Focal Loss implementation for improved detection accuracy
- SAR-specific training optimizations
- Comprehensive training guide and documentation
- Repository hygiene improvements and standardized structure

### Changed
- Upgraded from YOLOv8n to YOLOv8s for better accuracy
- Enhanced training configuration for >93% covered human detection accuracy
- Improved dataset configuration with pet detection capabilities

### Fixed
- Training pipeline path resolution issues
- Model download and validation processes

## [1.0.0] - 2024-01-15

### Added
- Initial release of ARGUS SAR System
- Real-time human detection using YOLO models
- Geolocation integration with GPS coordinate mapping
- Evidence management and automated capture
- Multi-platform support (Desktop, Web, Mobile)
- DJI O4 Air Unit integration
- NVIDIA Jetson optimization
- Web-based operator interface
- Desktop application wrapper
- Privacy and security compliance features

### Core Features
- YOLO-based AI models for aerial and ground-based search
- Precise GPS coordinate mapping with terrain intersection
- Automated evidence capture, annotation, and export
- Real-time processing capabilities
- Offline operation support for remote locations
- Cross-platform compatibility (x86_64, ARM64)
- CUDA acceleration support
- RESTful API with WebSocket real-time updates

### Hardware Support
- DJI drone systems integration
- NVIDIA Jetson series (Orin, Xavier, Nano)
- CUDA-compatible GPUs (GTX 1060+, RTX series)
- USB webcams, IP cameras, RTSP streams
- Android/iOS mobile device compatibility

### Documentation
- Comprehensive README with quickstart guide
- Operational runbook for field deployment
- API documentation and reference
- Privacy impact assessment template
- Contributing guidelines
- Training and model development guides

### Security
- Data encryption at rest
- Role-based access control
- Privacy compliance (GDPR/CCPA)
- Audit logging capabilities
- Secure configuration management

---

## Release Notes Format

### Added
- New features and capabilities

### Changed
- Changes in existing functionality

### Deprecated
- Soon-to-be removed features

### Removed
- Removed features

### Fixed
- Bug fixes

### Security
- Security-related changes and fixes