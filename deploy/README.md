# Foresight SAR Deployment System

Comprehensive deployment and installation system for the Foresight Search and Rescue platform, supporting Windows and NVIDIA Jetson devices with interactive setup wizards and automated configuration.

## Overview

The deployment system provides multiple installation methods:

- **Interactive Setup Wizard**: Guided installation with platform detection
- **Platform-Specific Scripts**: Automated setup for Windows and Jetson
- **Docker Deployment**: Containerized deployment options
- **Offline Installation**: Air-gapped deployment support
- **Custom Configuration**: Flexible component selection

## Quick Start

### Interactive Setup Wizard

```bash
# Run the interactive setup wizard
python setup_wizard.py

# Non-interactive installation with defaults
python setup_wizard.py --non-interactive

# Use custom configuration file
python setup_wizard.py --config deployment_config.json

# Override platform detection
python setup_wizard.py --platform jetson
```

### Platform-Specific Installation

#### Windows
```powershell
# Run as Administrator
.\windows\setup_windows.ps1

# Development environment
.\windows\setup_windows.ps1 -Development

# Production deployment
.\windows\setup_windows.ps1 -Production -InstallPath "C:\SAR" -DataPath "D:\SAR-Data"

# Minimal installation
.\windows\setup_windows.ps1 -Minimal
```

#### NVIDIA Jetson
```bash
# Standard installation
sudo bash jetson/setup_jetson.sh

# Make executable first if needed
chmod +x jetson/setup_jetson.sh
sudo ./jetson/setup_jetson.sh
```

## Installation Modes

### Development Mode
- Full development environment
- Debugging tools and utilities
- IDE integrations
- Development dependencies
- Hot-reload capabilities

### Production Mode
- Optimized for operational deployment
- System service installation
- Security hardening
- Performance optimizations
- Monitoring and logging

### Minimal Mode
- Core components only
- Reduced resource usage
- Essential functionality
- Lightweight deployment

### Custom Mode
- Choose specific components
- Modular installation
- Tailored configurations
- Component-specific settings

## Components

### Setup Wizard (`setup_wizard.py`)

Interactive installation wizard with:

- **Platform Detection**: Automatic Windows/Jetson/Linux detection
- **System Requirements Check**: Memory, storage, GPU validation
- **Interactive Configuration**: Guided setup process
- **Component Selection**: Modular installation options
- **Post-Installation Testing**: Validation and verification
- **Comprehensive Logging**: Detailed installation logs

#### Features
- Colorized terminal output
- Progress tracking
- Error handling and recovery
- Configuration validation
- Offline asset management
- Service installation
- Shortcut creation

### Windows Setup (`windows/setup_windows.ps1`)

PowerShell-based Windows installation:

- **Chocolatey Integration**: Automated package management
- **System Dependencies**: Python, Git, Node.js, FFmpeg
- **GPU Support**: CUDA toolkit installation
- **Windows Services**: NSSM service wrapper
- **Firewall Configuration**: Port management
- **Registry Integration**: Programs and Features
- **Performance Optimization**: Power plans, exclusions

#### Windows-Specific Features
- Windows Defender exclusions
- High-performance power plan
- Desktop and Start Menu shortcuts
- Uninstaller creation
- Registry entries
- Service management scripts

### Jetson Setup (`jetson/setup_jetson.sh`)

Bash-based NVIDIA Jetson installation:

- **System Optimization**: Performance mode configuration
- **Docker Integration**: NVIDIA Container Runtime
- **GPU Acceleration**: CUDA and TensorRT support
- **System Services**: Systemd integration
- **Resource Management**: Memory and swap optimization
- **Monitoring Tools**: System status and health checks

#### Jetson-Specific Features
- Jetson performance modes
- GPU memory configuration
- Container runtime setup
- System limits optimization
- Network performance tuning
- Automated backup system

## Configuration

### Setup Wizard Configuration

```json
{
  "platform": "windows",
  "mode": "production",
  "install_path": "C:\\Program Files\\Foresight SAR",
  "data_path": "C:\\ProgramData\\Foresight SAR",
  "enable_gpu": true,
  "enable_docker": true,
  "enable_services": true,
  "offline_mode": false,
  "vault_integration": true,
  "custom_components": [
    "vision", "tracking", "geolocation", "reid", "packaging"
  ]
}
```

### Application Configuration

Generated during installation:

```json
{
  "app": {
    "name": "Foresight SAR System",
    "version": "1.0.0",
    "mode": "production",
    "debug": false
  },
  "server": {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4
  },
  "ui": {
    "host": "0.0.0.0",
    "port": 8080
  },
  "websocket": {
    "host": "0.0.0.0",
    "port": 5000
  },
  "database": {
    "type": "sqlite",
    "path": "/var/lib/foresight/data/foresight.db"
  },
  "logging": {
    "level": "INFO",
    "file": "/var/lib/foresight/logs/foresight.log"
  },
  "models": {
    "path": "/var/lib/foresight/models"
  },
  "evidence": {
    "path": "/var/lib/foresight/evidence"
  },
  "gpu": {
    "enabled": true
  },
  "vault": {
    "enabled": true,
    "url": "https://vault.example.com:8200",
    "key_name": "evidence-signing-key"
  }
}
```

## Directory Structure

### Installation Layout

```
Installation Directory/
├── main.py                    # Main application entry point
├── requirements.txt           # Python dependencies
├── configs/                   # Configuration templates
├── src/                       # Source code
├── ui/                        # User interface
├── vision/                    # Computer vision modules
├── tracking/                  # Object tracking
├── geolocation/              # Geolocation services
├── reid/                     # Re-identification
├── packaging/                # Evidence packaging
├── connection/               # Device connections
├── tools/                    # Utility tools
├── assets/                   # Static assets
├── Start.bat                 # Windows start script
├── Stop.bat                  # Windows stop script
├── Status.bat                # Windows status script
└── Uninstall.ps1            # Windows uninstaller
```

### Data Directory Layout

```
Data Directory/
├── config/
│   ├── settings.json         # Main configuration
│   ├── camera_calib.yaml     # Camera calibration
│   ├── privacy.yaml          # Privacy settings
│   └── encryption.key        # Encryption key
├── data/
│   ├── foresight.db          # Application database
│   └── cache/                # Cached data
├── models/
│   ├── yolov8n.pt           # Detection model
│   ├── reid_model.pth       # Re-ID model
│   └── optimized/           # Optimized models
├── evidence/
│   ├── packages/            # Evidence packages
│   └── temp/                # Temporary files
├── logs/
│   ├── foresight.log        # Application logs
│   ├── access.log           # Access logs
│   └── error.log            # Error logs
└── backups/
    ├── config/              # Configuration backups
    └── evidence/            # Evidence backups
```

## System Requirements

### Minimum Requirements
- **Memory**: 8GB RAM
- **Storage**: 50GB available space
- **Python**: 3.8 or higher
- **Operating System**: Windows 10+ or Ubuntu 18.04+

### Recommended Requirements
- **Memory**: 16GB RAM
- **Storage**: 100GB SSD
- **GPU**: NVIDIA GPU with 4GB+ VRAM
- **Network**: Gigabit Ethernet

### Jetson Requirements
- **Device**: Jetson Nano, TX2, Xavier, or Orin
- **JetPack**: 4.6+ or 5.0+
- **Storage**: 64GB+ microSD or eMMC
- **Power**: Adequate power supply for device

## Installation Process

### Pre-Installation
1. **System Check**: Verify requirements
2. **Platform Detection**: Identify target platform
3. **Permission Check**: Ensure administrative privileges
4. **Network Check**: Verify internet connectivity (if not offline)

### Installation Steps
1. **Directory Creation**: Create installation and data directories
2. **Dependency Installation**: Install system and Python dependencies
3. **Application Deployment**: Copy application files
4. **Configuration**: Generate configuration files
5. **Service Installation**: Install system services
6. **Shortcut Creation**: Create desktop and menu shortcuts
7. **Testing**: Run post-installation tests
8. **Validation**: Verify installation integrity

### Post-Installation
1. **Service Start**: Start application services
2. **Web Interface**: Access web interface
3. **Configuration**: Customize settings
4. **Testing**: Verify functionality

## Service Management

### Windows Services

```powershell
# Start service
net start ForesightSAR
# or
Start-Service ForesightSAR

# Stop service
net stop ForesightSAR
# or
Stop-Service ForesightSAR

# Check status
Get-Service ForesightSAR

# Service configuration
sc config ForesightSAR start= auto
```

### Linux/Jetson Services

```bash
# Start service
sudo systemctl start foresight-sar

# Stop service
sudo systemctl stop foresight-sar

# Enable auto-start
sudo systemctl enable foresight-sar

# Check status
sudo systemctl status foresight-sar

# View logs
journalctl -u foresight-sar -f
```

## Docker Deployment

### Jetson Docker Setup

```bash
# Build image
docker build -f jetson/Dockerfile -t foresight-sar:jetson .

# Run container
docker run -d \
  --name foresight-sar \
  --runtime nvidia \
  -p 8000:8000 \
  -p 8080:8080 \
  -p 5000:5000 \
  -v /opt/foresight/data:/app/data \
  foresight-sar:jetson

# Docker Compose
docker-compose -f jetson/docker-compose.yml up -d
```

### Docker Configuration

```yaml
# jetson/docker-compose.yml
version: '3.8'
services:
  foresight-sar:
    build:
      context: ..
      dockerfile: jetson/Dockerfile
    runtime: nvidia
    ports:
      - "8000:8000"
      - "8080:8080"
      - "5000:5000"
    volumes:
      - /opt/foresight/data:/app/data
      - /opt/foresight/models:/app/models
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all
    restart: unless-stopped
```

## Offline Installation

### Preparing Offline Assets

```bash
# Download offline assets
wget https://github.com/foresight-sar/assets/releases/latest/download/offline-assets.zip

# Extract assets
unzip offline-assets.zip -d deploy/offline_assets/

# Run offline installation
python setup_wizard.py --offline-mode
```

### Offline Asset Structure

```
offline_assets/
├── python_packages/          # Python wheels
├── system_packages/          # System packages
├── models/                   # Pre-trained models
├── docker_images/           # Docker images
└── documentation/           # Offline documentation
```

## Monitoring and Maintenance

### System Monitoring

```bash
# Jetson monitoring
foresight-monitor

# Manual monitoring
htop                         # System resources
nvtop                        # GPU usage
docker stats                 # Container stats
journalctl -u foresight-sar  # Service logs
```

### Backup and Recovery

```bash
# Create backup
foresight-backup

# Manual backup
tar -czf backup_$(date +%Y%m%d).tar.gz \
  /opt/foresight/config \
  /opt/foresight/data \
  /opt/foresight/evidence

# Restore backup
tar -xzf backup_20240115.tar.gz -C /
```

### Updates and Upgrades

```bash
# Update application
git pull origin main
pip install -r requirements.txt
sudo systemctl restart foresight-sar

# Update system packages
sudo apt update && sudo apt upgrade

# Update Docker images
docker-compose pull
docker-compose up -d
```

## Troubleshooting

### Common Issues

#### Installation Failures

1. **Permission Denied**
   ```bash
   # Ensure running as administrator/root
   sudo python setup_wizard.py
   ```

2. **Missing Dependencies**
   ```bash
   # Install missing system packages
   sudo apt install python3-dev build-essential
   ```

3. **GPU Not Detected**
   ```bash
   # Check NVIDIA drivers
   nvidia-smi
   
   # Install NVIDIA drivers
   sudo apt install nvidia-driver-470
   ```

#### Service Issues

1. **Service Won't Start**
   ```bash
   # Check service status
   systemctl status foresight-sar
   
   # Check logs
   journalctl -u foresight-sar --no-pager
   ```

2. **Port Conflicts**
   ```bash
   # Check port usage
   netstat -tulpn | grep :8000
   
   # Kill conflicting process
   sudo kill -9 <PID>
   ```

3. **Permission Issues**
   ```bash
   # Fix file permissions
   sudo chown -R foresight:foresight /opt/foresight
   sudo chmod -R 755 /opt/foresight
   ```

#### Performance Issues

1. **High Memory Usage**
   ```bash
   # Monitor memory
   free -h
   
   # Adjust worker count
   # Edit config: server.workers = 2
   ```

2. **Slow GPU Performance**
   ```bash
   # Check GPU utilization
   nvidia-smi
   
   # Enable maximum performance
   sudo jetson_clocks
   sudo nvpmodel -m 0
   ```

### Debug Mode

```bash
# Enable debug logging
export FORESIGHT_DEBUG=1
python main.py

# Verbose installation
python setup_wizard.py --verbose

# Test installation
python setup_wizard.py --test-only
```

### Log Analysis

```bash
# Application logs
tail -f /var/lib/foresight/logs/foresight.log

# System logs
journalctl -u foresight-sar -f

# Docker logs
docker logs -f foresight-sar

# Error analysis
grep ERROR /var/lib/foresight/logs/foresight.log
```

## Security Considerations

### Network Security
- Configure firewall rules
- Use HTTPS for web interface
- Implement VPN for remote access
- Regular security updates

### Data Protection
- Enable encryption at rest
- Secure evidence storage
- Access control and authentication
- Audit logging

### System Hardening
- Disable unnecessary services
- Configure user permissions
- Regular security patches
- Monitoring and alerting

## Performance Optimization

### System Optimization
- High-performance power plans
- GPU memory optimization
- Network tuning
- Storage optimization

### Application Optimization
- Model optimization (TensorRT)
- Batch processing
- Caching strategies
- Resource pooling

## Integration

### Enterprise Integration
- Active Directory authentication
- LDAP integration
- SIEM integration
- Backup system integration

### Cloud Integration
- Cloud storage backends
- Remote monitoring
- Centralized logging
- Distributed deployment

## Contributing

When contributing to the deployment system:

1. Test on target platforms
2. Update documentation
3. Follow security best practices
4. Maintain backward compatibility
5. Add comprehensive error handling

## License

This deployment system is part of the Foresight SAR System and follows the same licensing terms.

## Support

For deployment issues:

1. Check the troubleshooting section
2. Review system requirements
3. Test with minimal configuration
4. Check platform-specific documentation
5. Contact system administrators

## Changelog

### Version 1.0.0
- Initial release
- Interactive setup wizard
- Windows and Jetson support
- Docker deployment
- Offline installation
- Service management
- Comprehensive documentation