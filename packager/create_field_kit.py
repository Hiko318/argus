#!/usr/bin/env python3
"""
Field Kit Creator for Foresight SAR
Creates complete field deployment packages with all necessary components
"""

import os
import sys
import shutil
import subprocess
import tempfile
import tarfile
import zipfile
from pathlib import Path
import argparse
import json
import logging
import yaml
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FieldKitCreator:
    def __init__(self, source_dir, output_dir, version="1.0.0", kit_type="standard"):
        self.source_dir = Path(source_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.version = version
        self.kit_type = kit_type
        self.build_date = datetime.utcnow().isoformat()
        
        # Kit configurations
        self.kit_configs = {
            'minimal': {
                'description': 'Minimal deployment for basic SAR operations',
                'components': ['app', 'config', 'docs'],
                'platforms': ['jetson_nano'],
                'size_limit_mb': 500
            },
            'standard': {
                'description': 'Standard deployment with full features',
                'components': ['app', 'config', 'docs', 'models', 'scripts'],
                'platforms': ['jetson_orin', 'jetson_xavier', 'windows'],
                'size_limit_mb': 2000
            },
            'complete': {
                'description': 'Complete deployment with all components and tools',
                'components': ['app', 'config', 'docs', 'models', 'scripts', 'tools', 'training'],
                'platforms': ['jetson_orin', 'jetson_xavier', 'windows', 'linux'],
                'size_limit_mb': 5000
            }
        }
        
        # Validate inputs
        if not self.source_dir.exists():
            raise ValueError(f"Source directory does not exist: {self.source_dir}")
        
        if kit_type not in self.kit_configs:
            raise ValueError(f"Invalid kit type: {kit_type}. Available: {list(self.kit_configs.keys())}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Source: {self.source_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Version: {self.version}")
        logger.info(f"Kit Type: {self.kit_type}")
    
    def check_dependencies(self):
        """Check if required tools are available"""
        logger.info("Checking dependencies...")
        
        required_tools = ['tar', 'zip']
        optional_tools = ['docker', 'git']
        
        for tool in required_tools:
            try:
                subprocess.run([tool, '--version'], 
                              capture_output=True, text=True, check=True)
                logger.info(f"{tool} ✓")
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.error(f"{tool} not found (required)")
                return False
        
        for tool in optional_tools:
            try:
                subprocess.run([tool, '--version'], 
                              capture_output=True, text=True, check=True)
                logger.info(f"{tool} ✓")
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.warning(f"{tool} not found (optional)")
        
        return True
    
    def create_kit_structure(self):
        """Create field kit directory structure"""
        logger.info("Creating field kit structure...")
        
        kit_dir = self.output_dir / f'foresight-sar-fieldkit-{self.version}'
        kit_dir.mkdir(exist_ok=True)
        
        # Create directory structure based on kit type
        config = self.kit_configs[self.kit_type]
        
        directories = {
            'app': 'Application files and source code',
            'config': 'Configuration files and templates',
            'docs': 'Documentation and user guides',
            'models': 'AI models and weights',
            'scripts': 'Installation and management scripts',
            'tools': 'Additional tools and utilities',
            'training': 'Training data and scripts',
            'platforms': 'Platform-specific packages',
            'licenses': 'License files and legal documents'
        }
        
        for component in config['components']:
            if component in directories:
                comp_dir = kit_dir / component
                comp_dir.mkdir(exist_ok=True)
                
                # Create README for each component
                readme_content = f"# {component.title()}\n\n{directories[component]}\n"
                with open(comp_dir / 'README.md', 'w') as f:
                    f.write(readme_content)
        
        # Always create platforms and licenses directories
        (kit_dir / 'platforms').mkdir(exist_ok=True)
        (kit_dir / 'licenses').mkdir(exist_ok=True)
        
        logger.info(f"Field kit structure created: {kit_dir}")
        return kit_dir
    
    def copy_application_files(self, kit_dir):
        """Copy application files to field kit"""
        logger.info("Copying application files...")
        
        config = self.kit_configs[self.kit_type]
        
        # Copy source code if app component is included
        if 'app' in config['components']:
            app_dir = kit_dir / 'app'
            
            # Copy main source code
            if (self.source_dir / 'src').exists():
                shutil.copytree(self.source_dir / 'src', app_dir / 'src', dirs_exist_ok=True)
                logger.info("Source code copied")
            
            # Copy requirements and setup files
            for file_name in ['requirements.txt', 'setup.py', 'pyproject.toml', 'main.py']:
                src_file = self.source_dir / file_name
                if src_file.exists():
                    shutil.copy2(src_file, app_dir)
                    logger.info(f"{file_name} copied")
        
        # Copy configuration files
        if 'config' in config['components']:
            config_dir = kit_dir / 'config'
            
            if (self.source_dir / 'config').exists():
                for config_file in (self.source_dir / 'config').rglob('*'):
                    if config_file.is_file():
                        rel_path = config_file.relative_to(self.source_dir / 'config')
                        dest_file = config_dir / rel_path
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(config_file, dest_file)
                logger.info("Configuration files copied")
        
        # Copy documentation
        if 'docs' in config['components']:
            docs_dir = kit_dir / 'docs'
            
            # Copy main documentation files
            doc_files = ['README.md', 'LICENSE', 'CHANGELOG.md', 'INSTALL.md']
            for doc_file in doc_files:
                src_file = self.source_dir / doc_file
                if src_file.exists():
                    shutil.copy2(src_file, docs_dir)
            
            # Copy docs directory if exists
            if (self.source_dir / 'docs').exists():
                for doc_file in (self.source_dir / 'docs').rglob('*'):
                    if doc_file.is_file():
                        rel_path = doc_file.relative_to(self.source_dir / 'docs')
                        dest_file = docs_dir / rel_path
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(doc_file, dest_file)
            
            logger.info("Documentation copied")
        
        # Copy models (if included and exists)
        if 'models' in config['components']:
            models_dir = kit_dir / 'models'
            
            if (self.source_dir / 'models').exists():
                # Only copy model configuration and small files
                # Large model files should be downloaded separately
                for model_file in (self.source_dir / 'models').rglob('*'):
                    if model_file.is_file() and model_file.stat().st_size < 100 * 1024 * 1024:  # < 100MB
                        rel_path = model_file.relative_to(self.source_dir / 'models')
                        dest_file = models_dir / rel_path
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(model_file, dest_file)
                
                # Create model download script
                download_script = f'''
#!/bin/bash
# Model Download Script for Foresight SAR

echo "Downloading AI models..."

# Create models directory
mkdir -p models/weights

# Download YOLOv8 models
echo "Downloading YOLOv8 models..."
wget -O models/weights/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
wget -O models/weights/yolov8s.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
wget -O models/weights/yolov8m.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt

# Download face recognition models
echo "Downloading face recognition models..."
mkdir -p models/face
wget -O models/face/face_recognition_model.pkl https://github.com/ageitgey/face_recognition_models/raw/master/face_recognition_models/models/dlib_face_recognition_resnet_model_v1.dat

# Download additional models as needed
echo "Model download completed!"
echo "Total models downloaded: $(find models -name '*.pt' -o -name '*.pkl' -o -name '*.dat' | wc -l)"
'''
                
                with open(models_dir / 'download_models.sh', 'w') as f:
                    f.write(download_script)
                (models_dir / 'download_models.sh').chmod(0o755)
                
                logger.info("Models configuration copied")
        
        # Copy scripts
        if 'scripts' in config['components']:
            scripts_dir = kit_dir / 'scripts'
            
            if (self.source_dir / 'scripts').exists():
                for script_file in (self.source_dir / 'scripts').rglob('*'):
                    if script_file.is_file():
                        rel_path = script_file.relative_to(self.source_dir / 'scripts')
                        dest_file = scripts_dir / rel_path
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(script_file, dest_file)
                        # Make scripts executable
                        if script_file.suffix in ['.sh', '.py']:
                            dest_file.chmod(0o755)
                logger.info("Scripts copied")
        
        # Copy training materials
        if 'training' in config['components']:
            training_dir = kit_dir / 'training'
            
            if (self.source_dir / 'training').exists():
                for training_file in (self.source_dir / 'training').rglob('*'):
                    if training_file.is_file():
                        rel_path = training_file.relative_to(self.source_dir / 'training')
                        dest_file = training_dir / rel_path
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(training_file, dest_file)
                logger.info("Training materials copied")
    
    def create_platform_packages(self, kit_dir):
        """Create platform-specific packages"""
        logger.info("Creating platform-specific packages...")
        
        platforms_dir = kit_dir / 'platforms'
        config = self.kit_configs[self.kit_type]
        
        for platform in config['platforms']:
            platform_dir = platforms_dir / platform
            platform_dir.mkdir(exist_ok=True)
            
            if platform.startswith('jetson'):
                # Create Jetson package
                self.create_jetson_package(platform_dir, platform)
            elif platform == 'windows':
                # Create Windows package
                self.create_windows_package(platform_dir)
            elif platform == 'linux':
                # Create Linux package
                self.create_linux_package(platform_dir)
        
        logger.info("Platform packages created")
    
    def create_jetson_package(self, platform_dir, jetson_model):
        """Create Jetson-specific package"""
        logger.info(f"Creating {jetson_model} package...")
        
        # Copy Jetson Dockerfile
        if (self.source_dir / 'docker' / 'jetson.Dockerfile').exists():
            shutil.copy2(self.source_dir / 'docker' / 'jetson.Dockerfile', platform_dir)
        
        # Copy Jetson build script
        if (self.source_dir / 'packager' / 'build_jetson.py').exists():
            shutil.copy2(self.source_dir / 'packager' / 'build_jetson.py', platform_dir)
        
        # Create Jetson-specific installation script
        install_script = f'''
#!/bin/bash
# Jetson Installation Script for {jetson_model}

set -e

echo "Installing Foresight SAR on {jetson_model}..."

# Check Jetson model
JETSON_MODEL=$(cat /proc/device-tree/model 2>/dev/null | tr -d '\0' || echo "unknown")
echo "Detected: $JETSON_MODEL"

# Run the main Jetson packager
python3 build_jetson.py --source ../.. --output ./dist --jetson-model {jetson_model}

# Extract and install the package
cd dist
tar -xzf foresight-sar-*-jetson.tar.gz
cd foresight-sar-*/
sudo ./scripts/install.sh

echo "Installation completed!"
echo "Start services with: sudo systemctl start foresight-sar"
'''
        
        with open(platform_dir / 'install.sh', 'w') as f:
            f.write(install_script)
        (platform_dir / 'install.sh').chmod(0o755)
        
        # Create platform-specific README
        readme_content = f'''
# {jetson_model.replace('_', ' ').title()} Deployment

This directory contains deployment files for {jetson_model}.

## Quick Installation

1. Run the installation script:
   ```bash
   ./install.sh
   ```

2. Or manually build and install:
   ```bash
   python3 build_jetson.py --source ../.. --output ./dist
   cd dist
   tar -xzf foresight-sar-*-jetson.tar.gz
   cd foresight-sar-*/
   sudo ./scripts/install.sh
   ```

## Requirements

- {jetson_model.replace('_', ' ').title()}
- JetPack 5.0+
- Python 3.8+
- 4GB+ RAM
- 10GB+ storage

## Support

See the main documentation for troubleshooting and support.
'''
        
        with open(platform_dir / 'README.md', 'w') as f:
            f.write(readme_content)
    
    def create_windows_package(self, platform_dir):
        """Create Windows-specific package"""
        logger.info("Creating Windows package...")
        
        # Copy Windows build script
        if (self.source_dir / 'packager' / 'build_windows.py').exists():
            shutil.copy2(self.source_dir / 'packager' / 'build_windows.py', platform_dir)
        
        # Create Windows installation script
        install_script = f'''
@echo off
REM Windows Installation Script for Foresight SAR

echo Installing Foresight SAR on Windows...

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM Run the Windows packager
python build_windows.py --source ..\..\ --output .\dist

REM Run the installer
cd dist
for %%f in (foresight-sar-*-setup.exe) do (
    echo Running installer: %%f
    start /wait %%f
)

echo Installation completed!
echo Start the application from the Start Menu or Desktop shortcut.
pause
'''
        
        with open(platform_dir / 'install.bat', 'w') as f:
            f.write(install_script)
        
        # Create platform-specific README
        readme_content = f'''
# Windows Deployment

This directory contains deployment files for Windows.

## Quick Installation

1. Run the installation script:
   ```cmd
   install.bat
   ```

2. Or manually build and install:
   ```cmd
   python build_windows.py --source ..\..\ --output .\dist
   cd dist
   foresight-sar-{self.version}-setup.exe
   ```

## Requirements

- Windows 10/11 (64-bit)
- Python 3.8+
- 8GB+ RAM
- 10GB+ storage
- Optional: NVIDIA GPU with CUDA support

## Support

See the main documentation for troubleshooting and support.
'''
        
        with open(platform_dir / 'README.md', 'w') as f:
            f.write(readme_content)
    
    def create_linux_package(self, platform_dir):
        """Create Linux-specific package"""
        logger.info("Creating Linux package...")
        
        # Copy deployment Dockerfile
        if (self.source_dir / 'docker' / 'deploy.Dockerfile').exists():
            shutil.copy2(self.source_dir / 'docker' / 'deploy.Dockerfile', platform_dir)
        
        # Create Linux installation script
        install_script = f'''
#!/bin/bash
# Linux Installation Script for Foresight SAR

set -e

echo "Installing Foresight SAR on Linux..."

# Check system
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "Detected OS: $NAME $VERSION"
else
    echo "Warning: Could not detect OS version"
fi

# Check Docker
if command -v docker &> /dev/null; then
    echo "Docker found, using containerized deployment"
    
    # Build and run with Docker
    docker build -f deploy.Dockerfile -t foresight-sar:latest ../..
    
    # Create docker-compose.yml
    cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  foresight-sar:
    image: foresight-sar:latest
    ports:
      - "8080:8080"
      - "9090:9090"
    volumes:
      - foresight-data:/var/lib/foresight
      - foresight-logs:/var/log/foresight
    environment:
      - FORESIGHT_ENV=production
    restart: unless-stopped

volumes:
  foresight-data:
  foresight-logs:
EOF
    
    # Start services
    docker-compose up -d
    
    echo "Services started with Docker Compose"
    echo "Access web interface: http://localhost:8080"
    
else
    echo "Docker not found, using native installation"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo "Error: Python 3 not found"
        exit 1
    fi
    
    # Install dependencies
    pip3 install -r ../../requirements.txt
    
    # Create systemd service
    sudo tee /etc/systemd/system/foresight-sar.service > /dev/null << 'EOF'
[Unit]
Description=Foresight SAR System
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PWD/../..
Environment=PYTHONPATH=$PWD/../../src
ExecStart=/usr/bin/python3 -m src.backend.app
Restart=always

[Install]
WantedBy=multi-user.target
EOF
    
    # Enable and start service
    sudo systemctl daemon-reload
    sudo systemctl enable foresight-sar
    sudo systemctl start foresight-sar
    
    echo "Service installed and started"
    echo "Check status: sudo systemctl status foresight-sar"
fi

echo "Installation completed!"
'''
        
        with open(platform_dir / 'install.sh', 'w') as f:
            f.write(install_script)
        (platform_dir / 'install.sh').chmod(0o755)
        
        # Create platform-specific README
        readme_content = f'''
# Linux Deployment

This directory contains deployment files for Linux.

## Quick Installation

1. Run the installation script:
   ```bash
   ./install.sh
   ```

## Docker Deployment (Recommended)

If Docker is available:
```bash
docker build -f deploy.Dockerfile -t foresight-sar:latest ../..
docker-compose up -d
```

## Native Installation

If Docker is not available:
```bash
pip3 install -r ../../requirements.txt
sudo ./install.sh
```

## Requirements

- Ubuntu 20.04+ / CentOS 8+ / Debian 11+
- Python 3.8+
- 4GB+ RAM
- 10GB+ storage
- Optional: Docker for containerized deployment

## Support

See the main documentation for troubleshooting and support.
'''
        
        with open(platform_dir / 'README.md', 'w') as f:
            f.write(readme_content)
    
    def create_field_kit_manifest(self, kit_dir):
        """Create field kit manifest with metadata"""
        logger.info("Creating field kit manifest...")
        
        config = self.kit_configs[self.kit_type]
        
        manifest = {
            'name': 'foresight-sar-fieldkit',
            'version': self.version,
            'kit_type': self.kit_type,
            'build_date': self.build_date,
            'description': config['description'],
            'components': config['components'],
            'platforms': config['platforms'],
            'size_limit_mb': config['size_limit_mb'],
            'requirements': {
                'minimum': {
                    'ram_gb': 4,
                    'storage_gb': 10,
                    'python': '3.8+'
                },
                'recommended': {
                    'ram_gb': 8,
                    'storage_gb': 20,
                    'python': '3.9+',
                    'gpu': 'NVIDIA with CUDA support'
                }
            },
            'installation': {
                'quick_start': 'See README.md for platform-specific installation instructions',
                'platforms': {
                    platform: f'platforms/{platform}/install.*'
                    for platform in config['platforms']
                }
            },
            'support': {
                'documentation': 'docs/',
                'github': 'https://github.com/foresight-sar/foresight',
                'issues': 'https://github.com/foresight-sar/foresight/issues'
            },
            'checksums': {}
        }
        
        # Calculate checksums for important files
        important_files = [
            'README.md',
            'app/requirements.txt',
            'config/privacy_defaults.json'
        ]
        
        for file_path in important_files:
            full_path = kit_dir / file_path
            if full_path.exists():
                with open(full_path, 'rb') as f:
                    content = f.read()
                    sha256_hash = hashlib.sha256(content).hexdigest()
                    manifest['checksums'][file_path] = sha256_hash
        
        with open(kit_dir / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info("Field kit manifest created")
    
    def create_field_kit_readme(self, kit_dir):
        """Create comprehensive field kit README"""
        logger.info("Creating field kit README...")
        
        config = self.kit_configs[self.kit_type]
        
        readme_content = f'''
# Foresight SAR - Field Kit {self.version}

**Kit Type:** {self.kit_type.title()}  
**Build Date:** {self.build_date}  
**Description:** {config['description']}  

## Overview

This field kit contains everything needed to deploy Foresight SAR in the field for search and rescue operations. The kit is designed for rapid deployment with minimal technical expertise required.

## What's Included

### Components
{chr(10).join(f'- **{comp.title()}**: {comp} files and configurations' for comp in config['components'])}

### Supported Platforms
{chr(10).join(f'- {platform.replace("_", " ").title()}' for platform in config['platforms'])}

## Quick Start

### 1. Choose Your Platform

Navigate to the appropriate platform directory:

```bash
cd platforms/[your-platform]
```

Available platforms:
{chr(10).join(f'- `{platform}/` - {platform.replace("_", " ").title()} deployment' for platform in config['platforms'])}

### 2. Run Installation

Each platform directory contains installation scripts:

**Linux/Jetson:**
```bash
./install.sh
```

**Windows:**
```cmd
install.bat
```

### 3. Verify Installation

After installation, verify the system is running:

```bash
# Check service status (Linux/Jetson)
sudo systemctl status foresight-sar

# Access web interface
http://localhost:8080
```

## System Requirements

### Minimum Requirements
- **RAM:** 4GB
- **Storage:** 10GB free space
- **Python:** 3.8+
- **Network:** Internet connection for initial setup

### Recommended Requirements
- **RAM:** 8GB+
- **Storage:** 20GB+ free space
- **Python:** 3.9+
- **GPU:** NVIDIA with CUDA support
- **Network:** Stable internet connection

### Platform-Specific Requirements

**NVIDIA Jetson:**
- JetPack 5.0+
- CUDA support enabled
- Proper cooling solution

**Windows:**
- Windows 10/11 (64-bit)
- Visual C++ Redistributable
- Optional: NVIDIA GPU drivers

**Linux:**
- Ubuntu 20.04+ / CentOS 8+ / Debian 11+
- Docker (recommended)
- systemd support

## Directory Structure

```
foresight-sar-fieldkit-{self.version}/
├── README.md                 # This file
├── manifest.json            # Kit metadata and checksums
├── LICENSE                  # License information
├── app/                     # Application files
│   ├── src/                # Source code
│   ├── requirements.txt    # Python dependencies
│   └── main.py            # Application entry point
├── config/                  # Configuration files
│   ├── privacy_defaults.json
│   ├── camera_intrinsics.json
│   └── jetson_config.yaml
├── docs/                    # Documentation
│   ├── user_guide.md
│   ├── api_reference.md
│   └── troubleshooting.md
├── platforms/               # Platform-specific packages
│   ├── jetson_orin/        # Jetson AGX Orin
│   ├── jetson_xavier/      # Jetson Xavier NX
│   ├── windows/            # Windows deployment
│   └── linux/              # Generic Linux
├── scripts/                 # Utility scripts
├── models/                  # AI models and configs
└── licenses/               # Legal documents
```

## Configuration

Key configuration files:

- `config/privacy_defaults.json` - Privacy and security settings
- `config/camera_intrinsics.json` - Camera calibration data
- `config/jetson_config.yaml` - Jetson-specific settings

## Usage

### Web Interface

Access the web interface at `http://localhost:8080` after installation.

### API Access

The REST API is available at `http://localhost:8080/api/v1/`

### Command Line

Use the management scripts in each platform directory for command-line control.

## Troubleshooting

### Common Issues

**Service won't start:**
1. Check logs: `sudo journalctl -u foresight-sar -f`
2. Verify Python dependencies: `pip3 list`
3. Check permissions: `ls -la /opt/foresight`

**Web interface not accessible:**
1. Check if service is running
2. Verify firewall settings
3. Check port 8080 availability

**GPU not detected:**
1. Verify CUDA installation: `nvcc --version`
2. Check GPU status: `nvidia-smi`
3. Restart services

**High memory usage:**
1. Check system resources: `htop`
2. Reduce batch sizes in configuration
3. Enable swap if needed

### Getting Help

- **Documentation:** See `docs/` directory
- **GitHub Issues:** https://github.com/foresight-sar/foresight/issues
- **Community:** https://github.com/foresight-sar/foresight/discussions

## Security Considerations

- Review privacy settings in `config/privacy_defaults.json`
- Enable authentication for production deployments
- Use HTTPS in production environments
- Regularly update dependencies
- Follow the kill switch procedures in emergencies

## License

See `licenses/` directory for complete license information.

## Support

For technical support and questions:

- GitHub: https://github.com/foresight-sar/foresight
- Issues: https://github.com/foresight-sar/foresight/issues
- Documentation: `docs/` directory

---

**Field Kit Version:** {self.version}  
**Build Date:** {self.build_date}  
**Kit Type:** {self.kit_type.title()}  
'''
        
        with open(kit_dir / 'README.md', 'w') as f:
            f.write(readme_content)
        
        logger.info("Field kit README created")
    
    def copy_licenses(self, kit_dir):
        """Copy license files to field kit"""
        logger.info("Copying license files...")
        
        licenses_dir = kit_dir / 'licenses'
        
        # Copy main license
        if (self.source_dir / 'LICENSE').exists():
            shutil.copy2(self.source_dir / 'LICENSE', licenses_dir)
        
        # Create third-party licenses file
        third_party_licenses = f'''
# Third-Party Licenses

This document contains the licenses for third-party software used in Foresight SAR.

## Python Packages

The following Python packages are used under their respective licenses:

### PyTorch
- License: BSD-3-Clause
- URL: https://github.com/pytorch/pytorch/blob/master/LICENSE

### OpenCV
- License: Apache-2.0
- URL: https://github.com/opencv/opencv/blob/master/LICENSE

### Ultralytics YOLOv8
- License: AGPL-3.0
- URL: https://github.com/ultralytics/ultralytics/blob/main/LICENSE

### Flask
- License: BSD-3-Clause
- URL: https://github.com/pallets/flask/blob/main/LICENSE.rst

### NumPy
- License: BSD-3-Clause
- URL: https://github.com/numpy/numpy/blob/main/LICENSE.txt

### Pillow
- License: HPND
- URL: https://github.com/python-pillow/Pillow/blob/main/LICENSE

## JavaScript Libraries

### React
- License: MIT
- URL: https://github.com/facebook/react/blob/main/LICENSE

### Axios
- License: MIT
- URL: https://github.com/axios/axios/blob/master/LICENSE

## System Dependencies

### CUDA
- License: NVIDIA Software License
- URL: https://docs.nvidia.com/cuda/eula/index.html

### TensorRT
- License: NVIDIA TensorRT License
- URL: https://docs.nvidia.com/deeplearning/tensorrt/sla/index.html

## Note

This is not an exhaustive list. Please check the requirements.txt file and package.json for a complete list of dependencies and their respective licenses.

For the most up-to-date license information, please refer to the individual package repositories.
'''
        
        with open(licenses_dir / 'THIRD_PARTY_LICENSES.md', 'w') as f:
            f.write(third_party_licenses)
        
        logger.info("License files copied")
    
    def calculate_kit_size(self, kit_dir):
        """Calculate total size of field kit"""
        total_size = 0
        file_count = 0
        
        for file_path in kit_dir.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
        
        size_mb = total_size / (1024 * 1024)
        
        logger.info(f"Field kit size: {size_mb:.1f} MB ({file_count} files)")
        
        # Check against size limit
        config = self.kit_configs[self.kit_type]
        if size_mb > config['size_limit_mb']:
            logger.warning(f"Kit size ({size_mb:.1f} MB) exceeds limit ({config['size_limit_mb']} MB)")
        
        return size_mb, file_count
    
    def create_packages(self, kit_dir):
        """Create distribution packages"""
        logger.info("Creating distribution packages...")
        
        packages = {}
        
        # Create TAR.GZ package
        tar_filename = f'foresight-sar-fieldkit-{self.version}-{self.kit_type}.tar.gz'
        tar_path = self.output_dir / tar_filename
        
        with tarfile.open(tar_path, 'w:gz') as tar:
            tar.add(kit_dir, arcname=kit_dir.name)
        
        packages['tar_gz'] = tar_path
        logger.info(f"TAR.GZ package created: {tar_path}")
        
        # Create ZIP package
        zip_filename = f'foresight-sar-fieldkit-{self.version}-{self.kit_type}.zip'
        zip_path = self.output_dir / zip_filename
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in kit_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(kit_dir.parent)
                    zipf.write(file_path, arcname)
        
        packages['zip'] = zip_path
        logger.info(f"ZIP package created: {zip_path}")
        
        # Calculate package sizes
        for package_type, package_path in packages.items():
            size_mb = package_path.stat().st_size / (1024 * 1024)
            logger.info(f"{package_type.upper()} size: {size_mb:.1f} MB")
        
        return packages
    
    def cleanup(self, kit_dir):
        """Clean up temporary files"""
        logger.info("Cleaning up temporary files...")
        
        if kit_dir.exists():
            shutil.rmtree(kit_dir)
            logger.info("Field kit directory removed")
    
    def create_field_kit(self, cleanup=True):
        """Main field kit creation function"""
        logger.info(f"Creating {self.kit_type} field kit...")
        
        try:
            # Check dependencies
            if not self.check_dependencies():
                raise RuntimeError("Dependency check failed")
            
            # Create kit structure
            kit_dir = self.create_kit_structure()
            
            # Copy application files
            self.copy_application_files(kit_dir)
            
            # Create platform packages
            self.create_platform_packages(kit_dir)
            
            # Copy licenses
            self.copy_licenses(kit_dir)
            
            # Create manifest
            self.create_field_kit_manifest(kit_dir)
            
            # Create README
            self.create_field_kit_readme(kit_dir)
            
            # Calculate kit size
            size_mb, file_count = self.calculate_kit_size(kit_dir)
            
            # Create distribution packages
            packages = self.create_packages(kit_dir)
            
            # Cleanup
            if cleanup:
                self.cleanup(kit_dir)
            
            logger.info("Field kit creation completed successfully!")
            
            # Print results
            logger.info("\nField Kit Results:")
            logger.info("=" * 50)
            logger.info(f"Kit Type: {self.kit_type.title()}")
            logger.info(f"Version: {self.version}")
            logger.info(f"Size: {size_mb:.1f} MB ({file_count} files)")
            logger.info(f"Components: {', '.join(self.kit_configs[self.kit_type]['components'])}")
            logger.info(f"Platforms: {', '.join(self.kit_configs[self.kit_type]['platforms'])}")
            logger.info("\nPackages:")
            for package_type, package_path in packages.items():
                logger.info(f"  {package_type.upper()}: {package_path}")
            
            return packages
            
        except Exception as e:
            logger.error(f"Field kit creation failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Create Foresight SAR Field Kit')
    parser.add_argument('--source', '-s', default='.', help='Source directory (default: current directory)')
    parser.add_argument('--output', '-o', default='./dist', help='Output directory (default: ./dist)')
    parser.add_argument('--version', '-v', default='1.0.0', help='Version number (default: 1.0.0)')
    parser.add_argument('--kit-type', '-k', choices=['minimal', 'standard', 'complete'], 
                       default='standard', help='Field kit type (default: standard)')
    parser.add_argument('--no-cleanup', action='store_true', help='Skip cleanup of temporary files')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        creator = FieldKitCreator(
            source_dir=args.source,
            output_dir=args.output,
            version=args.version,
            kit_type=args.kit_type
        )
        
        packages = creator.create_field_kit(cleanup=not args.no_cleanup)
        
        print("\n✅ Field kit creation completed successfully!")
        print(f"\n📦 Packages created:")
        for package_type, package_path in packages.items():
            print(f"   {package_type.upper()}: {package_path}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Field kit creation failed: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())