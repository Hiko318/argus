#!/usr/bin/env python3
"""
Jetson packaging script for Foresight SAR
Creates deployment packages for NVIDIA Jetson devices
"""

import os
import sys
import shutil
import subprocess
import tempfile
import tarfile
from pathlib import Path
import argparse
import json
import logging
import yaml
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JetsonPackager:
    def __init__(self, source_dir, output_dir, version="1.0.0", jetson_model="auto"):
        self.source_dir = Path(source_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.version = version
        self.jetson_model = jetson_model
        self.build_date = datetime.utcnow().isoformat()
        
        # Validate source directory
        if not self.source_dir.exists():
            raise ValueError(f"Source directory does not exist: {self.source_dir}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Source: {self.source_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Version: {self.version}")
        logger.info(f"Jetson Model: {self.jetson_model}")
    
    def check_dependencies(self):
        """Check if required tools are available"""
        logger.info("Checking dependencies...")
        
        # Check Python
        try:
            python_version = sys.version_info
            if python_version.major != 3 or python_version.minor < 8:
                raise RuntimeError(f"Python 3.8+ required, found {python_version.major}.{python_version.minor}")
            logger.info(f"Python {python_version.major}.{python_version.minor}.{python_version.micro} ✓")
        except Exception as e:
            logger.error(f"Python check failed: {e}")
            return False
        
        # Check Docker (optional)
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, check=True)
            logger.info(f"Docker {result.stdout.strip()} ✓")
            self.docker_available = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Docker not found. Container builds will be skipped.")
            self.docker_available = False
        
        # Check tar
        try:
            subprocess.run(['tar', '--version'], 
                          capture_output=True, text=True, check=True)
            logger.info("tar ✓")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("tar not found")
            return False
        
        return True
    
    def create_deployment_structure(self):
        """Create deployment directory structure"""
        logger.info("Creating deployment structure...")
        
        deploy_dir = self.output_dir / f'foresight-sar-{self.version}'
        deploy_dir.mkdir(exist_ok=True)
        
        # Create directory structure
        directories = [
            'app',
            'config',
            'scripts',
            'systemd',
            'docker',
            'docs',
            'models',
            'logs',
            'data'
        ]
        
        for directory in directories:
            (deploy_dir / directory).mkdir(exist_ok=True)
        
        logger.info(f"Deployment structure created: {deploy_dir}")
        return deploy_dir
    
    def copy_application_files(self, deploy_dir):
        """Copy application files to deployment directory"""
        logger.info("Copying application files...")
        
        # Copy source code
        src_dest = deploy_dir / 'app' / 'src'
        if (self.source_dir / 'src').exists():
            shutil.copytree(self.source_dir / 'src', src_dest, dirs_exist_ok=True)
            logger.info("Source code copied")
        
        # Copy configuration files
        config_dest = deploy_dir / 'config'
        if (self.source_dir / 'config').exists():
            for config_file in (self.source_dir / 'config').rglob('*'):
                if config_file.is_file():
                    rel_path = config_file.relative_to(self.source_dir / 'config')
                    dest_file = config_dest / rel_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(config_file, dest_file)
            logger.info("Configuration files copied")
        
        # Copy scripts
        scripts_dest = deploy_dir / 'scripts'
        if (self.source_dir / 'scripts').exists():
            for script_file in (self.source_dir / 'scripts').rglob('*'):
                if script_file.is_file():
                    rel_path = script_file.relative_to(self.source_dir / 'scripts')
                    dest_file = scripts_dest / rel_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(script_file, dest_file)
                    # Make scripts executable
                    if script_file.suffix in ['.sh', '.py']:
                        dest_file.chmod(0o755)
            logger.info("Scripts copied")
        
        # Copy Docker files
        docker_dest = deploy_dir / 'docker'
        if (self.source_dir / 'docker').exists():
            for docker_file in (self.source_dir / 'docker').rglob('*'):
                if docker_file.is_file():
                    rel_path = docker_file.relative_to(self.source_dir / 'docker')
                    dest_file = docker_dest / rel_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(docker_file, dest_file)
            logger.info("Docker files copied")
        
        # Copy requirements.txt
        if (self.source_dir / 'requirements.txt').exists():
            shutil.copy2(self.source_dir / 'requirements.txt', deploy_dir / 'app')
            logger.info("Requirements file copied")
        
        # Copy documentation
        docs_dest = deploy_dir / 'docs'
        doc_files = ['README.md', 'LICENSE', 'CHANGELOG.md']
        for doc_file in doc_files:
            if (self.source_dir / doc_file).exists():
                shutil.copy2(self.source_dir / doc_file, docs_dest)
        
        # Copy docs directory if exists
        if (self.source_dir / 'docs').exists():
            for doc_file in (self.source_dir / 'docs').rglob('*'):
                if doc_file.is_file():
                    rel_path = doc_file.relative_to(self.source_dir / 'docs')
                    dest_file = docs_dest / rel_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(doc_file, dest_file)
        
        logger.info("Documentation copied")
    
    def create_systemd_service(self, deploy_dir):
        """Create systemd service files"""
        logger.info("Creating systemd service files...")
        
        systemd_dir = deploy_dir / 'systemd'
        
        # Main service file
        service_content = f'''
[Unit]
Description=Foresight SAR System
Documentation=https://github.com/foresight-sar/foresight
After=network.target network-online.target
Wants=network-online.target
Requires=foresight-sar-worker.service

[Service]
Type=simple
User=foresight
Group=foresight
WorkingDirectory=/opt/foresight/app
Environment=PYTHONPATH=/opt/foresight/app/src
Environment=FORESIGHT_CONFIG=/opt/foresight/config
Environment=FORESIGHT_DATA=/var/lib/foresight
Environment=FORESIGHT_LOGS=/var/log/foresight
Environment=FORESIGHT_PLATFORM=jetson
Environment=FORESIGHT_ENV=production
Environment=CUDA_VISIBLE_DEVICES=0
ExecStartPre=/opt/foresight/scripts/pre_start.sh
ExecStart=/usr/bin/python3 -m gunicorn --config /opt/foresight/config/gunicorn.conf.py src.backend.app:app
ExecReload=/bin/kill -HUP $MAINPID
ExecStop=/bin/kill -TERM $MAINPID
Restart=always
RestartSec=10
KillMode=mixed
TimeoutStartSec=60
TimeoutStopSec=30

# Resource limits
LimitNOFILE=65536
LimitNPROC=32768
MemoryMax=4G
CPUQuota=400%

# Security settings
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/foresight /var/log/foresight /tmp /opt/foresight/data
PrivateTmp=true
PrivateDevices=false
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=foresight-sar

[Install]
WantedBy=multi-user.target
'''
        
        with open(systemd_dir / 'foresight-sar.service', 'w') as f:
            f.write(service_content)
        
        # Worker service file
        worker_service_content = f'''
[Unit]
Description=Foresight SAR Worker
Documentation=https://github.com/foresight-sar/foresight
After=network.target
PartOf=foresight-sar.service

[Service]
Type=simple
User=foresight
Group=foresight
WorkingDirectory=/opt/foresight/app
Environment=PYTHONPATH=/opt/foresight/app/src
Environment=FORESIGHT_CONFIG=/opt/foresight/config
Environment=FORESIGHT_DATA=/var/lib/foresight
Environment=FORESIGHT_LOGS=/var/log/foresight
Environment=FORESIGHT_PLATFORM=jetson
Environment=FORESIGHT_ENV=production
ExecStart=/usr/bin/python3 -m src.backend.worker
Restart=always
RestartSec=5
KillMode=mixed
TimeoutStopSec=15

# Resource limits
LimitNOFILE=32768
LimitNPROC=16384
MemoryMax=2G
CPUQuota=200%

# Security settings
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/foresight /var/log/foresight /tmp /opt/foresight/data
PrivateTmp=true

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=foresight-sar-worker

[Install]
WantedBy=multi-user.target
'''
        
        with open(systemd_dir / 'foresight-sar-worker.service', 'w') as f:
            f.write(worker_service_content)
        
        # Monitoring service
        monitor_service_content = f'''
[Unit]
Description=Foresight SAR System Monitor
Documentation=https://github.com/foresight-sar/foresight
After=foresight-sar.service
Wants=foresight-sar.service

[Service]
Type=simple
User=foresight
Group=foresight
WorkingDirectory=/opt/foresight
Environment=PYTHONPATH=/opt/foresight/app/src
Environment=FORESIGHT_CONFIG=/opt/foresight/config
ExecStart=/usr/bin/python3 /opt/foresight/scripts/jetson_monitor.py
Restart=always
RestartSec=30
KillMode=mixed
TimeoutStopSec=10

# Resource limits
LimitNOFILE=1024
LimitNPROC=512
MemoryMax=512M
CPUQuota=50%

# Security settings
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/log/foresight /tmp
PrivateTmp=true

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=foresight-sar-monitor

[Install]
WantedBy=multi-user.target
'''
        
        with open(systemd_dir / 'foresight-sar-monitor.service', 'w') as f:
            f.write(monitor_service_content)
        
        logger.info("Systemd service files created")
    
    def create_installation_scripts(self, deploy_dir):
        """Create installation and management scripts"""
        logger.info("Creating installation scripts...")
        
        scripts_dir = deploy_dir / 'scripts'
        
        # Installation script
        install_script = f'''
#!/bin/bash
# Foresight SAR Installation Script for Jetson
# Version: {self.version}

set -e

echo "Installing Foresight SAR {self.version} on Jetson..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (use sudo)"
    exit 1
fi

# Detect Jetson platform
JETSON_MODEL=$(cat /proc/device-tree/model 2>/dev/null | tr -d '\0' || echo "unknown")
echo "Detected Jetson: $JETSON_MODEL"

# Check system requirements
echo "Checking system requirements..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d" " -f2 | cut -d"." -f1,2)
if [ "$(echo "$PYTHON_VERSION >= 3.8" | bc -l)" -eq 0 ]; then
    echo "Error: Python 3.8+ required, found $PYTHON_VERSION"
    exit 1
fi
echo "Python $PYTHON_VERSION ✓"

# Check available memory
MEM_GB=$(free -g | awk '/^Mem:/ {{print $2}}')
if [ "$MEM_GB" -lt 4 ]; then
    echo "Warning: Less than 4GB RAM detected. Performance may be limited."
fi
echo "Memory: ${MEM_GB}GB"

# Check CUDA availability
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "CUDA $CUDA_VERSION ✓"
else
    echo "Warning: CUDA not found. GPU acceleration will not be available."
fi

# Create user and group
echo "Creating foresight user..."
if ! id "foresight" &>/dev/null; then
    useradd -r -s /bin/bash -d /opt/foresight -m foresight
    echo "User 'foresight' created"
else
    echo "User 'foresight' already exists"
fi

# Create directories
echo "Creating directories..."
mkdir -p /opt/foresight
mkdir -p /var/lib/foresight
mkdir -p /var/log/foresight
mkdir -p /etc/foresight

# Copy application files
echo "Installing application files..."
cp -r app/* /opt/foresight/
cp -r config/* /etc/foresight/
cp -r scripts/* /opt/foresight/scripts/

# Set permissions
echo "Setting permissions..."
chown -R foresight:foresight /opt/foresight
chown -R foresight:foresight /var/lib/foresight
chown -R foresight:foresight /var/log/foresight
chown -R root:root /etc/foresight
chmod -R 755 /opt/foresight/scripts
chmod +x /opt/foresight/scripts/*.sh
chmod +x /opt/foresight/scripts/*.py

# Install Python dependencies
echo "Installing Python dependencies..."
if [ -f "/opt/foresight/requirements.txt" ]; then
    pip3 install -r /opt/foresight/requirements.txt
    echo "Python dependencies installed"
else
    echo "Warning: requirements.txt not found"
fi

# Install systemd services
echo "Installing systemd services..."
cp systemd/*.service /etc/systemd/system/
systemctl daemon-reload

# Enable services
echo "Enabling services..."
systemctl enable foresight-sar.service
systemctl enable foresight-sar-worker.service
systemctl enable foresight-sar-monitor.service

# Optimize Jetson performance
echo "Optimizing Jetson performance..."
/opt/foresight/scripts/jetson_optimize.sh || echo "Optimization script failed"

# Create log rotation configuration
echo "Configuring log rotation..."
cat > /etc/logrotate.d/foresight-sar << 'EOF'
/var/log/foresight/*.log {{
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 foresight foresight
    postrotate
        systemctl reload foresight-sar.service > /dev/null 2>&1 || true
    endscript
}}
EOF

# Create firewall rules (if ufw is available)
if command -v ufw &> /dev/null; then
    echo "Configuring firewall..."
    ufw allow 8080/tcp comment "Foresight SAR Web Interface"
    ufw allow 9090/tcp comment "Foresight SAR Metrics"
fi

echo "Installation completed successfully!"
echo ""
echo "Next steps:"
echo "1. Review configuration in /etc/foresight/"
echo "2. Start services: sudo systemctl start foresight-sar"
echo "3. Check status: sudo systemctl status foresight-sar"
echo "4. View logs: sudo journalctl -u foresight-sar -f"
echo "5. Access web interface: http://$(hostname -I | awk '{{print $1}}'):8080"
echo ""
echo "For troubleshooting, see: /opt/foresight/docs/"
'''
        
        with open(scripts_dir / 'install.sh', 'w') as f:
            f.write(install_script)
        (scripts_dir / 'install.sh').chmod(0o755)
        
        # Uninstallation script
        uninstall_script = f'''
#!/bin/bash
# Foresight SAR Uninstallation Script for Jetson

set -e

echo "Uninstalling Foresight SAR..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (use sudo)"
    exit 1
fi

# Stop and disable services
echo "Stopping services..."
systemctl stop foresight-sar.service foresight-sar-worker.service foresight-sar-monitor.service 2>/dev/null || true
systemctl disable foresight-sar.service foresight-sar-worker.service foresight-sar-monitor.service 2>/dev/null || true

# Remove systemd service files
echo "Removing systemd services..."
rm -f /etc/systemd/system/foresight-sar*.service
systemctl daemon-reload

# Remove application files
echo "Removing application files..."
rm -rf /opt/foresight
rm -rf /etc/foresight

# Remove log rotation configuration
rm -f /etc/logrotate.d/foresight-sar

# Ask about data and logs
read -p "Remove data and logs? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf /var/lib/foresight
    rm -rf /var/log/foresight
    echo "Data and logs removed"
else
    echo "Data and logs preserved"
fi

# Ask about user removal
read -p "Remove foresight user? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    userdel -r foresight 2>/dev/null || true
    echo "User removed"
else
    echo "User preserved"
fi

# Remove firewall rules (if ufw is available)
if command -v ufw &> /dev/null; then
    echo "Removing firewall rules..."
    ufw delete allow 8080/tcp 2>/dev/null || true
    ufw delete allow 9090/tcp 2>/dev/null || true
fi

echo "Uninstallation completed!"
'''
        
        with open(scripts_dir / 'uninstall.sh', 'w') as f:
            f.write(uninstall_script)
        (scripts_dir / 'uninstall.sh').chmod(0o755)
        
        # Management script
        manage_script = f'''
#!/bin/bash
# Foresight SAR Management Script

SERVICE_NAME="foresight-sar"
WORKER_SERVICE="foresight-sar-worker"
MONITOR_SERVICE="foresight-sar-monitor"

case "$1" in
    start)
        echo "Starting Foresight SAR services..."
        sudo systemctl start $SERVICE_NAME $WORKER_SERVICE $MONITOR_SERVICE
        ;;
    stop)
        echo "Stopping Foresight SAR services..."
        sudo systemctl stop $SERVICE_NAME $WORKER_SERVICE $MONITOR_SERVICE
        ;;
    restart)
        echo "Restarting Foresight SAR services..."
        sudo systemctl restart $SERVICE_NAME $WORKER_SERVICE $MONITOR_SERVICE
        ;;
    status)
        echo "Foresight SAR service status:"
        sudo systemctl status $SERVICE_NAME $WORKER_SERVICE $MONITOR_SERVICE
        ;;
    logs)
        echo "Foresight SAR logs (press Ctrl+C to exit):"
        sudo journalctl -u $SERVICE_NAME -u $WORKER_SERVICE -u $MONITOR_SERVICE -f
        ;;
    enable)
        echo "Enabling Foresight SAR services..."
        sudo systemctl enable $SERVICE_NAME $WORKER_SERVICE $MONITOR_SERVICE
        ;;
    disable)
        echo "Disabling Foresight SAR services..."
        sudo systemctl disable $SERVICE_NAME $WORKER_SERVICE $MONITOR_SERVICE
        ;;
    health)
        echo "Checking Foresight SAR health..."
        curl -f http://localhost:8080/health || echo "Health check failed"
        ;;
    stats)
        echo "System statistics:"
        if [ -f "/var/log/foresight/jetson_stats.json" ]; then
            cat /var/log/foresight/jetson_stats.json | python3 -m json.tool
        else
            echo "Stats file not found"
        fi
        ;;
    update)
        echo "Updating Foresight SAR..."
        echo "Please download and run the latest installer"
        ;;
    *)
        echo "Usage: $0 {{start|stop|restart|status|logs|enable|disable|health|stats|update}}"
        echo ""
        echo "Commands:"
        echo "  start    - Start all services"
        echo "  stop     - Stop all services"
        echo "  restart  - Restart all services"
        echo "  status   - Show service status"
        echo "  logs     - Show live logs"
        echo "  enable   - Enable services at boot"
        echo "  disable  - Disable services at boot"
        echo "  health   - Check application health"
        echo "  stats    - Show system statistics"
        echo "  update   - Update instructions"
        exit 1
        ;;
esac
'''
        
        with open(scripts_dir / 'foresight-sar', 'w') as f:
            f.write(manage_script)
        (scripts_dir / 'foresight-sar').chmod(0o755)
        
        logger.info("Installation scripts created")
    
    def create_docker_compose(self, deploy_dir):
        """Create Docker Compose configuration"""
        logger.info("Creating Docker Compose configuration...")
        
        docker_compose_content = f'''
version: '3.8'

services:
  foresight-sar:
    build:
      context: .
      dockerfile: docker/jetson.Dockerfile
      args:
        BUILD_DATE: "{self.build_date}"
        VCS_REF: "main"
    image: foresight-sar:jetson-{self.version}
    container_name: foresight-sar
    restart: unless-stopped
    ports:
      - "8080:8080"
      - "9090:9090"
    volumes:
      - foresight-data:/var/lib/foresight
      - foresight-logs:/var/log/foresight
      - foresight-models:/opt/foresight/models
      - ./config:/etc/foresight:ro
    environment:
      - FORESIGHT_ENV=production
      - FORESIGHT_PLATFORM=jetson
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    depends_on:
      - redis
    networks:
      - foresight-network

  redis:
    image: redis:7-alpine
    container_name: foresight-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes
    networks:
      - foresight-network

  prometheus:
    image: prom/prometheus:latest
    container_name: foresight-prometheus
    restart: unless-stopped
    ports:
      - "9091:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    networks:
      - foresight-network

  grafana:
    image: grafana/grafana:latest
    container_name: foresight-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    networks:
      - foresight-network

volumes:
  foresight-data:
    driver: local
  foresight-logs:
    driver: local
  foresight-models:
    driver: local
  redis-data:
    driver: local
  prometheus-data:
    driver: local
  grafana-data:
    driver: local

networks:
  foresight-network:
    driver: bridge
'''
        
        with open(deploy_dir / 'docker-compose.yml', 'w') as f:
            f.write(docker_compose_content)
        
        # Create monitoring configuration
        monitoring_dir = deploy_dir / 'monitoring'
        monitoring_dir.mkdir(exist_ok=True)
        
        # Prometheus configuration
        prometheus_config = f'''
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'foresight-sar'
    static_configs:
      - targets: ['foresight-sar:9090']
    scrape_interval: 5s
    metrics_path: '/metrics'

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['host.docker.internal:9100']
'''
        
        with open(monitoring_dir / 'prometheus.yml', 'w') as f:
            f.write(prometheus_config)
        
        logger.info("Docker Compose configuration created")
    
    def create_deployment_manifest(self, deploy_dir):
        """Create deployment manifest with metadata"""
        logger.info("Creating deployment manifest...")
        
        manifest = {
            'name': 'foresight-sar',
            'version': self.version,
            'build_date': self.build_date,
            'platform': 'jetson',
            'jetson_model': self.jetson_model,
            'description': 'Foresight Search and Rescue System for NVIDIA Jetson',
            'requirements': {
                'python': '>=3.8',
                'cuda': '>=11.4',
                'memory_gb': 4,
                'storage_gb': 10,
                'jetson_models': ['orin_agx', 'orin_nx', 'xavier_nx', 'nano']
            },
            'services': {
                'foresight-sar': {
                    'description': 'Main application service',
                    'port': 8080,
                    'health_endpoint': '/health'
                },
                'foresight-sar-worker': {
                    'description': 'Background worker service'
                },
                'foresight-sar-monitor': {
                    'description': 'System monitoring service',
                    'port': 9090
                }
            },
            'installation': {
                'script': 'scripts/install.sh',
                'user': 'foresight',
                'directories': {
                    'app': '/opt/foresight',
                    'config': '/etc/foresight',
                    'data': '/var/lib/foresight',
                    'logs': '/var/log/foresight'
                }
            },
            'management': {
                'script': 'scripts/foresight-sar',
                'systemd_services': [
                    'foresight-sar.service',
                    'foresight-sar-worker.service',
                    'foresight-sar-monitor.service'
                ]
            },
            'docker': {
                'compose_file': 'docker-compose.yml',
                'image': f'foresight-sar:jetson-{self.version}',
                'dockerfile': 'docker/jetson.Dockerfile'
            }
        }
        
        with open(deploy_dir / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info("Deployment manifest created")
    
    def create_readme(self, deploy_dir):
        """Create deployment README"""
        logger.info("Creating deployment README...")
        
        readme_content = f'''
# Foresight SAR - Jetson Deployment Package

Version: {self.version}  
Build Date: {self.build_date}  
Platform: NVIDIA Jetson  
Target Model: {self.jetson_model}  

## Overview

This package contains everything needed to deploy Foresight SAR on NVIDIA Jetson devices.

## System Requirements

- NVIDIA Jetson device (AGX Orin, Orin NX, Xavier NX, or Nano)
- JetPack 5.0+ with CUDA support
- Ubuntu 20.04+ (L4T)
- Python 3.8+
- 4GB+ RAM (8GB+ recommended)
- 10GB+ free storage
- Internet connection for initial setup

## Quick Installation

1. Extract the package:
   ```bash
   tar -xzf foresight-sar-{self.version}-jetson.tar.gz
   cd foresight-sar-{self.version}
   ```

2. Run the installation script:
   ```bash
   sudo ./scripts/install.sh
   ```

3. Start the services:
   ```bash
   sudo systemctl start foresight-sar
   ```

4. Check the status:
   ```bash
   sudo systemctl status foresight-sar
   ```

5. Access the web interface:
   ```
   http://YOUR_JETSON_IP:8080
   ```

## Docker Deployment (Alternative)

If you prefer Docker deployment:

1. Install Docker and Docker Compose
2. Build and start containers:
   ```bash
   docker-compose up -d
   ```

## Management

Use the management script for common operations:

```bash
# Start services
sudo ./scripts/foresight-sar start

# Stop services
sudo ./scripts/foresight-sar stop

# View logs
sudo ./scripts/foresight-sar logs

# Check health
sudo ./scripts/foresight-sar health

# View system stats
sudo ./scripts/foresight-sar stats
```

## Configuration

Configuration files are located in `/etc/foresight/` after installation.

Key configuration files:
- `jetson_config.yaml` - Jetson-specific settings
- `performance_profiles.json` - Performance optimization profiles
- `thermal_config.json` - Thermal management settings
- `privacy_defaults.json` - Privacy and security settings

## Performance Optimization

The installation automatically optimizes your Jetson for best performance:
- Sets maximum performance mode
- Configures CPU and GPU clocks
- Optimizes memory settings
- Configures thermal management

## Monitoring

System monitoring is available through:
- Web interface: http://YOUR_JETSON_IP:8080
- Metrics endpoint: http://YOUR_JETSON_IP:9090/metrics
- System logs: `journalctl -u foresight-sar -f`
- Stats file: `/var/log/foresight/jetson_stats.json`

## Troubleshooting

### Service won't start
1. Check logs: `sudo journalctl -u foresight-sar -f`
2. Verify permissions: `ls -la /opt/foresight`
3. Check dependencies: `pip3 list | grep torch`

### High temperature warnings
1. Check thermal status: `sudo ./scripts/foresight-sar stats`
2. Ensure proper cooling
3. Reduce performance mode if needed

### GPU not detected
1. Verify CUDA installation: `nvcc --version`
2. Check GPU status: `nvidia-smi`
3. Restart services: `sudo systemctl restart foresight-sar`

### Memory issues
1. Check memory usage: `free -h`
2. Configure swap if needed
3. Reduce batch sizes in configuration

## Uninstallation

To remove Foresight SAR:

```bash
sudo ./scripts/uninstall.sh
```

## Support

- Documentation: `/opt/foresight/docs/`
- GitHub: https://github.com/foresight-sar/foresight
- Issues: https://github.com/foresight-sar/foresight/issues

## License

See LICENSE file for licensing information.
'''
        
        with open(deploy_dir / 'README.md', 'w') as f:
            f.write(readme_content)
        
        logger.info("Deployment README created")
    
    def create_tar_package(self, deploy_dir):
        """Create TAR package for deployment"""
        logger.info("Creating TAR package...")
        
        tar_filename = f'foresight-sar-{self.version}-jetson.tar.gz'
        tar_path = self.output_dir / tar_filename
        
        with tarfile.open(tar_path, 'w:gz') as tar:
            tar.add(deploy_dir, arcname=deploy_dir.name)
        
        # Calculate file size
        file_size_mb = tar_path.stat().st_size / (1024 * 1024)
        
        logger.info(f"TAR package created: {tar_path} ({file_size_mb:.1f} MB)")
        return tar_path
    
    def build_docker_image(self, deploy_dir):
        """Build Docker image for Jetson"""
        if not self.docker_available:
            logger.warning("Docker not available, skipping image build")
            return None
        
        logger.info("Building Docker image...")
        
        try:
            # Build image
            image_tag = f'foresight-sar:jetson-{self.version}'
            cmd = [
                'docker', 'build',
                '-f', 'docker/jetson.Dockerfile',
                '-t', image_tag,
                '--build-arg', f'BUILD_DATE={self.build_date}',
                '--build-arg', 'VCS_REF=main',
                str(deploy_dir)
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=deploy_dir)
            
            logger.info(f"Docker image built: {image_tag}")
            
            # Save image to tar file
            image_tar = self.output_dir / f'foresight-sar-{self.version}-jetson-docker.tar'
            save_cmd = ['docker', 'save', '-o', str(image_tar), image_tag]
            
            subprocess.run(save_cmd, check=True)
            
            # Compress the tar file
            compressed_tar = self.output_dir / f'foresight-sar-{self.version}-jetson-docker.tar.gz'
            with open(image_tar, 'rb') as f_in:
                with tarfile.open(compressed_tar, 'w:gz') as tar_out:
                    tarinfo = tarfile.TarInfo(name=image_tar.name)
                    tarinfo.size = image_tar.stat().st_size
                    tar_out.addfile(tarinfo, f_in)
            
            # Remove uncompressed tar
            image_tar.unlink()
            
            logger.info(f"Docker image saved: {compressed_tar}")
            return compressed_tar
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker build failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            return None
    
    def cleanup(self, deploy_dir):
        """Clean up temporary files"""
        logger.info("Cleaning up temporary files...")
        
        if deploy_dir.exists():
            shutil.rmtree(deploy_dir)
            logger.info("Deployment directory removed")
    
    def package(self, create_docker=True, cleanup=True):
        """Main packaging function"""
        logger.info("Starting Jetson packaging process...")
        
        try:
            # Check dependencies
            if not self.check_dependencies():
                raise RuntimeError("Dependency check failed")
            
            # Create deployment structure
            deploy_dir = self.create_deployment_structure()
            
            # Copy application files
            self.copy_application_files(deploy_dir)
            
            # Create systemd services
            self.create_systemd_service(deploy_dir)
            
            # Create installation scripts
            self.create_installation_scripts(deploy_dir)
            
            # Create Docker Compose configuration
            self.create_docker_compose(deploy_dir)
            
            # Create deployment manifest
            self.create_deployment_manifest(deploy_dir)
            
            # Create README
            self.create_readme(deploy_dir)
            
            # Create TAR package
            tar_path = self.create_tar_package(deploy_dir)
            
            results = {'tar_package': tar_path}
            
            # Build Docker image
            if create_docker:
                docker_image = self.build_docker_image(deploy_dir)
                results['docker_image'] = docker_image
            
            # Cleanup
            if cleanup:
                self.cleanup(deploy_dir)
            
            logger.info("Packaging completed successfully!")
            
            # Print results
            logger.info("\nPackaging Results:")
            logger.info("=" * 50)
            for package_type, path in results.items():
                if path:
                    logger.info(f"{package_type.replace('_', ' ').title()}: {path}")
                else:
                    logger.info(f"{package_type.replace('_', ' ').title()}: Not created")
            
            return results
            
        except Exception as e:
            logger.error(f"Packaging failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Package Foresight SAR for Jetson')
    parser.add_argument('--source', '-s', default='.', help='Source directory (default: current directory)')
    parser.add_argument('--output', '-o', default='./dist', help='Output directory (default: ./dist)')
    parser.add_argument('--version', '-v', default='1.0.0', help='Version number (default: 1.0.0)')
    parser.add_argument('--jetson-model', default='auto', help='Target Jetson model (default: auto)')
    parser.add_argument('--no-docker', action='store_true', help='Skip Docker image creation')
    parser.add_argument('--no-cleanup', action='store_true', help='Skip cleanup of temporary files')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        packager = JetsonPackager(
            source_dir=args.source,
            output_dir=args.output,
            version=args.version,
            jetson_model=args.jetson_model
        )
        
        results = packager.package(
            create_docker=not args.no_docker,
            cleanup=not args.no_cleanup
        )
        
        print("\n✅ Packaging completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Packaging failed: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())