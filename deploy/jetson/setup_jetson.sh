#!/bin/bash
# Foresight SAR System - Jetson Setup Script
# Configures NVIDIA Jetson device for optimal SAR operations

set -e

echo "=== Foresight SAR Jetson Setup ==="
echo "Configuring NVIDIA Jetson for SAR operations..."

# Check if running on Jetson
if [ ! -f "/etc/nv_tegra_release" ]; then
    echo "Warning: This script is designed for NVIDIA Jetson devices"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
sudo apt install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    unzip \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    python3-dev \
    python3-pip \
    python3-venv \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    libqtgui4 \
    libqtwebkit4 \
    libqt4-test \
    python3-pyqt5 \
    libopenblas-dev \
    liblapack-dev \
    libeigen3-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libdc1394-22-dev \
    libavresample-dev \
    libgphoto2-dev \
    libgphoto2-port12 \
    libtbb2 \
    libtbb-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-glog-dev \
    libgflags-dev \
    libgtest-dev \
    liblmdb-dev \
    libleveldb-dev \
    libsnappy-dev \
    libboost-all-dev \
    libcaffe-cuda-dev \
    redis-server \
    postgresql \
    postgresql-contrib \
    nginx \
    supervisor \
    htop \
    iotop \
    nvtop

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
fi

# Install Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Configure NVIDIA Container Runtime
echo "Configuring NVIDIA Container Runtime..."
sudo apt install -y nvidia-container-runtime

# Add nvidia runtime to Docker daemon
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "100m",
        "max-file": "3"
    }
}
EOF

# Restart Docker
sudo systemctl restart docker

# Configure Jetson performance mode
echo "Configuring Jetson performance settings..."
if command -v jetson_clocks &> /dev/null; then
    sudo jetson_clocks
fi

# Set power mode to maximum performance
if command -v nvpmodel &> /dev/null; then
    sudo nvpmodel -m 0  # Max performance mode
fi

# Increase swap space for memory-intensive operations
echo "Configuring swap space..."
SWAP_SIZE="4G"
if [ ! -f /swapfile ]; then
    sudo fallocate -l $SWAP_SIZE /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

# Configure GPU memory split
echo "Configuring GPU memory..."
if [ -f /boot/config.txt ]; then
    sudo sed -i 's/^gpu_mem=.*/gpu_mem=128/' /boot/config.txt
    if ! grep -q "gpu_mem=" /boot/config.txt; then
        echo "gpu_mem=128" | sudo tee -a /boot/config.txt
    fi
fi

# Create application directories
echo "Creating application directories..."
sudo mkdir -p /opt/foresight/{data,models,evidence,logs,cache}
sudo chown -R $USER:$USER /opt/foresight

# Configure system limits
echo "Configuring system limits..."
sudo tee /etc/security/limits.d/foresight.conf > /dev/null <<EOF
# Foresight SAR System limits
$USER soft nofile 65536
$USER hard nofile 65536
$USER soft nproc 32768
$USER hard nproc 32768
EOF

# Configure sysctl for network performance
echo "Configuring network performance..."
sudo tee /etc/sysctl.d/99-foresight.conf > /dev/null <<EOF
# Foresight SAR System network optimizations
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_congestion_control = bbr
EOF

sudo sysctl -p /etc/sysctl.d/99-foresight.conf

# Configure logrotate for application logs
echo "Configuring log rotation..."
sudo tee /etc/logrotate.d/foresight > /dev/null <<EOF
/opt/foresight/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    copytruncate
    maxage 30
}
EOF

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install --upgrade pip setuptools wheel

# Install Jetson-specific packages
echo "Installing Jetson-specific packages..."
pip3 install jetson-stats Jetson.GPIO

# Create systemd service for Foresight
echo "Creating systemd service..."
sudo tee /etc/systemd/system/foresight-sar.service > /dev/null <<EOF
[Unit]
Description=Foresight SAR System
After=docker.service
Requires=docker.service

[Service]
Type=forking
User=$USER
Group=$USER
WorkingDirectory=/opt/foresight
ExecStart=/usr/local/bin/docker-compose -f /opt/foresight/docker-compose.yml up -d
ExecStop=/usr/local/bin/docker-compose -f /opt/foresight/docker-compose.yml down
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable service but don't start yet
sudo systemctl daemon-reload
sudo systemctl enable foresight-sar.service

# Configure firewall
echo "Configuring firewall..."
sudo ufw --force enable
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 8000/tcp  # FastAPI
sudo ufw allow 8080/tcp  # Web interface
sudo ufw allow 5000/tcp  # WebSocket

# Create monitoring script
echo "Creating monitoring script..."
sudo tee /usr/local/bin/foresight-monitor > /dev/null <<'EOF'
#!/bin/bash
# Foresight SAR System Monitor

echo "=== Foresight SAR System Status ==="
echo "Date: $(date)"
echo

echo "=== System Resources ==="
echo "CPU Usage:"
top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1
echo
echo "Memory Usage:"
free -h
echo
echo "Disk Usage:"
df -h /
echo

echo "=== GPU Status ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
fi
echo

echo "=== Docker Containers ==="
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo

echo "=== Service Status ==="
systemctl is-active foresight-sar.service
echo

echo "=== Recent Logs ==="
journalctl -u foresight-sar.service --no-pager -n 10
EOF

sudo chmod +x /usr/local/bin/foresight-monitor

# Create backup script
echo "Creating backup script..."
sudo tee /usr/local/bin/foresight-backup > /dev/null <<'EOF'
#!/bin/bash
# Foresight SAR System Backup

BACKUP_DIR="/opt/foresight/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "Creating backup in $BACKUP_DIR..."

# Backup configuration
cp -r /opt/foresight/configs "$BACKUP_DIR/"

# Backup models
cp -r /opt/foresight/models "$BACKUP_DIR/"

# Backup recent evidence (last 7 days)
find /opt/foresight/evidence -mtime -7 -type f -exec cp {} "$BACKUP_DIR/evidence/" \;

# Create archive
tar -czf "$BACKUP_DIR.tar.gz" -C "$(dirname $BACKUP_DIR)" "$(basename $BACKUP_DIR)"
rm -rf "$BACKUP_DIR"

echo "Backup created: $BACKUP_DIR.tar.gz"

# Clean old backups (keep last 5)
find /opt/foresight/backups -name "*.tar.gz" -type f | sort -r | tail -n +6 | xargs rm -f
EOF

sudo chmod +x /usr/local/bin/foresight-backup

# Add cron job for daily backup
echo "Setting up daily backup..."
(crontab -l 2>/dev/null; echo "0 2 * * * /usr/local/bin/foresight-backup") | crontab -

echo
echo "=== Setup Complete ==="
echo "Jetson device configured for Foresight SAR operations"
echo
echo "Next steps:"
echo "1. Copy application files to /opt/foresight/"
echo "2. Configure application settings"
echo "3. Start service: sudo systemctl start foresight-sar.service"
echo "4. Monitor status: foresight-monitor"
echo
echo "Note: A reboot is recommended to apply all changes"
echo "Reboot now? (y/N)"
read -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo reboot
fi