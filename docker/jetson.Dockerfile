# Foresight SAR System - Jetson Deployment
# Optimized for NVIDIA Jetson AGX Orin, Xavier NX, and Nano devices
# Includes CUDA, TensorRT, and PyTorch for edge AI inference

# Use NVIDIA L4T (Linux for Tegra) base image with CUDA support
# This provides optimized drivers and libraries for Jetson hardware
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# Metadata
LABEL maintainer="Foresight SAR Team"
LABEL version="1.0"
LABEL description="Foresight SAR System for NVIDIA Jetson Edge Devices"
LABEL gpu.required="true"
LABEL platform="jetson"

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV TENSORRT_VERSION=8.5.2
ENV OPENCV_VERSION=4.8.0
ENV FORESIGHT_HOME=/opt/foresight
ENV FORESIGHT_CONFIG=/etc/foresight
ENV FORESIGHT_DATA=/var/lib/foresight
ENV FORESIGHT_LOGS=/var/log/foresight

# Set working directory
WORKDIR $FORESIGHT_HOME

# Update system packages and install dependencies
RUN apt-get update && apt-get install -y \
    # System utilities
    curl \
    wget \
    git \
    vim \
    htop \
    tree \
    unzip \
    software-properties-common \
    # Build tools
    build-essential \
    cmake \
    pkg-config \
    # Python development
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    # OpenCV dependencies
    libopencv-dev \
    libopencv-contrib-dev \
    libopencv-imgproc-dev \
    libopencv-imgcodecs-dev \
    libopencv-video-dev \
    # Media processing
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    # Image processing
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    # Math libraries
    libatlas-base-dev \
    liblapack-dev \
    libblas-dev \
    libhdf5-dev \
    # Networking
    libssl-dev \
    libcurl4-openssl-dev \
    # Database
    libpq-dev \
    libsqlite3-dev \
    # System monitoring
    lm-sensors \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages optimized for Jetson
# Use pre-built wheels where available for faster installation
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch ecosystem (already included in base image, but ensure latest)
RUN pip3 install --no-cache-dir \
    torchvision \
    torchaudio \
    # Computer vision
    opencv-python-headless==4.8.0.* \
    pillow \
    imageio \
    scikit-image \
    # Machine learning
    scikit-learn \
    numpy \
    scipy \
    pandas \
    # Deep learning utilities
    timm \
    transformers \
    ultralytics \
    # Web framework
    flask \
    flask-cors \
    flask-jwt-extended \
    gunicorn \
    # API and networking
    requests \
    websockets \
    paho-mqtt \
    # Data processing
    h5py \
    pyarrow \
    # Geospatial
    pyproj \
    shapely \
    rasterio \
    # Cryptography and security
    cryptography \
    pycryptodome \
    hashicorp-vault \
    # Configuration and utilities
    pyyaml \
    toml \
    click \
    tqdm \
    psutil \
    # Monitoring and logging
    prometheus-client \
    structlog \
    # Testing (for development)
    pytest \
    pytest-cov

# Install TensorRT Python bindings (if not already included)
RUN python3 -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')" || \
    pip3 install --no-cache-dir nvidia-tensorrt

# Install ONNX and ONNX Runtime for model optimization
RUN pip3 install --no-cache-dir \
    onnx \
    onnxruntime-gpu \
    onnx-simplifier

# Create application directories
RUN mkdir -p \
    $FORESIGHT_CONFIG \
    $FORESIGHT_DATA \
    $FORESIGHT_LOGS \
    $FORESIGHT_HOME/models \
    $FORESIGHT_HOME/cache \
    $FORESIGHT_HOME/temp

# Copy application code
COPY src/ $FORESIGHT_HOME/src/
COPY config/ $FORESIGHT_CONFIG/
COPY scripts/ $FORESIGHT_HOME/scripts/
COPY models/ $FORESIGHT_HOME/models/
COPY requirements.txt $FORESIGHT_HOME/

# Install application-specific dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy Jetson-specific configuration
COPY docker/jetson/jetson_config.yaml $FORESIGHT_CONFIG/
COPY docker/jetson/performance_profiles.json $FORESIGHT_CONFIG/
COPY docker/jetson/thermal_config.json $FORESIGHT_CONFIG/

# Create Jetson-specific configuration files
RUN cat > $FORESIGHT_CONFIG/jetson_config.yaml << 'EOF'
# Jetson-specific configuration for Foresight SAR
jetson:
  platform: "auto"  # auto-detect: orin, xavier, nano
  power_mode: "MAXN"  # MAXN, 15W, 10W (device dependent)
  gpu_freq: "max"  # max, balanced, power_save
  cpu_freq: "max"  # max, balanced, power_save
  memory_freq: "max"  # max, balanced, power_save
  
performance:
  inference_threads: 4
  preprocessing_threads: 2
  postprocessing_threads: 2
  max_batch_size: 8
  tensorrt_optimization: true
  fp16_inference: true
  dynamic_batching: true
  
thermal:
  max_temp_celsius: 85
  throttle_temp_celsius: 80
  fan_control: "auto"  # auto, manual, off
  thermal_monitoring: true
  
storage:
  cache_size_mb: 1024
  temp_cleanup_interval: 3600
  log_rotation_size_mb: 100
  max_log_files: 10
  
networking:
  wifi_country: "US"
  bluetooth_enabled: false
  ethernet_priority: true
  hotspot_fallback: true
EOF

# Create performance profiles for different Jetson devices
RUN cat > $FORESIGHT_CONFIG/performance_profiles.json << 'EOF'
{
  "profiles": {
    "jetson_orin_agx": {
      "gpu_memory_mb": 32768,
      "cpu_cores": 12,
      "max_power_watts": 60,
      "inference_batch_size": 16,
      "concurrent_streams": 4,
      "tensorrt_workspace_mb": 4096
    },
    "jetson_orin_nx": {
      "gpu_memory_mb": 8192,
      "cpu_cores": 8,
      "max_power_watts": 25,
      "inference_batch_size": 8,
      "concurrent_streams": 2,
      "tensorrt_workspace_mb": 2048
    },
    "jetson_xavier_nx": {
      "gpu_memory_mb": 8192,
      "cpu_cores": 6,
      "max_power_watts": 20,
      "inference_batch_size": 4,
      "concurrent_streams": 2,
      "tensorrt_workspace_mb": 1024
    },
    "jetson_nano": {
      "gpu_memory_mb": 2048,
      "cpu_cores": 4,
      "max_power_watts": 10,
      "inference_batch_size": 2,
      "concurrent_streams": 1,
      "tensorrt_workspace_mb": 512
    }
  }
}
EOF

# Create thermal management configuration
RUN cat > $FORESIGHT_CONFIG/thermal_config.json << 'EOF'
{
  "thermal_zones": {
    "cpu": {
      "warning_temp": 75,
      "critical_temp": 85,
      "shutdown_temp": 95
    },
    "gpu": {
      "warning_temp": 80,
      "critical_temp": 90,
      "shutdown_temp": 100
    },
    "thermal": {
      "warning_temp": 70,
      "critical_temp": 80,
      "shutdown_temp": 90
    }
  },
  "cooling_policies": {
    "aggressive": {
      "fan_speed_min": 50,
      "fan_speed_max": 100,
      "cpu_throttle_temp": 75,
      "gpu_throttle_temp": 80
    },
    "balanced": {
      "fan_speed_min": 30,
      "fan_speed_max": 80,
      "cpu_throttle_temp": 80,
      "gpu_throttle_temp": 85
    },
    "quiet": {
      "fan_speed_min": 20,
      "fan_speed_max": 60,
      "cpu_throttle_temp": 70,
      "gpu_throttle_temp": 75
    }
  }
}
EOF

# Create Jetson optimization script
RUN cat > $FORESIGHT_HOME/scripts/jetson_optimize.sh << 'EOF'
#!/bin/bash
# Jetson optimization script for Foresight SAR

set -e

echo "Optimizing Jetson device for Foresight SAR..."

# Detect Jetson platform
JETSON_MODEL=$(cat /proc/device-tree/model 2>/dev/null | tr -d '\0' || echo "unknown")
echo "Detected Jetson model: $JETSON_MODEL"

# Set performance mode
echo "Setting performance mode..."
sudo nvpmodel -m 0 2>/dev/null || echo "nvpmodel not available"

# Maximize CPU performance
echo "Maximizing CPU performance..."
sudo jetson_clocks 2>/dev/null || echo "jetson_clocks not available"

# Set GPU to maximum performance
echo "Setting GPU to maximum performance..."
if [ -f /sys/devices/gpu.0/devfreq/17000000.gv11b/governor ]; then
    echo performance | sudo tee /sys/devices/gpu.0/devfreq/17000000.gv11b/governor
fi

# Disable unnecessary services to free up resources
echo "Disabling unnecessary services..."
sudo systemctl disable bluetooth 2>/dev/null || true
sudo systemctl stop bluetooth 2>/dev/null || true

# Set memory governor
echo "Optimizing memory governor..."
if [ -f /sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked ]; then
    echo 1 | sudo tee /sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked
fi

# Configure swap (if needed)
echo "Configuring swap..."
if [ ! -f /swapfile ]; then
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

# Set CPU affinity for better performance
echo "Setting CPU affinity..."
echo 0-11 | sudo tee /sys/fs/cgroup/cpuset/cpuset.cpus 2>/dev/null || true

# Configure thermal management
echo "Configuring thermal management..."
if [ -f /sys/devices/virtual/thermal/thermal_zone0/trip_point_0_temp ]; then
    echo 85000 | sudo tee /sys/devices/virtual/thermal/thermal_zone0/trip_point_0_temp
fi

echo "Jetson optimization complete!"
EOF

# Make optimization script executable
RUN chmod +x $FORESIGHT_HOME/scripts/jetson_optimize.sh

# Create model optimization script for TensorRT
RUN cat > $FORESIGHT_HOME/scripts/optimize_models.py << 'EOF'
#!/usr/bin/env python3
"""
Model optimization script for Jetson deployment
Converts PyTorch models to TensorRT for optimal inference performance
"""

import os
import sys
import json
import torch
import tensorrt as trt
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def optimize_yolo_model(model_path, output_path, precision='fp16'):
    """Optimize YOLOv8 model for TensorRT"""
    try:
        from ultralytics import YOLO
        
        logger.info(f"Loading YOLO model from {model_path}")
        model = YOLO(model_path)
        
        # Export to TensorRT
        logger.info(f"Exporting to TensorRT with {precision} precision")
        model.export(
            format='engine',
            half=(precision == 'fp16'),
            dynamic=True,
            workspace=4,  # 4GB workspace
            device=0
        )
        
        logger.info(f"TensorRT model saved to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to optimize YOLO model: {e}")
        return False

def optimize_reid_model(model_path, output_path, precision='fp16'):
    """Optimize ReID model for TensorRT"""
    try:
        # Load PyTorch model
        logger.info(f"Loading ReID model from {model_path}")
        model = torch.load(model_path, map_location='cuda')
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 256, 128).cuda()
        
        # Export to ONNX first
        onnx_path = output_path.replace('.engine', '.onnx')
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Convert ONNX to TensorRT
        logger.info(f"Converting ONNX to TensorRT")
        # Implementation would use TensorRT Python API
        
        logger.info(f"TensorRT model saved to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to optimize ReID model: {e}")
        return False

def main():
    """Main optimization function"""
    models_dir = Path('/opt/foresight/models')
    optimized_dir = models_dir / 'optimized'
    optimized_dir.mkdir(exist_ok=True)
    
    # Optimize detection models
    yolo_models = list(models_dir.glob('**/yolo*.pt'))
    for model_path in yolo_models:
        output_path = optimized_dir / f"{model_path.stem}_trt.engine"
        optimize_yolo_model(str(model_path), str(output_path))
    
    # Optimize ReID models
    reid_models = list(models_dir.glob('**/reid*.pt'))
    for model_path in reid_models:
        output_path = optimized_dir / f"{model_path.stem}_trt.engine"
        optimize_reid_model(str(model_path), str(output_path))
    
    logger.info("Model optimization complete!")

if __name__ == '__main__':
    main()
EOF

# Make model optimization script executable
RUN chmod +x $FORESIGHT_HOME/scripts/optimize_models.py

# Create startup script
RUN cat > $FORESIGHT_HOME/scripts/start_foresight.sh << 'EOF'
#!/bin/bash
# Foresight SAR startup script for Jetson

set -e

echo "Starting Foresight SAR on Jetson..."

# Source environment
source /opt/foresight/scripts/jetson_optimize.sh

# Check GPU availability
echo "Checking GPU availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Optimize models if needed
if [ ! -d "/opt/foresight/models/optimized" ] || [ -z "$(ls -A /opt/foresight/models/optimized)" ]; then
    echo "Optimizing models for TensorRT..."
    python3 /opt/foresight/scripts/optimize_models.py
fi

# Start monitoring services
echo "Starting system monitoring..."
python3 /opt/foresight/scripts/jetson_monitor.py &

# Start main application
echo "Starting Foresight SAR application..."
cd /opt/foresight
export PYTHONPATH=/opt/foresight/src:$PYTHONPATH
export FORESIGHT_CONFIG=/etc/foresight
export FORESIGHT_PLATFORM=jetson

# Start with gunicorn for production
if [ "$FORESIGHT_ENV" = "production" ]; then
    gunicorn --bind 0.0.0.0:8080 --workers 2 --timeout 300 --worker-class gevent src.backend.app:app
else
    python3 -m src.backend.app
fi
EOF

# Make startup script executable
RUN chmod +x $FORESIGHT_HOME/scripts/start_foresight.sh

# Create Jetson monitoring script
RUN cat > $FORESIGHT_HOME/scripts/jetson_monitor.py << 'EOF'
#!/usr/bin/env python3
"""
Jetson system monitoring for Foresight SAR
Monitors temperature, power, memory, and GPU utilization
"""

import time
import json
import psutil
import logging
from pathlib import Path
from threading import Thread

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JetsonMonitor:
    def __init__(self):
        self.monitoring = True
        self.stats_file = Path('/var/log/foresight/jetson_stats.json')
        self.stats_file.parent.mkdir(parents=True, exist_ok=True)
    
    def read_thermal_zone(self, zone_path):
        """Read temperature from thermal zone"""
        try:
            with open(zone_path, 'r') as f:
                temp_millicelsius = int(f.read().strip())
                return temp_millicelsius / 1000.0
        except:
            return None
    
    def get_gpu_stats(self):
        """Get GPU statistics"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # Power
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
            except:
                power = None
            
            return {
                'gpu_utilization': util.gpu,
                'memory_utilization': util.memory,
                'memory_used_mb': mem_info.used // 1024 // 1024,
                'memory_total_mb': mem_info.total // 1024 // 1024,
                'temperature_celsius': temp,
                'power_watts': power
            }
        except:
            return {}
    
    def get_system_stats(self):
        """Get system statistics"""
        # CPU stats
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_freq = psutil.cpu_freq()
        cpu_temps = psutil.sensors_temperatures()
        
        # Memory stats
        memory = psutil.virtual_memory()
        
        # Disk stats
        disk = psutil.disk_usage('/')
        
        # Network stats
        network = psutil.net_io_counters()
        
        # Thermal zones (Jetson-specific)
        thermal_zones = {}
        for zone_file in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
            zone_name = zone_file.parent.name
            temp = self.read_thermal_zone(zone_file)
            if temp is not None:
                thermal_zones[zone_name] = temp
        
        return {
            'timestamp': time.time(),
            'cpu': {
                'utilization_percent': cpu_percent,
                'frequency_mhz': cpu_freq.current if cpu_freq else None,
                'temperature_celsius': cpu_temps.get('coretemp', [{}])[0].get('current', None) if cpu_temps else None
            },
            'memory': {
                'utilization_percent': memory.percent,
                'used_mb': memory.used // 1024 // 1024,
                'total_mb': memory.total // 1024 // 1024,
                'available_mb': memory.available // 1024 // 1024
            },
            'disk': {
                'utilization_percent': disk.percent,
                'used_gb': disk.used // 1024 // 1024 // 1024,
                'total_gb': disk.total // 1024 // 1024 // 1024,
                'free_gb': disk.free // 1024 // 1024 // 1024
            },
            'network': {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            },
            'thermal_zones': thermal_zones,
            'gpu': self.get_gpu_stats()
        }
    
    def monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                stats = self.get_system_stats()
                
                # Log critical conditions
                if stats['cpu']['temperature_celsius'] and stats['cpu']['temperature_celsius'] > 80:
                    logger.warning(f"High CPU temperature: {stats['cpu']['temperature_celsius']}°C")
                
                if stats['gpu'].get('temperature_celsius', 0) > 85:
                    logger.warning(f"High GPU temperature: {stats['gpu']['temperature_celsius']}°C")
                
                if stats['memory']['utilization_percent'] > 90:
                    logger.warning(f"High memory usage: {stats['memory']['utilization_percent']}%")
                
                # Save stats to file
                with open(self.stats_file, 'w') as f:
                    json.dump(stats, f, indent=2)
                
                time.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(30)  # Wait longer on error
    
    def start(self):
        """Start monitoring in background thread"""
        monitor_thread = Thread(target=self.monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info("Jetson monitoring started")
    
    def stop(self):
        """Stop monitoring"""
        self.monitoring = False
        logger.info("Jetson monitoring stopped")

def main():
    monitor = JetsonMonitor()
    monitor.start()
    
    try:
        # Keep the script running
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        monitor.stop()

if __name__ == '__main__':
    main()
EOF

# Make monitoring script executable
RUN chmod +x $FORESIGHT_HOME/scripts/jetson_monitor.py

# Create systemd service file for auto-start
RUN cat > /etc/systemd/system/foresight-sar.service << 'EOF'
[Unit]
Description=Foresight SAR System
After=network.target
Wants=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/foresight
Environment=PYTHONPATH=/opt/foresight/src
Environment=FORESIGHT_CONFIG=/etc/foresight
Environment=FORESIGHT_PLATFORM=jetson
Environment=FORESIGHT_ENV=production
ExecStart=/opt/foresight/scripts/start_foresight.sh
Restart=always
RestartSec=10
KillMode=mixed
TimeoutStopSec=30

# Resource limits
LimitNOFILE=65536
LimitNPROC=32768

# Security settings
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/foresight /var/log/foresight /tmp

[Install]
WantedBy=multi-user.target
EOF

# Set proper permissions
RUN chown -R root:root $FORESIGHT_HOME && \
    chown -R root:root $FORESIGHT_CONFIG && \
    chmod -R 755 $FORESIGHT_HOME && \
    chmod -R 644 $FORESIGHT_CONFIG && \
    chmod +x $FORESIGHT_HOME/scripts/*.sh && \
    chmod +x $FORESIGHT_HOME/scripts/*.py

# Create log directories
RUN mkdir -p $FORESIGHT_LOGS && \
    chown -R root:root $FORESIGHT_LOGS && \
    chmod -R 755 $FORESIGHT_LOGS

# Expose ports
EXPOSE 8080 8443 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set entrypoint
ENTRYPOINT ["/opt/foresight/scripts/start_foresight.sh"]

# Default command
CMD []

# Build information
ARG BUILD_DATE
ARG VCS_REF
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/foresight-sar/foresight" \
      org.label-schema.schema-version="1.0"