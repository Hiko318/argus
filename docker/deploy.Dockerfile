# Foresight SAR System - Production Deployment
# Multi-stage build for optimized production container
# Supports both CPU and GPU inference with CUDA/TensorRT

# Build stage
FROM python:3.13-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Production stage
FROM nvidia/cuda:11.8-runtime-ubuntu22.04 as production

# Metadata
LABEL maintainer="Foresight SAR Team"
LABEL version="1.0"
LABEL description="Foresight SAR System - Production Deployment"
LABEL gpu.optional="true"
LABEL platform="linux/amd64"

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src
ENV FORESIGHT_HOME=/app
ENV FORESIGHT_CONFIG=/etc/foresight
ENV FORESIGHT_DATA=/var/lib/foresight
ENV FORESIGHT_LOGS=/var/log/foresight
ENV FORESIGHT_ENV=production
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Python runtime
    python3.10 \
    python3.10-venv \
    python3-pip \
    # System utilities
    curl \
    wget \
    git \
    vim \
    htop \
    tree \
    unzip \
    ca-certificates \
    gnupg \
    lsb-release \
    # OpenCV dependencies
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Media processing
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
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
    # Networking and security
    libssl-dev \
    libcurl4-openssl-dev \
    # Database
    libpq-dev \
    libsqlite3-dev \
    # Process management
    supervisor \
    # Monitoring
    prometheus-node-exporter \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create application user for security
RUN groupadd -r foresight && useradd -r -g foresight -d /app -s /bin/bash foresight

# Create application directories
RUN mkdir -p \
    $FORESIGHT_HOME \
    $FORESIGHT_CONFIG \
    $FORESIGHT_DATA \
    $FORESIGHT_LOGS \
    $FORESIGHT_HOME/models \
    $FORESIGHT_HOME/cache \
    $FORESIGHT_HOME/temp \
    /var/run/foresight

# Copy application code
COPY --chown=foresight:foresight src/ $FORESIGHT_HOME/src/
COPY --chown=foresight:foresight config/ $FORESIGHT_CONFIG/
COPY --chown=foresight:foresight scripts/ $FORESIGHT_HOME/scripts/
COPY --chown=foresight:foresight models/ $FORESIGHT_HOME/models/
COPY --chown=foresight:foresight requirements.txt $FORESIGHT_HOME/

# Create production configuration
RUN cat > $FORESIGHT_CONFIG/production.yaml << 'EOF'
# Production configuration for Foresight SAR
app:
  name: "foresight-sar"
  version: "1.0.0"
  environment: "production"
  debug: false
  
server:
  host: "0.0.0.0"
  port: 8080
  workers: 4
  worker_class: "gevent"
  worker_connections: 1000
  timeout: 300
  keepalive: 2
  max_requests: 1000
  max_requests_jitter: 100
  
logging:
  level: "INFO"
  format: "json"
  file: "/var/log/foresight/app.log"
  max_size_mb: 100
  backup_count: 10
  
security:
  secret_key: "${FORESIGHT_SECRET_KEY}"
  jwt_secret: "${FORESIGHT_JWT_SECRET}"
  cors_origins: ["*"]
  rate_limiting: true
  max_requests_per_minute: 100
  
database:
  url: "${DATABASE_URL:-sqlite:///var/lib/foresight/foresight.db}"
  pool_size: 10
  max_overflow: 20
  pool_timeout: 30
  
redis:
  url: "${REDIS_URL:-redis://localhost:6379/0}"
  max_connections: 10
  
models:
  device: "auto"  # auto, cpu, cuda
  batch_size: 8
  max_batch_size: 32
  inference_timeout: 30
  model_cache_size: 4
  
monitoring:
  metrics_enabled: true
  metrics_port: 9090
  health_check_interval: 30
  
storage:
  upload_path: "/var/lib/foresight/uploads"
  max_file_size_mb: 100
  allowed_extensions: [".jpg", ".jpeg", ".png", ".mp4", ".avi"]
  cleanup_interval: 3600
  
geolocation:
  dem_cache_size_mb: 1024
  coordinate_precision: 6
  
privacy:
  face_blur_default: true
  data_retention_days: 30
  audit_log_retention_days: 365
  anonymization_enabled: true
EOF

# Create supervisor configuration
RUN cat > /etc/supervisor/conf.d/foresight.conf << 'EOF'
[supervisord]
nodaemon=true
user=root
logfile=/var/log/foresight/supervisord.log
pidfile=/var/run/foresight/supervisord.pid

[program:foresight-app]
command=/opt/venv/bin/gunicorn --config /app/gunicorn.conf.py src.backend.app:app
directory=/app
user=foresight
autostart=true
autorestart=true
stdout_logfile=/var/log/foresight/app.log
stderr_logfile=/var/log/foresight/app_error.log
environment=PYTHONPATH="/app/src",FORESIGHT_CONFIG="/etc/foresight",FORESIGHT_ENV="production"

[program:foresight-worker]
command=/opt/venv/bin/python -m src.backend.worker
directory=/app
user=foresight
autostart=true
autorestart=true
stdout_logfile=/var/log/foresight/worker.log
stderr_logfile=/var/log/foresight/worker_error.log
environment=PYTHONPATH="/app/src",FORESIGHT_CONFIG="/etc/foresight",FORESIGHT_ENV="production"
numprocs=2
process_name=%(program_name)s_%(process_num)02d

[program:foresight-monitor]
command=/opt/venv/bin/python /app/scripts/system_monitor.py
directory=/app
user=foresight
autostart=true
autorestart=true
stdout_logfile=/var/log/foresight/monitor.log
stderr_logfile=/var/log/foresight/monitor_error.log
environment=PYTHONPATH="/app/src",FORESIGHT_CONFIG="/etc/foresight"

[program:prometheus-exporter]
command=/usr/bin/prometheus-node-exporter --web.listen-address=:9100
autostart=true
autorestart=true
stdout_logfile=/var/log/foresight/prometheus.log
stderr_logfile=/var/log/foresight/prometheus_error.log
EOF

# Create Gunicorn configuration
RUN cat > $FORESIGHT_HOME/gunicorn.conf.py << 'EOF'
# Gunicorn configuration for Foresight SAR
import multiprocessing
import os

# Server socket
bind = "0.0.0.0:8080"
backlog = 2048

# Worker processes
workers = int(os.environ.get('GUNICORN_WORKERS', multiprocessing.cpu_count() * 2 + 1))
worker_class = "gevent"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
preload_app = True

# Timeout
timeout = 300
keepalive = 2
graceful_timeout = 30

# Logging
accesslog = "/var/log/foresight/access.log"
errorlog = "/var/log/foresight/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "foresight-sar"

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Performance
worker_tmp_dir = "/dev/shm"

# Hooks
def on_starting(server):
    server.log.info("Starting Foresight SAR server")

def on_reload(server):
    server.log.info("Reloading Foresight SAR server")

def worker_int(worker):
    worker.log.info("Worker received INT or QUIT signal")

def pre_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_worker_init(worker):
    worker.log.info("Worker initialized (pid: %s)", worker.pid)

def worker_abort(worker):
    worker.log.info("Worker received SIGABRT signal")
EOF

# Create system monitoring script
RUN cat > $FORESIGHT_HOME/scripts/system_monitor.py << 'EOF'
#!/usr/bin/env python3
"""
System monitoring for Foresight SAR production deployment
Monitors system resources, application health, and performance metrics
"""

import time
import json
import psutil
import logging
import requests
from pathlib import Path
from threading import Thread
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemMonitor:
    def __init__(self):
        self.monitoring = True
        self.stats_file = Path('/var/log/foresight/system_stats.json')
        self.stats_file.parent.mkdir(parents=True, exist_ok=True)
        self.alert_thresholds = {
            'cpu_percent': 80,
            'memory_percent': 85,
            'disk_percent': 90,
            'gpu_temp_celsius': 85,
            'response_time_ms': 5000
        }
    
    def get_gpu_stats(self):
        """Get GPU statistics if available"""
        try:
            import pynvml
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            
            gpus = []
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                # Memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                # Power
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except:
                    power = None
                
                gpus.append({
                    'index': i,
                    'utilization_percent': util.gpu,
                    'memory_utilization_percent': util.memory,
                    'memory_used_mb': mem_info.used // 1024 // 1024,
                    'memory_total_mb': mem_info.total // 1024 // 1024,
                    'temperature_celsius': temp,
                    'power_watts': power
                })
            
            return gpus
        except:
            return []
    
    def check_application_health(self):
        """Check application health endpoint"""
        try:
            start_time = time.time()
            response = requests.get('http://localhost:8080/health', timeout=10)
            response_time = (time.time() - start_time) * 1000
            
            return {
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'status_code': response.status_code,
                'response_time_ms': response_time,
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'response_time_ms': None,
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def get_process_stats(self):
        """Get Foresight process statistics"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
            try:
                if 'foresight' in ' '.join(proc.info['cmdline'] or []).lower():
                    processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_percent': proc.info['memory_percent']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return processes
    
    def get_system_stats(self):
        """Get comprehensive system statistics"""
        # CPU stats
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_freq = psutil.cpu_freq()
        cpu_count = psutil.cpu_count()
        load_avg = psutil.getloadavg()
        
        # Memory stats
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk stats
        disk = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # Network stats
        network = psutil.net_io_counters()
        
        # Process stats
        processes = self.get_process_stats()
        
        # Application health
        app_health = self.check_application_health()
        
        # GPU stats
        gpus = self.get_gpu_stats()
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'system': {
                'cpu': {
                    'utilization_percent': cpu_percent,
                    'frequency_mhz': cpu_freq.current if cpu_freq else None,
                    'core_count': cpu_count,
                    'load_average': {
                        '1min': load_avg[0],
                        '5min': load_avg[1],
                        '15min': load_avg[2]
                    }
                },
                'memory': {
                    'utilization_percent': memory.percent,
                    'used_gb': memory.used / 1024**3,
                    'total_gb': memory.total / 1024**3,
                    'available_gb': memory.available / 1024**3
                },
                'swap': {
                    'utilization_percent': swap.percent,
                    'used_gb': swap.used / 1024**3,
                    'total_gb': swap.total / 1024**3
                },
                'disk': {
                    'utilization_percent': disk.percent,
                    'used_gb': disk.used / 1024**3,
                    'total_gb': disk.total / 1024**3,
                    'free_gb': disk.free / 1024**3,
                    'io': {
                        'read_bytes': disk_io.read_bytes if disk_io else 0,
                        'write_bytes': disk_io.write_bytes if disk_io else 0,
                        'read_count': disk_io.read_count if disk_io else 0,
                        'write_count': disk_io.write_count if disk_io else 0
                    }
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv,
                    'errors_in': network.errin,
                    'errors_out': network.errout,
                    'drops_in': network.dropin,
                    'drops_out': network.dropout
                }
            },
            'gpus': gpus,
            'processes': processes,
            'application': app_health
        }
    
    def check_alerts(self, stats):
        """Check for alert conditions"""
        alerts = []
        
        # CPU alert
        if stats['system']['cpu']['utilization_percent'] > self.alert_thresholds['cpu_percent']:
            alerts.append(f"High CPU usage: {stats['system']['cpu']['utilization_percent']:.1f}%")
        
        # Memory alert
        if stats['system']['memory']['utilization_percent'] > self.alert_thresholds['memory_percent']:
            alerts.append(f"High memory usage: {stats['system']['memory']['utilization_percent']:.1f}%")
        
        # Disk alert
        if stats['system']['disk']['utilization_percent'] > self.alert_thresholds['disk_percent']:
            alerts.append(f"High disk usage: {stats['system']['disk']['utilization_percent']:.1f}%")
        
        # GPU temperature alerts
        for gpu in stats['gpus']:
            if gpu['temperature_celsius'] > self.alert_thresholds['gpu_temp_celsius']:
                alerts.append(f"High GPU {gpu['index']} temperature: {gpu['temperature_celsius']}°C")
        
        # Application response time alert
        if (stats['application']['response_time_ms'] and 
            stats['application']['response_time_ms'] > self.alert_thresholds['response_time_ms']):
            alerts.append(f"Slow application response: {stats['application']['response_time_ms']:.0f}ms")
        
        # Application health alert
        if stats['application']['status'] != 'healthy':
            alerts.append(f"Application unhealthy: {stats['application'].get('error', 'Unknown error')}")
        
        return alerts
    
    def monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                stats = self.get_system_stats()
                
                # Check for alerts
                alerts = self.check_alerts(stats)
                if alerts:
                    for alert in alerts:
                        logger.warning(f"ALERT: {alert}")
                
                # Save stats to file
                with open(self.stats_file, 'w') as f:
                    json.dump(stats, f, indent=2)
                
                # Log summary
                logger.info(
                    f"System: CPU {stats['system']['cpu']['utilization_percent']:.1f}% | "
                    f"Memory {stats['system']['memory']['utilization_percent']:.1f}% | "
                    f"Disk {stats['system']['disk']['utilization_percent']:.1f}% | "
                    f"App {stats['application']['status']}"
                )
                
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def start(self):
        """Start monitoring in background thread"""
        monitor_thread = Thread(target=self.monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info("System monitoring started")
    
    def stop(self):
        """Stop monitoring"""
        self.monitoring = False
        logger.info("System monitoring stopped")

def main():
    monitor = SystemMonitor()
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

# Create health check script
RUN cat > $FORESIGHT_HOME/scripts/health_check.py << 'EOF'
#!/usr/bin/env python3
"""
Health check script for Foresight SAR
Used by Docker health checks and load balancers
"""

import sys
import requests
import json
from pathlib import Path

def check_health():
    """Perform comprehensive health check"""
    try:
        # Check main application
        response = requests.get('http://localhost:8080/health', timeout=10)
        if response.status_code != 200:
            print(f"Application health check failed: {response.status_code}")
            return False
        
        # Check if models are loaded
        try:
            models_response = requests.get('http://localhost:8080/api/models/status', timeout=5)
            if models_response.status_code == 200:
                models_data = models_response.json()
                if not models_data.get('models_loaded', False):
                    print("Models not loaded")
                    return False
        except:
            pass  # Models endpoint might not be available
        
        # Check system resources
        stats_file = Path('/var/log/foresight/system_stats.json')
        if stats_file.exists():
            with open(stats_file) as f:
                stats = json.load(f)
            
            # Check critical thresholds
            if stats['system']['memory']['utilization_percent'] > 95:
                print("Critical memory usage")
                return False
            
            if stats['system']['disk']['utilization_percent'] > 95:
                print("Critical disk usage")
                return False
        
        print("Health check passed")
        return True
        
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

if __name__ == '__main__':
    if check_health():
        sys.exit(0)
    else:
        sys.exit(1)
EOF

# Create startup script
RUN cat > $FORESIGHT_HOME/scripts/start_production.sh << 'EOF'
#!/bin/bash
# Production startup script for Foresight SAR

set -e

echo "Starting Foresight SAR in production mode..."

# Wait for dependencies
echo "Waiting for dependencies..."
if [ -n "$DATABASE_URL" ]; then
    echo "Waiting for database..."
    # Add database wait logic here
fi

if [ -n "$REDIS_URL" ]; then
    echo "Waiting for Redis..."
    # Add Redis wait logic here
fi

# Initialize database if needed
echo "Initializing database..."
cd /app
python -c "from src.backend.database import init_db; init_db()" || echo "Database initialization skipped"

# Download models if needed
echo "Checking models..."
if [ ! -f "/app/models/yolo_detection.pt" ]; then
    echo "Downloading detection models..."
    python /app/scripts/download_models.py
fi

# Set proper permissions
chown -R foresight:foresight /var/lib/foresight /var/log/foresight

# Start supervisor
echo "Starting services with supervisor..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/foresight.conf
EOF

# Create model download script
RUN cat > $FORESIGHT_HOME/scripts/download_models.py << 'EOF'
#!/usr/bin/env python3
"""
Model download script for production deployment
Downloads pre-trained models from remote storage
"""

import os
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_file(url, destination):
    """Download file with progress"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded {destination.name}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False

def main():
    """Download required models"""
    models_dir = Path('/app/models')
    
    # Model URLs (replace with actual URLs)
    model_urls = {
        'yolo_detection.pt': os.environ.get('YOLO_MODEL_URL', ''),
        'reid_model.pt': os.environ.get('REID_MODEL_URL', ''),
        'face_embedding.pt': os.environ.get('FACE_MODEL_URL', '')
    }
    
    for model_name, url in model_urls.items():
        if url:
            model_path = models_dir / model_name
            if not model_path.exists():
                logger.info(f"Downloading {model_name}...")
                download_file(url, model_path)
            else:
                logger.info(f"{model_name} already exists")
        else:
            logger.warning(f"No URL provided for {model_name}")

if __name__ == '__main__':
    main()
EOF

# Make scripts executable
RUN chmod +x $FORESIGHT_HOME/scripts/*.sh && \
    chmod +x $FORESIGHT_HOME/scripts/*.py

# Set proper ownership
RUN chown -R foresight:foresight $FORESIGHT_HOME && \
    chown -R foresight:foresight $FORESIGHT_DATA && \
    chown -R foresight:foresight $FORESIGHT_LOGS

# Create log directories
RUN mkdir -p $FORESIGHT_LOGS && \
    touch $FORESIGHT_LOGS/app.log && \
    touch $FORESIGHT_LOGS/worker.log && \
    touch $FORESIGHT_LOGS/monitor.log && \
    chown -R foresight:foresight $FORESIGHT_LOGS

# Expose ports
EXPOSE 8080 9090 9100

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python /app/scripts/health_check.py || exit 1

# Set working directory
WORKDIR $FORESIGHT_HOME

# Switch to application user
USER foresight

# Set entrypoint
ENTRYPOINT ["/app/scripts/start_production.sh"]

# Default command
CMD []

# Build information
ARG BUILD_DATE
ARG VCS_REF
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/foresight-sar/foresight" \
      org.label-schema.schema-version="1.0"