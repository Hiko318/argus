# Docker Setup for Foresight SAR System

This directory contains Docker configurations for different deployment scenarios.

## Available Configurations

### Development (`Dockerfile.dev`)
- Full development environment with all dependencies
- Includes Electron app dependencies
- Suitable for local development and testing
- CPU-based inference by default

### Jetson (`Dockerfile.jetson`)
- Optimized for NVIDIA Jetson devices
- Based on L4T (Linux for Tegra) with JetPack
- GPU acceleration and TensorRT optimization enabled
- Minimal web interface (no Electron)

## Quick Start

### Development Environment

```bash
# Build and run development container
docker-compose up foresight-dev

# Access the application
# Web interface: http://localhost:8004
# API docs: http://localhost:8004/docs
```

### GPU-Accelerated Development

```bash
# Requires NVIDIA Docker runtime
docker-compose --profile gpu up foresight-gpu
```

### Production with Load Balancer

```bash
# Full production stack with Nginx
docker-compose --profile production up
```

### With Redis Cache

```bash
# Include Redis for caching
docker-compose --profile cache up foresight-dev redis
```

## Building Individual Images

### Development Image
```bash
docker build -f docker/Dockerfile.dev -t foresight:dev .
```

### Jetson Image
```bash
# Build on Jetson device or with ARM64 emulation
docker build -f docker/Dockerfile.jetson -t foresight:jetson .
```

## Running Containers

### Basic Development
```bash
docker run -p 8004:8004 -v $(pwd)/data:/app/data foresight:dev
```

### Jetson Deployment
```bash
docker run --runtime nvidia -p 8004:8004 \
  -v /path/to/data:/app/data \
  -e ENABLE_TENSORRT=true \
  foresight:jetson
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SAR_SERVICE_PORT` | 8004 | API service port |
| `ENABLE_GPU` | false | Enable GPU acceleration |
| `ENABLE_TENSORRT` | false | Enable TensorRT optimization |
| `DETECTION_CONFIDENCE_THRESHOLD` | 0.5 | Detection confidence threshold |
| `GEOLOCATION_PRECISION` | high | Geolocation precision level |
| `JETSON_OPTIMIZATION` | false | Enable Jetson-specific optimizations |

## Volume Mounts

- `/app/data` - Training data and samples
- `/app/models` - Model files and cache
- `/app/configs` - Configuration files
- `/app/src` - Source code (development only)

## Networking

- Port 8004: Main API service
- Port 3000: Development web server
- Port 80/443: Production Nginx (with profile)
- Port 6379: Redis cache (with profile)

## Profiles

- `gpu` - GPU-accelerated containers
- `cache` - Include Redis caching
- `production` - Production deployment with Nginx

## Health Checks

Containers include health checks that verify:
- API service responsiveness
- Model loading status
- GPU availability (if enabled)

## Troubleshooting

### GPU Issues
```bash
# Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Check GPU access in container
docker exec -it <container> nvidia-smi
```

### Permission Issues
```bash
# Fix volume permissions
sudo chown -R $(id -u):$(id -g) data/ models/
```

### Memory Issues
```bash
# Increase Docker memory limit
# Docker Desktop: Settings > Resources > Memory

# Monitor container memory usage
docker stats
```

## Security Considerations

- Containers run as non-root user in production
- Secrets should be passed via environment variables or Docker secrets
- Network access is restricted to necessary ports
- Regular security updates for base images