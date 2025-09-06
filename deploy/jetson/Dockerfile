# Foresight SAR System - Jetson Deployment
# Multi-stage build for NVIDIA Jetson with CUDA and TensorRT support

# Stage 1: Base Jetson image with CUDA/TensorRT
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV TRT_LOGGER_LEVEL=WARNING

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Python dependencies
FROM base as python-deps

# Upgrade pip and install wheel
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch and torchvision (pre-built for Jetson)
RUN python3 -m pip install --no-cache-dir \
    numpy==1.24.3 \
    Pillow==10.0.0 \
    scipy==1.10.1

# Install OpenCV with CUDA support (pre-built for Jetson)
RUN python3 -m pip install --no-cache-dir opencv-python==4.8.0.76

# Install YOLO and tracking dependencies
RUN python3 -m pip install --no-cache-dir \
    ultralytics==8.0.196 \
    filterpy==1.4.5 \
    scikit-learn==1.3.0 \
    lap==0.4.0

# Install communication and utility dependencies
RUN python3 -m pip install --no-cache-dir \
    fastapi==0.103.1 \
    uvicorn==0.23.2 \
    websockets==11.0.3 \
    pydantic==2.3.0 \
    requests==2.31.0 \
    aiofiles==23.2.1

# Install geospatial dependencies
RUN python3 -m pip install --no-cache-dir \
    pyproj==3.6.0 \
    geopy==2.3.0

# Install cryptography for evidence packaging
RUN python3 -m pip install --no-cache-dir \
    cryptography==41.0.4

# Stage 3: TensorRT optimization
FROM python-deps as tensorrt-stage

# Install TensorRT Python bindings (if not already included)
RUN python3 -m pip install --no-cache-dir pycuda==2022.2.2

# Create TensorRT optimization script
RUN mkdir -p /opt/tensorrt
COPY tensorrt_optimizer.py /opt/tensorrt/

# Stage 4: Application setup
FROM tensorrt-stage as app

# Create application directory
WORKDIR /app

# Create non-root user for security
RUN useradd -m -u 1000 foresight && \
    chown -R foresight:foresight /app

# Copy application code
COPY --chown=foresight:foresight . /app/

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/logs /app/cache && \
    chown -R foresight:foresight /app

# Install application dependencies
RUN python3 -m pip install --no-cache-dir -e .

# Copy and set up entrypoint script
COPY --chown=foresight:foresight deploy/jetson/entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# Switch to non-root user
USER foresight

# Expose ports
EXPOSE 8000 8001 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["production"]

# Labels for metadata
LABEL maintainer="Foresight SAR Team"
LABEL version="1.0.0"
LABEL description="Foresight SAR System for NVIDIA Jetson"
LABEL gpu.required="true"
LABEL cuda.version="11.4"
LABEL tensorrt.version="8.5"