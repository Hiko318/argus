# Model Optimization Tools

This directory contains tools for optimizing machine learning models for deployment in the Foresight SAR System, particularly for Jetson edge devices.

## Overview

The optimization pipeline converts PyTorch models to optimized formats (ONNX, TensorRT) for efficient inference on resource-constrained devices. The tools support various model types including YOLOv8 detection models and ReID embedding models.

## Tools

### 1. `export_to_onnx.py`

Exports PyTorch models to ONNX format with validation and benchmarking.

**Features:**
- Automatic model type detection (YOLOv8, ReID, generic PyTorch)
- Dynamic shape support
- Input size configuration
- Model validation and benchmarking
- Batch processing capabilities

**Usage:**
```bash
# Export single model
python export_to_onnx.py --model models/yolov8n.pt --output models/yolov8n.onnx

# Export with custom input size
python export_to_onnx.py --model models/reid_model.pt --output models/reid_model.onnx --input-size 256 128

# Batch export
python export_to_onnx.py --input models/ --output onnx_models/ --pattern "*.pt"

# Export with dynamic shapes
python export_to_onnx.py --model models/yolov8s.pt --output models/yolov8s.onnx --dynamic
```

### 2. `convert_to_tensorrt.sh`

Converts ONNX models to TensorRT engines optimized for specific Jetson devices.

**Features:**
- Multiple precision modes (FP32, FP16, INT8)
- Dynamic shape support
- Device-specific optimizations
- DLA (Deep Learning Accelerator) support
- Calibration data support for INT8
- Validation and benchmarking

**Usage:**
```bash
# Basic conversion
./convert_to_tensorrt.sh --precision fp16 models/yolov8n.onnx models/yolov8n_fp16.trt

# Jetson Orin optimization
./convert_to_tensorrt.sh --precision fp16 --device jetson_orin --workspace 4GB models/yolov8s.onnx models/yolov8s_orin.trt

# INT8 with calibration
./convert_to_tensorrt.sh --precision int8 --calibration-data calibration/ models/yolov8n.onnx models/yolov8n_int8.trt

# Dynamic shapes
./convert_to_tensorrt.sh --precision fp16 --min-shapes 1x3x320x320 --opt-shapes 1x3x640x640 --max-shapes 1x3x1280x1280 models/yolov8n.onnx models/yolov8n_dynamic.trt
```

### 3. `optimize_models.py`

Complete optimization pipeline that combines ONNX export and TensorRT conversion with intelligent configuration selection.

**Features:**
- End-to-end optimization pipeline
- Automatic model configuration detection
- Device-specific optimization
- Batch processing
- Performance benchmarking
- Deployment configuration generation
- Multiple precision and batch size variants

**Usage:**
```bash
# Optimize single model
python optimize_models.py --model models/yolov8n.pt --output optimized/ --target jetson_orin

# Optimize all models in directory
python optimize_models.py --input models/ --output optimized/ --target jetson_xavier

# Custom precision and batch size
python optimize_models.py --model models/reid_model.pt --output optimized/ --precision fp16 --batch-size 8

# Skip validation for faster processing
python optimize_models.py --input models/ --output optimized/ --no-validate
```

## Supported Models

### YOLOv8 Detection Models
- **YOLOv8n**: Nano model, optimized for speed
- **YOLOv8s**: Small model, balanced speed/accuracy
- **YOLOv8m**: Medium model, higher accuracy
- **YOLOv8l**: Large model, best accuracy
- **YOLOv8x**: Extra large model, maximum accuracy

### ReID Models
- Person re-identification models
- Feature embedding models
- Custom ReID architectures

### Generic PyTorch Models
- Classification models
- Segmentation models
- Custom architectures

## Target Devices

### Jetson Nano
- **GPU Memory**: 4GB
- **Recommended Precision**: FP16
- **Max Workspace**: 1GB
- **DLA Cores**: 2
- **Optimal Models**: YOLOv8n, lightweight ReID

### Jetson Xavier NX/AGX
- **GPU Memory**: 8-32GB
- **Recommended Precision**: FP16
- **Max Workspace**: 4GB
- **DLA Cores**: 2
- **Optimal Models**: YOLOv8s/m, standard ReID

### Jetson Orin Nano/NX/AGX
- **GPU Memory**: 8-64GB
- **Recommended Precision**: FP16
- **Max Workspace**: 8GB
- **DLA Cores**: 0 (GPU-focused)
- **Optimal Models**: All YOLOv8 variants, complex ReID

## Optimization Strategies

### Precision Selection

1. **FP32**: Full precision, largest models, highest accuracy
2. **FP16**: Half precision, 2x memory reduction, minimal accuracy loss
3. **INT8**: 8-bit quantization, 4x memory reduction, requires calibration

### Batch Size Optimization

- **Batch Size 1**: Real-time inference, lowest latency
- **Batch Size 4-8**: Balanced throughput/latency
- **Batch Size 16+**: Maximum throughput, higher latency

### Memory Management

- Automatic batch size limiting based on device memory
- Workspace size optimization
- Memory pool configuration

## Performance Benchmarking

The tools provide comprehensive benchmarking:

- **Inference Time**: Average and standard deviation
- **Throughput**: Frames per second (FPS)
- **Memory Usage**: GPU memory consumption
- **Model Size**: File size comparison

### Benchmark Results Format

```json
{
  "model_name": "yolov8n",
  "benchmarks": {
    "onnx": {
      "avg_time_ms": 15.2,
      "std_time_ms": 1.1,
      "fps": 65.8
    },
    "tensorrt": {
      "avg_time_ms": 8.7,
      "fps": 114.9
    }
  }
}
```

## Deployment Configuration

The optimization pipeline generates deployment configurations:

```json
{
  "target_device": "jetson_orin",
  "device_specs": {
    "gpu_memory_gb": 64,
    "recommended_precision": "fp16",
    "max_workspace_gb": 8
  },
  "optimized_models": {
    "yolov8n.pt": {
      "variant_name": "yolov8n_fp16_b1",
      "files": {
        "onnx": "optimized/yolov8n/yolov8n_fp16_b1.onnx",
        "tensorrt": "optimized/yolov8n/yolov8n_fp16_b1.trt"
      },
      "precision": "fp16",
      "batch_size": 1,
      "benchmarks": {...}
    }
  }
}
```

## Integration with Foresight

Optimized models integrate seamlessly with the Foresight SAR System:

1. **Detection Pipeline**: Optimized YOLOv8 models for person detection
2. **ReID Pipeline**: Optimized embedding models for person re-identification
3. **Geolocation**: GPU-accelerated coordinate transformations
4. **Real-time Processing**: Low-latency inference for live video streams

## Requirements

### Python Dependencies
```bash
pip install torch torchvision onnx onnxruntime-gpu ultralytics numpy
```

### System Dependencies
```bash
# TensorRT (Jetson devices)
sudo apt install tensorrt

# CUDA (if not using Jetson)
# Follow NVIDIA CUDA installation guide
```

### Jetson Setup
```bash
# Install JetPack (includes TensorRT, CUDA, cuDNN)
sudo apt update
sudo apt install nvidia-jetpack

# Verify installation
trtexec --help
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use smaller models
   - Enable memory optimization

2. **TensorRT Build Failures**
   - Check ONNX model validity
   - Verify TensorRT version compatibility
   - Reduce workspace size

3. **Performance Issues**
   - Enable GPU acceleration
   - Use appropriate precision
   - Optimize batch size

### Debug Mode

Enable verbose logging for detailed information:

```bash
python optimize_models.py --model models/yolov8n.pt --output optimized/ --verbose
```

### Validation

Always validate optimized models:

```bash
# Test ONNX model
python -c "import onnx; onnx.checker.check_model('model.onnx')"

# Test TensorRT engine
trtexec --loadEngine=model.trt --iterations=10
```

## Best Practices

1. **Model Selection**
   - Choose appropriate model size for target device
   - Consider accuracy vs. speed trade-offs
   - Test multiple variants

2. **Optimization Strategy**
   - Start with FP16 precision
   - Use INT8 only if memory is critical
   - Optimize batch size for use case

3. **Validation**
   - Always validate optimized models
   - Compare accuracy with original models
   - Benchmark on target hardware

4. **Deployment**
   - Use deployment configurations
   - Monitor performance in production
   - Update models as needed

## Contributing

When adding new optimization tools:

1. Follow existing code structure
2. Add comprehensive error handling
3. Include validation and benchmarking
4. Update documentation
5. Test on multiple Jetson devices

## License

These tools are part of the Foresight SAR System and follow the same licensing terms.

## Support

For issues with model optimization:

1. Check the troubleshooting section
2. Enable verbose logging
3. Verify system requirements
4. Test with known working models
5. Report issues with full logs and system information