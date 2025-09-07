# Jetson TensorRT Optimization Guide

This guide provides comprehensive instructions for optimizing PyTorch models for deployment on NVIDIA Jetson devices using TensorRT.

## Overview

The Jetson TensorRT Optimizer is a comprehensive pipeline that:
- Automatically detects and exports PyTorch models to ONNX format
- Generates optimized TensorRT engines for Jetson devices
- Provides performance benchmarking and validation
- Creates deployment-ready configurations

## Features

### 🚀 **Automatic Model Detection**
- Supports YOLOv8, ReID, and custom PyTorch models
- Intelligent model type detection based on architecture
- Batch processing for multiple models

### ⚡ **TensorRT Optimization**
- FP32, FP16, and INT8 precision modes
- Dynamic shape support for flexible input sizes
- DLA (Deep Learning Accelerator) utilization
- Configurable workspace memory allocation

### 📊 **Performance Benchmarking**
- Automated inference speed testing
- Memory usage profiling
- Comparison between ONNX and TensorRT performance
- Detailed metrics reporting

### 🎯 **Device-Specific Optimization**
- Jetson Nano, Xavier NX, Orin NX, and Orin AGX support
- Device-specific memory and compute constraints
- Optimal configuration recommendations

## Installation

### Prerequisites

```bash
# On development machine (for ONNX export)
pip install torch torchvision onnx onnxruntime

# On Jetson device (for TensorRT optimization)
sudo apt-get update
sudo apt-get install python3-pip
pip3 install pycuda

# TensorRT is pre-installed on JetPack
# Verify installation:
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

### Dependencies

```python
# Required packages
torch>=1.12.0
torchvision>=0.13.0
onnx>=1.12.0
onnxruntime>=1.12.0  # Optional, for validation
pycuda>=2022.1       # For TensorRT on Jetson
numpy>=1.21.0
Pillow>=8.3.0
```

## Quick Start

### Basic Usage

```python
from tools.jetson_tensorrt_optimizer import (
    JetsonTensorRTOptimizer,
    OptimizationConfig,
    JetsonDevice
)

# Configure optimization
config = OptimizationConfig(
    device=JetsonDevice.ORIN_NX,
    precision='fp16',
    batch_size=1,
    input_shape=(1, 3, 640, 640)
)

# Create optimizer
optimizer = JetsonTensorRTOptimizer(config, './optimized_models')

# Optimize a model
result = optimizer.optimize_model('path/to/model.pt', 'optimized_model')

if result['success']:
    print(f"✅ Optimization completed!")
    print(f"TensorRT engine: {result['files']['tensorrt']}")
else:
    print(f"❌ Optimization failed: {result['error']}")
```

### Batch Optimization

```python
# Optimize all models in a directory
results = optimizer.optimize_directory('./models')

for result in results:
    if result['success']:
        print(f"✅ {result['model_name']} optimized successfully")
    else:
        print(f"❌ {result['model_name']} failed: {result['error']}")
```

## Configuration Options

### Device Selection

```python
from tools.jetson_tensorrt_optimizer import JetsonDevice

# Available devices
JetsonDevice.NANO        # Jetson Nano
JetsonDevice.XAVIER_NX   # Jetson Xavier NX
JetsonDevice.ORIN_NX     # Jetson Orin NX
JetsonDevice.ORIN_AGX    # Jetson Orin AGX
```

### Precision Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `fp32` | Full precision | Maximum accuracy, slower inference |
| `fp16` | Half precision | Balanced accuracy/speed, 2x speedup |
| `int8` | 8-bit quantization | Maximum speed, requires calibration |

### Optimization Configuration

```python
config = OptimizationConfig(
    device=JetsonDevice.ORIN_NX,     # Target device
    precision='fp16',                 # Precision mode
    batch_size=1,                     # Inference batch size
    input_shape=(1, 3, 640, 640),     # Model input shape
    workspace_size=4,                 # TensorRT workspace (GB)
    enable_dla=True,                  # Use Deep Learning Accelerator
    dynamic_shapes=False,             # Enable dynamic input shapes
    validate_output=True,             # Validate ONNX export
    benchmark_iterations=100,         # Benchmark iterations
    calibration_dataset=None          # INT8 calibration data
)
```

## Supported Model Types

### YOLOv8 Models

```python
# YOLOv8 detection models
- yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
- Custom trained YOLOv8 models
- Input shape: (batch, 3, height, width)
- Typical sizes: 640x640, 416x416
```

### ReID Models

```python
# Re-identification models
- Face embedding models
- Person re-identification models
- Feature extraction networks
- Input shape: (batch, 3, height, width)
```

### Custom PyTorch Models

```python
# Requirements for custom models:
- Must be torch.nn.Module
- Support torch.jit.trace() or torch.onnx.export()
- Fixed input shapes (or dynamic shape support)
- Compatible with TensorRT operations
```

## Performance Optimization

### Device-Specific Recommendations

#### Jetson Nano
```python
config = OptimizationConfig(
    device=JetsonDevice.NANO,
    precision='fp16',           # FP16 for better performance
    batch_size=1,               # Single batch only
    workspace_size=512,         # Limited memory (512MB)
    enable_dla=True,            # Use DLA to save GPU memory
    input_shape=(1, 3, 416, 416) # Smaller input for speed
)
```

#### Jetson Xavier NX
```python
config = OptimizationConfig(
    device=JetsonDevice.XAVIER_NX,
    precision='fp16',           # Good balance
    batch_size=2,               # Small batches
    workspace_size=2,           # 2GB workspace
    enable_dla=True,            # DLA available
    input_shape=(2, 3, 640, 640)
)
```

#### Jetson Orin NX/AGX
```python
config = OptimizationConfig(
    device=JetsonDevice.ORIN_AGX,
    precision='fp16',           # Or INT8 for maximum speed
    batch_size=4,               # Larger batches possible
    workspace_size=8,           # More memory available
    enable_dla=False,           # GPU more powerful
    dynamic_shapes=True,        # Advanced features
    input_shape=(4, 3, 640, 640)
)
```

### Memory Optimization

```python
# For memory-constrained environments
config = OptimizationConfig(
    workspace_size=512,         # Minimal workspace
    batch_size=1,               # Single inference
    enable_dla=True,            # Offload to DLA
    precision='int8'            # Maximum quantization
)
```

### Speed Optimization

```python
# For maximum inference speed
config = OptimizationConfig(
    precision='fp16',           # Good speed/accuracy balance
    batch_size=4,               # Batch processing
    workspace_size=8,           # Large workspace
    enable_dla=False,           # Use full GPU power
    dynamic_shapes=False        # Fixed shapes for optimization
)
```

## Deployment Workflow

### 1. Model Preparation

```bash
# Ensure models are properly trained and saved
# Verify model compatibility
python -c "import torch; model = torch.load('model.pt'); print(model)"
```

### 2. Optimization

```python
# Run optimization pipeline
optimizer = JetsonTensorRTOptimizer(config, output_dir)
result = optimizer.optimize_model('model.pt', 'optimized_model')
```

### 3. Validation

```python
# Check optimization results
if result['success']:
    # Review benchmark results
    benchmarks = result['benchmarks']
    print(f"TensorRT FPS: {benchmarks['tensorrt_fps']}")
    print(f"Speedup: {benchmarks['speedup']:.2f}x")
    
    # Validate accuracy (implement your validation logic)
    validate_model_accuracy(result['files']['tensorrt'])
else:
    print(f"Optimization failed: {result['error']}")
```

### 4. Deployment

```bash
# Copy optimized models to Jetson device
scp -r optimized_models/ jetson@192.168.1.100:~/models/

# On Jetson device, test the optimized model
python3 test_inference.py --model models/optimized_model.engine
```

## Troubleshooting

### Common Issues

#### ONNX Export Failures
```python
# Issue: Model contains unsupported operations
# Solution: Simplify model or use torch.jit.trace()

# Check model compatibility
torch.onnx.export(
    model, dummy_input, 'test.onnx',
    verbose=True,  # Enable verbose logging
    do_constant_folding=True
)
```

#### TensorRT Build Failures
```python
# Issue: Unsupported ONNX operations
# Solution: Update TensorRT or modify model

# Check TensorRT version compatibility
import tensorrt as trt
print(f"TensorRT version: {trt.__version__}")
```

#### Memory Issues
```python
# Issue: Out of memory during optimization
# Solution: Reduce workspace size or batch size

config = OptimizationConfig(
    workspace_size=512,  # Reduce from default
    batch_size=1         # Single batch
)
```

#### Performance Issues
```python
# Issue: Poor inference performance
# Solutions:
# 1. Enable DLA for memory-bound models
# 2. Use FP16 instead of FP32
# 3. Optimize input preprocessing
# 4. Use larger batch sizes if memory allows
```

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use validation to catch issues early
config = OptimizationConfig(
    validate_output=True,
    benchmark_iterations=10  # Quick benchmark for debugging
)
```

## Best Practices

### 1. Model Design
- Use standard operations supported by TensorRT
- Avoid dynamic control flow when possible
- Design for fixed input shapes if dynamic shapes aren't needed
- Test model export early in development

### 2. Optimization Strategy
- Start with FP16 precision for good balance
- Use INT8 only when maximum speed is required
- Profile memory usage before deploying
- Validate accuracy after optimization

### 3. Deployment
- Test on actual target hardware
- Monitor inference performance in production
- Keep original models for comparison
- Document optimization settings for reproducibility

### 4. Maintenance
- Regularly update TensorRT for latest optimizations
- Re-optimize models when retraining
- Monitor for performance regressions
- Keep optimization logs for debugging

## Performance Benchmarks

### Expected Speedups

| Model Type | Device | Precision | Typical Speedup |
|------------|--------|-----------|----------------|
| YOLOv8n | Orin NX | FP16 | 2-3x |
| YOLOv8s | Orin AGX | FP16 | 2-4x |
| ReID | Xavier NX | FP16 | 1.5-2x |
| Custom CNN | Nano | FP16 | 1.5-2.5x |

*Note: Actual performance depends on model complexity, input size, and system configuration.*

### Memory Usage

| Device | Available GPU Memory | Recommended Workspace |
|--------|---------------------|----------------------|
| Nano | 128MB | 64-128MB |
| Xavier NX | 8GB | 1-2GB |
| Orin NX | 8GB | 2-4GB |
| Orin AGX | 32GB | 4-8GB |

## API Reference

### JetsonTensorRTOptimizer

```python
class JetsonTensorRTOptimizer:
    def __init__(self, config: OptimizationConfig, output_dir: str)
    def optimize_model(self, model_path: str, output_name: str) -> Dict
    def optimize_directory(self, models_dir: str) -> List[Dict]
    def benchmark_model(self, engine_path: str) -> Dict
```

### OptimizationConfig

```python
@dataclass
class OptimizationConfig:
    device: JetsonDevice
    precision: str = 'fp16'
    batch_size: int = 1
    input_shape: Tuple[int, ...] = (1, 3, 640, 640)
    workspace_size: int = 4
    enable_dla: bool = True
    dynamic_shapes: bool = False
    validate_output: bool = True
    benchmark_iterations: int = 100
    calibration_dataset: Optional[str] = None
```

## Examples

See `examples/jetson_optimization_example.py` for comprehensive usage examples including:
- Single model optimization
- Batch processing
- Custom configurations
- Complete deployment workflow

## Support

For issues and questions:
1. Check this documentation
2. Review example code
3. Enable debug logging
4. Consult NVIDIA TensorRT documentation
5. Check Jetson developer forums

---

*This guide is part of the Foresight AI project. For updates and contributions, see the project repository.*