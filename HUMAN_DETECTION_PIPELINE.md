# Human Detection Pipeline

Complete human detection pipeline with YOLOv8 integration, real-time inference, and multi-object tracking optimized for edge devices.

## Overview

This pipeline provides a comprehensive solution for human detection in aerial and SAR imagery with the following features:

- **YOLOv8 Integration**: Fine-tuned model for aerial/SAR datasets
- **Real-time Inference**: Optimized for 25+ FPS performance
- **Multi-Object Tracking**: SORT/DeepSORT algorithms with ID assignment
- **Edge Optimization**: TensorRT/ONNX support for Jetson and mobile devices
- **Performance Monitoring**: Real-time FPS tracking and optimization
- **Video Processing**: Support for video files and live camera feeds

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Video Input   │───▶│   YOLO Detector  │───▶│   SORT Tracker │
│  (Camera/File)  │    │  (YOLOv8 Model)  │    │ (ID Assignment) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │ Edge Optimization│    │   Visualization │
                       │ (TensorRT/ONNX)  │    │  (Annotations)  │
                       └──────────────────┘    └─────────────────┘
```

## Components

### 1. Detection Pipeline (`detection_pipeline.py`)
Core pipeline that integrates YOLO detection with tracking:
- `DetectionPipeline`: Main pipeline class
- `HumanDetection`: Detection result dataclass
- `DetectionFrame`: Frame-level results
- Real-time optimization methods

### 2. YOLO Detector (`detector.py`)
YOLOv8 model wrapper with optimization:
- Model loading and inference
- Device management (CPU/GPU)
- TensorRT support detection
- Performance monitoring

### 3. Multi-Object Tracker (`tracker.py`)
SORT/DeepSORT implementation:
- `Track`: Individual track representation
- `SORTTracker`: Basic SORT algorithm
- `DeepSORTTracker`: Enhanced with appearance features
- Kalman filter motion prediction
- Hungarian algorithm for data association

### 4. Edge Optimizer (`edge_optimizer.py`)
Model optimization for edge devices:
- TensorRT optimization
- ONNX conversion
- Dynamic quantization
- Mobile optimization
- Performance benchmarking

## Installation

### Prerequisites
```bash
# Python 3.8+
pip install ultralytics opencv-python numpy scipy filterpy

# For TensorRT (NVIDIA devices)
pip install tensorrt

# For ONNX optimization
pip install onnx onnxruntime
```

### Edge Device Setup
For Jetson devices, see `EDGE_DEVICE_SETUP.md` for detailed installation instructions.

## Usage

### 1. Basic Human Detection Demo
```bash
# Run with default YOLOv8 model
python -m src.backend.human_detection_demo --benchmark --verbose

# Use trained aerial model
python -m src.backend.human_detection_demo --model "trained_models/aerial_yolov8n_20250904_130108/yolov8n_aerial_trained.pt" --benchmark

# Enable TensorRT optimization
python -m src.backend.human_detection_demo --tensorrt --benchmark
```

### 2. Optimized Pipeline
```bash
# Run performance benchmark
python -m src.backend.final_human_detection_pipeline --benchmark --benchmark-frames 100

# Process video file
python -m src.backend.final_human_detection_pipeline --video "test_video.mp4" --output "output.mp4" --display

# Live camera feed
python -m src.backend.final_human_detection_pipeline --camera 0 --display

# With optimizations
python -m src.backend.final_human_detection_pipeline --tensorrt --aerial --target-fps 30 --benchmark
```

### 3. Model Optimization
```bash
# Optimize model for edge deployment
python -m src.backend.optimize_detection_pipeline --model "yolov8n.pt" --output-dir "optimized_models" --target-fps 30

# Benchmark different optimizations
python -m src.backend.optimize_detection_pipeline --model "trained_models/aerial_yolov8n_20250904_130108/yolov8n_aerial_trained.pt" --benchmark-frames 50
```

### 4. Training Custom Models
```bash
# Train on aerial/SAR datasets
python -m src.backend.train_aerial_model --epochs 50 --batch-size 16 --optimize-edge
```

## Performance Results

### Benchmark Results (Windows CPU)
- **Original YOLOv8n**: ~7.5 FPS
- **Optimized ONNX**: ~6.7 FPS
- **Quantized Model**: ~7.6 FPS
- **Target Performance**: 25-30 FPS (requires GPU/edge optimization)

### Optimization Recommendations
1. **GPU Acceleration**: Use CUDA-enabled devices
2. **TensorRT**: Enable for NVIDIA GPUs/Jetson
3. **Model Size**: Use YOLOv8n (nano) for speed
4. **Input Resolution**: Reduce to 416x416 for faster inference
5. **Confidence Threshold**: Increase to 0.5+ for fewer detections
6. **Tracking Parameters**: Reduce max_disappeared and max_distance

## Configuration

### Pipeline Parameters
```python
pipeline = OptimizedHumanDetectionPipeline(
    model_path="yolov8n.pt",           # Model path
    device="cuda",                     # Device (cuda/cpu)
    confidence_threshold=0.5,          # Detection confidence
    tracker_type="sort",               # Tracker type
    enable_tensorrt=True,              # TensorRT optimization
    aerial_optimized=True,             # Aerial mode
    target_fps=25.0                    # Target FPS
)
```

### Optimization Config
```python
config = OptimizationConfig(
    target_device="cuda",              # Target device
    enable_tensorrt=True,              # Enable TensorRT
    enable_onnx=True,                  # Enable ONNX
    enable_quantization=True,          # Enable quantization
    tensorrt_precision="fp16",         # Precision mode
    input_shape=(1, 3, 416, 416),     # Input dimensions
    target_fps=30.0                    # Performance target
)
```

## API Reference

### DetectionPipeline
```python
class DetectionPipeline:
    def __init__(self, model_path, device=None, confidence_threshold=0.25, ...)
    def process_frame(self, frame) -> DetectionFrame
    def draw_annotations(self, frame, detection_result) -> np.ndarray
    def get_performance_stats(self) -> Dict[str, Any]
    def optimize_for_realtime(self, target_fps=30.0)
```

### OptimizedHumanDetectionPipeline
```python
class OptimizedHumanDetectionPipeline:
    def __init__(self, model_path, device=None, confidence_threshold=0.5, ...)
    def process_frame(self, frame) -> Dict[str, Any]
    def process_video(self, video_path, output_path=None, ...) -> Dict[str, Any]
    def benchmark(self, num_frames=100) -> Dict[str, Any]
    def get_performance_stats(self) -> Dict[str, Any]
```

### EdgeOptimizer
```python
class EdgeOptimizer:
    def __init__(self, config: OptimizationConfig)
    def optimize_model(self, model_path: str) -> Dict[str, str]
    def benchmark_model(self, model_path: str) -> BenchmarkResult
```

## File Structure

```
src/backend/
├── detection_pipeline.py          # Main detection pipeline
├── detector.py                     # YOLO detector wrapper
├── tracker.py                      # SORT/DeepSORT tracker
├── edge_optimizer.py               # Edge optimization
├── human_detection_demo.py         # Basic demo script
├── final_human_detection_pipeline.py  # Optimized pipeline
├── optimize_detection_pipeline.py  # Optimization script
└── train_aerial_model.py          # Model training

models/
├── yolov8n.pt                     # Base YOLOv8 nano model
├── yolov8s.pt                     # YOLOv8 small model
└── yolov8m.pt                     # YOLOv8 medium model

trained_models/
└── aerial_yolov8n_*/              # Trained aerial models
    ├── yolov8n_aerial_trained.pt  # Trained PyTorch model
    ├── yolov8n_aerial_trained_onnx.onnx  # ONNX optimized
    └── yolov8n_aerial_trained_quantized.pt  # Quantized model

optimized_models/
├── optimization_benchmark.json    # Benchmark results
└── *.pt, *.onnx                  # Optimized models
```

## Troubleshooting

### Common Issues

1. **Low FPS Performance**
   - Enable GPU acceleration
   - Reduce input resolution
   - Increase confidence threshold
   - Use lighter model (YOLOv8n)

2. **Memory Issues**
   - Reduce batch size
   - Use quantized models
   - Lower input resolution

3. **TensorRT Errors**
   - Check CUDA compatibility
   - Verify TensorRT installation
   - Use compatible model format

4. **Tracking Issues**
   - Adjust tracker parameters
   - Increase IoU threshold
   - Reduce max_disappeared frames

### Performance Optimization Tips

1. **Hardware**
   - Use NVIDIA GPU with CUDA
   - Consider Jetson devices for edge deployment
   - Ensure sufficient RAM (8GB+)

2. **Software**
   - Enable TensorRT optimization
   - Use ONNX runtime
   - Apply dynamic quantization

3. **Model**
   - Use YOLOv8n for speed
   - Train on specific datasets
   - Optimize input resolution

4. **Pipeline**
   - Adjust confidence thresholds
   - Optimize tracker parameters
   - Enable real-time optimizations

## Future Enhancements

- [ ] Multi-camera support
- [ ] Real-time streaming integration
- [ ] Advanced tracking algorithms (ByteTrack, FairMOT)
- [ ] Mobile deployment (Android/iOS)
- [ ] Cloud inference integration
- [ ] Advanced analytics and reporting
- [ ] Integration with drone control systems

## License

This project is part of the Foresight AI surveillance system. See main project license for details.

## Support

For technical support and questions:
- Check the troubleshooting section
- Review performance optimization tips
- Consult the API reference
- See `EDGE_DEVICE_SETUP.md` for edge deployment