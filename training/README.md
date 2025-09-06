# YOLOv8 Training Pipeline for SAR Operations

This module provides a complete training pipeline for YOLOv8 models optimized for Search and Rescue (SAR) operations. It includes training scripts, evaluation tools, dataset converters, and configuration management.

## Features

- **YOLOv8 Training**: Optimized for SAR scenarios with custom configurations
- **Dataset Conversion**: Support for COCO, Pascal VOC, CSV, and YOLOv5 formats
- **Model Evaluation**: Comprehensive evaluation with SAR-specific metrics
- **Cross-Platform**: Both Bash and PowerShell scripts for Linux/macOS and Windows
- **Performance Optimization**: TensorRT export and benchmarking tools
- **Automated Pipeline**: End-to-end training automation with validation

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install ultralytics torch torchvision opencv-python pyyaml

# For GPU training (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Basic Training

**Linux/macOS:**
```bash
# Make script executable
chmod +x train.sh

# Basic training
./train.sh

# Custom configuration
./train.sh --config custom_config.yaml --epochs 200 --batch-size 32
```

**Windows:**
```powershell
# Basic training
.\train.ps1

# Custom configuration
.\train.ps1 -Config custom_config.yaml -Epochs 200 -BatchSize 32
```

## File Structure

```
training/
├── train_config.yaml      # Main training configuration
├── train.py              # Python training script
├── train.sh              # Bash training automation
├── train.ps1             # PowerShell training automation
├── evaluate.py           # Model evaluation script
├── dataset_converter.py  # Dataset format converter
└── README.md             # This file
```

## Configuration

### Training Configuration (`train_config.yaml`)

The main configuration file contains:

- **Model Settings**: Architecture, input size, class definitions
- **Dataset Paths**: Training, validation, and test data locations
- **Hyperparameters**: Learning rate, batch size, epochs, optimizers
- **SAR Optimizations**: Weather robustness, low-light enhancement
- **Export Settings**: ONNX, TensorRT, and mobile formats

### Key Configuration Sections

```yaml
# Model configuration
model: yolov8n.pt
input_size: [640, 640]
num_classes: 4

# SAR-specific classes
class_names:
  0: person
  1: vehicle
  2: debris
  3: structure

# Training hyperparameters
epochs: 100
batch_size: 16
learning_rate: 0.01
optimizer: AdamW

# SAR optimizations
weather_augmentation: true
low_light_enhancement: true
thermal_simulation: true
```

## Dataset Preparation

### Supported Formats

1. **COCO Format**: JSON annotations with image references
2. **Pascal VOC**: XML annotations with bounding boxes
3. **CSV Format**: Simple CSV with image paths and coordinates
4. **YOLOv5**: Text files with normalized coordinates

### Dataset Conversion

```python
# Convert COCO to YOLOv8
python dataset_converter.py \
    --input-format coco \
    --input-path /path/to/coco/dataset \
    --output-path /path/to/yolov8/dataset \
    --train-split 0.8 \
    --val-split 0.15 \
    --test-split 0.05

# Convert Pascal VOC to YOLOv8
python dataset_converter.py \
    --input-format voc \
    --input-path /path/to/voc/dataset \
    --output-path /path/to/yolov8/dataset
```

### Expected Dataset Structure

```
datasets/sar_dataset/
├── dataset.yaml          # Dataset configuration
├── images/
│   ├── train/           # Training images
│   ├── val/             # Validation images
│   └── test/            # Test images
└── labels/
    ├── train/           # Training labels (.txt)
    ├── val/             # Validation labels (.txt)
    └── test/            # Test labels (.txt)
```

## Training Scripts

### Python Script (`train.py`)

Direct Python training with full control:

```python
# Basic training
python train.py --config train_config.yaml

# Resume training
python train.py --config train_config.yaml --resume runs/train/sar_yolov8/weights/last.pt

# Custom parameters
python train.py \
    --config train_config.yaml \
    --epochs 200 \
    --batch-size 32 \
    --device 0 \
    --export-onnx \
    --export-tensorrt
```

### Automation Scripts

**Bash Script (`train.sh`)**

```bash
# Full pipeline with validation and benchmarking
./train.sh --validate --export --benchmark

# High-performance training
./train.sh --model l --batch-size 64 --epochs 300 --device 0

# Resume training
./train.sh --resume runs/train/sar_yolov8/weights/last.pt
```

**PowerShell Script (`train.ps1`)**

```powershell
# Full pipeline with validation and benchmarking
.\train.ps1 -Validate -Export -Benchmark

# High-performance training
.\train.ps1 -Model l -BatchSize 64 -Epochs 300 -Device 0

# Resume training
.\train.ps1 -Resume runs/train/sar_yolov8/weights/last.pt
```

## Model Evaluation

### Evaluation Script (`evaluate.py`)

```python
# Evaluate trained model
python evaluate.py \
    --model runs/train/sar_yolov8/weights/best.pt \
    --data datasets/sar_dataset \
    --export-results

# Benchmark performance
python evaluate.py \
    --model runs/train/sar_yolov8/weights/best.pt \
    --data datasets/sar_dataset \
    --benchmark \
    --export-results

# Compare multiple models
python evaluate.py \
    --models model1.pt model2.pt model3.pt \
    --data datasets/sar_dataset \
    --compare
```

### Evaluation Metrics

- **Detection Metrics**: mAP@0.5, mAP@0.5:0.95, Precision, Recall
- **Performance Metrics**: FPS, Inference time, Memory usage
- **SAR-Specific**: Weather robustness, Low-light performance
- **Class-wise Analysis**: Per-class precision and recall

## Advanced Features

### Multi-GPU Training

```bash
# Use multiple GPUs
./train.sh --device 0,1,2,3 --batch-size 64
```

### Mixed Precision Training

```yaml
# In train_config.yaml
amp: true  # Automatic Mixed Precision
```

### Custom Augmentations

```yaml
# SAR-specific augmentations
augmentation:
  weather_effects: true
  thermal_noise: true
  low_light_simulation: true
  fog_simulation: true
  rain_simulation: true
```

### Model Export

```python
# Export to multiple formats
python train.py \
    --config train_config.yaml \
    --export-onnx \
    --export-tensorrt \
    --export-coreml \
    --export-tflite
```

## Performance Optimization

### TensorRT Optimization

```bash
# Export to TensorRT for Jetson deployment
python train.py --config train_config.yaml --export-tensorrt

# Optimize for specific hardware
python train.py \
    --config train_config.yaml \
    --export-tensorrt \
    --tensorrt-workspace 4 \
    --tensorrt-precision fp16
```

### Model Pruning

```python
# Prune model for edge deployment
python train.py \
    --config train_config.yaml \
    --prune 0.3 \
    --export-onnx
```

## Monitoring and Logging

### TensorBoard Integration

```bash
# Start TensorBoard
tensorboard --logdir runs/train

# View training progress at http://localhost:6006
```

### Weights & Biases Integration

```yaml
# In train_config.yaml
wandb:
  enabled: true
  project: "sar-yolov8"
  name: "sar-training-run"
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   ./train.sh --batch-size 8
   
   # Use gradient accumulation
   ./train.sh --batch-size 8 --accumulate 4
   ```

2. **Dataset Loading Errors**
   ```bash
   # Validate dataset structure
   python dataset_converter.py --validate /path/to/dataset
   ```

3. **Slow Training**
   ```bash
   # Enable mixed precision
   ./train.sh --amp
   
   # Use multiple workers
   ./train.sh --workers 8
   ```

### Performance Tips

- Use SSD storage for datasets
- Enable mixed precision training (AMP)
- Use appropriate batch size for your GPU memory
- Consider gradient accumulation for large effective batch sizes
- Use multiple data loading workers

## Integration with Foresight

The trained models integrate seamlessly with the Foresight SAR system:

```python
# Load trained model in Foresight
from vision.yolo_infer import YOLOInference

# Initialize with trained model
detector = YOLOInference(
    model_path="runs/train/sar_yolov8/weights/best.pt",
    device="cuda:0"
)

# Use in detection pipeline
detections = detector.detect(frame)
```

## Contributing

When contributing to the training pipeline:

1. Test with multiple model sizes (n, s, m, l, x)
2. Validate on different hardware configurations
3. Ensure cross-platform compatibility
4. Add appropriate logging and error handling
5. Update documentation for new features

## License

This training pipeline is part of the Foresight SAR system and follows the same licensing terms.