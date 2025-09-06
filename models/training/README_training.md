# Foresight SAR Model Training Guide

This directory contains the complete training pipeline for Foresight SAR models, including object detection (YOLOv8), person re-identification, and face embedding models.

## Overview

The training pipeline supports:

- **Object Detection**: YOLOv8 models for aerial people detection
- **Person Re-ID**: Person re-identification across multiple camera views
- **Face Embedding**: Face recognition and verification models
- **Dataset Preparation**: Conversion from various formats to YOLO format
- **Data Augmentation**: Specialized augmentations for aerial imagery

## Directory Structure

```
models/training/
├── train.sh                    # Main training wrapper script
├── dataset_prep.py             # Dataset preparation and conversion
├── augment_data.py             # Data augmentation utilities
├── README_training.md          # This file
├── reid/                       # Person re-ID and face embedding
│   ├── train_reid.py          # Person re-ID training
│   ├── train_face_embedding.py # Face embedding training
│   └── utils.py               # Utility functions
└── configs/                    # Training configurations
    ├── yolo_aerial.yaml
    ├── reid_config.yaml
    └── face_config.yaml
```

## Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install ultralytics torch torchvision opencv-python pillow numpy matplotlib
pip install albumentations scikit-learn tqdm pyyaml

# For advanced augmentations
pip install imgaug
```

### 2. Dataset Preparation

#### Convert COCO Dataset
```bash
python dataset_prep.py \
    --input /path/to/coco/dataset \
    --output /path/to/yolo/dataset \
    --format coco \
    --classes person:0 \
    --augment --aug-factor 3
```

#### Convert Pascal VOC Dataset
```bash
python dataset_prep.py \
    --input /path/to/pascal/dataset \
    --output /path/to/yolo/dataset \
    --format pascal \
    --classes person:0 vehicle:1
```

#### Prepare Custom Dataset
```bash
# Your dataset structure should be:
data/
└── your_dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── dataset.yaml
```

### 3. Object Detection Training

#### Basic Training
```bash
./train.sh --dataset aerial_people --model yolov8n --epochs 100
```

#### Advanced Training
```bash
./train.sh \
    --dataset my_aerial_data \
    --model yolov8m \
    --epochs 200 \
    --batch-size 32 \
    --image-size 1024 \
    --device 0 \
    --experiment aerial_detection_v2
```

#### Resume Training
```bash
./train.sh --resume runs/train/exp1/weights/last.pt
```

### 4. Person Re-Identification Training

#### Dataset Structure for Re-ID
```
data/reid_dataset/
├── train/
│   ├── person_001/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   ├── person_002/
│   └── ...
├── val/
└── test/
```

#### Train Re-ID Model
```bash
python reid/train_reid.py \
    --data-dir /path/to/reid/dataset \
    --output-dir /path/to/output \
    --backbone resnet50 \
    --batch-size 32 \
    --epochs 120
```

### 5. Face Embedding Training

#### Dataset Structure for Face Recognition
```
data/face_dataset/
├── train/
│   ├── identity_001/
│   │   ├── face_001.jpg
│   │   ├── face_002.jpg
│   │   └── ...
│   ├── identity_002/
│   └── ...
├── val/
└── test/
```

#### Train Face Embedding Model
```bash
python reid/train_face_embedding.py \
    --data-dir /path/to/face/dataset \
    --output-dir /path/to/output \
    --backbone resnet50 \
    --batch-size 64 \
    --epochs 100 \
    --margin 0.5 \
    --scale 64.0
```

## Configuration Files

### YOLOv8 Configuration (configs/yolo_aerial.yaml)
```yaml
model: yolov8n
epochs: 100
batch_size: 16
image_size: 640
device: auto
augment: true
validation_split: 0.2
test_split: 0.1
seed: 42
```

### Re-ID Configuration (configs/reid_config.yaml)
```yaml
model_type: person_reid
backbone: resnet50
embedding_dim: 512
batch_size: 32
learning_rate: 0.0003
epochs: 120
margin: 0.3
device: auto
seed: 42
```

### Face Embedding Configuration (configs/face_config.yaml)
```yaml
backbone: resnet50
embedding_dim: 512
batch_size: 64
learning_rate: 0.1
epochs: 100
margin: 0.5
scale: 64.0
input_size: [112, 112]
device: auto
seed: 42
```

## Data Augmentation

The training pipeline includes specialized augmentations for aerial imagery:

### Object Detection Augmentations
- Random rotation (±15°)
- Horizontal/vertical flips
- Brightness/contrast adjustments
- Gaussian noise and blur
- Random fog and sun flare effects
- Random scaling and cropping

### Re-ID Augmentations
- Random horizontal flip
- Random rotation (±10°)
- Color jitter
- Random erasing
- Normalization

### Face Augmentations
- Random horizontal flip
- Random rotation (±15°)
- Color jitter
- Random grayscale
- Normalization to [-1, 1]

## Model Architectures

### Supported Backbones

#### Object Detection
- YOLOv8n (nano) - 3.2M parameters
- YOLOv8s (small) - 11.2M parameters
- YOLOv8m (medium) - 25.9M parameters
- YOLOv8l (large) - 43.7M parameters
- YOLOv8x (extra large) - 68.2M parameters

#### Re-ID and Face Embedding
- ResNet50 - 25.6M parameters
- ResNet101 - 44.5M parameters
- MobileNetV2 - 3.5M parameters
- EfficientNet-B0 - 5.3M parameters

## Training Strategies

### Learning Rate Scheduling
- Warmup: Linear increase for first 10 epochs
- Step decay: Reduce by factor of 0.1 every 40 epochs
- Cosine annealing: Smooth decay to minimum LR

### Loss Functions

#### Object Detection
- YOLOv8 loss (combination of classification, objectness, and box regression)

#### Person Re-ID
- Cross-entropy loss for classification
- Triplet loss for metric learning
- Center loss for feature clustering

#### Face Embedding
- ArcFace loss for improved face recognition
- Focal loss for handling class imbalance

## Evaluation Metrics

### Object Detection
- mAP@0.5 (mean Average Precision at IoU 0.5)
- mAP@0.5:0.95 (mean Average Precision at IoU 0.5 to 0.95)
- Precision, Recall, F1-score

### Re-ID
- Rank-1, Rank-5, Rank-10 accuracy
- mean Average Precision (mAP)
- Cumulative Matching Characteristics (CMC)

### Face Embedding
- Verification accuracy
- Equal Error Rate (EER)
- Area Under Curve (AUC)

## Hardware Requirements

### Minimum Requirements
- GPU: 8GB VRAM (RTX 3070 or equivalent)
- RAM: 16GB system memory
- Storage: 100GB free space

### Recommended Requirements
- GPU: 24GB VRAM (RTX 4090 or A6000)
- RAM: 32GB system memory
- Storage: 500GB SSD

### Jetson Deployment
- Jetson AGX Orin: Full training capability
- Jetson Orin NX: Limited batch sizes
- Jetson Nano: Inference only

## Troubleshooting

### Common Issues

#### Out of Memory (OOM)
```bash
# Reduce batch size
./train.sh --batch-size 8

# Use gradient accumulation
./train.sh --batch-size 8 --accumulate 4

# Use mixed precision
./train.sh --amp
```

#### Slow Training
```bash
# Increase number of workers
./train.sh --workers 8

# Use faster data loading
./train.sh --cache ram

# Enable multi-GPU training
./train.sh --device 0,1,2,3
```

#### Poor Convergence
```bash
# Adjust learning rate
./train.sh --lr 0.001

# Use warmup
./train.sh --warmup-epochs 10

# Increase epochs
./train.sh --epochs 200
```

### Performance Optimization

#### Data Loading
- Use SSD storage for datasets
- Increase `num_workers` based on CPU cores
- Enable `pin_memory` for GPU training
- Use dataset caching for small datasets

#### Memory Management
- Use gradient checkpointing for large models
- Enable mixed precision training (AMP)
- Clear cache between epochs
- Use data parallel training for multiple GPUs

## Model Export and Deployment

### Export Trained Models

#### YOLOv8 Export
```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/train/exp1/weights/best.pt')

# Export to different formats
model.export(format='onnx')        # ONNX
model.export(format='tensorrt')    # TensorRT
model.export(format='coreml')      # CoreML
model.export(format='tflite')      # TensorFlow Lite
```

#### Re-ID Model Export
```python
import torch
from reid.train_reid import PersonReIDModel

# Load checkpoint
checkpoint = torch.load('best_model.pth')
model = PersonReIDModel(**checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])

# Export to ONNX
torch.onnx.export(
    model,
    torch.randn(1, 3, 256, 128),
    'person_reid.onnx',
    opset_version=11
)
```

### Integration with Foresight SAR

#### Model Configuration
```yaml
# config/models.yaml
models:
  detection:
    path: models/detection/best.pt
    type: yolov8
    confidence: 0.5
    iou_threshold: 0.45
  
  reid:
    path: models/reid/best_model.pth
    type: person_reid
    embedding_dim: 512
  
  face:
    path: models/face/face_embedding_model.pth
    type: face_embedding
    embedding_dim: 512
```

#### Usage in Pipeline
```python
from src.backend.models import ModelManager

# Initialize model manager
model_manager = ModelManager('config/models.yaml')

# Load models
detection_model = model_manager.load_detection_model()
reid_model = model_manager.load_reid_model()
face_model = model_manager.load_face_model()

# Run inference
detections = detection_model(image)
embeddings = reid_model(person_crops)
face_embeddings = face_model(face_crops)
```

## Best Practices

### Dataset Quality
- Ensure balanced class distribution
- Include diverse lighting conditions
- Add various weather conditions
- Include different altitudes and angles
- Validate annotation quality

### Training Process
- Start with pretrained models
- Use progressive resizing
- Monitor validation metrics
- Save checkpoints regularly
- Use early stopping to prevent overfitting

### Model Selection
- Balance accuracy vs. speed requirements
- Consider deployment constraints
- Test on representative data
- Validate on edge cases
- Benchmark inference speed

## Advanced Features

### Multi-GPU Training
```bash
# Data parallel training
./train.sh --device 0,1,2,3

# Distributed training
torchrun --nproc_per_node=4 train.py
```

### Mixed Precision Training
```bash
# Enable automatic mixed precision
./train.sh --amp
```

### Knowledge Distillation
```bash
# Train student model with teacher guidance
./train.sh --teacher-model teacher.pt --distillation-alpha 0.7
```

### Hyperparameter Optimization
```bash
# Use Optuna for hyperparameter search
python optimize_hyperparams.py --trials 100
```

## Monitoring and Logging

### TensorBoard Integration
```bash
# Start TensorBoard
tensorboard --logdir runs/

# View in browser
# http://localhost:6006
```

### Weights & Biases Integration
```bash
# Install wandb
pip install wandb

# Login and train with logging
wandb login
./train.sh --wandb-project foresight-sar
```

### Custom Metrics
```python
# Add custom metrics to training loop
def custom_metric(predictions, targets):
    # Your custom metric calculation
    return metric_value

# Log during training
logger.log({'custom_metric': custom_metric(pred, target)})
```

## Contributing

When adding new models or features:

1. Follow the existing code structure
2. Add comprehensive documentation
3. Include unit tests
4. Update this README
5. Test on multiple datasets
6. Validate performance benchmarks

## Support

For issues and questions:

- Check the troubleshooting section
- Review existing GitHub issues
- Create detailed bug reports
- Include system specifications
- Provide reproducible examples

## License

This training pipeline is part of the Foresight SAR project and follows the same licensing terms.