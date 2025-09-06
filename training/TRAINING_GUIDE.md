# Enhanced SAR Training Guide

## Achieving >93% Accuracy for Covered Humans and Pet Detection

This guide provides step-by-step instructions to train your SAR model for improved covered human detection (>93% accuracy) and household pet detection (dogs and cats).

## Prerequisites

### Required Dependencies
```bash
pip install ultralytics>=8.0.0
pip install albumentations>=1.3.0
pip install opencv-python>=4.8.0
pip install torch>=2.0.0
pip install torchvision>=0.15.0
pip install pyyaml>=6.0
pip install tqdm>=4.65.0
pip install numpy>=1.24.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
```

### Hardware Requirements
- **Minimum**: 8GB RAM, GTX 1060 or equivalent
- **Recommended**: 16GB+ RAM, RTX 3070 or better
- **Optimal**: 32GB+ RAM, RTX 4080/4090 or A100

## Step 1: Prepare Your Dataset

### 1.1 Organize Training Data

Create the following directory structure:
```
data/
├── raw_images/
│   ├── covered_humans/     # Images with partially occluded humans
│   ├── clear_humans/       # Images with clearly visible humans
│   ├── pets/              # Images with dogs and cats
│   └── annotations.json   # COCO format annotations
└── processed/             # Output from preparation script
```

### 1.2 Data Collection Guidelines

**For Covered Humans (Target: >93% accuracy):**
- Collect 2000+ images with humans under various occlusion conditions:
  - Vegetation coverage (trees, bushes, grass)
  - Shadow occlusion
  - Debris coverage
  - Partial structural occlusion
  - Weather conditions (fog, rain)
- Ensure diverse lighting conditions
- Include various human poses and orientations
- Maintain 70% covered humans, 30% clear humans ratio

**For Pet Detection:**
- Collect 1500+ images each for dogs and cats:
  - Various breeds and sizes
  - Different poses (sitting, lying, running)
  - Indoor and outdoor environments
  - Different lighting conditions
  - Include pets with humans in same frame

### 1.3 Run Data Preparation

```bash
# Navigate to training directory
cd training/

# Prepare enhanced dataset with synthetic occlusion
python prepare_enhanced_dataset.py \
    --source-dir ../data/raw_images \
    --output-dir ../data/processed \
    --train-ratio 0.7 \
    --val-ratio 0.2 \
    --test-ratio 0.1 \
    --augment-factor 4
```

This will:
- Apply synthetic occlusion to simulate covered humans
- Generate 4x augmented versions of training data
- Create proper train/val/test splits
- Generate YOLO format annotations

## Step 2: Configure Training Parameters

### 2.1 Update Dataset Configuration

Ensure `data/training/dataset.yaml` contains:
```yaml
path: data/processed
train: images/train
val: images/val
test: images/test
nc: 6
names:
  0: person
  1: vehicle
  2: structure
  3: debris
  4: dog
  5: cat
```

### 2.2 Training Configuration

The enhanced `configs/train_config.yaml` is already optimized for:
- **Model**: YOLOv8s (better than nano for accuracy)
- **Image Size**: 736px (higher resolution for better detection)
- **Epochs**: 150 (sufficient for convergence)
- **Batch Size**: 12 (balanced for memory and performance)
- **Advanced Loss Functions**: Focal Loss for hard examples
- **Enhanced Augmentation**: Optimized for SAR scenarios

## Step 3: Training Process

### 3.1 Start Enhanced Training

```bash
# Use the enhanced training script
python train_enhanced.py \
    --config ../configs/train_config.yaml \
    --dataset ../data/processed/dataset.yaml \
    --name sar_enhanced_v1 \
    --resume-from-checkpoint path/to/checkpoint.pt  # Optional
```

### 3.2 Monitor Training Progress

The enhanced trainer provides:
- **Real-time metrics**: mAP, precision, recall per class
- **Loss tracking**: Box loss, class loss, focal loss
- **Validation curves**: Performance on covered humans specifically
- **Learning rate scheduling**: Automatic adjustment
- **Early stopping**: Prevents overfitting

### 3.3 Key Metrics to Watch

**For Covered Human Detection (Target: >93%):**
- `mAP50_person`: Should reach >0.93
- `recall_person`: Should be >0.90
- `precision_person`: Should be >0.95

**For Pet Detection:**
- `mAP50_dog`: Should reach >0.85
- `mAP50_cat`: Should reach >0.85

## Step 4: Advanced Optimization Techniques

### 4.1 Hyperparameter Tuning

If initial results don't meet targets, try:

```bash
# Hyperparameter search
python train_enhanced.py \
    --config ../configs/train_config.yaml \
    --dataset ../data/processed/dataset.yaml \
    --name sar_tuning \
    --hyperparameter-search \
    --search-iterations 20
```

### 4.2 Model Architecture Upgrades

For even better accuracy, consider:

1. **YOLOv8m or YOLOv8l**: Larger models for higher accuracy
2. **Custom backbone**: ResNet50 or EfficientNet
3. **Ensemble methods**: Combine multiple models

### 4.3 Advanced Data Augmentation

Add more challenging scenarios:
```python
# In prepare_enhanced_dataset.py, add:
A.RandomWeather(rain=(0.1, 0.3), snow=(0.1, 0.2), p=0.3),
A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=0.2),
A.RandomToneCurve(scale=0.1, p=0.3)
```

## Step 5: Evaluation and Validation

### 5.1 Comprehensive Testing

```bash
# Test on validation set
python -c "
from ultralytics import YOLO
model = YOLO('runs/detect/sar_enhanced_v1/weights/best.pt')
results = model.val(data='../data/processed/dataset.yaml')
print(f'Overall mAP: {results.box.map}')
print(f'Person mAP: {results.box.maps[0]}')
print(f'Dog mAP: {results.box.maps[4]}')
print(f'Cat mAP: {results.box.maps[5]}')
"
```

### 5.2 Covered Human Specific Testing

Create a test script for covered humans:
```python
# test_covered_humans.py
import cv2
from ultralytics import YOLO

model = YOLO('runs/detect/sar_enhanced_v1/weights/best.pt')

# Test on covered human images
covered_human_images = ['path/to/covered1.jpg', 'path/to/covered2.jpg']

for img_path in covered_human_images:
    results = model(img_path, conf=0.25)
    # Check if person detected with high confidence
    for r in results:
        for box in r.boxes:
            if box.cls == 0 and box.conf > 0.93:  # Person class with >93% confidence
                print(f"✓ Covered human detected: {box.conf:.3f}")
```

## Step 6: Deployment Optimization

### 6.1 Model Export

```bash
# Export for different deployment scenarios
python -c "
from ultralytics import YOLO
model = YOLO('runs/detect/sar_enhanced_v1/weights/best.pt')

# For Jetson deployment
model.export(format='engine', device=0, half=True)

# For CPU deployment
model.export(format='onnx', opset=11)

# For mobile deployment
model.export(format='tflite', int8=True)
"
```

### 6.2 Performance Benchmarking

```bash
# Benchmark inference speed
python -c "
from ultralytics import YOLO
import time

model = YOLO('runs/detect/sar_enhanced_v1/weights/best.pt')

# Warm up
for _ in range(10):
    model('path/to/test_image.jpg')

# Benchmark
start_time = time.time()
for _ in range(100):
    results = model('path/to/test_image.jpg')
end_time = time.time()

print(f'Average inference time: {(end_time - start_time) / 100 * 1000:.2f}ms')
"
```

## Expected Results

### Performance Targets
- **Covered Human Detection**: >93% mAP50
- **Dog Detection**: >85% mAP50
- **Cat Detection**: >85% mAP50
- **Overall mAP**: >88%
- **Inference Speed**: <50ms on RTX 3070

### Training Timeline
- **Data Preparation**: 2-4 hours
- **Training (150 epochs)**: 8-12 hours on RTX 3070
- **Validation & Testing**: 1-2 hours
- **Total**: 12-18 hours

## Troubleshooting

### Common Issues

1. **Low Accuracy on Covered Humans**:
   - Increase augmentation factor to 6-8
   - Add more diverse occlusion types
   - Increase training epochs to 200
   - Use YOLOv8m instead of YOLOv8s

2. **Poor Pet Detection**:
   - Collect more diverse pet images
   - Increase pet class weights in loss function
   - Add pet-specific augmentations

3. **Overfitting**:
   - Reduce learning rate
   - Increase dropout
   - Add more validation data
   - Use early stopping

4. **Memory Issues**:
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

### Performance Optimization

1. **For Higher Accuracy**:
   - Use YOLOv8x model
   - Increase image resolution to 1024px
   - Implement test-time augmentation
   - Use model ensembling

2. **For Faster Inference**:
   - Use YOLOv8n model
   - Reduce image resolution to 640px
   - Enable TensorRT optimization
   - Use quantization

## Next Steps

After achieving target performance:
1. **Deploy to production environment**
2. **Set up continuous monitoring**
3. **Implement feedback loop for model improvement**
4. **Consider active learning for edge cases**
5. **Expand to additional animal classes if needed**

## Support

For issues or questions:
1. Check training logs in `runs/detect/sar_enhanced_v1/`
2. Review validation metrics and loss curves
3. Test on individual challenging images
4. Consider consulting SAR domain experts for data quality

---

**Remember**: Achieving >93% accuracy on covered humans requires high-quality, diverse training data and proper hyperparameter tuning. The enhanced training pipeline provides the tools, but data quality is crucial for success.