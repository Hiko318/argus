# Enhanced SAR Model Training Guide

## 🎯 Objective
Train an enhanced SAR (Search and Rescue) model to achieve:
- **>93% accuracy** on covered/partially occluded humans
- **Pet detection** capabilities (dogs and cats)
- Optimized performance for aerial SAR operations

## ✅ Completed Enhancements

### 1. Enhanced Training Configuration
- **Model**: Upgraded from YOLOv8n to YOLOv8s for better accuracy
- **Classes**: Extended to 6 classes (human, vehicle, structure, debris, dog, cat)
- **Image Size**: Increased to 736px for better detection
- **Training**: 150 epochs with optimized hyperparameters
- **Batch Size**: 12 (optimized for available hardware)

### 2. Advanced Training Tools
- **Enhanced Training Script** (`train_enhanced.py`)
  - Focal Loss implementation for hard examples
  - Advanced data augmentation pipeline
  - SAR-specific optimizations
  - Real-time monitoring and logging

- **Data Preparation Script** (`prepare_enhanced_dataset.py`)
  - Synthetic occlusion generation
  - Vegetation coverage simulation
  - Shadow and debris augmentation
  - YOLO format conversion

### 3. SAR-Specific Optimizations
- **Covered Human Detection**:
  - Synthetic occlusion augmentation
  - Vegetation coverage simulation
  - Shadow handling improvements
  - Debris occlusion training

- **Pet Detection**:
  - Dog class (class 4)
  - Cat class (class 5)
  - Household pet identification
  - Multi-species detection

## 📊 Training Requirements

### Required Training Data
- **2000+ images** with covered humans (various occlusion types)
- **1500+ images** with dogs (various breeds and poses)
- **1500+ images** with cats (various breeds and poses)
- **1000+ images** with vehicles and structures
- **500+ images** with debris

### Covered Human Scenarios
- Vegetation coverage (trees, bushes, grass)
- Shadow occlusion
- Debris coverage
- Partial structural occlusion
- Weather conditions (fog, rain)

### Pet Detection Scenarios
- Various dog breeds and sizes
- Different cat breeds and colors
- Indoor and outdoor environments
- Different lighting conditions
- Pets with humans in same frame

## 🚀 Training Process

### Step 1: Data Collection
1. Collect training images as specified above
2. Create COCO format annotations
3. Organize data in `data/raw_images/` directory

### Step 2: Data Preparation
```bash
cd training
python prepare_enhanced_dataset.py \
    --source-dir ../data/raw_images \
    --output-dir ../data/processed \
    --augment-factor 4
```

### Step 3: Enhanced Training
```bash
python train_enhanced.py \
    --config ../configs/train_config.yaml \
    --dataset ../data/processed/dataset.yaml \
    --name sar_enhanced_v1
```

### Step 4: Monitor Training
- Watch for **mAP50_person > 0.93** (covered humans)
- Check **pet detection accuracy > 0.85**
- Monitor loss curves and validation metrics
- Use TensorBoard for real-time monitoring

### Step 5: Evaluate Results
```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/detect/sar_enhanced_v1/weights/best.pt')

# Evaluate on test set
results = model.val(data='../data/processed/dataset.yaml')

# Check specific metrics
print(f'Person mAP: {results.box.maps[0]:.3f}')
print(f'Dog mAP: {results.box.maps[4]:.3f}')
print(f'Cat mAP: {results.box.maps[5]:.3f}')
```

## 🔧 Configuration Files

### Training Configuration (`configs/train_config.yaml`)
- Optimized hyperparameters for SAR scenarios
- Enhanced augmentation settings
- Focal Loss configuration
- Multi-scale training setup

### Dataset Configuration (`data/training/dataset.yaml`)
- 6-class detection setup
- Proper train/val/test splits
- Class names and indices

## 📈 Expected Performance

### Target Metrics
- **Human Detection (Covered)**: >93% mAP50
- **Dog Detection**: >85% mAP50
- **Cat Detection**: >85% mAP50
- **Overall mAP**: >88%
- **Inference Speed**: <50ms per image

### Key Improvements
1. **Better Occlusion Handling**: Synthetic augmentation improves covered human detection
2. **Larger Model**: YOLOv8s provides better feature extraction than nano
3. **Higher Resolution**: 736px training improves small object detection
4. **Focal Loss**: Addresses class imbalance and hard examples
5. **SAR Optimizations**: Aerial imagery specific enhancements

## 🛠️ Troubleshooting

### Common Issues
1. **Insufficient Training Data**: Collect more diverse examples
2. **Class Imbalance**: Use weighted sampling or focal loss
3. **Overfitting**: Increase augmentation or reduce model complexity
4. **Low Accuracy**: Check data quality and annotation accuracy

### Performance Optimization
- Use mixed precision training for faster convergence
- Implement gradient accumulation for larger effective batch sizes
- Use learning rate scheduling for better convergence
- Monitor validation metrics to prevent overfitting

## 📁 File Structure
```
foresight/
├── configs/
│   └── train_config.yaml          # Enhanced training configuration
├── data/
│   ├── raw_images/                 # Original training images
│   ├── processed/                  # Processed YOLO format data
│   └── training/
│       ├── dataset.yaml            # Dataset configuration
│       └── yolo_format/            # YOLO format directories
├── training/
│   ├── train_enhanced.py           # Enhanced training script
│   ├── prepare_enhanced_dataset.py # Data preparation script
│   └── demo_enhanced_model.py      # Demo and guidance script
└── runs/
    └── detect/
        └── sar_enhanced_v1/        # Training results
```

## 🎯 Success Criteria

### Model Performance
- [x] Enhanced configuration created
- [x] Advanced training pipeline implemented
- [x] SAR-specific optimizations added
- [ ] Training data collected
- [ ] Model trained with >93% covered human accuracy
- [ ] Pet detection validated
- [ ] Model deployed for SAR operations

### Technical Achievements
- ✅ YOLOv8s model configuration
- ✅ 6-class detection setup
- ✅ Focal Loss implementation
- ✅ Advanced augmentation pipeline
- ✅ Synthetic occlusion generation
- ✅ Comprehensive training guide

---

**Status**: Enhanced SAR model configuration complete. Ready for training once data is collected.

**Next Action**: Collect training data as specified and run the enhanced training pipeline.