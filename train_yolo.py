#!/usr/bin/env python3
"""
YOLO Training Script for SAR Dataset
Foresight SAR Application - Object Detection Model Training

Usage:
    python train_yolo.py
    
Or run directly with YOLO CLI:
    yolo task=detect mode=train model=yolov8n.pt data=sar_dataset.yaml epochs=80 imgsz=1280 batch=12
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dataset_structure():
    """Verify that the dataset structure exists"""
    required_dirs = [
        'data/sar/images/train',
        'data/sar/images/val',
        'data/sar/labels/train',
        'data/sar/labels/val'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        logger.warning(f"Missing directories: {missing_dirs}")
        logger.info("Creating missing directories...")
        for dir_path in missing_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    return len(missing_dirs) == 0

def check_dataset_files():
    """Check if dataset has any training files"""
    train_images = list(Path('data/sar/images/train').glob('*'))
    val_images = list(Path('data/sar/images/val').glob('*'))
    
    if not train_images:
        logger.warning("No training images found in data/sar/images/train/")
        logger.info("Please add your training images and corresponding label files")
        return False
    
    if not val_images:
        logger.warning("No validation images found in data/sar/images/val/")
        logger.info("Please add your validation images and corresponding label files")
        return False
    
    logger.info(f"Found {len(train_images)} training images and {len(val_images)} validation images")
    return True

def train_yolo_model():
    """Train YOLO model with SAR dataset"""
    try:
        # Check if dataset configuration exists
        dataset_config = 'data/training/sar_dataset.yaml'
        if not Path(dataset_config).exists():
            logger.error(f"{dataset_config} not found. Please ensure the dataset configuration file exists.")
            return False
        
        # Check dataset structure
        check_dataset_structure()
        
        # Check for actual dataset files
        if not check_dataset_files():
            logger.warning("Dataset appears to be empty. Training will proceed but may fail.")
            logger.info("To add sample data, place images in data/sar/images/train/ and labels in data/sar/labels/train/")
        
        # Initialize YOLO model
        logger.info("Initializing YOLOv11x model for maximum accuracy...")
        model = YOLO('yolo11x.pt')  # Load pretrained YOLOv11x model for highest accuracy
        
        # Training parameters
        training_args = {
            'data': 'data/training/sar_dataset.yaml',
            'epochs': 200,  # More epochs for maximum accuracy
            'imgsz': 1280,
            'batch': 12,
            'device': 'cpu',  # Use CPU for training
            'workers': 4,
            'project': 'runs/detect',
            'name': 'sar_training',
            'save': True,
            'save_period': 10,  # Save checkpoint every 10 epochs
            'cache': False,  # Don't cache images (saves RAM)
            'verbose': True,
            'seed': 42,  # For reproducible results
            'deterministic': True,
            'single_cls': False,  # Multi-class detection
            'rect': False,  # Rectangular training
            'cos_lr': False,  # Cosine learning rate scheduler
            'close_mosaic': 10,  # Disable mosaic augmentation for last N epochs
            'resume': False,  # Resume training from last checkpoint
            'amp': True,  # Automatic Mixed Precision training
            'fraction': 1.0,  # Use full dataset
            'profile': False,  # Profile ONNX and TensorRT speeds during training
            'freeze': None,  # Freeze layers: backbone=10, first3=0,1,2
            'lr0': 0.001,  # Very low initial learning rate for precision
            'lrf': 0.0001,  # Extremely low final learning rate
            'momentum': 0.95,  # Higher momentum for stability
            'weight_decay': 0.001,  # Higher weight decay for generalization
            'warmup_epochs': 10.0,  # Longer warmup for stability
            'warmup_momentum': 0.9,  # Higher warmup momentum
            'warmup_bias_lr': 0.05,  # Lower warmup bias lr for precision
            'box': 0.1,  # Higher box loss for precise localization
            'cls': 1.0,  # Maximum cls loss for confident classification
            'dfl': 2.5,  # Higher DFL loss for better distribution
            'pose': 12.0,  # Pose loss gain
            'kobj': 2.0,  # Keypoint obj loss gain
            'label_smoothing': 0.0,  # Label smoothing
            'nbs': 64,  # Nominal batch size
            'overlap_mask': True,  # Masks should overlap during training
            'mask_ratio': 4,  # Mask downsample ratio
            'dropout': 0.0,  # Use dropout regularization
            'val': True,  # Validate/test during training
            'conf': 0.99,  # Maximum confidence threshold (99% - closest to 100%)
        }
        
        logger.info("Starting YOLO training...")
        logger.info(f"Training parameters: {training_args}")
        
        # Start training
        results = model.train(**training_args)
        
        logger.info("Training completed successfully!")
        logger.info(f"Results saved to: {results.save_dir}")
        logger.info(f"Best model saved as: {results.save_dir}/weights/best.pt")
        
        # Validate the model
        logger.info("Running validation...")
        val_results = model.val()
        logger.info(f"Validation mAP50: {val_results.box.map50:.4f}")
        logger.info(f"Validation mAP50-95: {val_results.box.map:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        return False

def main():
    """Main training function"""
    logger.info("SAR YOLO Training Script")
    logger.info("=" * 50)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8+ is required for ultralytics")
        return
    
    # Start training
    success = train_yolo_model()
    
    if success:
        logger.info("\n" + "="*50)
        logger.info("Training completed successfully!")
        logger.info("Next steps:")
        logger.info("1. Check results in runs/detect/sar_training/")
        logger.info("2. Use best.pt for inference")
        logger.info("3. Export to ONNX/TensorRT for deployment")
        logger.info("="*50)
    else:
        logger.error("Training failed. Please check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()