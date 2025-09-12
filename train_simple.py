#!/usr/bin/env python3
"""
Simple YOLOv11 Training Script
Direct parameter approach to avoid YAML parsing issues
"""

import os
import sys
from ultralytics import YOLO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_yolo_simple():
    """
    Train YOLOv11 with direct parameters
    """
    try:
        logger.info("Starting YOLOv11 Training (Simple Mode)")
        logger.info("=" * 50)
        
        # Initialize model
        logger.info("Loading YOLOv11x model...")
        model = YOLO('yolo11x.pt')
        
        # Training parameters
        train_params = {
            'data': 'data/training/sar_dataset.yaml',
            'epochs': 5,  # Reduced for quick test
            'imgsz': 640,  # Reduced image size for faster training
            'batch': 4,    # Smaller batch for CPU
            'device': 'cpu',
            'workers': 2,
            'project': 'runs/detect',
            'name': 'simple_training',
            'save': True,
            'verbose': True,
            'patience': 10,
            'lr0': 0.001,
            'conf': 0.99
        }
        
        logger.info(f"Training parameters: {train_params}")
        
        # Start training
        logger.info("Starting training...")
        results = model.train(**train_params)
        
        logger.info("Training completed successfully!")
        logger.info(f"Results saved to: {results.save_dir}")
        
        # Validate the model
        logger.info("Running validation...")
        val_results = model.val()
        logger.info(f"Validation mAP50: {val_results.box.map50:.4f}")
        logger.info(f"Validation mAP50-95: {val_results.box.map:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    success = train_yolo_simple()
    
    if success:
        logger.info("\n" + "=" * 50)
        logger.info("Training completed successfully!")
        logger.info("Model saved in runs/detect/simple_training/")
        logger.info("You can now use the trained model for inference.")
    else:
        logger.error("Training failed. Check the logs above.")
        sys.exit(1)