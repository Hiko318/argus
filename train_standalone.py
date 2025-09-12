#!/usr/bin/env python3
"""
Standalone YOLOv11 Training Script
Creates dataset configuration programmatically to avoid YAML issues
"""

import os
import sys
import tempfile
import yaml
from ultralytics import YOLO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_temp_dataset_config():
    """
    Create a temporary dataset configuration file
    """
    dataset_config = {
        'path': os.path.abspath('data/training'),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 3,
        'names': {
            0: 'person',
            1: 'dog', 
            2: 'cat'
        }
    }
    
    # Create temporary YAML file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(dataset_config, temp_file, default_flow_style=False)
    temp_file.close()
    
    logger.info(f"Created temporary dataset config: {temp_file.name}")
    return temp_file.name

def train_yolo_standalone():
    """
    Train YOLOv11 with programmatically created dataset config
    """
    temp_config = None
    try:
        logger.info("Starting YOLOv11 Training (Standalone Mode)")
        logger.info("=" * 50)
        
        # Create temporary dataset config
        temp_config = create_temp_dataset_config()
        
        # Initialize model
        logger.info("Loading YOLOv11x model...")
        model = YOLO('yolo11x.pt')
        
        # Training parameters
        train_params = {
            'data': temp_config,
            'epochs': 3,  # Very short for quick test
            'imgsz': 320,  # Small image size for fast training
            'batch': 2,    # Very small batch for CPU
            'device': 'cpu',
            'workers': 1,
            'project': 'runs/detect',
            'name': 'standalone_training',
            'save': True,
            'verbose': True,
            'patience': 5,
            'lr0': 0.01,
            'conf': 0.99,
            'plots': False,  # Disable plots to avoid potential issues
            'cache': False   # Disable caching
        }
        
        logger.info(f"Training parameters: {train_params}")
        
        # Start training
        logger.info("Starting training...")
        results = model.train(**train_params)
        
        logger.info("Training completed successfully!")
        logger.info(f"Results saved to: {results.save_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False
    
    finally:
        # Clean up temporary file
        if temp_config and os.path.exists(temp_config):
            try:
                os.unlink(temp_config)
                logger.info(f"Cleaned up temporary config: {temp_config}")
            except:
                pass

if __name__ == "__main__":
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Check if training data exists
    train_dir = 'data/training/images/train'
    if not os.path.exists(train_dir) or len(os.listdir(train_dir)) == 0:
        logger.warning(f"No training images found in {train_dir}")
        logger.info("Please run 'python create_sample_data.py' first to create sample data")
        sys.exit(1)
    
    success = train_yolo_standalone()
    
    if success:
        logger.info("\n" + "=" * 50)
        logger.info("Training completed successfully!")
        logger.info("Model saved in runs/detect/standalone_training/")
        logger.info("You can now use the trained model for inference.")
    else:
        logger.error("Training failed. Check the logs above.")
        sys.exit(1)