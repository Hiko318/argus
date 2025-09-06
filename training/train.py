#!/usr/bin/env python3
"""
YOLOv8 Training Script for SAR Operations

This script provides comprehensive training capabilities for YOLOv8 models
optimized for Search and Rescue operations, including person detection
in aerial imagery.

Usage:
    python train.py --config train_config.yaml
    python train.py --config train_config.yaml --resume runs/train/sar_yolov8/weights/last.pt
    python train.py --data datasets/custom --epochs 50
"""

import argparse
import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
except ImportError:
    print("Error: Ultralytics not installed. Install with: pip install ultralytics")
    sys.exit(1)

import torch
import torch.backends.cudnn as cudnn
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SARTrainer:
    """
    SAR-optimized YOLOv8 trainer with specialized configurations
    for aerial imagery and person detection.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize trainer with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.model = None
        self.device = self._setup_device()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load and validate training configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def _setup_device(self) -> str:
        """Setup and validate training device."""
        if torch.cuda.is_available():
            device = f"cuda:{self.config.get('device', 0)}"
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Using GPU: {gpu_name}")
            
            # Enable optimizations
            cudnn.benchmark = True
            if self.config.get('deterministic', False):
                cudnn.deterministic = True
                torch.manual_seed(self.config.get('seed', 0))
        else:
            device = 'cpu'
            logger.warning("CUDA not available, using CPU (training will be slow)")
            
        return device
    
    def _validate_dataset(self) -> bool:
        """Validate dataset structure and files."""
        data_config = self.config.get('data', {})
        dataset_path = Path(data_config.get('path', 'datasets/sar_dataset'))
        
        required_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
        
        for dir_name in required_dirs:
            dir_path = dataset_path / dir_name
            if not dir_path.exists():
                logger.error(f"Missing required directory: {dir_path}")
                return False
            
            # Check if directory has files
            files = list(dir_path.glob('*'))
            if not files:
                logger.warning(f"Directory {dir_path} is empty")
        
        logger.info("Dataset validation passed")
        return True
    
    def _create_dataset_yaml(self) -> str:
        """Create dataset YAML file for YOLOv8."""
        data_config = self.config.get('data', {})
        dataset_path = Path(data_config.get('path', 'datasets/sar_dataset'))
        
        dataset_yaml = {
            'path': str(dataset_path.absolute()),
            'train': data_config.get('train', 'images/train'),
            'val': data_config.get('val', 'images/val'),
            'test': data_config.get('test', 'images/test'),
            'nc': data_config.get('nc', 4),
            'names': data_config.get('names', {
                0: 'person',
                1: 'vehicle', 
                2: 'structure',
                3: 'debris'
            })
        }
        
        yaml_path = dataset_path / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)
        
        logger.info(f"Created dataset YAML: {yaml_path}")
        return str(yaml_path)
    
    def _apply_sar_optimizations(self) -> Dict[str, Any]:
        """Apply SAR-specific training optimizations."""
        sar_config = self.config.get('sar_config', {})
        
        # Modify hyperparameters for aerial imagery
        hyp = self.config.get('hyp', {}).copy()
        
        # Adjust augmentation for aerial views
        hyp.update({
            'degrees': 0.0,      # No rotation for aerial imagery
            'perspective': 0.0,   # No perspective changes
            'flipud': 0.0,       # No vertical flipping
            'fliplr': 0.5,       # Keep horizontal flipping
        })
        
        # Adjust loss weights for SAR priorities
        if 'person_weight' in sar_config:
            hyp['cls'] *= sar_config['person_weight']
        
        logger.info("Applied SAR-specific optimizations")
        return hyp
    
    def _setup_callbacks(self):
        """Setup training callbacks for monitoring and logging."""
        callbacks_config = self.config.get('callbacks', {})
        
        # Setup TensorBoard logging
        if callbacks_config.get('tensorboard', True):
            log_dir = self.config.get('logging', {}).get('log_dir', 'logs/training')
            os.makedirs(log_dir, exist_ok=True)
            logger.info(f"TensorBoard logging enabled: {log_dir}")
    
    def train(self, resume: Optional[str] = None) -> str:
        """
        Execute training with SAR optimizations.
        
        Args:
            resume: Path to checkpoint to resume from
            
        Returns:
            Path to best model weights
        """
        logger.info("Starting SAR YOLOv8 training...")
        
        # Validate dataset
        if not self._validate_dataset():
            raise ValueError("Dataset validation failed")
        
        # Create dataset YAML
        dataset_yaml = self._create_dataset_yaml()
        
        # Setup callbacks
        self._setup_callbacks()
        
        # Apply SAR optimizations
        optimized_hyp = self._apply_sar_optimizations()
        
        # Initialize model
        model_path = resume if resume else self.config.get('model', 'yolov8n.pt')
        self.model = YOLO(model_path)
        
        logger.info(f"Initialized model: {model_path}")
        
        # Prepare training arguments
        train_args = {
            'data': dataset_yaml,
            'epochs': self.config.get('epochs', 100),
            'batch': self.config.get('batch_size', 16),
            'imgsz': self.config.get('imgsz', 640),
            'device': self.device,
            'workers': self.config.get('workers', 8),
            'project': self.config.get('project', 'runs/train'),
            'name': self.config.get('name', 'sar_yolov8'),
            'exist_ok': self.config.get('exist_ok', True),
            'pretrained': self.config.get('pretrained', True),
            'optimizer': self.config.get('optimizer', 'SGD'),
            'verbose': self.config.get('verbose', True),
            'seed': self.config.get('seed', 0),
            'deterministic': self.config.get('deterministic', True),
            'single_cls': self.config.get('single_cls', False),
            'rect': self.config.get('rect', False),
            'cos_lr': self.config.get('cos_lr', False),
            'close_mosaic': self.config.get('close_mosaic', 10),
            'resume': bool(resume),
            'amp': self.config.get('amp', True),
            'fraction': self.config.get('fraction', 1.0),
            'profile': self.config.get('profile', False),
            'freeze': self.config.get('freeze', None),
            'multi_scale': self.config.get('multi_scale', False),
            'overlap_mask': self.config.get('overlap_mask', True),
            'mask_ratio': self.config.get('mask_ratio', 4),
            'dropout': self.config.get('dropout', 0.0),
            'val': self.config.get('val', True),
            'split': self.config.get('split', 'val'),
            'save_json': self.config.get('save_json', False),
            'save_hybrid': self.config.get('save_hybrid', False),
            'conf': self.config.get('conf', None),
            'iou': self.config.get('iou', 0.7),
            'max_det': self.config.get('max_det', 300),
            'half': self.config.get('half', False),
            'dnn': self.config.get('dnn', False),
            'plots': self.config.get('plots', True)
        }
        
        # Add hyperparameters
        for key, value in optimized_hyp.items():
            train_args[key] = value
        
        # Start training
        logger.info(f"Training arguments: {train_args}")
        
        try:
            results = self.model.train(**train_args)
            
            # Get best model path
            best_model_path = self.model.trainer.best
            logger.info(f"Training completed. Best model: {best_model_path}")
            
            return str(best_model_path)
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def export_model(self, model_path: str, formats: list = None) -> Dict[str, str]:
        """
        Export trained model to deployment formats.
        
        Args:
            model_path: Path to trained model weights
            formats: List of export formats ['onnx', 'tensorrt', etc.]
            
        Returns:
            Dictionary mapping format to exported file path
        """
        if formats is None:
            formats = self.config.get('export_config', {}).get('formats', ['onnx'])
        
        logger.info(f"Exporting model to formats: {formats}")
        
        model = YOLO(model_path)
        exported_paths = {}
        
        export_config = self.config.get('export_config', {})
        
        for fmt in formats:
            try:
                logger.info(f"Exporting to {fmt.upper()}...")
                
                export_args = {
                    'format': fmt,
                    'optimize': export_config.get('optimize', True),
                    'half': export_config.get('half', True),
                    'dynamic': export_config.get('dynamic', False),
                    'simplify': export_config.get('simplify', True),
                }
                
                if fmt == 'onnx':
                    export_args['opset'] = export_config.get('opset', 11)
                elif fmt == 'tensorrt':
                    export_args['workspace'] = export_config.get('workspace', 4)
                
                exported_path = model.export(**export_args)
                exported_paths[fmt] = str(exported_path)
                logger.info(f"Exported {fmt.upper()}: {exported_path}")
                
            except Exception as e:
                logger.error(f"Failed to export {fmt}: {e}")
        
        return exported_paths

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='YOLOv8 SAR Training')
    parser.add_argument('--config', type=str, default='train_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--export', action='store_true',
                       help='Export model after training')
    parser.add_argument('--data', type=str, default=None,
                       help='Override dataset path')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--device', type=str, default=None,
                       help='Override training device')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = SARTrainer(args.config)
    
    # Apply command line overrides
    if args.data:
        trainer.config['data']['path'] = args.data
    if args.epochs:
        trainer.config['epochs'] = args.epochs
    if args.batch_size:
        trainer.config['batch_size'] = args.batch_size
    if args.device:
        trainer.config['device'] = args.device
    
    try:
        # Train model
        best_model_path = trainer.train(resume=args.resume)
        
        # Export if requested
        if args.export:
            exported_paths = trainer.export_model(best_model_path)
            logger.info(f"Exported models: {exported_paths}")
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()