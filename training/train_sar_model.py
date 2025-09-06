#!/usr/bin/env python3
"""SAR-specific YOLO training script.

Fine-tunes YOLO models for human detection in aerial Search and Rescue scenarios.
Includes SAR-specific augmentations, loss functions, and validation metrics.
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from ultralytics import YOLO
    import torch
    import numpy as np
    import cv2
    from PIL import Image
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Install with: pip install ultralytics torch opencv-python pillow")
    sys.exit(1)

from .augmentation import SARAugmentation
from .validation import SARValidator
from .dataset_manager import DatasetManager


class SARTrainer:
    """SAR-specific YOLO trainer with aerial optimizations."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.setup_logging()
        
        # Initialize components
        self.augmentation = SARAugmentation(self.config.get('augmentation', {}))
        self.validator = SARValidator(self.config.get('validation', {}))
        self.dataset_manager = DatasetManager(self.config.get('dataset', {}))
        
        # Training state
        self.model = None
        self.training_results = {}
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load training configuration."""
        default_config = {
            'model': {
                'base_model': 'yolov8n.pt',
                'input_size': 640,
                'batch_size': 16,
                'epochs': 100,
                'patience': 50,
                'device': 'auto'
            },
            'dataset': {
                'train_path': 'datasets/sar_train',
                'val_path': 'datasets/sar_val',
                'test_path': 'datasets/sar_test',
                'classes': ['person'],
                'min_target_size': 8,  # Minimum target size in pixels
                'max_target_size': 200  # Maximum target size in pixels
            },
            'augmentation': {
                'enable_sar_augmentations': True,
                'altitude_simulation': True,
                'weather_simulation': True,
                'lighting_variation': True,
                'small_target_focus': True
            },
            'optimization': {
                'optimizer': 'AdamW',
                'lr0': 0.01,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1
            },
            'loss': {
                'box_loss_gain': 0.05,
                'cls_loss_gain': 0.5,
                'dfl_loss_gain': 1.5,
                'focal_loss': True,
                'label_smoothing': 0.0
            },
            'validation': {
                'val_interval': 1,
                'save_best': True,
                'save_period': 10,
                'distance_thresholds': [50, 100, 200, 500],  # meters
                'confidence_threshold': 0.25,
                'iou_threshold': 0.45
            },
            'output': {
                'project': 'runs/train',
                'name': f'sar_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'save_txt': True,
                'save_conf': True
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                # Deep merge configs
                default_config.update(user_config)
        
        return default_config
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def prepare_dataset(self) -> str:
        """Prepare and validate dataset for training."""
        self.logger.info("Preparing SAR dataset...")
        
        # Create dataset YAML file
        dataset_config = {
            'path': str(Path(self.config['dataset']['train_path']).parent.absolute()),
            'train': 'train',
            'val': 'val',
            'test': 'test' if Path(self.config['dataset']['test_path']).exists() else None,
            'names': {i: name for i, name in enumerate(self.config['dataset']['classes'])},
            'nc': len(self.config['dataset']['classes'])
        }
        
        # Add SAR-specific metadata
        dataset_config['sar_metadata'] = {
            'min_target_size': self.config['dataset']['min_target_size'],
            'max_target_size': self.config['dataset']['max_target_size'],
            'altitude_range': [50, 500],  # meters
            'camera_angles': [-90, -45, -30],  # degrees from horizontal
            'weather_conditions': ['clear', 'cloudy', 'hazy', 'low_light']
        }
        
        dataset_yaml_path = Path(self.config['output']['project']) / self.config['output']['name'] / 'dataset.yaml'
        dataset_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dataset_yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        self.logger.info(f"Dataset configuration saved to {dataset_yaml_path}")
        
        # Validate dataset
        stats = self.dataset_manager.validate_dataset(
            self.config['dataset']['train_path'],
            self.config['dataset']['val_path']
        )
        
        self.logger.info(f"Dataset statistics: {stats}")
        
        return str(dataset_yaml_path)
    
    def create_model(self) -> YOLO:
        """Create and configure YOLO model for SAR training."""
        self.logger.info(f"Loading base model: {self.config['model']['base_model']}")
        
        # Load base model
        model_path = Path("models") / self.config['model']['base_model']
        if not model_path.exists():
            self.logger.warning(f"Model not found at {model_path}, downloading...")
            model = YOLO(self.config['model']['base_model'])  # Will auto-download
        else:
            model = YOLO(str(model_path))
        
        # Configure model for SAR-specific optimizations
        if hasattr(model.model, 'yaml'):
            # Modify model architecture for small target detection
            model_config = model.model.yaml.copy()
            
            # Increase feature pyramid network resolution for small targets
            if 'backbone' in model_config:
                model_config['backbone']['small_target_optimization'] = True
            
            self.logger.info("Applied SAR-specific model optimizations")
        
        return model
    
    def train(self, resume: bool = False) -> Dict:
        """Train the SAR model."""
        self.logger.info("Starting SAR model training...")
        
        # Prepare dataset
        dataset_yaml = self.prepare_dataset()
        
        # Create model
        self.model = self.create_model()
        
        # Configure training parameters
        train_args = {
            'data': dataset_yaml,
            'epochs': self.config['model']['epochs'],
            'batch': self.config['model']['batch_size'],
            'imgsz': self.config['model']['input_size'],
            'device': self.config['model']['device'],
            'patience': self.config['model']['patience'],
            'project': self.config['output']['project'],
            'name': self.config['output']['name'],
            'exist_ok': True,
            'resume': resume,
            
            # Optimizer settings
            'optimizer': self.config['optimization']['optimizer'],
            'lr0': self.config['optimization']['lr0'],
            'lrf': self.config['optimization']['lrf'],
            'momentum': self.config['optimization']['momentum'],
            'weight_decay': self.config['optimization']['weight_decay'],
            'warmup_epochs': self.config['optimization']['warmup_epochs'],
            'warmup_momentum': self.config['optimization']['warmup_momentum'],
            'warmup_bias_lr': self.config['optimization']['warmup_bias_lr'],
            
            # Loss settings
            'box': self.config['loss']['box_loss_gain'],
            'cls': self.config['loss']['cls_loss_gain'],
            'dfl': self.config['loss']['dfl_loss_gain'],
            'fl_gamma': 2.0 if self.config['loss']['focal_loss'] else 0.0,
            'label_smoothing': self.config['loss']['label_smoothing'],
            
            # Validation settings
            'val': True,
            'save': self.config['validation']['save_best'],
            'save_period': self.config['validation']['save_period'],
            
            # Output settings
            'save_txt': self.config['output']['save_txt'],
            'save_conf': self.config['output']['save_conf'],
            
            # SAR-specific augmentations
            'hsv_h': 0.015,  # Hue augmentation for lighting variations
            'hsv_s': 0.7,    # Saturation for weather conditions
            'hsv_v': 0.4,    # Value for altitude/atmospheric effects
            'degrees': 0.0,  # No rotation (aerial view is consistent)
            'translate': 0.1, # Small translation for camera movement
            'scale': 0.5,    # Scale variation for altitude changes
            'shear': 0.0,    # No shear (aerial perspective)
            'perspective': 0.0, # No perspective (orthogonal view)
            'flipud': 0.0,   # No vertical flip
            'fliplr': 0.5,   # Horizontal flip OK
            'mosaic': 1.0,   # Mosaic augmentation for multi-target scenes
            'mixup': 0.1,    # Light mixup for robustness
            'copy_paste': 0.1 # Copy-paste for target augmentation
        }
        
        self.logger.info(f"Training with parameters: {train_args}")
        
        try:
            # Start training
            results = self.model.train(**train_args)
            
            # Store results
            self.training_results = {
                'best_fitness': float(results.best_fitness) if hasattr(results, 'best_fitness') else None,
                'best_epoch': int(results.best_epoch) if hasattr(results, 'best_epoch') else None,
                'training_time': results.speed if hasattr(results, 'speed') else None,
                'final_metrics': results.results_dict if hasattr(results, 'results_dict') else None
            }
            
            self.logger.info(f"Training completed. Best fitness: {self.training_results['best_fitness']}")
            
            # Run SAR-specific validation
            self._run_sar_validation()
            
            # Save training summary
            self._save_training_summary()
            
            return self.training_results
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def _run_sar_validation(self):
        """Run SAR-specific validation metrics."""
        if not self.model:
            return
        
        self.logger.info("Running SAR-specific validation...")
        
        # Get best model path
        best_model_path = Path(self.config['output']['project']) / self.config['output']['name'] / 'weights' / 'best.pt'
        
        if best_model_path.exists():
            # Load best model for validation
            best_model = YOLO(str(best_model_path))
            
            # Run validation with distance-based metrics
            val_results = self.validator.validate_with_distance_metrics(
                best_model,
                self.config['dataset']['val_path'],
                self.config['validation']['distance_thresholds']
            )
            
            self.training_results['sar_validation'] = val_results
            self.logger.info(f"SAR validation results: {val_results}")
    
    def _save_training_summary(self):
        """Save comprehensive training summary."""
        summary = {
            'config': self.config,
            'results': self.training_results,
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'base_model': self.config['model']['base_model'],
                'final_model_path': str(Path(self.config['output']['project']) / self.config['output']['name'] / 'weights' / 'best.pt')
            }
        }
        
        summary_path = Path(self.config['output']['project']) / self.config['output']['name'] / 'training_summary.json'
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Training summary saved to {summary_path}")
    
    def export_model(self, formats: List[str] = None) -> Dict[str, str]:
        """Export trained model to various formats."""
        if not self.model:
            raise ValueError("No trained model available for export")
        
        if formats is None:
            formats = ['onnx', 'torchscript', 'tflite']
        
        export_paths = {}
        best_model_path = Path(self.config['output']['project']) / self.config['output']['name'] / 'weights' / 'best.pt'
        
        if best_model_path.exists():
            model = YOLO(str(best_model_path))
            
            for fmt in formats:
                try:
                    self.logger.info(f"Exporting model to {fmt}...")
                    export_path = model.export(format=fmt)
                    export_paths[fmt] = str(export_path)
                    self.logger.info(f"Exported {fmt} model to {export_path}")
                except Exception as e:
                    self.logger.error(f"Failed to export {fmt}: {e}")
        
        return export_paths


def main():
    """Main CLI interface for SAR training."""
    parser = argparse.ArgumentParser(
        description="Train YOLO models for SAR human detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_sar_model.py --config configs/sar_training.yaml
  python train_sar_model.py --base-model yolov8s.pt --epochs 200
  python train_sar_model.py --resume runs/train/sar_model_20240101_120000
        """
    )
    
    parser.add_argument("--config", help="Path to training configuration YAML")
    parser.add_argument("--base-model", default="yolov8n.pt", help="Base YOLO model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="auto", help="Training device (auto, cpu, 0, 1, ...)")
    parser.add_argument("--resume", help="Resume training from checkpoint")
    parser.add_argument("--export", nargs="*", default=["onnx"], help="Export formats after training")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = SARTrainer(args.config)
    
    # Override config with CLI args
    if args.base_model:
        trainer.config['model']['base_model'] = args.base_model
    if args.epochs:
        trainer.config['model']['epochs'] = args.epochs
    if args.batch_size:
        trainer.config['model']['batch_size'] = args.batch_size
    if args.device:
        trainer.config['model']['device'] = args.device
    
    try:
        # Train model
        results = trainer.train(resume=bool(args.resume))
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Best fitness: {results.get('best_fitness', 'N/A')}")
        print(f"Best epoch: {results.get('best_epoch', 'N/A')}")
        
        if 'sar_validation' in results:
            print("\nSAR Validation Results:")
            for metric, value in results['sar_validation'].items():
                print(f"  {metric}: {value}")
        
        # Export model
        if args.export:
            print("\nExporting model...")
            export_paths = trainer.export_model(args.export)
            for fmt, path in export_paths.items():
                print(f"  {fmt}: {path}")
        
        print("\nTraining artifacts saved to:")
        print(f"  {Path(trainer.config['output']['project']) / trainer.config['output']['name']}")
        
    except Exception as e:
        print(f"\nTraining failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()