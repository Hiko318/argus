#!/usr/bin/env python3
"""
SAR-Optimized YOLOv8 Training Script
Fine-tunes YOLOv8 models for Search and Rescue operations with comprehensive metrics tracking.

Features:
- SAR-specific hyperparameter optimization
- Distance-binned recall metrics
- Real-time performance monitoring
- Model size and inference speed tracking
- Automated model validation and export
"""

import os
import json
import yaml
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix
import wandb
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sar_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SARMetricsTracker:
    """Tracks SAR-specific metrics during training."""
    
    def __init__(self, distance_bins: List[Tuple[int, int]] = None):
        self.distance_bins = distance_bins or [(0, 50), (50, 200), (200, float('inf'))]
        self.metrics_history = []
        self.best_metrics = {}
        
    def calculate_distance_binned_recall(self, predictions, targets, image_metadata):
        """Calculate recall metrics binned by distance."""
        
        distance_recalls = {f"recall@{bin_start}-{bin_end}m": [] 
                          for bin_start, bin_end in self.distance_bins}
        
        for pred, target, metadata in zip(predictions, targets, image_metadata):
            # Extract distance information from metadata
            distance = metadata.get('distance', 100)  # Default to 100m if not available
            
            # Determine which distance bin this belongs to
            for bin_start, bin_end in self.distance_bins:
                if bin_start <= distance < bin_end:
                    bin_key = f"recall@{bin_start}-{bin_end}m"
                    
                    # Calculate recall for this prediction
                    if len(target) > 0:  # Has ground truth
                        # Simple IoU-based matching (simplified for example)
                        matches = self.match_predictions_to_targets(pred, target)
                        recall = len(matches) / len(target) if len(target) > 0 else 0
                        distance_recalls[bin_key].append(recall)
                    break
        
        # Average recalls for each distance bin
        avg_distance_recalls = {}
        for bin_key, recalls in distance_recalls.items():
            avg_distance_recalls[bin_key] = np.mean(recalls) if recalls else 0.0
        
        return avg_distance_recalls
    
    def match_predictions_to_targets(self, predictions, targets, iou_threshold=0.5):
        """Match predictions to targets using IoU threshold."""
        
        matches = []
        
        for pred in predictions:
            best_iou = 0
            best_target_idx = -1
            
            for i, target in enumerate(targets):
                iou = self.calculate_iou(pred[:4], target[:4])  # First 4 elements are bbox
                if iou > best_iou and iou > iou_threshold:
                    best_iou = iou
                    best_target_idx = i
            
            if best_target_idx >= 0:
                matches.append((pred, targets[best_target_idx]))
        
        return matches
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes in YOLO format."""
        
        # Convert YOLO format (x_center, y_center, width, height) to corners
        def yolo_to_corners(box):
            x_center, y_center, width, height = box
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            return x1, y1, x2, y2
        
        x1_1, y1_1, x2_1, y2_1 = yolo_to_corners(box1)
        x1_2, y1_2, x2_2, y2_2 = yolo_to_corners(box2)
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update_metrics(self, epoch, metrics_dict):
        """Update metrics history."""
        
        metrics_dict['epoch'] = epoch
        metrics_dict['timestamp'] = datetime.now().isoformat()
        self.metrics_history.append(metrics_dict)
        
        # Update best metrics
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                if key not in self.best_metrics or value > self.best_metrics[key]:
                    self.best_metrics[key] = value
    
    def save_metrics(self, output_path: str):
        """Save metrics to file."""
        
        metrics_data = {
            'history': self.metrics_history,
            'best_metrics': self.best_metrics,
            'distance_bins': self.distance_bins
        }
        
        with open(output_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Metrics saved to {output_path}")

class SARTrainer:
    """SAR-optimized YOLOv8 trainer."""
    
    def __init__(self, config_path: str, use_wandb: bool = False):
        self.config = self.load_config(config_path)
        self.metrics_tracker = SARMetricsTracker()
        self.use_wandb = use_wandb
        
        if self.use_wandb:
            wandb.init(
                project="sar-yolov8",
                config=self.config,
                name=f"sar_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def load_config(self, config_path: str) -> Dict:
        """Load training configuration."""
        
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        # Set default SAR-optimized parameters
        default_config = {
            'model': 'yolov8n.pt',
            'data': 'data/training/sar_dataset.yaml',
            'epochs': 100,
            'imgsz': 1280,
            'batch': 16,
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 0.05,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'label_smoothing': 0.0,
            'nbs': 64,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'plots': True,
            'save_json': True,
            'save_hybrid': False,
            'conf': 0.001,
            'iou': 0.7,
            'max_det': 300,
            'half': False,
            'device': '',
            'workers': 8,
            'project': 'runs/train',
            'name': 'sar_yolov8',
            'exist_ok': False,
            'pretrained': True,
            'optimizer': 'SGD',
            'verbose': True,
            'seed': 0,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': False,
            'close_mosaic': 10,
            'resume': False,
            'amp': True,
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'multi_scale': False,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
            'auto_augment': 'randaugment',
            'erasing': 0.4,
            'crop_fraction': 1.0
        }
        
        # Merge with defaults
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
        
        return config
    
    def setup_model(self) -> YOLO:
        """Initialize and configure YOLO model."""
        
        model = YOLO(self.config['model'])
        
        # Configure model for SAR-specific requirements
        if hasattr(model.model, 'yaml'):
            # Adjust confidence thresholds for SAR scenarios
            model.model.yaml['conf'] = 0.001  # Lower confidence for distant objects
            model.model.yaml['iou'] = 0.7     # Higher IoU to reduce false positives
        
        logger.info(f"Initialized model: {self.config['model']}")
        return model
    
    def train(self) -> Dict:
        """Execute training with SAR-specific optimizations."""
        
        logger.info("Starting SAR-optimized YOLOv8 training...")
        
        # Setup model
        model = self.setup_model()
        
        # Custom callback for metrics tracking
        def on_train_epoch_end(trainer):
            """Callback executed at the end of each training epoch."""
            
            epoch = trainer.epoch
            metrics = trainer.metrics
            
            # Extract standard metrics
            epoch_metrics = {
                'train_loss': float(trainer.loss.item()) if hasattr(trainer, 'loss') else 0.0,
                'learning_rate': trainer.optimizer.param_groups[0]['lr'],
            }
            
            # Add validation metrics if available
            if hasattr(trainer, 'validator') and trainer.validator.metrics:
                val_metrics = trainer.validator.metrics
                epoch_metrics.update({
                    'val_map50': val_metrics.box.map50,
                    'val_map': val_metrics.box.map,
                    'val_precision': val_metrics.box.mp,
                    'val_recall': val_metrics.box.mr,
                })
            
            # Calculate inference speed
            if hasattr(trainer, 'speed'):
                epoch_metrics['inference_speed_ms'] = trainer.speed['inference']
                epoch_metrics['nms_speed_ms'] = trainer.speed['nms']
                epoch_metrics['total_speed_ms'] = trainer.speed['inference'] + trainer.speed['nms']
                epoch_metrics['fps'] = 1000.0 / epoch_metrics['total_speed_ms'] if epoch_metrics['total_speed_ms'] > 0 else 0
            
            # Update metrics tracker
            self.metrics_tracker.update_metrics(epoch, epoch_metrics)
            
            # Log to wandb if enabled
            if self.use_wandb:
                wandb.log(epoch_metrics, step=epoch)
            
            # Print progress
            if epoch % 10 == 0 or epoch == 1:
                logger.info(f"Epoch {epoch}: mAP@0.5={epoch_metrics.get('val_map50', 0):.3f}, "
                           f"Recall={epoch_metrics.get('val_recall', 0):.3f}, "
                           f"FPS={epoch_metrics.get('fps', 0):.1f}")
        
        # Add callback to model
        model.add_callback('on_train_epoch_end', on_train_epoch_end)
        
        # Start training
        start_time = time.time()
        
        results = model.train(
            data=self.config['data'],
            epochs=self.config['epochs'],
            imgsz=self.config['imgsz'],
            batch=self.config['batch'],
            lr0=self.config['lr0'],
            lrf=self.config['lrf'],
            momentum=self.config['momentum'],
            weight_decay=self.config['weight_decay'],
            warmup_epochs=self.config['warmup_epochs'],
            warmup_momentum=self.config['warmup_momentum'],
            warmup_bias_lr=self.config['warmup_bias_lr'],
            box=self.config['box'],
            cls=self.config['cls'],
            dfl=self.config['dfl'],
            pose=self.config['pose'],
            kobj=self.config['kobj'],
            label_smoothing=self.config['label_smoothing'],
            nbs=self.config['nbs'],
            overlap_mask=self.config['overlap_mask'],
            mask_ratio=self.config['mask_ratio'],
            dropout=self.config['dropout'],
            val=self.config['val'],
            plots=self.config['plots'],
            save_json=self.config['save_json'],
            save_hybrid=self.config['save_hybrid'],
            conf=self.config['conf'],
            iou=self.config['iou'],
            max_det=self.config['max_det'],
            half=self.config['half'],
            device=self.config['device'],
            workers=self.config['workers'],
            project=self.config['project'],
            name=self.config['name'],
            exist_ok=self.config['exist_ok'],
            pretrained=self.config['pretrained'],
            optimizer=self.config['optimizer'],
            verbose=self.config['verbose'],
            seed=self.config['seed'],
            deterministic=self.config['deterministic'],
            single_cls=self.config['single_cls'],
            rect=self.config['rect'],
            cos_lr=self.config['cos_lr'],
            close_mosaic=self.config['close_mosaic'],
            resume=self.config['resume'],
            amp=self.config['amp'],
            fraction=self.config['fraction'],
            profile=self.config['profile'],
            freeze=self.config['freeze'],
            multi_scale=self.config['multi_scale'],
            hsv_h=self.config['hsv_h'],
            hsv_s=self.config['hsv_s'],
            hsv_v=self.config['hsv_v'],
            degrees=self.config['degrees'],
            translate=self.config['translate'],
            scale=self.config['scale'],
            shear=self.config['shear'],
            perspective=self.config['perspective'],
            flipud=self.config['flipud'],
            fliplr=self.config['fliplr'],
            mosaic=self.config['mosaic'],
            mixup=self.config['mixup'],
            copy_paste=self.config['copy_paste'],
            auto_augment=self.config['auto_augment'],
            erasing=self.config['erasing'],
            crop_fraction=self.config['crop_fraction']
        )
        
        training_time = time.time() - start_time
        
        # Post-training analysis
        final_metrics = self.analyze_training_results(model, results, training_time)
        
        # Save comprehensive metrics
        self.save_training_summary(final_metrics)
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Best mAP@0.5: {final_metrics.get('best_map50', 0):.3f}")
        logger.info(f"Final model size: {final_metrics.get('model_size_mb', 0):.1f} MB")
        logger.info(f"Average inference speed: {final_metrics.get('avg_fps', 0):.1f} FPS")
        
        return final_metrics
    
    def analyze_training_results(self, model: YOLO, results, training_time: float) -> Dict:
        """Analyze and compile comprehensive training results."""
        
        # Get model information
        model_path = Path(self.config['project']) / self.config['name'] / 'weights' / 'best.pt'
        model_size_mb = model_path.stat().st_size / (1024 * 1024) if model_path.exists() else 0
        
        # Calculate parameter count
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
        # Performance metrics
        best_metrics = self.metrics_tracker.best_metrics
        avg_fps = np.mean([m.get('fps', 0) for m in self.metrics_tracker.metrics_history if m.get('fps', 0) > 0])
        
        final_metrics = {
            'training_time_seconds': training_time,
            'model_size_mb': model_size_mb,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'avg_fps': avg_fps,
            'best_map50': best_metrics.get('val_map50', 0),
            'best_map': best_metrics.get('val_map', 0),
            'best_precision': best_metrics.get('val_precision', 0),
            'best_recall': best_metrics.get('val_recall', 0),
            'final_learning_rate': best_metrics.get('learning_rate', 0),
            'config': self.config,
            'training_completed_at': datetime.now().isoformat()
        }
        
        return final_metrics
    
    def save_training_summary(self, final_metrics: Dict):
        """Save comprehensive training summary."""
        
        # Save metrics history
        metrics_path = Path(self.config['project']) / self.config['name'] / 'sar_metrics.json'
        self.metrics_tracker.save_metrics(str(metrics_path))
        
        # Save final summary
        summary_path = Path(self.config['project']) / self.config['name'] / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        # Create training visualization
        self.create_training_plots()
        
        logger.info(f"Training summary saved to {summary_path}")
    
    def create_training_plots(self):
        """Create comprehensive training visualization plots."""
        
        if not self.metrics_tracker.metrics_history:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('SAR YOLOv8 Training Results', fontsize=16)
        
        epochs = [m['epoch'] for m in self.metrics_tracker.metrics_history]
        
        # Training loss
        train_losses = [m.get('train_loss', 0) for m in self.metrics_tracker.metrics_history]
        axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # mAP metrics
        map50_values = [m.get('val_map50', 0) for m in self.metrics_tracker.metrics_history]
        map_values = [m.get('val_map', 0) for m in self.metrics_tracker.metrics_history]
        axes[0, 1].plot(epochs, map50_values, 'g-', label='mAP@0.5')
        axes[0, 1].plot(epochs, map_values, 'r-', label='mAP@0.5:0.95')
        axes[0, 1].set_title('Mean Average Precision')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mAP')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision and Recall
        precision_values = [m.get('val_precision', 0) for m in self.metrics_tracker.metrics_history]
        recall_values = [m.get('val_recall', 0) for m in self.metrics_tracker.metrics_history]
        axes[0, 2].plot(epochs, precision_values, 'purple', label='Precision')
        axes[0, 2].plot(epochs, recall_values, 'orange', label='Recall')
        axes[0, 2].set_title('Precision and Recall')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Learning rate
        lr_values = [m.get('learning_rate', 0) for m in self.metrics_tracker.metrics_history]
        axes[1, 0].plot(epochs, lr_values, 'brown', label='Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Inference speed (FPS)
        fps_values = [m.get('fps', 0) for m in self.metrics_tracker.metrics_history if m.get('fps', 0) > 0]
        fps_epochs = [m['epoch'] for m in self.metrics_tracker.metrics_history if m.get('fps', 0) > 0]
        if fps_values:
            axes[1, 1].plot(fps_epochs, fps_values, 'cyan', label='FPS')
            axes[1, 1].set_title('Inference Speed')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Frames Per Second')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        # Model performance summary
        best_metrics = self.metrics_tracker.best_metrics
        metrics_names = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']
        metrics_values = [
            best_metrics.get('val_map50', 0),
            best_metrics.get('val_map', 0),
            best_metrics.get('val_precision', 0),
            best_metrics.get('val_recall', 0)
        ]
        
        bars = axes[1, 2].bar(metrics_names, metrics_values, 
                             color=['green', 'red', 'purple', 'orange'])
        axes[1, 2].set_title('Best Metrics Summary')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plots
        plots_path = Path(self.config['project']) / self.config['name'] / 'sar_training_plots.png'
        plt.savefig(plots_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training plots saved to {plots_path}")

def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(description='SAR-optimized YOLOv8 Training')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Model to use (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)')
    parser.add_argument('--data', type=str, default='data/training/sar_dataset.yaml',
                       help='Path to dataset configuration')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=1280,
                       help='Image size for training')
    
    args = parser.parse_args()
    
    # Create config if not exists
    config_path = Path(args.config)
    if not config_path.exists():
        # Create default config
        default_config = {
            'model': args.model,
            'data': args.data,
            'epochs': args.epochs,
            'batch': args.batch,
            'imgsz': args.imgsz
        }
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        logger.info(f"Created default config at {config_path}")
    
    # Initialize trainer
    trainer = SARTrainer(str(config_path), use_wandb=args.wandb)
    
    # Start training
    results = trainer.train()
    
    # Print final summary
    print("\n" + "="*50)
    print("SAR YOLOv8 Training Complete!")
    print("="*50)
    print(f"Best mAP@0.5: {results.get('best_map50', 0):.3f}")
    print(f"Best mAP@0.5:0.95: {results.get('best_map', 0):.3f}")
    print(f"Model size: {results.get('model_size_mb', 0):.1f} MB")
    print(f"Average FPS: {results.get('avg_fps', 0):.1f}")
    print(f"Training time: {results.get('training_time_seconds', 0):.1f} seconds")
    print("="*50)

if __name__ == "__main__":
    main()