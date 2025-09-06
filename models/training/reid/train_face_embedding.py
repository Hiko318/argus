#!/usr/bin/env python3
"""
Foresight SAR Face Embedding Training Script

This script trains face embedding models for face recognition and verification
in SAR operations. Uses ArcFace loss for improved face recognition performance.

Author: Foresight SAR Team
Date: 2024
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import time
import random
import math

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms as transforms
    from torchvision import models
    import numpy as np
    from PIL import Image
    import cv2
    from sklearn.metrics import accuracy_score, roc_auc_score
    from tqdm import tqdm
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install: pip install torch torchvision numpy pillow opencv-python scikit-learn tqdm matplotlib")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class FaceTrainingConfig:
    """Face embedding training configuration."""
    backbone: str = 'resnet50'  # 'resnet50', 'resnet101', 'mobilenet_v2', 'efficientnet_b0'
    embedding_dim: int = 512
    num_classes: int = 10000  # Number of identities in training set
    batch_size: int = 64
    learning_rate: float = 0.1
    weight_decay: float = 5e-4
    epochs: int = 100
    warmup_epochs: int = 5
    step_size: int = 30
    gamma: float = 0.1
    margin: float = 0.5  # ArcFace margin
    scale: float = 64.0  # ArcFace scale
    device: str = 'auto'
    num_workers: int = 4
    seed: int = 42
    save_interval: int = 10
    eval_interval: int = 5
    input_size: Tuple[int, int] = (112, 112)  # Standard face recognition input size
    
class ArcFaceLoss(nn.Module):
    """ArcFace loss for face recognition."""
    
    def __init__(self, embedding_dim: int, num_classes: int, 
                 margin: float = 0.5, scale: float = 64.0):
        super(ArcFaceLoss, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        
        # Weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass for ArcFace loss."""
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cosine = F.linear(embeddings, weight)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # Compute phi (cosine with margin)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Apply threshold
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Create one-hot labels
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Apply margin to target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        return F.cross_entropy(output, labels)

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass for focal loss."""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class FaceEmbeddingModel(nn.Module):
    """Face embedding model with various backbones."""
    
    def __init__(self, backbone: str = 'resnet50', embedding_dim: int = 512, 
                 dropout: float = 0.5, input_size: Tuple[int, int] = (112, 112)):
        super(FaceEmbeddingModel, self).__init__()
        
        self.backbone = self._build_backbone(backbone)
        self.embedding_dim = embedding_dim
        self.input_size = input_size
        
        # Get backbone output dimension
        backbone_dim = self._get_backbone_dim(backbone)
        
        # Feature layers
        self.bn1 = nn.BatchNorm2d(backbone_dim)
        self.dropout = nn.Dropout2d(dropout)
        self.fc = nn.Linear(backbone_dim * 7 * 7, embedding_dim)  # Assuming 7x7 feature map
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        
        self._init_weights()
    
    def _build_backbone(self, backbone: str) -> nn.Module:
        """Build backbone network."""
        if backbone == 'resnet50':
            model = models.resnet50(pretrained=True)
            # Remove avgpool and fc layers
            model = nn.Sequential(*list(model.children())[:-2])
        elif backbone == 'resnet101':
            model = models.resnet101(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-2])
        elif backbone == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=True)
            model = model.features
        elif backbone == 'efficientnet_b0':
            try:
                from torchvision.models import efficientnet_b0
                model = efficientnet_b0(pretrained=True)
                model = model.features
            except ImportError:
                logger.warning("EfficientNet not available, falling back to ResNet50")
                model = models.resnet50(pretrained=True)
                model = nn.Sequential(*list(model.children())[:-2])
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        return model
    
    def _get_backbone_dim(self, backbone: str) -> int:
        """Get backbone output dimension."""
        if 'resnet50' in backbone:
            return 2048
        elif 'resnet101' in backbone:
            return 2048
        elif 'mobilenet' in backbone:
            return 1280
        elif 'efficientnet' in backbone:
            return 1280
        else:
            return 2048  # Default
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Backbone features
        features = self.backbone(x)
        
        # Batch normalization and dropout
        features = self.bn1(features)
        features = self.dropout(features)
        
        # Flatten
        features = features.view(features.size(0), -1)
        
        # Embedding layer
        embeddings = self.fc(features)
        embeddings = self.bn2(embeddings)
        
        return embeddings

class FaceDataset(Dataset):
    """Dataset for face recognition training."""
    
    def __init__(self, data_dir: Path, split: str = 'train', transform=None, 
                 min_images_per_identity: int = 5):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.min_images_per_identity = min_images_per_identity
        
        # Load dataset
        self.samples = self._load_samples()
        self.identity_to_idx = self._build_identity_mapping()
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
        logger.info(f"Number of identities: {len(self.identity_to_idx)}")
    
    def _load_samples(self) -> List[Tuple[Path, str]]:
        """Load dataset samples."""
        samples = []
        split_dir = self.data_dir / self.split
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        # Count images per identity
        identity_counts = {}
        for identity_dir in split_dir.iterdir():
            if identity_dir.is_dir():
                identity_id = identity_dir.name
                image_count = len(list(identity_dir.glob('*.[jp][pn]g')))
                identity_counts[identity_id] = image_count
        
        # Filter identities with sufficient images
        valid_identities = {
            identity: count for identity, count in identity_counts.items()
            if count >= self.min_images_per_identity
        }
        
        logger.info(f"Found {len(valid_identities)} identities with >= {self.min_images_per_identity} images")
        
        # Load samples from valid identities
        for identity_dir in split_dir.iterdir():
            if identity_dir.is_dir() and identity_dir.name in valid_identities:
                identity_id = identity_dir.name
                for image_file in identity_dir.glob('*'):
                    if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        samples.append((image_file, identity_id))
        
        return samples
    
    def _build_identity_mapping(self) -> Dict[str, int]:
        """Build mapping from identity ID to class index."""
        identities = sorted(set(identity for _, identity in self.samples))
        return {identity: idx for idx, identity in enumerate(identities)}
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path, identity_id = self.samples[idx]
        
        # Load and preprocess image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get class label
        label = self.identity_to_idx[identity_id]
        
        return image, label

class FaceTrainer:
    """Trainer for face embedding models."""
    
    def __init__(self, config: FaceTrainingConfig, output_dir: Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        if config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config.device)
        
        logger.info(f"Using device: {self.device}")
        
        # Set random seeds
        self._set_seeds(config.seed)
        
        # Initialize model, loss, and optimizer
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.training_history = []
    
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def build_model(self, num_classes: int):
        """Build and initialize model."""
        self.model = FaceEmbeddingModel(
            backbone=self.config.backbone,
            embedding_dim=self.config.embedding_dim,
            dropout=0.5,
            input_size=self.config.input_size
        ).to(self.device)
        
        # Loss function (ArcFace)
        self.criterion = ArcFaceLoss(
            embedding_dim=self.config.embedding_dim,
            num_classes=num_classes,
            margin=self.config.margin,
            scale=self.config.scale
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.SGD(
            [
                {'params': self.model.parameters()},
                {'params': self.criterion.parameters()}
            ],
            lr=self.config.learning_rate,
            momentum=0.9,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.step_size,
            gamma=self.config.gamma
        )
        
        logger.info(f"Model built with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def get_transforms(self, split: str) -> transforms.Compose:
        """Get data transforms for different splits."""
        if split == 'train':
            return transforms.Compose([
                transforms.Resize(self.config.input_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1] normalization
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.config.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.criterion.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.config.epochs}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            embeddings = self.model(images)
            loss = self.criterion(embeddings, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            
            # For accuracy, we need to compute predictions from ArcFace output
            with torch.no_grad():
                # Normalize embeddings and weights
                norm_embeddings = F.normalize(embeddings, p=2, dim=1)
                norm_weights = F.normalize(self.criterion.weight, p=2, dim=1)
                
                # Compute cosine similarity
                logits = F.linear(norm_embeddings, norm_weights) * self.config.scale
                _, predicted = logits.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
        
        # Compute epoch metrics
        epoch_metrics = {
            'loss': total_loss / len(train_loader),
            'accuracy': 100. * correct / total
        }
        
        return epoch_metrics
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        self.criterion.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Evaluating"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                embeddings = self.model(images)
                loss = self.criterion(embeddings, labels)
                
                total_loss += loss.item()
                
                # Compute predictions
                norm_embeddings = F.normalize(embeddings, p=2, dim=1)
                norm_weights = F.normalize(self.criterion.weight, p=2, dim=1)
                logits = F.linear(norm_embeddings, norm_weights) * self.config.scale
                
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'criterion_state_dict': self.criterion.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': metrics,
            'training_history': self.training_history
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with accuracy: {metrics['accuracy']:.2f}%")
        
        # Save latest model
        latest_path = self.output_dir / 'latest_model.pth'
        torch.save(checkpoint, latest_path)
        
        # Save model for inference (model only)
        inference_checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'embedding_dim': self.config.embedding_dim,
            'input_size': self.config.input_size
        }
        inference_path = self.output_dir / 'face_embedding_model.pth'
        torch.save(inference_checkpoint, inference_path)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop."""
        logger.info("Starting face embedding training...")
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            if epoch % self.config.eval_interval == 0:
                val_metrics = self.evaluate(val_loader)
                
                # Log metrics
                logger.info(f"Epoch {epoch + 1}/{self.config.epochs}:")
                logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
                logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
                
                # Save training history
                self.training_history.append({
                    'epoch': epoch + 1,
                    'train': train_metrics,
                    'val': val_metrics
                })
                
                # Check if best model
                is_best = val_metrics['accuracy'] > self.best_accuracy
                if is_best:
                    self.best_accuracy = val_metrics['accuracy']
                
                # Save checkpoint
                if epoch % self.config.save_interval == 0 or is_best:
                    self.save_checkpoint(epoch + 1, val_metrics, is_best)
            
            # Update learning rate
            self.scheduler.step()
        
        logger.info(f"Training completed! Best accuracy: {self.best_accuracy:.2f}%")
        
        # Save final training plot
        self._plot_training_history()
    
    def _plot_training_history(self):
        """Plot training history."""
        if not self.training_history:
            return
        
        epochs = [h['epoch'] for h in self.training_history]
        train_loss = [h['train']['loss'] for h in self.training_history]
        val_loss = [h['val']['loss'] for h in self.training_history]
        train_acc = [h['train']['accuracy'] for h in self.training_history]
        val_acc = [h['val']['accuracy'] for h in self.training_history]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss plot
        ax1.plot(epochs, train_loss, label='Train Loss')
        ax1.plot(epochs, val_loss, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, train_acc, label='Train Acc')
        ax2.plot(epochs, val_acc, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training history plot saved to: {self.output_dir / 'training_history.png'}")

def main():
    parser = argparse.ArgumentParser(description="Foresight SAR Face Embedding Training")
    
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Dataset directory')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for models and logs')
    parser.add_argument('--config', type=str,
                       help='Training configuration file (YAML)')
    parser.add_argument('--backbone', type=str, default='resnet50',
                       choices=['resnet50', 'resnet101', 'mobilenet_v2', 'efficientnet_b0'],
                       help='Backbone architecture')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Learning rate')
    parser.add_argument('--margin', type=float, default=0.5,
                       help='ArcFace margin')
    parser.add_argument('--scale', type=float, default=64.0,
                       help='ArcFace scale')
    parser.add_argument('--device', type=str, default='auto',
                       help='Training device')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--resume', type=str,
                       help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = FaceTrainingConfig(**config_dict)
    else:
        config = FaceTrainingConfig(
            backbone=args.backbone,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            margin=args.margin,
            scale=args.scale,
            device=args.device,
            seed=args.seed
        )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = output_dir / 'face_training_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config.__dict__, f, default_flow_style=False)
    
    # Create datasets
    trainer = FaceTrainer(config, output_dir)
    train_transform = trainer.get_transforms('train')
    val_transform = trainer.get_transforms('val')
    
    train_dataset = FaceDataset(args.data_dir, 'train', train_transform)
    val_dataset = FaceDataset(args.data_dir, 'val', val_transform)
    
    # Update config with actual number of classes
    config.num_classes = len(train_dataset.identity_to_idx)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Create trainer
    trainer = FaceTrainer(config, output_dir)
    trainer.build_model(config.num_classes)
    
    # Resume training if requested
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.criterion.load_state_dict(checkpoint['criterion_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        trainer.training_history = checkpoint.get('training_history', [])
        logger.info(f"Resumed training from epoch {trainer.current_epoch}")
    
    # Start training
    trainer.train(train_loader, val_loader)
    
    logger.info(f"Training completed. Models saved to: {output_dir}")

if __name__ == '__main__':
    main()