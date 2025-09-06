#!/usr/bin/env python3
"""
Foresight SAR Person Re-Identification Training Script

This script trains person re-identification models for tracking individuals
across multiple camera views in SAR operations. Supports both person re-ID
and face embedding models.

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
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install: pip install torch torchvision numpy pillow opencv-python scikit-learn tqdm")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    model_type: str = 'person_reid'  # 'person_reid' or 'face_embedding'
    backbone: str = 'resnet50'  # 'resnet50', 'resnet101', 'efficientnet_b0', etc.
    embedding_dim: int = 512
    num_classes: int = 1000  # Number of identities in training set
    batch_size: int = 32
    learning_rate: float = 0.0003
    weight_decay: float = 5e-4
    epochs: int = 120
    warmup_epochs: int = 10
    step_size: int = 40
    gamma: float = 0.1
    margin: float = 0.3  # For triplet loss
    device: str = 'auto'
    num_workers: int = 4
    seed: int = 42
    save_interval: int = 10
    eval_interval: int = 5
    
class TripletLoss(nn.Module):
    """Triplet loss for metric learning."""
    
    def __init__(self, margin: float = 0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute triplet loss."""
        # Get all pairwise distances
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
        
        # Create masks for positive and negative pairs
        labels = labels.unsqueeze(1)
        mask_pos = (labels == labels.t()).float()
        mask_neg = (labels != labels.t()).float()
        
        # Remove diagonal (self-comparisons)
        mask_pos = mask_pos - torch.eye(mask_pos.size(0), device=mask_pos.device)
        
        # Hard negative mining
        hardest_positive_dist = (pairwise_dist * mask_pos).max(dim=1)[0]
        hardest_negative_dist = (pairwise_dist + 1e6 * mask_pos).min(dim=1)[0]
        
        # Compute triplet loss
        y = torch.ones_like(hardest_positive_dist)
        loss = self.ranking_loss(hardest_negative_dist, hardest_positive_dist, y)
        
        return loss

class CenterLoss(nn.Module):
    """Center loss for feature learning."""
    
    def __init__(self, num_classes: int, feat_dim: int, device: str = 'cuda'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute center loss."""
        batch_size = features.size(0)
        
        # Compute distances to centers
        distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(features, self.centers.t(), beta=1, alpha=-2)
        
        # Select distances for ground truth classes
        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        
        return loss

class PersonReIDModel(nn.Module):
    """Person Re-Identification model."""
    
    def __init__(self, backbone: str = 'resnet50', embedding_dim: int = 512, 
                 num_classes: int = 1000, dropout: float = 0.5):
        super(PersonReIDModel, self).__init__()
        
        self.backbone = self._build_backbone(backbone)
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # Get backbone output dimension
        if 'resnet' in backbone:
            backbone_dim = 2048 if 'resnet50' in backbone or 'resnet101' in backbone else 512
        elif 'efficientnet' in backbone:
            backbone_dim = 1280  # EfficientNet-B0
        else:
            backbone_dim = 2048  # Default
        
        # Feature embedding layers
        self.bottleneck = nn.BatchNorm1d(backbone_dim)
        self.bottleneck.bias.requires_grad_(False)
        
        self.embedding = nn.Linear(backbone_dim, embedding_dim, bias=False)
        self.embedding_bn = nn.BatchNorm1d(embedding_dim)
        self.embedding_bn.bias.requires_grad_(False)
        
        # Classification layer
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _build_backbone(self, backbone: str) -> nn.Module:
        """Build backbone network."""
        if backbone == 'resnet50':
            model = models.resnet50(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1])  # Remove FC layer
        elif backbone == 'resnet101':
            model = models.resnet101(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1])
        elif backbone == 'efficientnet_b0':
            try:
                from torchvision.models import efficientnet_b0
                model = efficientnet_b0(pretrained=True)
                model.classifier = nn.Identity()  # Remove classifier
            except ImportError:
                logger.warning("EfficientNet not available, falling back to ResNet50")
                model = models.resnet50(pretrained=True)
                model = nn.Sequential(*list(model.children())[:-1])
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        return model
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass."""
        # Backbone features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Bottleneck
        features = self.bottleneck(features)
        
        # Embedding
        embeddings = self.embedding(features)
        embeddings = self.embedding_bn(embeddings)
        embeddings = self.dropout(embeddings)
        
        if return_features:
            return embeddings
        
        # Classification
        logits = self.classifier(embeddings)
        
        return logits, embeddings

class ReIDDataset(Dataset):
    """Dataset for person re-identification."""
    
    def __init__(self, data_dir: Path, split: str = 'train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
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
        
        # Assume directory structure: split/identity_id/image_files
        for identity_dir in split_dir.iterdir():
            if identity_dir.is_dir():
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
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get class label
        label = self.identity_to_idx[identity_id]
        
        return image, label

class ReIDTrainer:
    """Trainer for person re-identification models."""
    
    def __init__(self, config: TrainingConfig, output_dir: Path):
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
        
        # Initialize model, losses, and optimizer
        self.model = None
        self.criterion_ce = None
        self.criterion_triplet = None
        self.criterion_center = None
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
        self.model = PersonReIDModel(
            backbone=self.config.backbone,
            embedding_dim=self.config.embedding_dim,
            num_classes=num_classes,
            dropout=0.5
        ).to(self.device)
        
        # Loss functions
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_triplet = TripletLoss(margin=self.config.margin)
        self.criterion_center = CenterLoss(
            num_classes=num_classes,
            feat_dim=self.config.embedding_dim,
            device=self.device
        )
        
        # Optimizer
        params = [
            {'params': self.model.parameters()},
            {'params': self.criterion_center.parameters(), 'lr': self.config.learning_rate * 0.5}
        ]
        
        self.optimizer = optim.Adam(
            params,
            lr=self.config.learning_rate,
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
                transforms.Resize((256, 128)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_ce_loss = 0.0
        total_triplet_loss = 0.0
        total_center_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.config.epochs}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            logits, embeddings = self.model(images)
            
            # Compute losses
            ce_loss = self.criterion_ce(logits, labels)
            triplet_loss = self.criterion_triplet(embeddings, labels)
            center_loss = self.criterion_center(embeddings, labels)
            
            # Combined loss
            loss = ce_loss + 0.5 * triplet_loss + 0.0005 * center_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_triplet_loss += triplet_loss.item()
            total_center_loss += center_loss.item()
            
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
            'ce_loss': total_ce_loss / len(train_loader),
            'triplet_loss': total_triplet_loss / len(train_loader),
            'center_loss': total_center_loss / len(train_loader),
            'accuracy': 100. * correct / total
        }
        
        return epoch_metrics
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Evaluating"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                logits, embeddings = self.model(images)
                loss = self.criterion_ce(logits, labels)
                
                total_loss += loss.item()
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
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop."""
        logger.info("Starting training...")
        
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

def main():
    parser = argparse.ArgumentParser(description="Foresight SAR Person Re-ID Training")
    
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Dataset directory')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for models and logs')
    parser.add_argument('--config', type=str,
                       help='Training configuration file (YAML)')
    parser.add_argument('--model-type', type=str, default='person_reid',
                       choices=['person_reid', 'face_embedding'],
                       help='Model type to train')
    parser.add_argument('--backbone', type=str, default='resnet50',
                       choices=['resnet50', 'resnet101', 'efficientnet_b0'],
                       help='Backbone architecture')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=120,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0003,
                       help='Learning rate')
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
        config = TrainingConfig(**config_dict)
    else:
        config = TrainingConfig(
            model_type=args.model_type,
            backbone=args.backbone,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            device=args.device,
            seed=args.seed
        )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = output_dir / 'training_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config.__dict__, f, default_flow_style=False)
    
    # Create datasets
    train_transform = ReIDTrainer(config, output_dir).get_transforms('train')
    val_transform = ReIDTrainer(config, output_dir).get_transforms('val')
    
    train_dataset = ReIDDataset(args.data_dir, 'train', train_transform)
    val_dataset = ReIDDataset(args.data_dir, 'val', val_transform)
    
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
    trainer = ReIDTrainer(config, output_dir)
    trainer.build_model(config.num_classes)
    
    # Resume training if requested
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
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