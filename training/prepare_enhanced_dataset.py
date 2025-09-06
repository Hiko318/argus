#!/usr/bin/env python3
"""
Enhanced Dataset Preparation Script for SAR Operations

This script helps prepare training datasets with focus on:
- Covered/occluded human detection
- Pet detection (dogs and cats)
- Data augmentation for challenging scenarios
- Synthetic occlusion generation

Usage:
    python prepare_enhanced_dataset.py --source-dir /path/to/images --output-dir /path/to/output
    python prepare_enhanced_dataset.py --augment-existing --dataset-dir /path/to/existing/dataset
"""

import argparse
import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml
from tqdm import tqdm
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedDatasetPreparator:
    """
    Enhanced dataset preparation with focus on covered humans and pets.
    """
    
    def __init__(self, source_dir: str, output_dir: str):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.class_mapping = {
            'person': 0,
            'human': 0,
            'people': 0,
            'vehicle': 1,
            'car': 1,
            'truck': 1,
            'structure': 2,
            'building': 2,
            'debris': 3,
            'dog': 4,
            'dogs': 4,
            'canine': 4,
            'cat': 5,
            'cats': 5,
            'feline': 5
        }
        
        # Create output directories
        self._create_directories()
        
        # Setup augmentation pipeline for covered humans
        self.occlusion_augmentation = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.MotionBlur(blur_limit=7, p=0.2),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=3, p=0.4),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, p=0.2),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
    def _create_directories(self):
        """Create necessary output directories."""
        dirs = [
            'images/train', 'images/val', 'images/test',
            'labels/train', 'labels/val', 'labels/test'
        ]
        
        for dir_name in dirs:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created output directories in {self.output_dir}")
    
    def _apply_synthetic_occlusion(self, image: np.ndarray, bboxes: List[List[float]], 
                                 class_labels: List[int]) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """
        Apply synthetic occlusion to simulate covered humans.
        
        Args:
            image: Input image
            bboxes: List of bounding boxes in YOLO format
            class_labels: List of class labels
            
        Returns:
            Tuple of (augmented_image, new_bboxes, new_class_labels)
        """
        h, w = image.shape[:2]
        
        # Focus on person bboxes for occlusion
        person_indices = [i for i, label in enumerate(class_labels) if label == 0]
        
        if not person_indices or np.random.random() > 0.6:  # 60% chance of occlusion
            return image, bboxes, class_labels
        
        # Select random person to occlude
        person_idx = np.random.choice(person_indices)
        bbox = bboxes[person_idx]
        
        # Convert YOLO format to pixel coordinates
        x_center, y_center, width, height = bbox
        x_center *= w
        y_center *= h
        width *= w
        height *= h
        
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        # Generate occlusion parameters
        occlusion_types = ['vegetation', 'shadow', 'debris', 'partial_cover']
        occlusion_type = np.random.choice(occlusion_types)
        
        if occlusion_type == 'vegetation':
            # Simulate vegetation occlusion with green patches
            mask = np.zeros((h, w), dtype=np.uint8)
            num_patches = np.random.randint(2, 5)
            for _ in range(num_patches):
                patch_w = np.random.randint(width // 4, width // 2)
                patch_h = np.random.randint(height // 4, height // 2)
                patch_x = np.random.randint(max(0, x1), min(w - patch_w, x2))
                patch_y = np.random.randint(max(0, y1), min(h - patch_h, y2))
                cv2.rectangle(mask, (patch_x, patch_y), (patch_x + patch_w, patch_y + patch_h), 255, -1)
            
            # Apply green vegetation color
            vegetation_color = np.random.randint(20, 80, 3)  # Dark green range
            vegetation_color[1] = np.random.randint(80, 150)  # Higher green component
            image[mask > 0] = vegetation_color
            
        elif occlusion_type == 'shadow':
            # Simulate shadow occlusion
            shadow_intensity = np.random.uniform(0.3, 0.7)
            shadow_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Create irregular shadow shape
            shadow_w = int(width * np.random.uniform(0.5, 0.8))
            shadow_h = int(height * np.random.uniform(0.6, 0.9))
            shadow_x = np.random.randint(max(0, x1), min(w - shadow_w, x2))
            shadow_y = np.random.randint(max(0, y1), min(h - shadow_h, y2))
            
            cv2.ellipse(shadow_mask, (shadow_x + shadow_w//2, shadow_y + shadow_h//2),
                       (shadow_w//2, shadow_h//2), 0, 0, 360, 255, -1)
            
            image[shadow_mask > 0] = (image[shadow_mask > 0] * shadow_intensity).astype(np.uint8)
            
        elif occlusion_type == 'debris':
            # Simulate debris occlusion
            num_debris = np.random.randint(1, 4)
            for _ in range(num_debris):
                debris_w = np.random.randint(width // 6, width // 3)
                debris_h = np.random.randint(height // 6, height // 3)
                debris_x = np.random.randint(max(0, x1), min(w - debris_w, x2))
                debris_y = np.random.randint(max(0, y1), min(h - debris_h, y2))
                
                # Random debris color (brown/gray)
                debris_color = np.random.randint(40, 120, 3)
                cv2.rectangle(image, (debris_x, debris_y), 
                            (debris_x + debris_w, debris_y + debris_h), debris_color.tolist(), -1)
        
        elif occlusion_type == 'partial_cover':
            # Simulate partial covering (e.g., by other objects)
            cover_ratio = np.random.uniform(0.2, 0.5)  # Cover 20-50% of the person
            cover_w = int(width * cover_ratio)
            cover_h = int(height * cover_ratio)
            
            # Random position within the bbox
            cover_x = np.random.randint(x1, max(x1 + 1, x2 - cover_w))
            cover_y = np.random.randint(y1, max(y1 + 1, y2 - cover_h))
            
            # Random cover color
            cover_color = np.random.randint(0, 255, 3)
            cv2.rectangle(image, (cover_x, cover_y), 
                        (cover_x + cover_w, cover_y + cover_h), cover_color.tolist(), -1)
        
        return image, bboxes, class_labels
    
    def _convert_annotations(self, annotation_file: str) -> List[Tuple[str, List[List[float]], List[int]]]:
        """
        Convert various annotation formats to YOLO format.
        
        Args:
            annotation_file: Path to annotation file
            
        Returns:
            List of (image_path, bboxes, class_labels)
        """
        annotations = []
        
        if annotation_file.endswith('.json'):
            # COCO format
            with open(annotation_file, 'r') as f:
                coco_data = json.load(f)
            
            # Process COCO annotations
            for image_info in coco_data.get('images', []):
                image_id = image_info['id']
                image_path = image_info['file_name']
                width = image_info['width']
                height = image_info['height']
                
                bboxes = []
                class_labels = []
                
                for ann in coco_data.get('annotations', []):
                    if ann['image_id'] == image_id:
                        # Convert COCO bbox to YOLO format
                        x, y, w, h = ann['bbox']
                        x_center = (x + w / 2) / width
                        y_center = (y + h / 2) / height
                        w_norm = w / width
                        h_norm = h / height
                        
                        bboxes.append([x_center, y_center, w_norm, h_norm])
                        
                        # Map category to our class system
                        category_name = next((cat['name'] for cat in coco_data.get('categories', []) 
                                            if cat['id'] == ann['category_id']), 'unknown')
                        class_id = self.class_mapping.get(category_name.lower(), -1)
                        if class_id >= 0:
                            class_labels.append(class_id)
                        else:
                            # Skip unknown classes
                            bboxes.pop()
                
                if bboxes:  # Only add if we have valid annotations
                    annotations.append((image_path, bboxes, class_labels))
        
        return annotations
    
    def prepare_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.2, 
                       test_ratio: float = 0.1, augment_factor: int = 3):
        """
        Prepare enhanced dataset with augmentation.
        
        Args:
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            augment_factor: Number of augmented versions per original image
        """
        logger.info("Starting enhanced dataset preparation...")
        
        # Find annotation files
        annotation_files = list(self.source_dir.glob('*.json'))
        if not annotation_files:
            logger.error("No annotation files found in source directory")
            return
        
        all_annotations = []
        for ann_file in annotation_files:
            annotations = self._convert_annotations(str(ann_file))
            all_annotations.extend(annotations)
        
        logger.info(f"Found {len(all_annotations)} annotated images")
        
        # Shuffle and split data
        np.random.shuffle(all_annotations)
        
        n_total = len(all_annotations)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_data = all_annotations[:n_train]
        val_data = all_annotations[n_train:n_train + n_val]
        test_data = all_annotations[n_train + n_val:]
        
        # Process each split
        splits = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        for split_name, split_data in splits.items():
            logger.info(f"Processing {split_name} split ({len(split_data)} images)...")
            
            for idx, (image_path, bboxes, class_labels) in enumerate(tqdm(split_data)):
                # Load image
                full_image_path = self.source_dir / image_path
                if not full_image_path.exists():
                    logger.warning(f"Image not found: {full_image_path}")
                    continue
                
                image = cv2.imread(str(full_image_path))
                if image is None:
                    logger.warning(f"Failed to load image: {full_image_path}")
                    continue
                
                # Save original image and labels
                self._save_image_and_labels(image, bboxes, class_labels, 
                                          f"{split_name}_{idx:06d}", split_name)
                
                # Generate augmented versions (more for training)
                aug_count = augment_factor if split_name == 'train' else 1
                
                for aug_idx in range(aug_count):
                    # Apply synthetic occlusion
                    aug_image, aug_bboxes, aug_labels = self._apply_synthetic_occlusion(
                        image.copy(), bboxes.copy(), class_labels.copy())
                    
                    # Apply additional augmentations
                    try:
                        augmented = self.occlusion_augmentation(
                            image=aug_image, bboxes=aug_bboxes, class_labels=aug_labels)
                        aug_image = augmented['image']
                        aug_bboxes = augmented['bboxes']
                        aug_labels = augmented['class_labels']
                    except Exception as e:
                        logger.warning(f"Augmentation failed for {image_path}: {e}")
                        continue
                    
                    # Save augmented version
                    self._save_image_and_labels(aug_image, aug_bboxes, aug_labels,
                                              f"{split_name}_{idx:06d}_aug_{aug_idx}", split_name)
        
        # Create dataset YAML
        self._create_dataset_yaml()
        
        logger.info("Enhanced dataset preparation completed!")
    
    def _save_image_and_labels(self, image: np.ndarray, bboxes: List[List[float]], 
                              class_labels: List[int], filename: str, split: str):
        """
        Save image and corresponding YOLO format labels.
        
        Args:
            image: Image array
            bboxes: List of bounding boxes in YOLO format
            class_labels: List of class labels
            filename: Base filename (without extension)
            split: Dataset split (train/val/test)
        """
        # Save image
        image_path = self.output_dir / 'images' / split / f"{filename}.jpg"
        cv2.imwrite(str(image_path), image)
        
        # Save labels
        label_path = self.output_dir / 'labels' / split / f"{filename}.txt"
        with open(label_path, 'w') as f:
            for bbox, class_id in zip(bboxes, class_labels):
                f.write(f"{class_id} {' '.join(map(str, bbox))}\n")
    
    def _create_dataset_yaml(self):
        """
        Create dataset YAML file for YOLOv8.
        """
        dataset_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 6,
            'names': {
                0: 'person',
                1: 'vehicle',
                2: 'structure', 
                3: 'debris',
                4: 'dog',
                5: 'cat'
            }
        }
        
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        logger.info(f"Created dataset YAML: {yaml_path}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Enhanced Dataset Preparation for SAR')
    parser.add_argument('--source-dir', type=str, required=True,
                       help='Source directory containing images and annotations')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for prepared dataset')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Ratio of data for training (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='Ratio of data for validation (default: 0.2)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='Ratio of data for testing (default: 0.1)')
    parser.add_argument('--augment-factor', type=int, default=3,
                       help='Number of augmented versions per original image (default: 3)')
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.001:
        logger.error("Train, validation, and test ratios must sum to 1.0")
        sys.exit(1)
    
    # Initialize preparator
    preparator = EnhancedDatasetPreparator(args.source_dir, args.output_dir)
    
    # Prepare dataset
    preparator.prepare_dataset(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        augment_factor=args.augment_factor
    )
    
    logger.info("Dataset preparation completed successfully!")

if __name__ == '__main__':
    main()