#!/usr/bin/env python3
"""
SAR Dataset Preparation Script
Prepares training data for YOLOv8 with SAR-specific scenarios and augmentations.

Features:
- Aerial and ground-level imagery processing
- Environmental challenge simulation (water, rubble, occlusion)
- Albumentations-based augmentation pipeline
- YOLO format annotation generation
- Dataset validation and statistics
"""

import os
import json
import yaml
import shutil
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_prep.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SARDatasetConfig:
    """Configuration for SAR dataset preparation."""
    
    # Paths
    source_data_dir: str = "data/raw"
    output_dir: str = "data/training"
    
    # Dataset splits
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    
    # Image processing
    target_size: Tuple[int, int] = (1280, 1280)
    min_object_size: int = 32  # Minimum bounding box size
    
    # Augmentation settings
    augmentation_factor: int = 3  # How many augmented versions per original
    
    # SAR-specific classes
    classes: Dict[str, int] = None
    
    def __post_init__(self):
        if self.classes is None:
            self.classes = {
                'person': 0,
                'vehicle': 1,
                'aircraft': 2,
                'boat': 3,
                'structure': 4,
                'debris': 5,
                'equipment': 6,
                'animal': 7,
                'signal': 8,
                'clothing': 9
            }

class SARDatasetPreparator:
    """Prepares SAR-specific datasets for YOLOv8 training."""
    
    def __init__(self, config: SARDatasetConfig):
        self.config = config
        self.setup_directories()
        self.setup_augmentation_pipeline()
        
    def setup_directories(self):
        """Create necessary directory structure."""
        base_dir = Path(self.config.output_dir)
        
        # Create main directories
        for split in ['train', 'val', 'test']:
            (base_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (base_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Create metadata directory
        (base_dir / 'metadata').mkdir(exist_ok=True)
        
        logger.info(f"Created directory structure in {base_dir}")
    
    def setup_augmentation_pipeline(self):
        """Setup Albumentations pipeline for SAR-specific augmentations."""
        
        # Base augmentations for all scenarios
        base_transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        ]
        
        # Weather condition augmentations
        weather_transforms = [
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.3),
            A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, 
                        drop_color=(200, 200, 200), blur_value=7, brightness_coefficient=0.7, 
                        rain_type=None, p=0.2),
            A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, p=0.2),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, 
                           num_flare_circles_lower=6, num_flare_circles_upper=10, 
                           src_radius=400, src_color=(255, 255, 255), p=0.2),
        ]
        
        # Environmental challenge augmentations
        environmental_transforms = [
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.MotionBlur(blur_limit=7, p=0.2),
            A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.3),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.3),
            A.CoarseDropout(max_holes=8, max_height=64, max_width=64, 
                          min_holes=1, min_height=32, min_width=32, 
                          fill_value=0, mask_fill_value=0, p=0.3),
        ]
        
        # Geometric augmentations (careful with bounding boxes)
        geometric_transforms = [
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, 
                             border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
            A.Perspective(scale=(0.05, 0.1), keep_size=True, p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
        ]
        
        # Combine all transforms
        all_transforms = (
            base_transforms + 
            weather_transforms + 
            environmental_transforms + 
            geometric_transforms
        )
        
        # Create different augmentation pipelines for different scenarios
        self.augmentation_pipelines = {
            'aerial': A.Compose([
                A.OneOf(weather_transforms, p=0.5),
                A.OneOf(base_transforms[:2], p=0.7),  # Flip and brightness
                A.OneOf(environmental_transforms[:3], p=0.4),  # Blur and noise
                A.Resize(self.config.target_size[0], self.config.target_size[1]),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
            
            'ground': A.Compose([
                A.OneOf(base_transforms, p=0.8),
                A.OneOf(environmental_transforms, p=0.6),
                A.OneOf(geometric_transforms[:2], p=0.3),  # Avoid elastic for ground
                A.Resize(self.config.target_size[0], self.config.target_size[1]),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
            
            'water': A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.4, p=0.8),
                A.HorizontalFlip(p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.4),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.6),
                A.Resize(self.config.target_size[0], self.config.target_size[1]),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
            
            'rubble': A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.9),
                A.GaussNoise(var_limit=(20.0, 60.0), p=0.5),
                A.Cutout(num_holes=12, max_h_size=96, max_w_size=96, p=0.4),
                A.HorizontalFlip(p=0.5),
                A.Resize(self.config.target_size[0], self.config.target_size[1]),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        }
        
        logger.info("Augmentation pipelines configured for SAR scenarios")
    
    def load_annotations(self, annotation_file: str) -> List[Dict]:
        """Load annotations from various formats (COCO, YOLO, custom)."""
        
        annotation_path = Path(annotation_file)
        
        if annotation_path.suffix.lower() == '.json':
            # COCO format
            with open(annotation_path, 'r') as f:
                coco_data = json.load(f)
            return self.convert_coco_to_yolo(coco_data)
        
        elif annotation_path.suffix.lower() == '.txt':
            # YOLO format
            return self.load_yolo_annotations(annotation_path)
        
        else:
            raise ValueError(f"Unsupported annotation format: {annotation_path.suffix}")
    
    def convert_coco_to_yolo(self, coco_data: Dict) -> List[Dict]:
        """Convert COCO format annotations to YOLO format."""
        
        annotations = []
        
        # Create image ID to filename mapping
        image_info = {img['id']: img for img in coco_data['images']}
        
        # Group annotations by image
        image_annotations = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(ann)
        
        # Convert each image's annotations
        for image_id, anns in image_annotations.items():
            img_info = image_info[image_id]
            img_width = img_info['width']
            img_height = img_info['height']
            
            yolo_boxes = []
            class_labels = []
            
            for ann in anns:
                # Convert COCO bbox [x, y, width, height] to YOLO [x_center, y_center, width, height]
                x, y, w, h = ann['bbox']
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                norm_width = w / img_width
                norm_height = h / img_height
                
                # Map category ID to our class system
                category_id = ann['category_id']
                class_name = self.map_category_to_sar_class(category_id, coco_data['categories'])
                
                if class_name in self.config.classes:
                    yolo_boxes.append([x_center, y_center, norm_width, norm_height])
                    class_labels.append(self.config.classes[class_name])
            
            if yolo_boxes:  # Only include images with valid annotations
                annotations.append({
                    'image_path': img_info['file_name'],
                    'boxes': yolo_boxes,
                    'class_labels': class_labels,
                    'scenario': self.detect_scenario(img_info['file_name'])
                })
        
        return annotations
    
    def map_category_to_sar_class(self, category_id: int, categories: List[Dict]) -> str:
        """Map COCO category to SAR class."""
        
        # Find category name
        category_name = None
        for cat in categories:
            if cat['id'] == category_id:
                category_name = cat['name'].lower()
                break
        
        if not category_name:
            return 'unknown'
        
        # Mapping logic
        if 'person' in category_name or 'human' in category_name:
            return 'person'
        elif any(vehicle in category_name for vehicle in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']):
            return 'vehicle'
        elif any(aircraft in category_name for aircraft in ['airplane', 'helicopter', 'drone']):
            return 'aircraft'
        elif 'boat' in category_name or 'ship' in category_name:
            return 'boat'
        elif any(structure in category_name for structure in ['building', 'house', 'tent']):
            return 'structure'
        elif 'debris' in category_name or 'rubble' in category_name:
            return 'debris'
        elif any(equipment in category_name for equipment in ['equipment', 'tool', 'gear']):
            return 'equipment'
        elif any(animal in category_name for animal in ['dog', 'cat', 'horse', 'cow', 'animal']):
            return 'animal'
        elif any(signal in category_name for signal in ['flag', 'flare', 'signal', 'mirror']):
            return 'signal'
        elif 'clothing' in category_name or 'clothes' in category_name:
            return 'clothing'
        else:
            return 'unknown'
    
    def detect_scenario(self, image_path: str) -> str:
        """Detect scenario type from image path or metadata."""
        
        path_lower = image_path.lower()
        
        if any(keyword in path_lower for keyword in ['aerial', 'drone', 'uav', 'bird']):
            return 'aerial'
        elif any(keyword in path_lower for keyword in ['water', 'ocean', 'lake', 'river', 'sea']):
            return 'water'
        elif any(keyword in path_lower for keyword in ['rubble', 'debris', 'disaster', 'earthquake']):
            return 'rubble'
        else:
            return 'ground'
    
    def augment_image(self, image: np.ndarray, boxes: List[List[float]], 
                     class_labels: List[int], scenario: str) -> List[Tuple[np.ndarray, List[List[float]], List[int]]]:
        """Apply augmentations to image and annotations."""
        
        augmented_data = []
        pipeline = self.augmentation_pipelines.get(scenario, self.augmentation_pipelines['ground'])
        
        for _ in range(self.config.augmentation_factor):
            try:
                augmented = pipeline(
                    image=image,
                    bboxes=boxes,
                    class_labels=class_labels
                )
                
                # Filter out boxes that are too small after augmentation
                valid_boxes = []
                valid_labels = []
                
                for box, label in zip(augmented['bboxes'], augmented['class_labels']):
                    x_center, y_center, width, height = box
                    
                    # Convert to pixel coordinates to check size
                    pixel_width = width * self.config.target_size[0]
                    pixel_height = height * self.config.target_size[1]
                    
                    if pixel_width >= self.config.min_object_size and pixel_height >= self.config.min_object_size:
                        valid_boxes.append(box)
                        valid_labels.append(label)
                
                if valid_boxes:  # Only keep augmentation if it has valid boxes
                    augmented_data.append((
                        augmented['image'],
                        valid_boxes,
                        valid_labels
                    ))
            
            except Exception as e:
                logger.warning(f"Augmentation failed: {e}")
                continue
        
        return augmented_data
    
    def save_yolo_annotation(self, boxes: List[List[float]], class_labels: List[int], 
                           output_path: str):
        """Save annotations in YOLO format."""
        
        with open(output_path, 'w') as f:
            for box, class_id in zip(boxes, class_labels):
                x_center, y_center, width, height = box
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def split_dataset(self, annotations: List[Dict]) -> Dict[str, List[Dict]]:
        """Split dataset into train/val/test sets."""
        
        # Shuffle annotations
        random.shuffle(annotations)
        
        total_count = len(annotations)
        train_count = int(total_count * self.config.train_ratio)
        val_count = int(total_count * self.config.val_ratio)
        
        splits = {
            'train': annotations[:train_count],
            'val': annotations[train_count:train_count + val_count],
            'test': annotations[train_count + val_count:]
        }
        
        logger.info(f"Dataset split: train={len(splits['train'])}, "
                   f"val={len(splits['val'])}, test={len(splits['test'])}")
        
        return splits
    
    def process_dataset(self, source_annotations: str):
        """Main processing pipeline."""
        
        logger.info("Starting SAR dataset preparation...")
        
        # Load annotations
        annotations = self.load_annotations(source_annotations)
        logger.info(f"Loaded {len(annotations)} annotations")
        
        # Split dataset
        splits = self.split_dataset(annotations)
        
        # Process each split
        for split_name, split_annotations in splits.items():
            logger.info(f"Processing {split_name} split ({len(split_annotations)} images)...")
            
            split_stats = {
                'total_images': 0,
                'total_augmented': 0,
                'class_distribution': {class_name: 0 for class_name in self.config.classes.keys()},
                'scenario_distribution': {'aerial': 0, 'ground': 0, 'water': 0, 'rubble': 0}
            }
            
            for i, annotation in enumerate(tqdm(split_annotations, desc=f"Processing {split_name}")):
                try:
                    # Load image
                    image_path = Path(self.config.source_data_dir) / annotation['image_path']
                    if not image_path.exists():
                        logger.warning(f"Image not found: {image_path}")
                        continue
                    
                    image = cv2.imread(str(image_path))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Original image
                    original_name = f"{split_name}_{i:06d}_original"
                    self.save_processed_image(
                        image, annotation['boxes'], annotation['class_labels'],
                        split_name, original_name
                    )
                    
                    split_stats['total_images'] += 1
                    split_stats['scenario_distribution'][annotation['scenario']] += 1
                    
                    # Count classes
                    for class_id in annotation['class_labels']:
                        class_name = list(self.config.classes.keys())[class_id]
                        split_stats['class_distribution'][class_name] += 1
                    
                    # Augmented versions
                    if split_name == 'train':  # Only augment training data
                        augmented_data = self.augment_image(
                            image, annotation['boxes'], annotation['class_labels'],
                            annotation['scenario']
                        )
                        
                        for j, (aug_image, aug_boxes, aug_labels) in enumerate(augmented_data):
                            aug_name = f"{split_name}_{i:06d}_aug_{j:02d}"
                            self.save_processed_image(
                                aug_image, aug_boxes, aug_labels,
                                split_name, aug_name
                            )
                            
                            split_stats['total_augmented'] += 1
                            
                            # Count augmented classes
                            for class_id in aug_labels:
                                class_name = list(self.config.classes.keys())[class_id]
                                split_stats['class_distribution'][class_name] += 1
                
                except Exception as e:
                    logger.error(f"Error processing {annotation['image_path']}: {e}")
                    continue
            
            # Save split statistics
            stats_path = Path(self.config.output_dir) / 'metadata' / f'{split_name}_stats.json'
            with open(stats_path, 'w') as f:
                json.dump(split_stats, f, indent=2)
            
            logger.info(f"Completed {split_name}: {split_stats['total_images']} original + "
                       f"{split_stats['total_augmented']} augmented images")
        
        # Generate final dataset statistics
        self.generate_dataset_summary()
        logger.info("SAR dataset preparation completed!")
    
    def save_processed_image(self, image: np.ndarray, boxes: List[List[float]], 
                           class_labels: List[int], split: str, name: str):
        """Save processed image and annotations."""
        
        # Save image
        image_path = Path(self.config.output_dir) / 'images' / split / f'{name}.jpg'
        image_pil = Image.fromarray(image)
        image_pil.save(image_path, quality=95)
        
        # Save annotations
        label_path = Path(self.config.output_dir) / 'labels' / split / f'{name}.txt'
        self.save_yolo_annotation(boxes, class_labels, str(label_path))
    
    def generate_dataset_summary(self):
        """Generate comprehensive dataset summary."""
        
        summary = {
            'dataset_name': 'SAR YOLOv8 Dataset',
            'created_at': datetime.now().isoformat(),
            'config': {
                'target_size': self.config.target_size,
                'augmentation_factor': self.config.augmentation_factor,
                'min_object_size': self.config.min_object_size
            },
            'classes': self.config.classes,
            'splits': {}
        }
        
        # Load split statistics
        for split in ['train', 'val', 'test']:
            stats_path = Path(self.config.output_dir) / 'metadata' / f'{split}_stats.json'
            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    summary['splits'][split] = json.load(f)
        
        # Save summary
        summary_path = Path(self.config.output_dir) / 'metadata' / 'dataset_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate visualization
        self.create_dataset_visualization(summary)
        
        logger.info(f"Dataset summary saved to {summary_path}")
    
    def create_dataset_visualization(self, summary: Dict):
        """Create visualization of dataset statistics."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SAR Dataset Statistics', fontsize=16)
        
        # Class distribution across all splits
        all_classes = {}
        for split_data in summary['splits'].values():
            for class_name, count in split_data['class_distribution'].items():
                all_classes[class_name] = all_classes.get(class_name, 0) + count
        
        axes[0, 0].bar(all_classes.keys(), all_classes.values())
        axes[0, 0].set_title('Class Distribution (All Splits)')
        axes[0, 0].set_xlabel('Classes')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Split distribution
        split_counts = {split: data['total_images'] + data.get('total_augmented', 0) 
                       for split, data in summary['splits'].items()}
        axes[0, 1].pie(split_counts.values(), labels=split_counts.keys(), autopct='%1.1f%%')
        axes[0, 1].set_title('Dataset Split Distribution')
        
        # Scenario distribution
        all_scenarios = {}
        for split_data in summary['splits'].values():
            for scenario, count in split_data.get('scenario_distribution', {}).items():
                all_scenarios[scenario] = all_scenarios.get(scenario, 0) + count
        
        axes[1, 0].bar(all_scenarios.keys(), all_scenarios.values())
        axes[1, 0].set_title('Scenario Distribution')
        axes[1, 0].set_xlabel('Scenarios')
        axes[1, 0].set_ylabel('Count')
        
        # Augmentation impact (train split only)
        if 'train' in summary['splits']:
            train_data = summary['splits']['train']
            original = train_data['total_images']
            augmented = train_data.get('total_augmented', 0)
            
            axes[1, 1].bar(['Original', 'Augmented'], [original, augmented])
            axes[1, 1].set_title('Training Data Augmentation')
            axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = Path(self.config.output_dir) / 'metadata' / 'dataset_visualization.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Dataset visualization saved to {viz_path}")

def main():
    """Main execution function."""
    
    # Configuration
    config = SARDatasetConfig(
        source_data_dir="data/raw",
        output_dir="data/training",
        target_size=(1280, 1280),
        augmentation_factor=3
    )
    
    # Initialize preparator
    preparator = SARDatasetPreparator(config)
    
    # Process dataset (assuming COCO format annotations)
    # Replace with your actual annotation file path
    annotation_file = "data/raw/annotations.json"
    
    if not Path(annotation_file).exists():
        logger.error(f"Annotation file not found: {annotation_file}")
        logger.info("Please provide annotations in COCO format or modify the script for your format")
        return
    
    preparator.process_dataset(annotation_file)

if __name__ == "__main__":
    main()