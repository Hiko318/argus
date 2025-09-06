#!/usr/bin/env python3
"""
Foresight SAR Dataset Preparation Script

This script converts various dataset formats to YOLO format for training
aerial people detection models. Supports multiple input formats and provides
data augmentation capabilities.

Author: Foresight SAR Team
Date: 2024
"""

import os
import sys
import json
import yaml
import shutil
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import random
import math

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter
    import albumentations as A
    from sklearn.model_selection import train_test_split
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install: pip install opencv-python pillow albumentations scikit-learn")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dataset_prep.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BoundingBox:
    """Represents a bounding box annotation."""
    x_center: float
    y_center: float
    width: float
    height: float
    class_id: int
    confidence: float = 1.0
    
    def to_yolo_format(self) -> str:
        """Convert to YOLO format string."""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"
    
    def to_absolute(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Convert to absolute coordinates (x1, y1, x2, y2)."""
        x1 = int((self.x_center - self.width / 2) * img_width)
        y1 = int((self.y_center - self.height / 2) * img_height)
        x2 = int((self.x_center + self.width / 2) * img_width)
        y2 = int((self.y_center + self.height / 2) * img_height)
        return x1, y1, x2, y2
    
    @classmethod
    def from_absolute(cls, x1: int, y1: int, x2: int, y2: int, 
                     img_width: int, img_height: int, class_id: int) -> 'BoundingBox':
        """Create from absolute coordinates."""
        x_center = (x1 + x2) / 2 / img_width
        y_center = (y1 + y2) / 2 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        return cls(x_center, y_center, width, height, class_id)

@dataclass
class DatasetSample:
    """Represents a single dataset sample."""
    image_path: Path
    annotations: List[BoundingBox]
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class DatasetConverter:
    """Base class for dataset format converters."""
    
    def __init__(self, class_mapping: Dict[str, int] = None):
        self.class_mapping = class_mapping or {'person': 0}
        self.reverse_mapping = {v: k for k, v in self.class_mapping.items()}
    
    def convert(self, input_path: Path, output_path: Path) -> List[DatasetSample]:
        """Convert dataset to internal format."""
        raise NotImplementedError

class COCOConverter(DatasetConverter):
    """Converter for COCO format datasets."""
    
    def convert(self, input_path: Path, output_path: Path) -> List[DatasetSample]:
        """Convert COCO dataset to internal format."""
        logger.info(f"Converting COCO dataset from {input_path}")
        
        # Load COCO annotations
        annotations_file = input_path / "annotations.json"
        if not annotations_file.exists():
            raise FileNotFoundError(f"COCO annotations file not found: {annotations_file}")
        
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Build mappings
        image_info = {img['id']: img for img in coco_data['images']}
        category_info = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Group annotations by image
        image_annotations = defaultdict(list)
        for ann in coco_data['annotations']:
            image_annotations[ann['image_id']].append(ann)
        
        samples = []
        for image_id, image_data in image_info.items():
            image_path = input_path / "images" / image_data['file_name']
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue
            
            annotations = []
            for ann in image_annotations[image_id]:
                category_name = category_info[ann['category_id']]
                if category_name in self.class_mapping:
                    x, y, w, h = ann['bbox']
                    bbox = BoundingBox.from_absolute(
                        x, y, x + w, y + h,
                        image_data['width'], image_data['height'],
                        self.class_mapping[category_name]
                    )
                    annotations.append(bbox)
            
            if annotations:  # Only include images with relevant annotations
                samples.append(DatasetSample(image_path, annotations))
        
        logger.info(f"Converted {len(samples)} samples from COCO format")
        return samples

class PascalVOCConverter(DatasetConverter):
    """Converter for Pascal VOC format datasets."""
    
    def convert(self, input_path: Path, output_path: Path) -> List[DatasetSample]:
        """Convert Pascal VOC dataset to internal format."""
        logger.info(f"Converting Pascal VOC dataset from {input_path}")
        
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            raise ImportError("xml.etree.ElementTree required for Pascal VOC conversion")
        
        annotations_dir = input_path / "Annotations"
        images_dir = input_path / "JPEGImages"
        
        if not annotations_dir.exists() or not images_dir.exists():
            raise FileNotFoundError("Pascal VOC directory structure not found")
        
        samples = []
        for xml_file in annotations_dir.glob("*.xml"):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get image info
            filename = root.find('filename').text
            image_path = images_dir / filename
            
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue
            
            size = root.find('size')
            img_width = int(size.find('width').text)
            img_height = int(size.find('height').text)
            
            # Parse annotations
            annotations = []
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name in self.class_mapping:
                    bbox_elem = obj.find('bndbox')
                    x1 = int(bbox_elem.find('xmin').text)
                    y1 = int(bbox_elem.find('ymin').text)
                    x2 = int(bbox_elem.find('xmax').text)
                    y2 = int(bbox_elem.find('ymax').text)
                    
                    bbox = BoundingBox.from_absolute(
                        x1, y1, x2, y2, img_width, img_height,
                        self.class_mapping[class_name]
                    )
                    annotations.append(bbox)
            
            if annotations:
                samples.append(DatasetSample(image_path, annotations))
        
        logger.info(f"Converted {len(samples)} samples from Pascal VOC format")
        return samples

class YOLOConverter(DatasetConverter):
    """Converter for existing YOLO format datasets."""
    
    def convert(self, input_path: Path, output_path: Path) -> List[DatasetSample]:
        """Convert YOLO dataset to internal format."""
        logger.info(f"Loading YOLO dataset from {input_path}")
        
        images_dir = input_path / "images"
        labels_dir = input_path / "labels"
        
        if not images_dir.exists():
            raise FileNotFoundError(f"YOLO images directory not found: {images_dir}")
        
        samples = []
        for image_file in images_dir.glob("*"):
            if image_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            
            label_file = labels_dir / f"{image_file.stem}.txt"
            annotations = []
            
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            bbox = BoundingBox(x_center, y_center, width, height, class_id)
                            annotations.append(bbox)
            
            samples.append(DatasetSample(image_file, annotations))
        
        logger.info(f"Loaded {len(samples)} samples from YOLO format")
        return samples

class DatasetAugmenter:
    """Handles dataset augmentation for aerial imagery."""
    
    def __init__(self, augmentation_factor: int = 2, seed: int = 42):
        self.augmentation_factor = augmentation_factor
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Define augmentation pipeline for aerial imagery
        self.transform = A.Compose([
            A.OneOf([
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=15, p=0.5),
            ], p=0.7),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5),
            ], p=0.6),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.3),
                A.MotionBlur(blur_limit=3, p=0.3),
            ], p=0.4),
            A.OneOf([
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),
                A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, p=0.2),
            ], p=0.3),
            A.RandomScale(scale_limit=0.1, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),  # Less common for aerial imagery
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def augment_sample(self, sample: DatasetSample) -> List[DatasetSample]:
        """Generate augmented versions of a sample."""
        augmented_samples = [sample]  # Include original
        
        # Load image
        image = cv2.imread(str(sample.image_path))
        if image is None:
            logger.warning(f"Could not load image: {sample.image_path}")
            return augmented_samples
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Prepare bboxes and labels for albumentations
        bboxes = []
        class_labels = []
        for bbox in sample.annotations:
            bboxes.append([bbox.x_center, bbox.y_center, bbox.width, bbox.height])
            class_labels.append(bbox.class_id)
        
        # Generate augmented versions
        for i in range(self.augmentation_factor - 1):
            try:
                transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
                
                # Create new annotations
                new_annotations = []
                for bbox_coords, class_id in zip(transformed['bboxes'], transformed['class_labels']):
                    new_bbox = BoundingBox(
                        x_center=bbox_coords[0],
                        y_center=bbox_coords[1],
                        width=bbox_coords[2],
                        height=bbox_coords[3],
                        class_id=class_id
                    )
                    new_annotations.append(new_bbox)
                
                # Create augmented sample with temporary path
                aug_sample = DatasetSample(
                    image_path=sample.image_path.parent / f"{sample.image_path.stem}_aug_{i}{sample.image_path.suffix}",
                    annotations=new_annotations,
                    metadata={'augmented': True, 'original': str(sample.image_path)}
                )
                
                # Save augmented image
                aug_image = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(aug_sample.image_path), aug_image)
                
                augmented_samples.append(aug_sample)
                
            except Exception as e:
                logger.warning(f"Failed to augment sample {sample.image_path}: {e}")
                continue
        
        return augmented_samples

class YOLODatasetBuilder:
    """Builds YOLO format datasets from converted samples."""
    
    def __init__(self, output_path: Path, class_names: List[str]):
        self.output_path = Path(output_path)
        self.class_names = class_names
        
        # Create directory structure
        for split in ['train', 'val', 'test']:
            (self.output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    def build_dataset(self, samples: List[DatasetSample], 
                     train_split: float = 0.7, val_split: float = 0.2, 
                     test_split: float = 0.1, seed: int = 42) -> Dict[str, int]:
        """Build YOLO dataset from samples."""
        logger.info(f"Building YOLO dataset with {len(samples)} samples")
        
        # Validate splits
        if abs(train_split + val_split + test_split - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test splits must sum to 1.0")
        
        # Split dataset
        random.seed(seed)
        random.shuffle(samples)
        
        n_train = int(len(samples) * train_split)
        n_val = int(len(samples) * val_split)
        
        train_samples = samples[:n_train]
        val_samples = samples[n_train:n_train + n_val]
        test_samples = samples[n_train + n_val:]
        
        # Process each split
        split_counts = {}
        for split_name, split_samples in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
            split_counts[split_name] = self._process_split(split_name, split_samples)
        
        # Create dataset.yaml
        self._create_dataset_yaml()
        
        logger.info(f"Dataset built successfully: {split_counts}")
        return split_counts
    
    def _process_split(self, split_name: str, samples: List[DatasetSample]) -> int:
        """Process a single dataset split."""
        logger.info(f"Processing {split_name} split with {len(samples)} samples")
        
        images_dir = self.output_path / 'images' / split_name
        labels_dir = self.output_path / 'labels' / split_name
        
        for i, sample in enumerate(samples):
            # Copy/move image
            target_image_path = images_dir / f"{split_name}_{i:06d}{sample.image_path.suffix}"
            
            if sample.image_path != target_image_path:
                if sample.metadata.get('augmented', False):
                    # Augmented images are already saved, just move
                    if sample.image_path.exists():
                        shutil.move(str(sample.image_path), str(target_image_path))
                else:
                    # Copy original images
                    shutil.copy2(str(sample.image_path), str(target_image_path))
            
            # Create label file
            label_path = labels_dir / f"{split_name}_{i:06d}.txt"
            with open(label_path, 'w') as f:
                for bbox in sample.annotations:
                    f.write(bbox.to_yolo_format() + '\n')
        
        return len(samples)
    
    def _create_dataset_yaml(self):
        """Create dataset.yaml configuration file."""
        dataset_config = {
            'path': str(self.output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        yaml_path = self.output_path / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        logger.info(f"Dataset configuration saved to: {yaml_path}")

def get_converter(format_type: str, class_mapping: Dict[str, int]) -> DatasetConverter:
    """Factory function to get appropriate converter."""
    converters = {
        'coco': COCOConverter,
        'pascal': PascalVOCConverter,
        'yolo': YOLOConverter
    }
    
    if format_type.lower() not in converters:
        raise ValueError(f"Unsupported format: {format_type}. Supported: {list(converters.keys())}")
    
    return converters[format_type.lower()](class_mapping)

def main():
    parser = argparse.ArgumentParser(
        description="Foresight SAR Dataset Preparation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert COCO dataset
  python dataset_prep.py --input /path/to/coco --output /path/to/yolo --format coco
  
  # Convert with augmentation
  python dataset_prep.py --input /path/to/data --output /path/to/yolo --format pascal --augment --aug-factor 3
  
  # Custom class mapping
  python dataset_prep.py --input /path/to/data --output /path/to/yolo --format coco --classes person:0 vehicle:1
        """
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input dataset directory')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output YOLO dataset directory')
    parser.add_argument('--format', '-f', type=str, required=True,
                       choices=['coco', 'pascal', 'yolo'],
                       help='Input dataset format')
    parser.add_argument('--classes', type=str, nargs='*',
                       default=['person:0'],
                       help='Class mapping in format "name:id" (default: person:0)')
    parser.add_argument('--train-split', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation set ratio (default: 0.2)')
    parser.add_argument('--test-split', type=float, default=0.1,
                       help='Test set ratio (default: 0.1)')
    parser.add_argument('--augment', action='store_true',
                       help='Enable data augmentation')
    parser.add_argument('--aug-factor', type=int, default=2,
                       help='Augmentation factor (default: 2)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse class mapping
    class_mapping = {}
    class_names = []
    for class_spec in args.classes:
        if ':' not in class_spec:
            raise ValueError(f"Invalid class specification: {class_spec}. Use format 'name:id'")
        name, class_id = class_spec.split(':', 1)
        class_mapping[name] = int(class_id)
        class_names.append(name)
    
    # Sort class names by ID
    class_names.sort(key=lambda x: class_mapping[x])
    
    logger.info(f"Class mapping: {class_mapping}")
    logger.info(f"Class names: {class_names}")
    
    # Validate paths
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Convert dataset
        converter = get_converter(args.format, class_mapping)
        samples = converter.convert(input_path, output_path)
        
        if not samples:
            logger.error("No samples found in input dataset")
            return 1
        
        # Apply augmentation if requested
        if args.augment:
            logger.info(f"Applying data augmentation with factor {args.aug_factor}")
            augmenter = DatasetAugmenter(args.aug_factor, args.seed)
            
            augmented_samples = []
            for sample in samples:
                augmented_samples.extend(augmenter.augment_sample(sample))
            
            samples = augmented_samples
            logger.info(f"Total samples after augmentation: {len(samples)}")
        
        # Build YOLO dataset
        builder = YOLODatasetBuilder(output_path, class_names)
        split_counts = builder.build_dataset(
            samples, args.train_split, args.val_split, args.test_split, args.seed
        )
        
        # Print summary
        logger.info("Dataset preparation completed successfully!")
        logger.info(f"Output directory: {output_path.absolute()}")
        logger.info(f"Dataset splits: {split_counts}")
        logger.info(f"Total samples: {sum(split_counts.values())}")
        logger.info(f"Classes: {len(class_names)} ({', '.join(class_names)})")
        
        return 0
        
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())