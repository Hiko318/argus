#!/usr/bin/env python3
"""
Dataset Converter for SAR YOLOv8 Training

This script provides utilities to convert various annotation formats
to YOLOv8 format for SAR training datasets.

Supported input formats:
- COCO JSON
- Pascal VOC XML
- YOLO v5 format
- Custom CSV format
- LabelMe JSON

Usage:
    python dataset_converter.py --input datasets/coco --format coco --output datasets/sar_dataset
    python dataset_converter.py --input annotations.csv --format csv --output datasets/sar_dataset
    python dataset_converter.py --input voc_dataset --format voc --output datasets/sar_dataset --split 0.8 0.1 0.1
"""

import argparse
import os
import json
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
import shutil
import random
from collections import defaultdict

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetConverter:
    """
    Universal dataset converter for SAR YOLOv8 training.
    Converts various annotation formats to YOLOv8 format.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize converter.
        
        Args:
            output_dir: Output directory for converted dataset
        """
        self.output_dir = Path(output_dir)
        self.class_mapping = {
            'person': 0,
            'human': 0,
            'people': 0,
            'pedestrian': 0,
            'vehicle': 1,
            'car': 1,
            'truck': 1,
            'motorcycle': 1,
            'bicycle': 1,
            'structure': 2,
            'building': 2,
            'house': 2,
            'debris': 3,
            'wreckage': 3,
            'damage': 3
        }
        self.stats = defaultdict(int)
        
    def setup_output_structure(self):
        """Create YOLOv8 dataset directory structure."""
        dirs = [
            'images/train',
            'images/val', 
            'images/test',
            'labels/train',
            'labels/val',
            'labels/test'
        ]
        
        for dir_path in dirs:
            (self.output_dir / dir_path).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created dataset structure in {self.output_dir}")
    
    def create_dataset_yaml(self, class_names: Dict[int, str] = None):
        """Create dataset.yaml file for YOLOv8."""
        if class_names is None:
            class_names = {0: 'person', 1: 'vehicle', 2: 'structure', 3: 'debris'}
        
        dataset_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(class_names),
            'names': class_names
        }
        
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            import yaml
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        logger.info(f"Created dataset.yaml: {yaml_path}")
    
    def convert_coco(self, coco_path: str, split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
        """
        Convert COCO format dataset to YOLOv8 format.
        
        Args:
            coco_path: Path to COCO dataset directory
            split_ratios: Train/val/test split ratios
        """
        logger.info(f"Converting COCO dataset from {coco_path}")
        
        coco_path = Path(coco_path)
        
        # Load COCO annotations
        annotations_file = coco_path / 'annotations' / 'instances_train2017.json'
        if not annotations_file.exists():
            # Try alternative paths
            for alt_path in ['annotations.json', 'instances.json']:
                alt_file = coco_path / alt_path
                if alt_file.exists():
                    annotations_file = alt_file
                    break
        
        if not annotations_file.exists():
            raise FileNotFoundError(f"COCO annotations file not found in {coco_path}")
        
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Build category mapping
        category_map = {}
        for cat in coco_data['categories']:
            cat_name = cat['name'].lower()
            if cat_name in self.class_mapping:
                category_map[cat['id']] = self.class_mapping[cat_name]
        
        # Process images and annotations
        image_annotations = defaultdict(list)
        for ann in coco_data['annotations']:
            if ann['category_id'] in category_map:
                image_annotations[ann['image_id']].append(ann)
        
        # Convert images
        image_files = []
        for img_info in coco_data['images']:
            if img_info['id'] in image_annotations:
                image_files.append((img_info, image_annotations[img_info['id']]))
        
        # Split dataset
        train_data, temp_data = train_test_split(image_files, test_size=1-split_ratios[0], random_state=42)
        val_ratio = split_ratios[1] / (split_ratios[1] + split_ratios[2])
        val_data, test_data = train_test_split(temp_data, test_size=1-val_ratio, random_state=42)
        
        # Process each split
        splits = {'train': train_data, 'val': val_data, 'test': test_data}
        
        for split_name, split_data in splits.items():
            logger.info(f"Processing {split_name} split: {len(split_data)} images")
            
            for img_info, annotations in split_data:
                # Copy image
                img_filename = img_info['file_name']
                src_img_path = coco_path / 'images' / img_filename
                if not src_img_path.exists():
                    # Try alternative image directories
                    for img_dir in ['train2017', 'val2017', 'test2017', '.']:
                        alt_path = coco_path / img_dir / img_filename
                        if alt_path.exists():
                            src_img_path = alt_path
                            break
                
                if not src_img_path.exists():
                    logger.warning(f"Image not found: {img_filename}")
                    continue
                
                dst_img_path = self.output_dir / 'images' / split_name / img_filename
                shutil.copy2(src_img_path, dst_img_path)
                
                # Convert annotations
                yolo_annotations = []
                img_width = img_info['width']
                img_height = img_info['height']
                
                for ann in annotations:
                    if ann['category_id'] not in category_map:
                        continue
                    
                    class_id = category_map[ann['category_id']]
                    bbox = ann['bbox']  # [x, y, width, height]
                    
                    # Convert to YOLO format (normalized center coordinates)
                    x_center = (bbox[0] + bbox[2] / 2) / img_width
                    y_center = (bbox[1] + bbox[3] / 2) / img_height
                    width = bbox[2] / img_width
                    height = bbox[3] / img_height
                    
                    yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                    self.stats[f'{split_name}_annotations'] += 1
                
                # Save label file
                label_filename = Path(img_filename).stem + '.txt'
                label_path = self.output_dir / 'labels' / split_name / label_filename
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                
                self.stats[f'{split_name}_images'] += 1
        
        logger.info(f"COCO conversion completed. Stats: {dict(self.stats)}")
    
    def convert_voc(self, voc_path: str, split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
        """
        Convert Pascal VOC format dataset to YOLOv8 format.
        
        Args:
            voc_path: Path to VOC dataset directory
            split_ratios: Train/val/test split ratios
        """
        logger.info(f"Converting VOC dataset from {voc_path}")
        
        voc_path = Path(voc_path)
        annotations_dir = voc_path / 'Annotations'
        images_dir = voc_path / 'JPEGImages'
        
        if not annotations_dir.exists() or not images_dir.exists():
            raise FileNotFoundError(f"VOC structure not found in {voc_path}")
        
        # Get all annotation files
        xml_files = list(annotations_dir.glob('*.xml'))
        
        # Split dataset
        train_files, temp_files = train_test_split(xml_files, test_size=1-split_ratios[0], random_state=42)
        val_ratio = split_ratios[1] / (split_ratios[1] + split_ratios[2])
        val_files, test_files = train_test_split(temp_files, test_size=1-val_ratio, random_state=42)
        
        splits = {'train': train_files, 'val': val_files, 'test': test_files}
        
        for split_name, xml_files in splits.items():
            logger.info(f"Processing {split_name} split: {len(xml_files)} files")
            
            for xml_file in xml_files:
                # Parse XML
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # Get image info
                filename = root.find('filename').text
                size = root.find('size')
                img_width = int(size.find('width').text)
                img_height = int(size.find('height').text)
                
                # Copy image
                src_img_path = images_dir / filename
                if not src_img_path.exists():
                    logger.warning(f"Image not found: {filename}")
                    continue
                
                dst_img_path = self.output_dir / 'images' / split_name / filename
                shutil.copy2(src_img_path, dst_img_path)
                
                # Convert annotations
                yolo_annotations = []
                for obj in root.findall('object'):
                    class_name = obj.find('name').text.lower()
                    if class_name not in self.class_mapping:
                        continue
                    
                    class_id = self.class_mapping[class_name]
                    bbox = obj.find('bndbox')
                    
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)
                    
                    # Convert to YOLO format
                    x_center = (xmin + xmax) / 2 / img_width
                    y_center = (ymin + ymax) / 2 / img_height
                    width = (xmax - xmin) / img_width
                    height = (ymax - ymin) / img_height
                    
                    yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                    self.stats[f'{split_name}_annotations'] += 1
                
                # Save label file
                label_filename = Path(filename).stem + '.txt'
                label_path = self.output_dir / 'labels' / split_name / label_filename
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                
                self.stats[f'{split_name}_images'] += 1
        
        logger.info(f"VOC conversion completed. Stats: {dict(self.stats)}")
    
    def convert_csv(self, csv_path: str, images_dir: str, split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
        """
        Convert CSV format dataset to YOLOv8 format.
        
        Expected CSV format:
        filename,class,xmin,ymin,xmax,ymax,width,height
        
        Args:
            csv_path: Path to CSV annotations file
            images_dir: Directory containing images
            split_ratios: Train/val/test split ratios
        """
        logger.info(f"Converting CSV dataset from {csv_path}")
        
        images_dir = Path(images_dir)
        
        # Read CSV
        annotations = defaultdict(list)
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row['filename']
                class_name = row['class'].lower()
                
                if class_name in self.class_mapping:
                    annotations[filename].append({
                        'class_id': self.class_mapping[class_name],
                        'xmin': float(row['xmin']),
                        'ymin': float(row['ymin']),
                        'xmax': float(row['xmax']),
                        'ymax': float(row['ymax']),
                        'img_width': float(row['width']),
                        'img_height': float(row['height'])
                    })
        
        # Split dataset
        image_files = list(annotations.keys())
        train_files, temp_files = train_test_split(image_files, test_size=1-split_ratios[0], random_state=42)
        val_ratio = split_ratios[1] / (split_ratios[1] + split_ratios[2])
        val_files, test_files = train_test_split(temp_files, test_size=1-val_ratio, random_state=42)
        
        splits = {'train': train_files, 'val': val_files, 'test': test_files}
        
        for split_name, files in splits.items():
            logger.info(f"Processing {split_name} split: {len(files)} files")
            
            for filename in files:
                # Copy image
                src_img_path = images_dir / filename
                if not src_img_path.exists():
                    logger.warning(f"Image not found: {filename}")
                    continue
                
                dst_img_path = self.output_dir / 'images' / split_name / filename
                shutil.copy2(src_img_path, dst_img_path)
                
                # Convert annotations
                yolo_annotations = []
                for ann in annotations[filename]:
                    # Convert to YOLO format
                    x_center = (ann['xmin'] + ann['xmax']) / 2 / ann['img_width']
                    y_center = (ann['ymin'] + ann['ymax']) / 2 / ann['img_height']
                    width = (ann['xmax'] - ann['xmin']) / ann['img_width']
                    height = (ann['ymax'] - ann['ymin']) / ann['img_height']
                    
                    yolo_annotations.append(f"{ann['class_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                    self.stats[f'{split_name}_annotations'] += 1
                
                # Save label file
                label_filename = Path(filename).stem + '.txt'
                label_path = self.output_dir / 'labels' / split_name / label_filename
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                
                self.stats[f'{split_name}_images'] += 1
        
        logger.info(f"CSV conversion completed. Stats: {dict(self.stats)}")
    
    def convert_yolo_v5(self, yolo_path: str):
        """
        Convert YOLOv5 format dataset to YOLOv8 format (mostly just copy).
        
        Args:
            yolo_path: Path to YOLOv5 dataset directory
        """
        logger.info(f"Converting YOLOv5 dataset from {yolo_path}")
        
        yolo_path = Path(yolo_path)
        
        # Copy directory structure
        for split in ['train', 'val', 'test']:
            src_images = yolo_path / 'images' / split
            src_labels = yolo_path / 'labels' / split
            
            if src_images.exists():
                dst_images = self.output_dir / 'images' / split
                shutil.copytree(src_images, dst_images, dirs_exist_ok=True)
                self.stats[f'{split}_images'] = len(list(dst_images.glob('*')))
            
            if src_labels.exists():
                dst_labels = self.output_dir / 'labels' / split
                shutil.copytree(src_labels, dst_labels, dirs_exist_ok=True)
                self.stats[f'{split}_annotations'] = sum(1 for f in dst_labels.glob('*.txt') if f.stat().st_size > 0)
        
        # Copy dataset.yaml if exists
        src_yaml = yolo_path / 'dataset.yaml'
        if src_yaml.exists():
            dst_yaml = self.output_dir / 'dataset.yaml'
            shutil.copy2(src_yaml, dst_yaml)
        
        logger.info(f"YOLOv5 conversion completed. Stats: {dict(self.stats)}")
    
    def augment_dataset(self, augmentation_factor: int = 2):
        """
        Apply data augmentation to increase dataset size.
        
        Args:
            augmentation_factor: Number of augmented versions per image
        """
        logger.info(f"Applying data augmentation with factor {augmentation_factor}")
        
        train_images_dir = self.output_dir / 'images' / 'train'
        train_labels_dir = self.output_dir / 'labels' / 'train'
        
        image_files = list(train_images_dir.glob('*'))
        
        for img_path in image_files:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Load corresponding label
            label_path = train_labels_dir / (img_path.stem + '.txt')
            if not label_path.exists():
                continue
            
            with open(label_path, 'r') as f:
                labels = f.read().strip().split('\n')
            
            # Apply augmentations
            for i in range(augmentation_factor):
                aug_img, aug_labels = self._apply_augmentation(img, labels)
                
                # Save augmented image
                aug_img_name = f"{img_path.stem}_aug_{i}{img_path.suffix}"
                aug_img_path = train_images_dir / aug_img_name
                cv2.imwrite(str(aug_img_path), aug_img)
                
                # Save augmented labels
                aug_label_name = f"{img_path.stem}_aug_{i}.txt"
                aug_label_path = train_labels_dir / aug_label_name
                with open(aug_label_path, 'w') as f:
                    f.write('\n'.join(aug_labels))
                
                self.stats['augmented_images'] += 1
        
        logger.info(f"Data augmentation completed. Generated {self.stats['augmented_images']} augmented images")
    
    def _apply_augmentation(self, img: np.ndarray, labels: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Apply random augmentation to image and adjust labels accordingly.
        
        Args:
            img: Input image
            labels: YOLO format labels
            
        Returns:
            Augmented image and adjusted labels
        """
        h, w = img.shape[:2]
        
        # Random horizontal flip
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            # Adjust labels for horizontal flip
            new_labels = []
            for label in labels:
                if label.strip():
                    parts = label.split()
                    class_id = parts[0]
                    x_center = 1.0 - float(parts[1])  # Flip x coordinate
                    y_center = parts[2]
                    width = parts[3]
                    height = parts[4]
                    new_labels.append(f"{class_id} {x_center:.6f} {y_center} {width} {height}")
            labels = new_labels
        
        # Random brightness adjustment
        brightness = random.uniform(0.8, 1.2)
        img = cv2.convertScaleAbs(img, alpha=brightness, beta=0)
        
        # Random noise
        if random.random() > 0.7:
            noise = np.random.randint(0, 25, img.shape, dtype=np.uint8)
            img = cv2.add(img, noise)
        
        return img, labels
    
    def validate_dataset(self) -> Dict[str, Any]:
        """
        Validate converted dataset structure and content.
        
        Returns:
            Validation results
        """
        logger.info("Validating converted dataset...")
        
        validation_results = {
            'structure_valid': True,
            'issues': [],
            'statistics': {}
        }
        
        # Check directory structure
        required_dirs = [
            'images/train', 'images/val', 'images/test',
            'labels/train', 'labels/val', 'labels/test'
        ]
        
        for dir_path in required_dirs:
            full_path = self.output_dir / dir_path
            if not full_path.exists():
                validation_results['structure_valid'] = False
                validation_results['issues'].append(f"Missing directory: {dir_path}")
        
        # Check dataset.yaml
        yaml_path = self.output_dir / 'dataset.yaml'
        if not yaml_path.exists():
            validation_results['issues'].append("Missing dataset.yaml file")
        
        # Count files and check consistency
        for split in ['train', 'val', 'test']:
            images_dir = self.output_dir / 'images' / split
            labels_dir = self.output_dir / 'labels' / split
            
            if images_dir.exists() and labels_dir.exists():
                image_files = set(f.stem for f in images_dir.glob('*'))
                label_files = set(f.stem for f in labels_dir.glob('*.txt'))
                
                validation_results['statistics'][f'{split}_images'] = len(image_files)
                validation_results['statistics'][f'{split}_labels'] = len(label_files)
                
                # Check for missing labels
                missing_labels = image_files - label_files
                if missing_labels:
                    validation_results['issues'].append(
                        f"{split}: {len(missing_labels)} images missing labels"
                    )
                
                # Check for orphaned labels
                orphaned_labels = label_files - image_files
                if orphaned_labels:
                    validation_results['issues'].append(
                        f"{split}: {len(orphaned_labels)} labels without images"
                    )
        
        validation_results['valid'] = validation_results['structure_valid'] and len(validation_results['issues']) == 0
        
        if validation_results['valid']:
            logger.info("Dataset validation passed")
        else:
            logger.warning(f"Dataset validation issues: {validation_results['issues']}")
        
        return validation_results

def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(description='Dataset Converter for SAR YOLOv8')
    parser.add_argument('--input', type=str, required=True,
                       help='Input dataset path')
    parser.add_argument('--format', type=str, required=True,
                       choices=['coco', 'voc', 'csv', 'yolo_v5'],
                       help='Input dataset format')
    parser.add_argument('--output', type=str, required=True,
                       help='Output dataset directory')
    parser.add_argument('--images-dir', type=str, default=None,
                       help='Images directory (for CSV format)')
    parser.add_argument('--split', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                       help='Train/val/test split ratios')
    parser.add_argument('--augment', type=int, default=0,
                       help='Data augmentation factor')
    parser.add_argument('--validate', action='store_true',
                       help='Validate converted dataset')
    
    args = parser.parse_args()
    
    # Validate split ratios
    if abs(sum(args.split) - 1.0) > 0.001:
        logger.error("Split ratios must sum to 1.0")
        return
    
    # Initialize converter
    converter = DatasetConverter(args.output)
    converter.setup_output_structure()
    
    try:
        # Convert dataset based on format
        if args.format == 'coco':
            converter.convert_coco(args.input, tuple(args.split))
        elif args.format == 'voc':
            converter.convert_voc(args.input, tuple(args.split))
        elif args.format == 'csv':
            if not args.images_dir:
                logger.error("--images-dir required for CSV format")
                return
            converter.convert_csv(args.input, args.images_dir, tuple(args.split))
        elif args.format == 'yolo_v5':
            converter.convert_yolo_v5(args.input)
        
        # Create dataset.yaml
        converter.create_dataset_yaml()
        
        # Apply augmentation if requested
        if args.augment > 0:
            converter.augment_dataset(args.augment)
        
        # Validate if requested
        if args.validate:
            validation_results = converter.validate_dataset()
            print(f"\nValidation Results: {json.dumps(validation_results, indent=2)}")
        
        logger.info(f"Dataset conversion completed successfully!")
        logger.info(f"Final statistics: {dict(converter.stats)}")
        
    except Exception as e:
        logger.error(f"Dataset conversion failed: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())