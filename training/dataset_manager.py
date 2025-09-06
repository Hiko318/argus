"""Dataset management for SAR training.

Handles dataset preparation, augmentation, splitting, and organization
for Search and Rescue specific training scenarios.
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import random
from dataclasses import dataclass
from collections import defaultdict, Counter

try:
    import cv2
    import numpy as np
    import yaml
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw, ImageFont
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Install with: pip install opencv-python numpy scikit-learn matplotlib pillow")


@dataclass
class DatasetInfo:
    """Dataset information and statistics."""
    name: str
    total_images: int
    total_annotations: int
    class_distribution: Dict[str, int]
    train_split: float
    val_split: float
    test_split: float
    image_sizes: List[Tuple[int, int]]
    annotation_density: float  # annotations per image
    

@dataclass
class AnnotationStats:
    """Statistics for annotations in dataset."""
    total_objects: int
    objects_per_class: Dict[str, int]
    bbox_sizes: Dict[str, List[float]]  # width, height statistics
    object_density_per_image: List[int]
    small_objects: int  # < 32x32 pixels
    medium_objects: int  # 32x32 to 96x96 pixels
    large_objects: int  # > 96x96 pixels
    

class DatasetManager:
    """Manage SAR training datasets."""
    
    def __init__(self, datasets_root: str = "datasets"):
        self.datasets_root = Path(datasets_root)
        self.datasets_root.mkdir(exist_ok=True)
        
        # SAR-specific class mapping
        self.sar_classes = {
            0: 'person',
            1: 'vehicle',
            2: 'debris',
            3: 'structure'
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def create_dataset_structure(self, dataset_name: str, 
                               train_ratio: float = 0.7,
                               val_ratio: float = 0.2,
                               test_ratio: float = 0.1) -> Path:
        """Create standard dataset directory structure."""
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        dataset_path = self.datasets_root / dataset_name
        
        # Create directory structure
        splits = ['train', 'val', 'test']
        subdirs = ['images', 'labels']
        
        for split in splits:
            for subdir in subdirs:
                (dataset_path / split / subdir).mkdir(parents=True, exist_ok=True)
        
        # Create dataset.yaml
        dataset_config = {
            'path': str(dataset_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.sar_classes),
            'names': list(self.sar_classes.values())
        }
        
        with open(dataset_path / 'dataset.yaml', 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        self.logger.info(f"Created dataset structure at {dataset_path}")
        return dataset_path
    
    def import_coco_dataset(self, coco_json_path: str, images_dir: str, 
                           dataset_name: str, class_mapping: Dict[int, str] = None) -> Path:
        """Import COCO format dataset and convert to YOLO format."""
        import json
        
        # Load COCO annotations
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        # Create dataset structure
        dataset_path = self.create_dataset_structure(dataset_name)
        
        # Build mappings
        image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
        image_id_to_size = {img['id']: (img['width'], img['height']) for img in coco_data['images']}
        
        # Group annotations by image
        annotations_by_image = defaultdict(list)
        for ann in coco_data['annotations']:
            annotations_by_image[ann['image_id']].append(ann)
        
        # Convert annotations
        converted_count = 0
        for image_id, filename in image_id_to_filename.items():
            src_image_path = Path(images_dir) / filename
            if not src_image_path.exists():
                self.logger.warning(f"Image not found: {src_image_path}")
                continue
            
            # Copy image to train split (we'll split later)
            dst_image_path = dataset_path / 'train' / 'images' / filename
            shutil.copy2(src_image_path, dst_image_path)
            
            # Convert annotations to YOLO format
            width, height = image_id_to_size[image_id]
            yolo_annotations = []
            
            for ann in annotations_by_image[image_id]:
                # Convert COCO bbox to YOLO format
                x, y, w, h = ann['bbox']
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                w_norm = w / width
                h_norm = h / height
                
                # Map class ID
                class_id = ann['category_id']
                if class_mapping:
                    class_id = class_mapping.get(class_id, 0)
                
                yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            
            # Save YOLO annotation file
            label_path = dataset_path / 'train' / 'labels' / f"{Path(filename).stem}.txt"
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            
            converted_count += 1
        
        self.logger.info(f"Converted {converted_count} images from COCO to YOLO format")
        
        # Split dataset
        self.split_dataset(dataset_path)
        
        return dataset_path
    
    def split_dataset(self, dataset_path: Path, 
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.2,
                     test_ratio: float = 0.1,
                     random_seed: int = 42) -> None:
        """Split dataset into train/val/test splits."""
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Get all images from train directory (initial import location)
        train_images_dir = dataset_path / 'train' / 'images'
        train_labels_dir = dataset_path / 'train' / 'labels'
        
        image_files = list(train_images_dir.glob('*'))
        image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        if not image_files:
            self.logger.warning("No images found to split")
            return
        
        # Shuffle files
        random.shuffle(image_files)
        
        # Calculate split indices
        total_files = len(image_files)
        train_end = int(total_files * train_ratio)
        val_end = train_end + int(total_files * val_ratio)
        
        # Split files
        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]
        
        # Move files to appropriate directories
        for split_name, files in [('val', val_files), ('test', test_files)]:
            split_images_dir = dataset_path / split_name / 'images'
            split_labels_dir = dataset_path / split_name / 'labels'
            
            for image_file in files:
                # Move image
                dst_image = split_images_dir / image_file.name
                shutil.move(str(image_file), str(dst_image))
                
                # Move corresponding label
                label_file = train_labels_dir / f"{image_file.stem}.txt"
                if label_file.exists():
                    dst_label = split_labels_dir / label_file.name
                    shutil.move(str(label_file), str(dst_label))
        
        self.logger.info(f"Split dataset: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    def analyze_dataset(self, dataset_path: Path) -> DatasetInfo:
        """Analyze dataset and return statistics."""
        dataset_path = Path(dataset_path)
        
        # Count images and annotations for each split
        total_images = 0
        total_annotations = 0
        class_distribution = Counter()
        image_sizes = []
        annotations_per_image = []
        
        for split in ['train', 'val', 'test']:
            images_dir = dataset_path / split / 'images'
            labels_dir = dataset_path / split / 'labels'
            
            if not images_dir.exists():
                continue
            
            for image_file in images_dir.glob('*'):
                if image_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                    continue
                
                total_images += 1
                
                # Get image size
                try:
                    with Image.open(image_file) as img:
                        image_sizes.append(img.size)  # (width, height)
                except Exception as e:
                    self.logger.warning(f"Could not read image {image_file}: {e}")
                    continue
                
                # Count annotations
                label_file = labels_dir / f"{image_file.stem}.txt"
                image_annotations = 0
                
                if label_file.exists():
                    with open(label_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                parts = line.split()
                                if len(parts) >= 5:
                                    class_id = int(parts[0])
                                    class_name = self.sar_classes.get(class_id, f'class_{class_id}')
                                    class_distribution[class_name] += 1
                                    total_annotations += 1
                                    image_annotations += 1
                
                annotations_per_image.append(image_annotations)
        
        # Calculate annotation density
        annotation_density = total_annotations / total_images if total_images > 0 else 0
        
        # Load dataset config for split ratios
        config_path = dataset_path / 'dataset.yaml'
        train_split = val_split = test_split = 0.0
        
        if config_path.exists():
            # Estimate split ratios from actual file counts
            train_count = len(list((dataset_path / 'train' / 'images').glob('*')))
            val_count = len(list((dataset_path / 'val' / 'images').glob('*')))
            test_count = len(list((dataset_path / 'test' / 'images').glob('*')))
            
            total_count = train_count + val_count + test_count
            if total_count > 0:
                train_split = train_count / total_count
                val_split = val_count / total_count
                test_split = test_count / total_count
        
        return DatasetInfo(
            name=dataset_path.name,
            total_images=total_images,
            total_annotations=total_annotations,
            class_distribution=dict(class_distribution),
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            image_sizes=image_sizes,
            annotation_density=annotation_density
        )
    
    def create_annotation_statistics(self, dataset_path: Path) -> AnnotationStats:
        """Create detailed annotation statistics."""
        dataset_path = Path(dataset_path)
        
        total_objects = 0
        objects_per_class = Counter()
        bbox_sizes = defaultdict(list)
        object_density_per_image = []
        small_objects = medium_objects = large_objects = 0
        
        for split in ['train', 'val', 'test']:
            images_dir = dataset_path / split / 'images'
            labels_dir = dataset_path / split / 'labels'
            
            if not images_dir.exists():
                continue
            
            for image_file in images_dir.glob('*'):
                if image_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                    continue
                
                # Get image dimensions
                try:
                    with Image.open(image_file) as img:
                        img_width, img_height = img.size
                except Exception:
                    continue
                
                label_file = labels_dir / f"{image_file.stem}.txt"
                objects_in_image = 0
                
                if label_file.exists():
                    with open(label_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                parts = line.split()
                                if len(parts) >= 5:
                                    class_id = int(parts[0])
                                    w_norm = float(parts[3])
                                    h_norm = float(parts[4])
                                    
                                    # Convert to pixel dimensions
                                    w_pixels = w_norm * img_width
                                    h_pixels = h_norm * img_height
                                    
                                    # Classify by size
                                    max_dim = max(w_pixels, h_pixels)
                                    if max_dim < 32:
                                        small_objects += 1
                                    elif max_dim < 96:
                                        medium_objects += 1
                                    else:
                                        large_objects += 1
                                    
                                    # Store statistics
                                    class_name = self.sar_classes.get(class_id, f'class_{class_id}')
                                    objects_per_class[class_name] += 1
                                    bbox_sizes[class_name].extend([w_pixels, h_pixels])
                                    
                                    total_objects += 1
                                    objects_in_image += 1
                
                object_density_per_image.append(objects_in_image)
        
        return AnnotationStats(
            total_objects=total_objects,
            objects_per_class=dict(objects_per_class),
            bbox_sizes=dict(bbox_sizes),
            object_density_per_image=object_density_per_image,
            small_objects=small_objects,
            medium_objects=medium_objects,
            large_objects=large_objects
        )
    
    def visualize_dataset_statistics(self, dataset_info: DatasetInfo, 
                                   annotation_stats: AnnotationStats,
                                   save_path: str = None) -> None:
        """Create visualization of dataset statistics."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Dataset Analysis: {dataset_info.name}', fontsize=16)
        
        # Class distribution
        axes[0, 0].bar(dataset_info.class_distribution.keys(), 
                      dataset_info.class_distribution.values())
        axes[0, 0].set_title('Class Distribution')
        axes[0, 0].set_ylabel('Number of Instances')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Split distribution
        splits = ['Train', 'Val', 'Test']
        split_values = [dataset_info.train_split, dataset_info.val_split, dataset_info.test_split]
        axes[0, 1].pie(split_values, labels=splits, autopct='%1.1f%%')
        axes[0, 1].set_title('Dataset Splits')
        
        # Object size distribution
        size_labels = ['Small (<32px)', 'Medium (32-96px)', 'Large (>96px)']
        size_values = [annotation_stats.small_objects, 
                      annotation_stats.medium_objects, 
                      annotation_stats.large_objects]
        axes[0, 2].bar(size_labels, size_values)
        axes[0, 2].set_title('Object Size Distribution')
        axes[0, 2].set_ylabel('Number of Objects')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Objects per image distribution
        axes[1, 0].hist(annotation_stats.object_density_per_image, bins=20, edgecolor='black')
        axes[1, 0].set_title('Objects per Image Distribution')
        axes[1, 0].set_xlabel('Number of Objects')
        axes[1, 0].set_ylabel('Number of Images')
        
        # Image size distribution
        if dataset_info.image_sizes:
            widths, heights = zip(*dataset_info.image_sizes)
            axes[1, 1].scatter(widths, heights, alpha=0.6)
            axes[1, 1].set_title('Image Size Distribution')
            axes[1, 1].set_xlabel('Width (pixels)')
            axes[1, 1].set_ylabel('Height (pixels)')
        
        # Annotation density
        axes[1, 2].text(0.5, 0.5, f'Total Images: {dataset_info.total_images}\n'
                                  f'Total Annotations: {dataset_info.total_annotations}\n'
                                  f'Avg Annotations/Image: {dataset_info.annotation_density:.2f}',
                        ha='center', va='center', fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='lightblue'))
        axes[1, 2].set_title('Dataset Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Dataset visualization saved to {save_path}")
        
        plt.show()
    
    def create_synthetic_sar_dataset(self, dataset_name: str, 
                                   num_images: int = 1000,
                                   image_size: Tuple[int, int] = (640, 640)) -> Path:
        """Create synthetic SAR dataset for testing."""
        dataset_path = self.create_dataset_structure(dataset_name)
        
        # Generate synthetic images with annotations
        for i in range(num_images):
            # Create synthetic aerial image
            image = self._generate_synthetic_aerial_image(image_size)
            
            # Add synthetic objects
            annotations = self._add_synthetic_objects(image, image_size)
            
            # Determine split
            if i < num_images * 0.7:
                split = 'train'
            elif i < num_images * 0.9:
                split = 'val'
            else:
                split = 'test'
            
            # Save image
            image_path = dataset_path / split / 'images' / f'synthetic_{i:06d}.jpg'
            cv2.imwrite(str(image_path), image)
            
            # Save annotations
            if annotations:
                label_path = dataset_path / split / 'labels' / f'synthetic_{i:06d}.txt'
                with open(label_path, 'w') as f:
                    f.write('\n'.join(annotations))
        
        self.logger.info(f"Created synthetic dataset with {num_images} images at {dataset_path}")
        return dataset_path
    
    def _generate_synthetic_aerial_image(self, size: Tuple[int, int]) -> np.ndarray:
        """Generate synthetic aerial background image."""
        width, height = size
        
        # Create base terrain texture
        image = np.random.randint(80, 120, (height, width, 3), dtype=np.uint8)
        
        # Add terrain features
        # Grass/vegetation areas
        for _ in range(random.randint(3, 8)):
            center = (random.randint(0, width), random.randint(0, height))
            radius = random.randint(20, 100)
            color = (random.randint(60, 100), random.randint(80, 140), random.randint(40, 80))
            cv2.circle(image, center, radius, color, -1)
        
        # Add some noise for realism
        noise = np.random.normal(0, 10, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def _add_synthetic_objects(self, image: np.ndarray, 
                             image_size: Tuple[int, int]) -> List[str]:
        """Add synthetic objects to image and return YOLO annotations."""
        width, height = image_size
        annotations = []
        
        # Add random number of objects
        num_objects = random.randint(0, 5)
        
        for _ in range(num_objects):
            # Random class (person is most common in SAR)
            class_weights = [0.7, 0.15, 0.1, 0.05]  # person, vehicle, debris, structure
            class_id = random.choices(range(len(class_weights)), weights=class_weights)[0]
            
            # Random position and size
            obj_width = random.randint(10, 50)
            obj_height = random.randint(15, 60)
            
            x = random.randint(obj_width//2, width - obj_width//2)
            y = random.randint(obj_height//2, height - obj_height//2)
            
            # Draw object (simple rectangle with color based on class)
            colors = [(0, 0, 255), (255, 0, 0), (0, 255, 255), (128, 128, 128)]  # person, vehicle, debris, structure
            color = colors[class_id]
            
            cv2.rectangle(image, 
                         (x - obj_width//2, y - obj_height//2),
                         (x + obj_width//2, y + obj_height//2),
                         color, -1)
            
            # Convert to YOLO format
            x_center = x / width
            y_center = y / height
            w_norm = obj_width / width
            h_norm = obj_height / height
            
            annotation = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            annotations.append(annotation)
        
        return annotations
    
    def export_dataset_report(self, dataset_path: Path, output_path: str = None) -> str:
        """Export comprehensive dataset report."""
        dataset_info = self.analyze_dataset(dataset_path)
        annotation_stats = self.create_annotation_statistics(dataset_path)
        
        report = []
        report.append(f"SAR Dataset Report: {dataset_info.name}")
        report.append("=" * 50)
        report.append("")
        
        # Basic statistics
        report.append("Dataset Overview:")
        report.append(f"  Total Images: {dataset_info.total_images}")
        report.append(f"  Total Annotations: {dataset_info.total_annotations}")
        report.append(f"  Average Annotations per Image: {dataset_info.annotation_density:.2f}")
        report.append("")
        
        # Split information
        report.append("Dataset Splits:")
        report.append(f"  Train: {dataset_info.train_split:.1%}")
        report.append(f"  Validation: {dataset_info.val_split:.1%}")
        report.append(f"  Test: {dataset_info.test_split:.1%}")
        report.append("")
        
        # Class distribution
        report.append("Class Distribution:")
        for class_name, count in dataset_info.class_distribution.items():
            percentage = (count / dataset_info.total_annotations) * 100
            report.append(f"  {class_name}: {count} ({percentage:.1f}%)")
        report.append("")
        
        # Object size analysis
        report.append("Object Size Analysis:")
        total_objects = (annotation_stats.small_objects + 
                        annotation_stats.medium_objects + 
                        annotation_stats.large_objects)
        if total_objects > 0:
            report.append(f"  Small objects (<32px): {annotation_stats.small_objects} ({annotation_stats.small_objects/total_objects:.1%})")
            report.append(f"  Medium objects (32-96px): {annotation_stats.medium_objects} ({annotation_stats.medium_objects/total_objects:.1%})")
            report.append(f"  Large objects (>96px): {annotation_stats.large_objects} ({annotation_stats.large_objects/total_objects:.1%})")
        report.append("")
        
        # Image size statistics
        if dataset_info.image_sizes:
            widths, heights = zip(*dataset_info.image_sizes)
            report.append("Image Size Statistics:")
            report.append(f"  Average size: {np.mean(widths):.0f}x{np.mean(heights):.0f}")
            report.append(f"  Size range: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Dataset report saved to {output_path}")
        
        return report_text


# Example usage
if __name__ == "__main__":
    # Example dataset management
    manager = DatasetManager("datasets")
    
    # Create a synthetic dataset for testing
    dataset_path = manager.create_synthetic_sar_dataset("sar_synthetic_test", num_images=100)
    
    # Analyze the dataset
    dataset_info = manager.analyze_dataset(dataset_path)
    annotation_stats = manager.create_annotation_statistics(dataset_path)
    
    # Generate report
    report = manager.export_dataset_report(dataset_path)
    print(report)
    
    print("Dataset management example completed successfully!")