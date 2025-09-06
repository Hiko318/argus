"""SAR-specific validation metrics and evaluation.

Provides specialized validation for Search and Rescue scenarios including
distance-based recall metrics, geo-error evaluation, and performance analysis.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import cv2
from dataclasses import dataclass
from collections import defaultdict

try:
    from ultralytics import YOLO
    import torch
    from sklearn.metrics import precision_recall_curve, average_precision_score
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Install with: pip install ultralytics torch scikit-learn matplotlib seaborn")


@dataclass
class DetectionResult:
    """Single detection result with metadata."""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    distance_estimate: Optional[float] = None  # Estimated distance in meters
    target_size: Optional[float] = None  # Target size in pixels
    

@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics for SAR."""
    # Standard metrics
    precision: float
    recall: float
    f1_score: float
    ap: float  # Average Precision
    map50: float  # mAP at IoU 0.5
    map50_95: float  # mAP at IoU 0.5:0.95
    
    # Distance-based metrics
    recall_at_distances: Dict[int, float]  # Recall at different distances
    precision_at_distances: Dict[int, float]  # Precision at different distances
    
    # Size-based metrics
    recall_by_size: Dict[str, float]  # small, medium, large targets
    precision_by_size: Dict[str, float]
    
    # SAR-specific metrics
    detection_rate: float  # Overall detection rate
    false_positive_rate: float
    missed_detection_rate: float
    geo_error_p50: Optional[float] = None  # Median geo-location error
    geo_error_p95: Optional[float] = None  # 95th percentile geo-location error
    

class SARValidator:
    """SAR-specific validation and evaluation."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.confidence_threshold = self.config.get('confidence_threshold', 0.25)
        self.iou_threshold = self.config.get('iou_threshold', 0.45)
        self.distance_thresholds = self.config.get('distance_thresholds', [50, 100, 200, 500])
        
        # Size categories (in pixels)
        self.size_categories = {
            'small': (0, 32),
            'medium': (32, 96),
            'large': (96, float('inf'))
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def validate_model(self, model: YOLO, val_dataset_path: str, 
                      save_results: bool = True) -> ValidationMetrics:
        """Comprehensive model validation."""
        self.logger.info(f"Starting validation on {val_dataset_path}")
        
        # Run standard YOLO validation
        results = model.val(data=val_dataset_path, save_json=True, save_txt=True)
        
        # Extract standard metrics
        standard_metrics = self._extract_standard_metrics(results)
        
        # Run SAR-specific validation
        sar_metrics = self._run_sar_validation(model, val_dataset_path)
        
        # Combine metrics
        validation_metrics = ValidationMetrics(
            **standard_metrics,
            **sar_metrics
        )
        
        if save_results:
            self._save_validation_results(validation_metrics, val_dataset_path)
        
        return validation_metrics
    
    def _extract_standard_metrics(self, results) -> Dict:
        """Extract standard metrics from YOLO validation results."""
        metrics = {}
        
        if hasattr(results, 'results_dict'):
            results_dict = results.results_dict
            metrics['precision'] = results_dict.get('metrics/precision(B)', 0.0)
            metrics['recall'] = results_dict.get('metrics/recall(B)', 0.0)
            metrics['map50'] = results_dict.get('metrics/mAP50(B)', 0.0)
            metrics['map50_95'] = results_dict.get('metrics/mAP50-95(B)', 0.0)
            
            # Calculate F1 score
            p, r = metrics['precision'], metrics['recall']
            metrics['f1_score'] = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
            metrics['ap'] = metrics['map50']  # Use mAP50 as AP
        else:
            # Fallback values
            metrics = {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'ap': 0.0,
                'map50': 0.0,
                'map50_95': 0.0
            }
        
        return metrics
    
    def _run_sar_validation(self, model: YOLO, val_dataset_path: str) -> Dict:
        """Run SAR-specific validation metrics."""
        # Load validation dataset
        val_images, val_labels = self._load_validation_data(val_dataset_path)
        
        all_detections = []
        all_ground_truths = []
        
        self.logger.info(f"Processing {len(val_images)} validation images...")
        
        for img_path, label_path in zip(val_images, val_labels):
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Run inference
            results = model(image, conf=self.confidence_threshold, iou=self.iou_threshold)
            
            # Extract detections
            detections = self._extract_detections(results[0], image.shape)
            
            # Load ground truth
            ground_truth = self._load_ground_truth(label_path, image.shape)
            
            all_detections.append(detections)
            all_ground_truths.append(ground_truth)
        
        # Calculate SAR-specific metrics
        sar_metrics = self._calculate_sar_metrics(all_detections, all_ground_truths)
        
        return sar_metrics
    
    def _load_validation_data(self, val_dataset_path: str) -> Tuple[List[Path], List[Path]]:
        """Load validation dataset file paths."""
        val_path = Path(val_dataset_path)
        
        if val_path.is_file() and val_path.suffix == '.yaml':
            # Load from YAML config
            import yaml
            with open(val_path, 'r') as f:
                config = yaml.safe_load(f)
            
            dataset_root = Path(config['path'])
            val_dir = dataset_root / config['val']
        else:
            val_dir = val_path
        
        # Find image and label files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        images_dir = val_dir / 'images' if (val_dir / 'images').exists() else val_dir
        labels_dir = val_dir / 'labels' if (val_dir / 'labels').exists() else val_dir
        
        image_files = []
        label_files = []
        
        for img_file in images_dir.iterdir():
            if img_file.suffix.lower() in image_extensions:
                label_file = labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    image_files.append(img_file)
                    label_files.append(label_file)
        
        self.logger.info(f"Found {len(image_files)} image-label pairs")
        return image_files, label_files
    
    def _extract_detections(self, result, image_shape: Tuple[int, int, int]) -> List[DetectionResult]:
        """Extract detections from YOLO result."""
        detections = []
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                # Calculate target size
                target_size = max(box[2] - box[0], box[3] - box[1])
                
                # Estimate distance (simple heuristic based on target size)
                distance_estimate = self._estimate_distance_from_size(target_size, image_shape)
                
                detection = DetectionResult(
                    bbox=tuple(box),
                    confidence=float(conf),
                    class_id=int(cls_id),
                    distance_estimate=distance_estimate,
                    target_size=float(target_size)
                )
                detections.append(detection)
        
        return detections
    
    def _load_ground_truth(self, label_path: Path, image_shape: Tuple[int, int, int]) -> List[DetectionResult]:
        """Load ground truth annotations."""
        ground_truth = []
        
        if not label_path.exists():
            return ground_truth
        
        h, w = image_shape[:2]
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * w
                    y_center = float(parts[2]) * h
                    width = float(parts[3]) * w
                    height = float(parts[4]) * h
                    
                    # Convert to x1, y1, x2, y2
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    
                    target_size = max(width, height)
                    distance_estimate = self._estimate_distance_from_size(target_size, image_shape)
                    
                    gt = DetectionResult(
                        bbox=(x1, y1, x2, y2),
                        confidence=1.0,  # Ground truth has confidence 1.0
                        class_id=class_id,
                        distance_estimate=distance_estimate,
                        target_size=target_size
                    )
                    ground_truth.append(gt)
        
        return ground_truth
    
    def _estimate_distance_from_size(self, target_size: float, image_shape: Tuple[int, int, int]) -> float:
        """Estimate distance based on target size (simple heuristic)."""
        # Assume average human height of 1.7m and typical camera parameters
        # This is a rough estimation - in practice, you'd use camera calibration
        
        image_height = image_shape[0]
        
        # Rough heuristic: larger targets are closer
        if target_size > 100:
            return 50  # Close range
        elif target_size > 50:
            return 100  # Medium range
        elif target_size > 20:
            return 200  # Far range
        else:
            return 500  # Very far range
    
    def _calculate_sar_metrics(self, all_detections: List[List[DetectionResult]], 
                              all_ground_truths: List[List[DetectionResult]]) -> Dict:
        """Calculate SAR-specific metrics."""
        # Initialize counters
        total_detections = 0
        total_ground_truths = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # Distance-based metrics
        distance_metrics = {dist: {'tp': 0, 'fp': 0, 'fn': 0, 'total_gt': 0} 
                           for dist in self.distance_thresholds}
        
        # Size-based metrics
        size_metrics = {size: {'tp': 0, 'fp': 0, 'fn': 0, 'total_gt': 0} 
                       for size in self.size_categories.keys()}
        
        for detections, ground_truths in zip(all_detections, all_ground_truths):
            total_detections += len(detections)
            total_ground_truths += len(ground_truths)
            
            # Match detections to ground truths
            matches = self._match_detections_to_ground_truth(detections, ground_truths)
            
            # Count matches
            matched_gt_indices = set()
            for det_idx, gt_idx in matches:
                if gt_idx is not None:
                    true_positives += 1
                    matched_gt_indices.add(gt_idx)
                    
                    # Update distance-based metrics
                    gt = ground_truths[gt_idx]
                    for dist_threshold in self.distance_thresholds:
                        if gt.distance_estimate <= dist_threshold:
                            distance_metrics[dist_threshold]['tp'] += 1
                    
                    # Update size-based metrics
                    size_category = self._get_size_category(gt.target_size)
                    size_metrics[size_category]['tp'] += 1
                else:
                    false_positives += 1
            
            # Count unmatched ground truths (false negatives)
            for gt_idx, gt in enumerate(ground_truths):
                if gt_idx not in matched_gt_indices:
                    false_negatives += 1
                    
                    # Update distance-based metrics
                    for dist_threshold in self.distance_thresholds:
                        if gt.distance_estimate <= dist_threshold:
                            distance_metrics[dist_threshold]['fn'] += 1
                    
                    # Update size-based metrics
                    size_category = self._get_size_category(gt.target_size)
                    size_metrics[size_category]['fn'] += 1
                
                # Count total ground truths for each category
                for dist_threshold in self.distance_thresholds:
                    if gt.distance_estimate <= dist_threshold:
                        distance_metrics[dist_threshold]['total_gt'] += 1
                
                size_category = self._get_size_category(gt.target_size)
                size_metrics[size_category]['total_gt'] += 1
        
        # Calculate final metrics
        detection_rate = true_positives / total_ground_truths if total_ground_truths > 0 else 0.0
        false_positive_rate = false_positives / total_detections if total_detections > 0 else 0.0
        missed_detection_rate = false_negatives / total_ground_truths if total_ground_truths > 0 else 0.0
        
        # Calculate distance-based recall and precision
        recall_at_distances = {}
        precision_at_distances = {}
        
        for dist, metrics in distance_metrics.items():
            recall = metrics['tp'] / metrics['total_gt'] if metrics['total_gt'] > 0 else 0.0
            precision = metrics['tp'] / (metrics['tp'] + metrics['fp']) if (metrics['tp'] + metrics['fp']) > 0 else 0.0
            recall_at_distances[dist] = recall
            precision_at_distances[dist] = precision
        
        # Calculate size-based recall and precision
        recall_by_size = {}
        precision_by_size = {}
        
        for size, metrics in size_metrics.items():
            recall = metrics['tp'] / metrics['total_gt'] if metrics['total_gt'] > 0 else 0.0
            precision = metrics['tp'] / (metrics['tp'] + metrics['fp']) if (metrics['tp'] + metrics['fp']) > 0 else 0.0
            recall_by_size[size] = recall
            precision_by_size[size] = precision
        
        return {
            'recall_at_distances': recall_at_distances,
            'precision_at_distances': precision_at_distances,
            'recall_by_size': recall_by_size,
            'precision_by_size': precision_by_size,
            'detection_rate': detection_rate,
            'false_positive_rate': false_positive_rate,
            'missed_detection_rate': missed_detection_rate
        }
    
    def _match_detections_to_ground_truth(self, detections: List[DetectionResult], 
                                         ground_truths: List[DetectionResult]) -> List[Tuple[int, Optional[int]]]:
        """Match detections to ground truth using IoU."""
        matches = []
        used_gt_indices = set()
        
        # Sort detections by confidence (highest first)
        sorted_detections = sorted(enumerate(detections), key=lambda x: x[1].confidence, reverse=True)
        
        for det_idx, detection in sorted_detections:
            best_iou = 0.0
            best_gt_idx = None
            
            for gt_idx, ground_truth in enumerate(ground_truths):
                if gt_idx in used_gt_indices:
                    continue
                
                # Calculate IoU
                iou = self._calculate_iou(detection.bbox, ground_truth.bbox)
                
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_gt_idx is not None:
                used_gt_indices.add(best_gt_idx)
                matches.append((det_idx, best_gt_idx))
            else:
                matches.append((det_idx, None))  # False positive
        
        return matches
    
    def _calculate_iou(self, bbox1: Tuple[float, float, float, float], 
                      bbox2: Tuple[float, float, float, float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
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
    
    def _get_size_category(self, target_size: float) -> str:
        """Get size category for a target."""
        for category, (min_size, max_size) in self.size_categories.items():
            if min_size <= target_size < max_size:
                return category
        return 'large'  # Default for very large targets
    
    def validate_with_distance_metrics(self, model: YOLO, val_dataset_path: str, 
                                     distance_thresholds: List[int]) -> Dict:
        """Validate model with specific distance thresholds."""
        self.distance_thresholds = distance_thresholds
        metrics = self.validate_model(model, val_dataset_path, save_results=False)
        
        return {
            'recall_at_distances': metrics.recall_at_distances,
            'precision_at_distances': metrics.precision_at_distances,
            'overall_recall': metrics.recall,
            'overall_precision': metrics.precision,
            'map50': metrics.map50
        }
    
    def _save_validation_results(self, metrics: ValidationMetrics, dataset_path: str):
        """Save validation results to file."""
        results = {
            'dataset_path': dataset_path,
            'timestamp': str(np.datetime64('now')),
            'metrics': {
                'standard': {
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1_score': metrics.f1_score,
                    'ap': metrics.ap,
                    'map50': metrics.map50,
                    'map50_95': metrics.map50_95
                },
                'distance_based': {
                    'recall_at_distances': metrics.recall_at_distances,
                    'precision_at_distances': metrics.precision_at_distances
                },
                'size_based': {
                    'recall_by_size': metrics.recall_by_size,
                    'precision_by_size': metrics.precision_by_size
                },
                'sar_specific': {
                    'detection_rate': metrics.detection_rate,
                    'false_positive_rate': metrics.false_positive_rate,
                    'missed_detection_rate': metrics.missed_detection_rate
                }
            }
        }
        
        # Save to JSON
        output_path = Path(f"validation_results_{np.datetime64('now').astype(str).replace(':', '-')}.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Validation results saved to {output_path}")
    
    def create_validation_report(self, metrics: ValidationMetrics, save_path: str = None) -> str:
        """Create a comprehensive validation report."""
        report = []
        report.append("SAR Model Validation Report")
        report.append("=" * 50)
        report.append("")
        
        # Standard metrics
        report.append("Standard Metrics:")
        report.append(f"  Precision: {metrics.precision:.3f}")
        report.append(f"  Recall: {metrics.recall:.3f}")
        report.append(f"  F1 Score: {metrics.f1_score:.3f}")
        report.append(f"  mAP@0.5: {metrics.map50:.3f}")
        report.append(f"  mAP@0.5:0.95: {metrics.map50_95:.3f}")
        report.append("")
        
        # Distance-based metrics
        report.append("Distance-based Recall:")
        for dist, recall in metrics.recall_at_distances.items():
            report.append(f"  ≤{dist}m: {recall:.3f}")
        report.append("")
        
        # Size-based metrics
        report.append("Size-based Performance:")
        for size, recall in metrics.recall_by_size.items():
            precision = metrics.precision_by_size[size]
            report.append(f"  {size.capitalize()}: Recall={recall:.3f}, Precision={precision:.3f}")
        report.append("")
        
        # SAR-specific metrics
        report.append("SAR-specific Metrics:")
        report.append(f"  Detection Rate: {metrics.detection_rate:.3f}")
        report.append(f"  False Positive Rate: {metrics.false_positive_rate:.3f}")
        report.append(f"  Missed Detection Rate: {metrics.missed_detection_rate:.3f}")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text


# Example usage
if __name__ == "__main__":
    # Example validation
    config = {
        'confidence_threshold': 0.25,
        'iou_threshold': 0.45,
        'distance_thresholds': [50, 100, 200, 500]
    }
    
    validator = SARValidator(config)
    print("SAR validator initialized successfully!")