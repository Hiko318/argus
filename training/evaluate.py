#!/usr/bin/env python3
"""
YOLOv8 Model Evaluation Script for SAR Operations

This script provides comprehensive evaluation capabilities for trained YOLOv8 models,
including SAR-specific metrics and performance analysis.

Usage:
    python evaluate.py --model runs/train/sar_yolov8/weights/best.pt --data datasets/sar_dataset
    python evaluate.py --model models/sar_yolov8.onnx --data test_images/ --format onnx
    python evaluate.py --model best.pt --benchmark --export-results
"""

import argparse
import os
import sys
import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging

try:
    from ultralytics import YOLO
    from ultralytics.utils.metrics import ConfusionMatrix
except ImportError:
    print("Error: Ultralytics not installed. Install with: pip install ultralytics")
    sys.exit(1)

import numpy as np
import cv2
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SARModelEvaluator:
    """
    Comprehensive model evaluator for SAR YOLOv8 models with specialized
    metrics for aerial imagery and person detection performance.
    """
    
    def __init__(self, model_path: str, model_format: str = 'pt'):
        """
        Initialize evaluator with model.
        
        Args:
            model_path: Path to model file
            model_format: Model format ('pt', 'onnx', 'tensorrt')
        """
        self.model_path = model_path
        self.model_format = model_format
        self.model = self._load_model()
        self.class_names = self._get_class_names()
        self.results = {}
        
    def _load_model(self):
        """Load model based on format."""
        try:
            if self.model_format == 'pt':
                model = YOLO(self.model_path)
            elif self.model_format == 'onnx':
                model = YOLO(self.model_path, task='detect')
            elif self.model_format == 'tensorrt':
                model = YOLO(self.model_path)
            else:
                raise ValueError(f"Unsupported model format: {self.model_format}")
            
            logger.info(f"Loaded {self.model_format.upper()} model: {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _get_class_names(self) -> Dict[int, str]:
        """Get class names from model."""
        try:
            if hasattr(self.model, 'names'):
                return self.model.names
            else:
                # Default SAR classes
                return {0: 'person', 1: 'vehicle', 2: 'structure', 3: 'debris'}
        except:
            return {0: 'person', 1: 'vehicle', 2: 'structure', 3: 'debris'}
    
    def evaluate_dataset(self, data_path: str, conf_threshold: float = 0.25,
                        iou_threshold: float = 0.45) -> Dict[str, Any]:
        """
        Evaluate model on dataset with comprehensive metrics.
        
        Args:
            data_path: Path to dataset or dataset YAML
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Evaluating model on dataset: {data_path}")
        
        try:
            # Run validation
            results = self.model.val(
                data=data_path,
                conf=conf_threshold,
                iou=iou_threshold,
                save_json=True,
                save_txt=True,
                plots=True
            )
            
            # Extract metrics
            metrics = {
                'mAP50': float(results.box.map50),
                'mAP50-95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'f1_score': 2 * (float(results.box.mp) * float(results.box.mr)) / 
                           (float(results.box.mp) + float(results.box.mr) + 1e-16)
            }
            
            # Per-class metrics
            if hasattr(results.box, 'maps'):
                class_metrics = {}
                for i, class_name in self.class_names.items():
                    if i < len(results.box.maps):
                        class_metrics[class_name] = {
                            'mAP50': float(results.box.maps[i]),
                            'precision': float(results.box.p[i]) if hasattr(results.box, 'p') else 0.0,
                            'recall': float(results.box.r[i]) if hasattr(results.box, 'r') else 0.0
                        }
                metrics['per_class'] = class_metrics
            
            # SAR-specific metrics
            sar_metrics = self._calculate_sar_metrics(results)
            metrics.update(sar_metrics)
            
            self.results['dataset_evaluation'] = metrics
            logger.info(f"Dataset evaluation completed. mAP50: {metrics['mAP50']:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Dataset evaluation failed: {e}")
            raise
    
    def _calculate_sar_metrics(self, results) -> Dict[str, Any]:
        """Calculate SAR-specific evaluation metrics."""
        sar_metrics = {}
        
        # Person detection priority metrics
        if 'person' in self.class_names.values():
            person_idx = list(self.class_names.values()).index('person')
            if hasattr(results.box, 'maps') and person_idx < len(results.box.maps):
                sar_metrics['person_mAP50'] = float(results.box.maps[person_idx])
                sar_metrics['person_detection_rate'] = float(results.box.r[person_idx]) if hasattr(results.box, 'r') else 0.0
        
        # Small object detection performance
        # Note: This would require access to ground truth annotations
        # For now, we'll use overall small object performance from results
        sar_metrics['small_object_performance'] = {
            'note': 'Small object metrics require ground truth analysis',
            'overall_small_map': 'Available in detailed results'
        }
        
        return sar_metrics
    
    def benchmark_performance(self, test_images: List[str], 
                            warmup_runs: int = 10, 
                            benchmark_runs: int = 100) -> Dict[str, Any]:
        """
        Benchmark model inference performance.
        
        Args:
            test_images: List of test image paths
            warmup_runs: Number of warmup inference runs
            benchmark_runs: Number of benchmark inference runs
            
        Returns:
            Performance benchmark results
        """
        logger.info(f"Benchmarking model performance with {benchmark_runs} runs")
        
        if not test_images:
            raise ValueError("No test images provided for benchmarking")
        
        # Warmup
        logger.info(f"Warming up with {warmup_runs} runs...")
        for i in range(warmup_runs):
            img_path = test_images[i % len(test_images)]
            _ = self.model(img_path, verbose=False)
        
        # Benchmark
        inference_times = []
        preprocessing_times = []
        postprocessing_times = []
        
        for i in range(benchmark_runs):
            img_path = test_images[i % len(test_images)]
            
            # Load and preprocess
            start_time = time.time()
            img = cv2.imread(img_path)
            if img is None:
                continue
            preprocess_time = time.time() - start_time
            
            # Inference
            start_time = time.time()
            results = self.model(img, verbose=False)
            inference_time = time.time() - start_time
            
            # Postprocessing (extract results)
            start_time = time.time()
            _ = results[0].boxes if results[0].boxes is not None else []
            postprocess_time = time.time() - start_time
            
            preprocessing_times.append(preprocess_time)
            inference_times.append(inference_time)
            postprocessing_times.append(postprocess_time)
        
        # Calculate statistics
        benchmark_results = {
            'inference': {
                'mean_ms': np.mean(inference_times) * 1000,
                'std_ms': np.std(inference_times) * 1000,
                'min_ms': np.min(inference_times) * 1000,
                'max_ms': np.max(inference_times) * 1000,
                'fps': 1.0 / np.mean(inference_times)
            },
            'preprocessing': {
                'mean_ms': np.mean(preprocessing_times) * 1000,
                'std_ms': np.std(preprocessing_times) * 1000
            },
            'postprocessing': {
                'mean_ms': np.mean(postprocessing_times) * 1000,
                'std_ms': np.std(postprocessing_times) * 1000
            },
            'total_pipeline': {
                'mean_ms': (np.mean(preprocessing_times) + 
                           np.mean(inference_times) + 
                           np.mean(postprocessing_times)) * 1000,
                'fps': 1.0 / (np.mean(preprocessing_times) + 
                             np.mean(inference_times) + 
                             np.mean(postprocessing_times))
            }
        }
        
        self.results['performance_benchmark'] = benchmark_results
        logger.info(f"Benchmark completed. Average inference: {benchmark_results['inference']['mean_ms']:.2f}ms, "
                   f"FPS: {benchmark_results['inference']['fps']:.1f}")
        
        return benchmark_results
    
    def analyze_detection_quality(self, test_images: List[str], 
                                output_dir: str = 'evaluation_output') -> Dict[str, Any]:
        """
        Analyze detection quality with visual outputs.
        
        Args:
            test_images: List of test image paths
            output_dir: Directory to save analysis outputs
            
        Returns:
            Detection quality analysis results
        """
        logger.info(f"Analyzing detection quality on {len(test_images)} images")
        
        os.makedirs(output_dir, exist_ok=True)
        
        detection_stats = {
            'total_images': len(test_images),
            'images_with_detections': 0,
            'total_detections': 0,
            'detections_per_class': {name: 0 for name in self.class_names.values()},
            'confidence_distribution': [],
            'detection_sizes': []
        }
        
        for i, img_path in enumerate(test_images):
            try:
                # Run inference
                results = self.model(img_path, save=False, verbose=False)
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    detection_stats['images_with_detections'] += 1
                    detection_stats['total_detections'] += len(result.boxes)
                    
                    # Analyze detections
                    for box in result.boxes:
                        # Confidence
                        conf = float(box.conf)
                        detection_stats['confidence_distribution'].append(conf)
                        
                        # Class
                        cls_id = int(box.cls)
                        if cls_id in self.class_names:
                            class_name = self.class_names[cls_id]
                            detection_stats['detections_per_class'][class_name] += 1
                        
                        # Size
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        detection_stats['detection_sizes'].append(area)
                
                # Save annotated image (every 10th image)
                if i % 10 == 0:
                    annotated = result.plot()
                    output_path = os.path.join(output_dir, f'annotated_{i:04d}.jpg')
                    cv2.imwrite(output_path, annotated)
                    
            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")
                continue
        
        # Calculate statistics
        if detection_stats['confidence_distribution']:
            detection_stats['confidence_stats'] = {
                'mean': np.mean(detection_stats['confidence_distribution']),
                'std': np.std(detection_stats['confidence_distribution']),
                'min': np.min(detection_stats['confidence_distribution']),
                'max': np.max(detection_stats['confidence_distribution'])
            }
        
        if detection_stats['detection_sizes']:
            detection_stats['size_stats'] = {
                'mean_area': np.mean(detection_stats['detection_sizes']),
                'std_area': np.std(detection_stats['detection_sizes']),
                'min_area': np.min(detection_stats['detection_sizes']),
                'max_area': np.max(detection_stats['detection_sizes'])
            }
        
        detection_stats['detection_rate'] = detection_stats['images_with_detections'] / detection_stats['total_images']
        detection_stats['avg_detections_per_image'] = detection_stats['total_detections'] / detection_stats['total_images']
        
        self.results['detection_quality'] = detection_stats
        logger.info(f"Detection quality analysis completed. Detection rate: {detection_stats['detection_rate']:.2f}")
        
        return detection_stats
    
    def generate_report(self, output_path: str = 'evaluation_report.json'):
        """
        Generate comprehensive evaluation report.
        
        Args:
            output_path: Path to save the report
        """
        report = {
            'model_info': {
                'path': self.model_path,
                'format': self.model_format,
                'class_names': self.class_names
            },
            'evaluation_timestamp': datetime.now().isoformat(),
            'results': self.results
        }
        
        # Save JSON report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate CSV summary
        csv_path = output_path.replace('.json', '_summary.csv')
        self._generate_csv_summary(csv_path)
        
        logger.info(f"Evaluation report saved: {output_path}")
        logger.info(f"CSV summary saved: {csv_path}")
    
    def _generate_csv_summary(self, csv_path: str):
        """Generate CSV summary of key metrics."""
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['Metric', 'Value', 'Unit'])
            
            # Dataset metrics
            if 'dataset_evaluation' in self.results:
                metrics = self.results['dataset_evaluation']
                writer.writerow(['mAP50', f"{metrics.get('mAP50', 0):.3f}", ''])
                writer.writerow(['mAP50-95', f"{metrics.get('mAP50-95', 0):.3f}", ''])
                writer.writerow(['Precision', f"{metrics.get('precision', 0):.3f}", ''])
                writer.writerow(['Recall', f"{metrics.get('recall', 0):.3f}", ''])
                writer.writerow(['F1 Score', f"{metrics.get('f1_score', 0):.3f}", ''])
            
            # Performance metrics
            if 'performance_benchmark' in self.results:
                perf = self.results['performance_benchmark']
                writer.writerow(['Inference Time', f"{perf['inference']['mean_ms']:.2f}", 'ms'])
                writer.writerow(['Inference FPS', f"{perf['inference']['fps']:.1f}", 'fps'])
                writer.writerow(['Total Pipeline FPS', f"{perf['total_pipeline']['fps']:.1f}", 'fps'])
            
            # Detection quality
            if 'detection_quality' in self.results:
                quality = self.results['detection_quality']
                writer.writerow(['Detection Rate', f"{quality.get('detection_rate', 0):.2f}", ''])
                writer.writerow(['Avg Detections/Image', f"{quality.get('avg_detections_per_image', 0):.1f}", ''])

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='YOLOv8 SAR Model Evaluation')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model file')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset YAML or test images directory')
    parser.add_argument('--format', type=str, default='pt',
                       choices=['pt', 'onnx', 'tensorrt'],
                       help='Model format')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    parser.add_argument('--analyze-quality', action='store_true',
                       help='Analyze detection quality')
    parser.add_argument('--export-results', action='store_true',
                       help='Export detailed results')
    parser.add_argument('--output-dir', type=str, default='evaluation_output',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = SARModelEvaluator(args.model, args.format)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Dataset evaluation
        if os.path.isfile(args.data) and args.data.endswith('.yaml'):
            logger.info("Running dataset evaluation...")
            evaluator.evaluate_dataset(args.data, args.conf, args.iou)
        
        # Get test images for other evaluations
        test_images = []
        if os.path.isdir(args.data):
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            for ext in extensions:
                test_images.extend(Path(args.data).glob(f'**/*{ext}'))
            test_images = [str(p) for p in test_images]
        
        # Performance benchmark
        if args.benchmark and test_images:
            logger.info("Running performance benchmark...")
            evaluator.benchmark_performance(test_images[:100])  # Limit to 100 images
        
        # Detection quality analysis
        if args.analyze_quality and test_images:
            logger.info("Analyzing detection quality...")
            evaluator.analyze_detection_quality(test_images[:50], args.output_dir)  # Limit to 50 images
        
        # Export results
        if args.export_results:
            report_path = os.path.join(args.output_dir, 'evaluation_report.json')
            evaluator.generate_report(report_path)
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()