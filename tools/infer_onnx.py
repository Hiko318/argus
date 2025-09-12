#!/usr/bin/env python3
"""
ONNX Model Inference Tool for FORESIGHT System

Tests ONNX model inference performance and accuracy.
Supports YOLOv8 ONNX models and custom detection models.

Usage:
    python infer_onnx.py --model yolov8n.onnx --image test_image.jpg
    python infer_onnx.py --model yolov8n.onnx --benchmark --num-runs 100
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import json

try:
    import numpy as np
    import cv2
except ImportError:
    print("Error: OpenCV and NumPy required. Install with: pip install opencv-python numpy")
    sys.exit(1)

try:
    import onnxruntime as ort
except ImportError:
    print("Error: ONNX Runtime required. Install with: pip install onnxruntime")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ONNXInferencer:
    """ONNX model inference handler."""
    
    def __init__(self, model_path: str, providers: Optional[List[str]] = None):
        """Initialize ONNX inferencer.
        
        Args:
            model_path: Path to ONNX model file
            providers: List of execution providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Set up execution providers
        if providers is None:
            providers = ['CPUExecutionProvider']
            if ort.get_device() == 'GPU':
                providers.insert(0, 'CUDAExecutionProvider')
        
        logger.info(f"Loading ONNX model: {model_path}")
        logger.info(f"Available providers: {ort.get_available_providers()}")
        logger.info(f"Using providers: {providers}")
        
        # Create inference session
        try:
            self.session = ort.InferenceSession(str(model_path), providers=providers)
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
        
        # Get model info
        self.input_info = self._get_input_info()
        self.output_info = self._get_output_info()
        
        logger.info(f"Model loaded successfully")
        logger.info(f"Input shape: {self.input_info['shape']}")
        logger.info(f"Output shapes: {[out['shape'] for out in self.output_info]}")
    
    def _get_input_info(self) -> Dict[str, Any]:
        """Get input tensor information."""
        input_info = self.session.get_inputs()[0]
        return {
            'name': input_info.name,
            'shape': input_info.shape,
            'type': input_info.type
        }
    
    def _get_output_info(self) -> List[Dict[str, Any]]:
        """Get output tensor information."""
        outputs = []
        for output in self.session.get_outputs():
            outputs.append({
                'name': output.name,
                'shape': output.shape,
                'type': output.type
            })
        return outputs
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for YOLO inference.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Preprocessed image tensor
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Get target size from model input
        input_shape = self.input_info['shape']
        if len(input_shape) == 4:  # NCHW format
            target_height, target_width = input_shape[2], input_shape[3]
        else:
            raise ValueError(f"Unexpected input shape: {input_shape}")
        
        # Resize image
        image_resized = cv2.resize(image, (target_width, target_height))
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        image_normalized = image_rgb.astype(np.float32) / 255.0
        
        # Convert to NCHW format
        image_tensor = np.transpose(image_normalized, (2, 0, 1))
        
        # Add batch dimension
        image_batch = np.expand_dims(image_tensor, axis=0)
        
        return image_batch
    
    def create_dummy_input(self) -> np.ndarray:
        """Create dummy input tensor for benchmarking.
        
        Returns:
            Random input tensor matching model requirements
        """
        input_shape = self.input_info['shape']
        
        # Handle dynamic batch size
        shape = []
        for dim in input_shape:
            if isinstance(dim, str) or dim == -1:
                shape.append(1)  # Use batch size 1 for dynamic dimensions
            else:
                shape.append(dim)
        
        # Create random input
        dummy_input = np.random.rand(*shape).astype(np.float32)
        return dummy_input
    
    def infer(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """Run inference on input tensor.
        
        Args:
            input_tensor: Preprocessed input tensor
            
        Returns:
            List of output tensors
        """
        input_name = self.input_info['name']
        
        # Run inference
        outputs = self.session.run(None, {input_name: input_tensor})
        
        return outputs
    
    def infer_image(self, image_path: str) -> Tuple[List[np.ndarray], float]:
        """Run inference on image file.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (outputs, inference_time_ms)
        """
        # Preprocess image
        input_tensor = self.preprocess_image(image_path)
        
        # Run inference with timing
        start_time = time.perf_counter()
        outputs = self.infer(input_tensor)
        end_time = time.perf_counter()
        
        inference_time_ms = (end_time - start_time) * 1000
        
        return outputs, inference_time_ms
    
    def benchmark(self, num_runs: int = 100, warmup_runs: int = 10) -> Dict[str, float]:
        """Benchmark model inference performance.
        
        Args:
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Dictionary with benchmark statistics
        """
        logger.info(f"Starting benchmark: {warmup_runs} warmup + {num_runs} runs")
        
        # Create dummy input
        dummy_input = self.create_dummy_input()
        
        # Warmup runs
        logger.info("Running warmup...")
        for _ in range(warmup_runs):
            _ = self.infer(dummy_input)
        
        # Benchmark runs
        logger.info("Running benchmark...")
        latencies = []
        
        for i in range(num_runs):
            start_time = time.perf_counter()
            outputs = self.infer(dummy_input)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            if (i + 1) % 20 == 0:
                logger.info(f"Completed {i + 1}/{num_runs} runs")
        
        # Calculate statistics
        latencies = np.array(latencies)
        stats = {
            'avg_latency_ms': float(np.mean(latencies)),
            'min_latency_ms': float(np.min(latencies)),
            'max_latency_ms': float(np.max(latencies)),
            'std_latency_ms': float(np.std(latencies)),
            'avg_fps': float(1000 / np.mean(latencies)),
            'num_runs': num_runs,
            'warmup_runs': warmup_runs
        }
        
        return stats
    
    def postprocess_yolo_outputs(self, outputs: List[np.ndarray], 
                                conf_threshold: float = 0.5,
                                iou_threshold: float = 0.45) -> List[Dict[str, Any]]:
        """Post-process YOLO model outputs.
        
        Args:
            outputs: Raw model outputs
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            
        Returns:
            List of detection dictionaries
        """
        if not outputs:
            return []
        
        # Assume first output contains detections
        detections = outputs[0]  # Shape: [batch, 84, 8400] for YOLOv8
        
        if len(detections.shape) != 3:
            logger.warning(f"Unexpected output shape: {detections.shape}")
            return []
        
        batch_size, num_classes_plus_coords, num_anchors = detections.shape
        
        results = []
        
        for batch_idx in range(batch_size):
            batch_detections = detections[batch_idx]  # [84, 8400]
            
            # Extract coordinates and scores
            # First 4 values are box coordinates (cx, cy, w, h)
            # Remaining values are class scores
            boxes = batch_detections[:4, :].T  # [8400, 4]
            scores = batch_detections[4:, :].T  # [8400, 80] for COCO classes
            
            # Get max class scores and indices
            max_scores = np.max(scores, axis=1)  # [8400]
            class_ids = np.argmax(scores, axis=1)  # [8400]
            
            # Filter by confidence threshold
            valid_indices = max_scores > conf_threshold
            
            if not np.any(valid_indices):
                continue
            
            valid_boxes = boxes[valid_indices]
            valid_scores = max_scores[valid_indices]
            valid_class_ids = class_ids[valid_indices]
            
            # Convert center format to corner format
            x1 = valid_boxes[:, 0] - valid_boxes[:, 2] / 2
            y1 = valid_boxes[:, 1] - valid_boxes[:, 3] / 2
            x2 = valid_boxes[:, 0] + valid_boxes[:, 2] / 2
            y2 = valid_boxes[:, 1] + valid_boxes[:, 3] / 2
            
            # Create detection results
            batch_results = []
            for i in range(len(valid_boxes)):
                detection = {
                    'bbox': [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
                    'confidence': float(valid_scores[i]),
                    'class_id': int(valid_class_ids[i])
                }
                batch_results.append(detection)
            
            results.extend(batch_results)
        
        return results

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="ONNX model inference tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model', '-m', required=True,
                       help='Path to ONNX model file')
    parser.add_argument('--image', '-i',
                       help='Path to input image for inference')
    parser.add_argument('--benchmark', '-b', action='store_true',
                       help='Run benchmark instead of single inference')
    parser.add_argument('--num-runs', type=int, default=100,
                       help='Number of benchmark runs')
    parser.add_argument('--warmup-runs', type=int, default=10,
                       help='Number of warmup runs')
    parser.add_argument('--providers', nargs='+',
                       choices=['CPUExecutionProvider', 'CUDAExecutionProvider'],
                       help='ONNX execution providers')
    parser.add_argument('--output', '-o',
                       help='Output file for results (JSON format)')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                       help='Confidence threshold for detections')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize inferencer
        inferencer = ONNXInferencer(args.model, args.providers)
        
        if args.benchmark:
            # Run benchmark
            logger.info("Running benchmark...")
            stats = inferencer.benchmark(args.num_runs, args.warmup_runs)
            
            # Print results
            print("\n" + "="*60)
            print("ONNX INFERENCE BENCHMARK RESULTS")
            print("="*60)
            print(f"Model: {args.model}")
            print(f"Providers: {inferencer.session.get_providers()}")
            print(f"Input shape: {inferencer.input_info['shape']}")
            print()
            print(f"Average FPS: {stats['avg_fps']:.2f}")
            print(f"Average Latency: {stats['avg_latency_ms']:.2f}ms")
            print(f"Min Latency: {stats['min_latency_ms']:.2f}ms")
            print(f"Max Latency: {stats['max_latency_ms']:.2f}ms")
            print(f"Std Latency: {stats['std_latency_ms']:.2f}ms")
            print(f"Total Runs: {stats['num_runs']} (+ {stats['warmup_runs']} warmup)")
            
            # Save results if requested
            if args.output:
                result_data = {
                    'model_path': args.model,
                    'providers': inferencer.session.get_providers(),
                    'input_info': inferencer.input_info,
                    'output_info': inferencer.output_info,
                    'benchmark_stats': stats,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                with open(args.output, 'w') as f:
                    json.dump(result_data, f, indent=2)
                
                logger.info(f"Results saved to {args.output}")
        
        elif args.image:
            # Run single inference
            logger.info(f"Running inference on {args.image}")
            outputs, inference_time = inferencer.infer_image(args.image)
            
            print(f"\nInference completed in {inference_time:.2f}ms")
            print(f"Output shapes: {[out.shape for out in outputs]}")
            
            # Try to post-process as YOLO outputs
            try:
                detections = inferencer.postprocess_yolo_outputs(
                    outputs, args.conf_threshold
                )
                
                if detections:
                    print(f"\nDetected {len(detections)} objects:")
                    for i, det in enumerate(detections[:10]):  # Show first 10
                        print(f"  {i+1}: class={det['class_id']}, conf={det['confidence']:.3f}, "
                              f"bbox={[f'{x:.1f}' for x in det['bbox']]}")
                    
                    if len(detections) > 10:
                        print(f"  ... and {len(detections) - 10} more")
                else:
                    print("No objects detected above confidence threshold")
                    
            except Exception as e:
                logger.warning(f"Could not post-process outputs as YOLO: {e}")
            
            # Save results if requested
            if args.output:
                result_data = {
                    'model_path': args.model,
                    'image_path': args.image,
                    'inference_time_ms': inference_time,
                    'output_shapes': [out.shape for out in outputs],
                    'detections': detections if 'detections' in locals() else None,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                with open(args.output, 'w') as f:
                    json.dump(result_data, f, indent=2, default=str)
                
                logger.info(f"Results saved to {args.output}")
        
        else:
            # Just show model info
            print(f"\nModel: {args.model}")
            print(f"Providers: {inferencer.session.get_providers()}")
            print(f"Input: {inferencer.input_info}")
            print(f"Outputs: {inferencer.output_info}")
            print("\nUse --image to run inference or --benchmark to test performance")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()