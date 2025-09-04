#!/usr/bin/env python3
"""
Optimize Detection Pipeline for Real-time Performance

This script optimizes the human detection pipeline for 30 FPS real-time performance
using TensorRT, ONNX, and quantization techniques.
"""

import logging
import argparse
import time
from pathlib import Path
from typing import Dict, Any

from src.backend.edge_optimizer import EdgeOptimizer, OptimizationConfig
from src.backend.detection_pipeline import create_pipeline
from src.backend.detector import YOLODetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def optimize_model_for_realtime(model_path: str, output_dir: str) -> Dict[str, str]:
    """
    Optimize YOLO model for real-time inference using various techniques.
    
    Args:
        model_path: Path to the original YOLO model
        output_dir: Directory to save optimized models
        
    Returns:
        Dictionary with paths to optimized models
    """
    logger.info(f"Starting optimization for model: {model_path}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Configuration for aggressive optimization
    config = OptimizationConfig(
        target_device="cuda",  # Use GPU for better performance
        enable_tensorrt=True,
        enable_onnx=True,
        enable_quantization=True,
        enable_mobile=False,  # Focus on desktop/edge devices
        tensorrt_precision="fp16",  # Use half precision for speed
        tensorrt_workspace_size=1 << 30,  # 1GB workspace
        input_shape=(1, 3, 416, 416),  # Smaller input size for speed
        target_fps=30.0,
        max_latency_ms=33.3,
        output_dir=str(output_path)
    )
    
    # Initialize edge optimizer with config
    optimizer = EdgeOptimizer(config)
    
    # Run optimization
    result = optimizer.optimize_model(model_path)
    
    logger.info(f"Optimization completed. Results: {result}")
    return result

def benchmark_optimized_models(model_paths: Dict[str, str], num_frames: int = 100) -> Dict[str, float]:
    """
    Benchmark different optimized models to find the best performing one.
    
    Args:
        model_paths: Dictionary of model names to paths
        num_frames: Number of frames to test
        
    Returns:
        Dictionary of model names to average FPS
    """
    results = {}
    
    for model_name, model_path in model_paths.items():
        if not Path(model_path).exists():
            logger.warning(f"Model not found: {model_path}")
            continue
            
        logger.info(f"Benchmarking {model_name}: {model_path}")
        
        try:
            # Create pipeline with optimized model
            pipeline = create_pipeline(
                model_path=model_path,
                confidence_threshold=0.5,  # Lower confidence for speed
                iou_threshold=0.4,   # Lower IoU for speed
                human_only=True,
                aerial_optimized=False    # Disable aerial mode for speed
            )
            
            # Run benchmark
            fps_values = []
            for i in range(num_frames):
                start_time = time.time()
                
                # Process dummy frame (640x640 RGB)
                import numpy as np
                dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                
                result = pipeline.process_frame(dummy_frame)
                
                end_time = time.time()
                fps = 1.0 / (end_time - start_time)
                fps_values.append(fps)
                
                if (i + 1) % 20 == 0:
                    avg_fps = sum(fps_values) / len(fps_values)
                    logger.info(f"{model_name} - Frame {i+1}/{num_frames}, Avg FPS: {avg_fps:.1f}")
            
            avg_fps = sum(fps_values) / len(fps_values)
            results[model_name] = avg_fps
            
            logger.info(f"{model_name} - Final Avg FPS: {avg_fps:.1f}")
            
        except Exception as e:
            logger.error(f"Error benchmarking {model_name}: {e}")
            results[model_name] = 0.0
    
    return results

def find_best_model(benchmark_results: Dict[str, float], target_fps: float = 30.0) -> str:
    """
    Find the best performing model that meets the target FPS.
    
    Args:
        benchmark_results: Dictionary of model names to FPS
        target_fps: Target FPS threshold
        
    Returns:
        Name of the best model
    """
    # Filter models that meet target FPS
    suitable_models = {name: fps for name, fps in benchmark_results.items() if fps >= target_fps}
    
    if suitable_models:
        # Return the fastest model that meets target
        best_model = max(suitable_models.items(), key=lambda x: x[1])
        logger.info(f"Best model for {target_fps} FPS: {best_model[0]} ({best_model[1]:.1f} FPS)")
        return best_model[0]
    else:
        # Return the fastest model overall
        if benchmark_results:
            best_model = max(benchmark_results.items(), key=lambda x: x[1])
            logger.warning(f"No model meets {target_fps} FPS target. Best available: {best_model[0]} ({best_model[1]:.1f} FPS)")
            return best_model[0]
        else:
            logger.error("No valid models found")
            return ""

def main():
    parser = argparse.ArgumentParser(description="Optimize detection pipeline for real-time performance")
    parser.add_argument("--model", required=True, help="Path to the YOLO model to optimize")
    parser.add_argument("--output-dir", default="optimized_models", help="Output directory for optimized models")
    parser.add_argument("--benchmark-frames", type=int, default=100, help="Number of frames for benchmarking")
    parser.add_argument("--target-fps", type=float, default=30.0, help="Target FPS for real-time performance")
    parser.add_argument("--skip-optimization", action="store_true", help="Skip optimization and only benchmark existing models")
    
    args = parser.parse_args()
    
    model_paths = {}
    
    if not args.skip_optimization:
        # Optimize the model
        logger.info("Starting model optimization...")
        optimization_result = optimize_model_for_realtime(args.model, args.output_dir)
        
        # Add optimized models to benchmark list
        if optimization_result:
            for key, path in optimization_result.items():
                if path and Path(path).exists():
                    model_paths[f"optimized_{key}"] = path
    
    # Add original model for comparison
    if Path(args.model).exists():
        model_paths["original"] = args.model
    
    # Add any existing optimized models
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        for model_file in output_dir.glob("*.pt"):
            model_paths[f"existing_{model_file.stem}"] = str(model_file)
        for model_file in output_dir.glob("*.onnx"):
            model_paths[f"existing_{model_file.stem}"] = str(model_file)
    
    if not model_paths:
        logger.error("No models found for benchmarking")
        return
    
    # Benchmark all models
    logger.info("Starting benchmark comparison...")
    benchmark_results = benchmark_optimized_models(model_paths, args.benchmark_frames)
    
    # Print results
    logger.info("\n=== Benchmark Results ===")
    for model_name, fps in sorted(benchmark_results.items(), key=lambda x: x[1], reverse=True):
        status = "✓" if fps >= args.target_fps else "✗"
        logger.info(f"{status} {model_name}: {fps:.1f} FPS")
    
    # Find best model
    best_model = find_best_model(benchmark_results, args.target_fps)
    if best_model:
        logger.info(f"\nRecommended model: {best_model}")
        logger.info(f"Model path: {model_paths[best_model]}")
    
    # Save results
    results_file = Path(args.output_dir) / "optimization_benchmark.json"
    import json
    with open(results_file, 'w') as f:
        json.dump({
            "benchmark_results": benchmark_results,
            "model_paths": model_paths,
            "best_model": best_model,
            "target_fps": args.target_fps,
            "timestamp": time.time()
        }, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()