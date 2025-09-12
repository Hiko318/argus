#!/usr/bin/env python3
"""
Model Inference Benchmark Script for FORESIGHT System

Benchmarks YOLO model inference performance on CPU and GPU.
Tests different model sizes and measures FPS, latency, and memory usage.
"""

import os
import sys
import time
import argparse
import logging
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from ultralytics import YOLO
    import torch
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Install with: pip install ultralytics torch")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    model_name: str
    device: str
    image_size: Tuple[int, int]
    batch_size: int
    avg_fps: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    memory_usage_mb: float
    gpu_memory_mb: Optional[float]
    total_inferences: int
    warmup_runs: int
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'device': self.device,
            'image_size': self.image_size,
            'batch_size': self.batch_size,
            'avg_fps': round(self.avg_fps, 2),
            'avg_latency_ms': round(self.avg_latency_ms, 2),
            'min_latency_ms': round(self.min_latency_ms, 2),
            'max_latency_ms': round(self.max_latency_ms, 2),
            'memory_usage_mb': round(self.memory_usage_mb, 2),
            'gpu_memory_mb': self.gpu_memory_mb,
            'total_inferences': self.total_inferences,
            'warmup_runs': self.warmup_runs
        }

class ModelBenchmark:
    """Benchmark YOLO models for inference performance."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.device = self._get_best_device()
        logger.info(f"Using device: {self.device}")
        
    def _get_best_device(self) -> str:
        """Determine the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _get_memory_usage(self) -> Tuple[float, Optional[float]]:
        """Get current memory usage in MB."""
        # System memory
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # GPU memory
        gpu_memory_mb = None
        if torch.cuda.is_available() and self.device == "cuda":
            gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            
        return memory_mb, gpu_memory_mb
    
    def _create_test_images(self, image_size: Tuple[int, int], batch_size: int) -> np.ndarray:
        """Create random test images."""
        if batch_size == 1:
            return np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
        else:
            return np.random.randint(0, 255, (batch_size, *image_size, 3), dtype=np.uint8)
    
    def benchmark_model(self, 
                       model_path: str,
                       image_size: Tuple[int, int] = (640, 640),
                       batch_size: int = 1,
                       num_runs: int = 100,
                       warmup_runs: int = 10) -> BenchmarkResult:
        """Benchmark a single model."""
        
        logger.info(f"Benchmarking {model_path} on {self.device}")
        logger.info(f"Image size: {image_size}, Batch size: {batch_size}")
        
        # Load model
        try:
            model = YOLO(model_path)
            model.to(self.device)
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            raise
        
        # Create test images
        test_images = self._create_test_images(image_size, batch_size)
        
        # Warmup runs
        logger.info(f"Running {warmup_runs} warmup iterations...")
        for _ in range(warmup_runs):
            _ = model(test_images, verbose=False)
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Benchmark runs
        logger.info(f"Running {num_runs} benchmark iterations...")
        latencies = []
        
        start_memory, start_gpu_memory = self._get_memory_usage()
        
        for i in range(num_runs):
            start_time = time.perf_counter()
            
            # Run inference
            results = model(test_images, verbose=False)
            
            # Synchronize if using GPU
            if torch.cuda.is_available() and self.device == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            if (i + 1) % 20 == 0:
                logger.info(f"Completed {i + 1}/{num_runs} iterations")
        
        end_memory, end_gpu_memory = self._get_memory_usage()
        
        # Calculate statistics
        avg_latency_ms = np.mean(latencies)
        min_latency_ms = np.min(latencies)
        max_latency_ms = np.max(latencies)
        avg_fps = 1000 / avg_latency_ms * batch_size
        
        memory_usage_mb = end_memory - start_memory
        gpu_memory_mb = None
        if start_gpu_memory is not None and end_gpu_memory is not None:
            gpu_memory_mb = end_gpu_memory - start_gpu_memory
        
        model_name = Path(model_path).stem
        
        result = BenchmarkResult(
            model_name=model_name,
            device=self.device,
            image_size=image_size,
            batch_size=batch_size,
            avg_fps=avg_fps,
            avg_latency_ms=avg_latency_ms,
            min_latency_ms=min_latency_ms,
            max_latency_ms=max_latency_ms,
            memory_usage_mb=memory_usage_mb,
            gpu_memory_mb=gpu_memory_mb,
            total_inferences=num_runs,
            warmup_runs=warmup_runs
        )
        
        logger.info(f"Benchmark complete: {avg_fps:.2f} FPS, {avg_latency_ms:.2f}ms avg latency")
        return result
    
    def benchmark_all_models(self, 
                           image_sizes: List[Tuple[int, int]] = [(640, 640)],
                           batch_sizes: List[int] = [1],
                           num_runs: int = 100) -> List[BenchmarkResult]:
        """Benchmark all available models."""
        
        # Find all .pt files in models directory
        model_files = list(self.models_dir.glob("*.pt"))
        if not model_files:
            logger.warning(f"No .pt files found in {self.models_dir}")
            return []
        
        results = []
        
        for model_file in model_files:
            for image_size in image_sizes:
                for batch_size in batch_sizes:
                    try:
                        result = self.benchmark_model(
                            str(model_file),
                            image_size=image_size,
                            batch_size=batch_size,
                            num_runs=num_runs
                        )
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Failed to benchmark {model_file}: {e}")
                        continue
        
        return results
    
    def save_results(self, results: List[BenchmarkResult], output_file: str):
        """Save benchmark results to JSON file."""
        output_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': self.device,
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / 1024**3,
                'gpu_available': torch.cuda.is_available(),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            },
            'results': [result.to_dict() for result in results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    def print_summary(self, results: List[BenchmarkResult]):
        """Print a summary of benchmark results."""
        if not results:
            logger.warning("No results to display")
            return
        
        print("\n" + "="*80)
        print("FORESIGHT MODEL INFERENCE BENCHMARK RESULTS")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Total benchmarks: {len(results)}")
        print()
        
        # Group by model
        by_model = {}
        for result in results:
            if result.model_name not in by_model:
                by_model[result.model_name] = []
            by_model[result.model_name].append(result)
        
        for model_name, model_results in by_model.items():
            print(f"Model: {model_name}")
            print("-" * 40)
            
            for result in model_results:
                print(f"  Image Size: {result.image_size[0]}x{result.image_size[1]}, Batch: {result.batch_size}")
                print(f"    FPS: {result.avg_fps:.2f}")
                print(f"    Latency: {result.avg_latency_ms:.2f}ms (min: {result.min_latency_ms:.2f}, max: {result.max_latency_ms:.2f})")
                print(f"    Memory: {result.memory_usage_mb:.2f}MB")
                if result.gpu_memory_mb is not None:
                    print(f"    GPU Memory: {result.gpu_memory_mb:.2f}MB")
                print()
            print()

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO model inference performance",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--models-dir", default="models", help="Directory containing model files")
    parser.add_argument("--model", help="Specific model file to benchmark")
    parser.add_argument("--image-size", type=int, nargs=2, default=[640, 640], help="Image size (width height)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of benchmark runs")
    parser.add_argument("--warmup-runs", type=int, default=10, help="Number of warmup runs")
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], help="Force specific device")
    
    args = parser.parse_args()
    
    benchmark = ModelBenchmark(args.models_dir)
    
    # Override device if specified
    if args.device:
        benchmark.device = args.device
        logger.info(f"Using forced device: {args.device}")
    
    try:
        if args.model:
            # Benchmark specific model
            result = benchmark.benchmark_model(
                args.model,
                image_size=tuple(args.image_size),
                batch_size=args.batch_size,
                num_runs=args.num_runs,
                warmup_runs=args.warmup_runs
            )
            results = [result]
        else:
            # Benchmark all models
            results = benchmark.benchmark_all_models(
                image_sizes=[tuple(args.image_size)],
                batch_sizes=[args.batch_size],
                num_runs=args.num_runs
            )
        
        # Print summary
        benchmark.print_summary(results)
        
        # Save results if requested
        if args.output:
            benchmark.save_results(results, args.output)
        
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()