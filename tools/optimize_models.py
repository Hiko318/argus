#!/usr/bin/env python3
"""
Model Optimization Pipeline for Foresight SAR System

This script provides a complete pipeline for optimizing models for deployment:
1. Export PyTorch models to ONNX
2. Convert ONNX models to TensorRT engines
3. Benchmark and validate optimized models
4. Generate deployment configurations

Usage:
    python optimize_models.py --input models/ --output optimized/ --target jetson_orin
    python optimize_models.py --model yolov8n.pt --precision fp16 --validate
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import torch
except ImportError:
    print("Error: PyTorch not found. Please install PyTorch.")
    sys.exit(1)

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    print("Warning: ONNX packages not found. Install with: pip install onnx onnxruntime")
    onnx = None
    ort = None

import numpy as np
from export_to_onnx import ModelExporter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Complete model optimization pipeline for Foresight SAR system."""
    
    # Jetson device specifications
    JETSON_SPECS = {
        'jetson_nano': {
            'gpu_memory_gb': 4,
            'recommended_precision': 'fp16',
            'max_workspace_gb': 1,
            'dla_cores': 2
        },
        'jetson_xavier': {
            'gpu_memory_gb': 32,
            'recommended_precision': 'fp16',
            'max_workspace_gb': 4,
            'dla_cores': 2
        },
        'jetson_orin': {
            'gpu_memory_gb': 64,
            'recommended_precision': 'fp16',
            'max_workspace_gb': 8,
            'dla_cores': 0  # Orin uses GPU primarily
        }
    }
    
    # Model-specific configurations
    MODEL_CONFIGS = {
        'yolov8n': {
            'input_size': (640, 640),
            'batch_sizes': [1, 4, 8],
            'precision_modes': ['fp16', 'int8'],
            'use_dla': True
        },
        'yolov8s': {
            'input_size': (640, 640),
            'batch_sizes': [1, 2, 4],
            'precision_modes': ['fp16', 'int8'],
            'use_dla': True
        },
        'yolov8m': {
            'input_size': (640, 640),
            'batch_sizes': [1, 2],
            'precision_modes': ['fp16'],
            'use_dla': False
        },
        'reid': {
            'input_size': (256, 128),
            'batch_sizes': [1, 8, 16],
            'precision_modes': ['fp16', 'int8'],
            'use_dla': True
        }
    }
    
    def __init__(self, target_device: str = 'jetson_orin', verbose: bool = False):
        self.target_device = target_device
        self.verbose = verbose
        self.exporter = ModelExporter(verbose=verbose)
        
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Validate target device
        if target_device not in self.JETSON_SPECS:
            logger.warning(f"Unknown target device: {target_device}. Using generic settings.")
            self.device_specs = {
                'gpu_memory_gb': 8,
                'recommended_precision': 'fp16',
                'max_workspace_gb': 2,
                'dla_cores': 0
            }
        else:
            self.device_specs = self.JETSON_SPECS[target_device]
    
    def detect_model_config(self, model_path: str) -> Dict:
        """Detect optimal configuration for a model."""
        model_name = Path(model_path).stem.lower()
        
        # Check for known model patterns
        for config_name, config in self.MODEL_CONFIGS.items():
            if config_name in model_name:
                return config.copy()
        
        # Default configuration
        if 'yolo' in model_name:
            return self.MODEL_CONFIGS['yolov8n'].copy()
        elif 'reid' in model_name:
            return self.MODEL_CONFIGS['reid'].copy()
        else:
            return {
                'input_size': (640, 640),
                'batch_sizes': [1],
                'precision_modes': ['fp16'],
                'use_dla': False
            }
    
    def optimize_single_model(self, model_path: str, output_dir: str,
                            precision: str = None, batch_size: int = None,
                            validate: bool = True) -> Dict:
        """Optimize a single model through the complete pipeline."""
        model_path = Path(model_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_name = model_path.stem
        results = {
            'model_name': model_name,
            'original_path': str(model_path),
            'target_device': self.target_device,
            'optimization_results': {}
        }
        
        logger.info(f"Optimizing model: {model_name}")
        
        # Get model configuration
        config = self.detect_model_config(str(model_path))
        logger.info(f"Using configuration: {config}")
        
        # Determine optimization parameters
        precisions = [precision] if precision else config['precision_modes']
        batch_sizes = [batch_size] if batch_size else config['batch_sizes']
        
        # Filter batch sizes based on device memory
        max_batch = self._get_max_batch_size(config['input_size'])
        batch_sizes = [bs for bs in batch_sizes if bs <= max_batch]
        
        for prec in precisions:
            for bs in batch_sizes:
                variant_name = f"{model_name}_{prec}_b{bs}"
                logger.info(f"Creating variant: {variant_name}")
                
                try:
                    variant_result = self._optimize_variant(
                        model_path, output_dir, variant_name, 
                        config, prec, bs, validate
                    )
                    results['optimization_results'][variant_name] = variant_result
                    
                except Exception as e:
                    logger.error(f"Failed to optimize variant {variant_name}: {e}")
                    results['optimization_results'][variant_name] = {
                        'success': False,
                        'error': str(e)
                    }
        
        # Save results
        results_file = output_dir / f"{model_name}_optimization_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _optimize_variant(self, model_path: Path, output_dir: Path, 
                         variant_name: str, config: Dict, precision: str, 
                         batch_size: int, validate: bool) -> Dict:
        """Optimize a single model variant."""
        result = {
            'precision': precision,
            'batch_size': batch_size,
            'input_size': config['input_size'],
            'success': False,
            'files': {},
            'benchmarks': {}
        }
        
        # Step 1: Export to ONNX
        onnx_path = output_dir / f"{variant_name}.onnx"
        logger.info(f"Exporting to ONNX: {onnx_path}")
        
        export_success = self.exporter.export_model(
            str(model_path),
            str(onnx_path),
            input_size=config['input_size'],
            batch_size=batch_size,
            validate=validate
        )
        
        if not export_success:
            result['error'] = 'ONNX export failed'
            return result
        
        result['files']['onnx'] = str(onnx_path)
        
        # Step 2: Convert to TensorRT
        trt_path = output_dir / f"{variant_name}.trt"
        logger.info(f"Converting to TensorRT: {trt_path}")
        
        trt_success = self._convert_to_tensorrt(
            onnx_path, trt_path, precision, batch_size, config
        )
        
        if not trt_success:
            result['error'] = 'TensorRT conversion failed'
            return result
        
        result['files']['tensorrt'] = str(trt_path)
        
        # Step 3: Benchmark models
        if validate:
            logger.info("Benchmarking optimized models")
            result['benchmarks'] = self._benchmark_models(
                str(model_path), str(onnx_path), str(trt_path), 
                config['input_size'], batch_size
            )
        
        result['success'] = True
        return result
    
    def _convert_to_tensorrt(self, onnx_path: Path, trt_path: Path, 
                           precision: str, batch_size: int, config: Dict) -> bool:
        """Convert ONNX model to TensorRT engine."""
        try:
            # Build TensorRT conversion command
            script_dir = Path(__file__).parent
            convert_script = script_dir / "convert_to_tensorrt.sh"
            
            if not convert_script.exists():
                logger.error(f"TensorRT conversion script not found: {convert_script}")
                return False
            
            cmd = [
                'bash', str(convert_script),
                '--precision', precision,
                '--batch-size', str(batch_size),
                '--device', self.target_device,
                '--workspace', f"{self.device_specs['max_workspace_gb']}GB"
            ]
            
            # Add DLA support if available and recommended
            if (self.device_specs['dla_cores'] > 0 and 
                config.get('use_dla', False) and 
                precision in ['fp16', 'int8']):
                cmd.extend(['--use-dla'])
            
            cmd.extend([str(onnx_path), str(trt_path)])
            
            logger.debug(f"TensorRT command: {' '.join(cmd)}")
            
            # Execute conversion
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                logger.info("TensorRT conversion successful")
                return True
            else:
                logger.error(f"TensorRT conversion failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("TensorRT conversion timed out")
            return False
        except Exception as e:
            logger.error(f"TensorRT conversion error: {e}")
            return False
    
    def _benchmark_models(self, pytorch_path: str, onnx_path: str, 
                         trt_path: str, input_size: Tuple[int, int], 
                         batch_size: int) -> Dict:
        """Benchmark different model formats."""
        benchmarks = {}
        
        # Create dummy input
        dummy_input = np.random.randn(batch_size, 3, input_size[0], input_size[1]).astype(np.float32)
        
        # Benchmark ONNX
        if ort is not None and os.path.exists(onnx_path):
            try:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                session = ort.InferenceSession(onnx_path, providers=providers)
                
                # Warmup
                for _ in range(5):
                    session.run(None, {'input': dummy_input})
                
                # Benchmark
                times = []
                for _ in range(100):
                    start = time.time()
                    session.run(None, {'input': dummy_input})
                    times.append(time.time() - start)
                
                benchmarks['onnx'] = {
                    'avg_time_ms': np.mean(times) * 1000,
                    'std_time_ms': np.std(times) * 1000,
                    'fps': batch_size / np.mean(times)
                }
                
            except Exception as e:
                logger.warning(f"ONNX benchmark failed: {e}")
        
        # Benchmark TensorRT (using trtexec)
        if os.path.exists(trt_path):
            try:
                cmd = [
                    'trtexec',
                    f'--loadEngine={trt_path}',
                    '--warmUp=10',
                    '--iterations=100'
                ]
                
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300
                )
                
                if result.returncode == 0:
                    # Parse trtexec output
                    output = result.stdout
                    for line in output.split('\n'):
                        if 'mean' in line and 'ms' in line:
                            # Extract timing information
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if 'mean' in part and i + 2 < len(parts):
                                    try:
                                        mean_time = float(parts[i + 2])
                                        benchmarks['tensorrt'] = {
                                            'avg_time_ms': mean_time,
                                            'fps': batch_size * 1000 / mean_time
                                        }
                                        break
                                    except ValueError:
                                        continue
                
            except Exception as e:
                logger.warning(f"TensorRT benchmark failed: {e}")
        
        return benchmarks
    
    def _get_max_batch_size(self, input_size: Tuple[int, int]) -> int:
        """Estimate maximum batch size based on device memory."""
        # Rough estimation based on input size and available memory
        memory_gb = self.device_specs['gpu_memory_gb']
        
        # Estimate memory per sample (in GB)
        # This is a rough approximation
        pixels = input_size[0] * input_size[1] * 3  # RGB
        memory_per_sample = pixels * 4 / (1024**3)  # 4 bytes per float32, convert to GB
        
        # Reserve memory for model weights and intermediate activations
        available_memory = memory_gb * 0.6  # Use 60% of available memory
        
        max_batch = int(available_memory / (memory_per_sample * 10))  # Factor of 10 for safety
        
        return max(1, min(max_batch, 16))  # Clamp between 1 and 16
    
    def optimize_directory(self, input_dir: str, output_dir: str, 
                          model_pattern: str = "*.pt") -> Dict:
        """Optimize all models in a directory."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return {}
        
        # Find all model files
        model_files = list(input_dir.glob(model_pattern))
        
        if not model_files:
            logger.warning(f"No model files found in {input_dir} with pattern {model_pattern}")
            return {}
        
        logger.info(f"Found {len(model_files)} models to optimize")
        
        results = {
            'input_directory': str(input_dir),
            'output_directory': str(output_dir),
            'target_device': self.target_device,
            'models': {}
        }
        
        for model_file in model_files:
            logger.info(f"Processing: {model_file.name}")
            
            model_output_dir = output_dir / model_file.stem
            model_result = self.optimize_single_model(
                str(model_file), str(model_output_dir)
            )
            
            results['models'][model_file.name] = model_result
        
        # Save overall results
        results_file = output_dir / "optimization_summary.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate deployment configuration
        self._generate_deployment_config(results, output_dir)
        
        return results
    
    def _generate_deployment_config(self, results: Dict, output_dir: Path):
        """Generate deployment configuration files."""
        config = {
            'target_device': self.target_device,
            'device_specs': self.device_specs,
            'optimized_models': {}
        }
        
        for model_name, model_result in results['models'].items():
            if 'optimization_results' in model_result:
                best_variant = self._select_best_variant(model_result['optimization_results'])
                if best_variant:
                    config['optimized_models'][model_name] = best_variant
        
        # Save deployment config
        config_file = output_dir / "deployment_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Deployment configuration saved: {config_file}")
    
    def _select_best_variant(self, variants: Dict) -> Optional[Dict]:
        """Select the best performing variant for deployment."""
        best_variant = None
        best_score = 0
        
        for variant_name, variant_data in variants.items():
            if not variant_data.get('success', False):
                continue
            
            # Score based on performance and precision
            score = 0
            
            # Prefer TensorRT if available
            if 'tensorrt' in variant_data.get('files', {}):
                score += 100
            
            # Prefer higher FPS
            benchmarks = variant_data.get('benchmarks', {})
            if 'tensorrt' in benchmarks:
                fps = benchmarks['tensorrt'].get('fps', 0)
                score += fps
            elif 'onnx' in benchmarks:
                fps = benchmarks['onnx'].get('fps', 0)
                score += fps * 0.8  # Slight penalty for ONNX
            
            # Prefer FP16 over FP32, INT8 gets bonus if performance is good
            if variant_data.get('precision') == 'fp16':
                score += 10
            elif variant_data.get('precision') == 'int8':
                score += 15  # Higher bonus for INT8
            
            if score > best_score:
                best_score = score
                best_variant = {
                    'variant_name': variant_name,
                    'files': variant_data.get('files', {}),
                    'precision': variant_data.get('precision'),
                    'batch_size': variant_data.get('batch_size'),
                    'benchmarks': variant_data.get('benchmarks', {}),
                    'score': score
                }
        
        return best_variant

def main():
    parser = argparse.ArgumentParser(
        description='Optimize models for Foresight SAR System deployment'
    )
    
    parser.add_argument('--model', '-m',
                       help='Path to single model file to optimize')
    parser.add_argument('--input', '-i',
                       help='Input directory containing models to optimize')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory for optimized models')
    parser.add_argument('--target', '-t', 
                       choices=['jetson_nano', 'jetson_xavier', 'jetson_orin'],
                       default='jetson_orin',
                       help='Target Jetson device')
    parser.add_argument('--precision', '-p',
                       choices=['fp32', 'fp16', 'int8'],
                       help='Precision mode (if not specified, uses model defaults)')
    parser.add_argument('--batch-size', '-b', type=int,
                       help='Batch size (if not specified, uses model defaults)')
    parser.add_argument('--pattern', default='*.pt',
                       help='File pattern for batch processing')
    parser.add_argument('--no-validate', action='store_true',
                       help='Skip model validation and benchmarking')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    if not args.model and not args.input:
        parser.error('Either --model or --input must be specified')
    
    # Create optimizer
    optimizer = ModelOptimizer(
        target_device=args.target,
        verbose=args.verbose
    )
    
    # Run optimization
    if args.model:
        # Single model optimization
        results = optimizer.optimize_single_model(
            args.model,
            args.output,
            precision=args.precision,
            batch_size=args.batch_size,
            validate=not args.no_validate
        )
        
        print("\nOptimization Results:")
        print(json.dumps(results, indent=2))
        
    else:
        # Batch optimization
        results = optimizer.optimize_directory(
            args.input,
            args.output,
            args.pattern
        )
        
        print(f"\nOptimized {len(results.get('models', {}))} models")
        print(f"Results saved to: {args.output}")
    
    logger.info("Model optimization completed!")

if __name__ == '__main__':
    main()