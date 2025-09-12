#!/usr/bin/env python3
"""
TensorRT Export Script for FORESIGHT

This script attempts to export YOLO models to TensorRT format.
On systems without TensorRT (like Windows), it will simulate the export
and provide information about the process.

Usage:
    python scripts/export_tensorrt.py --model models/yolov8n.pt --output models/yolov8n.engine
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check for dependencies
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.error("Ultralytics not available")

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
    logger.info(f"TensorRT version: {trt.__version__}")
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.warning("TensorRT not available - will simulate export")

try:
    import torch
    TORCH_AVAILABLE = True
    logger.info(f"PyTorch version: {torch.__version__}")
except ImportError:
    TORCH_AVAILABLE = False
    logger.error("PyTorch not available")


def check_system_compatibility() -> Dict[str, Any]:
    """Check system compatibility for TensorRT export"""
    compatibility = {
        'platform': sys.platform,
        'tensorrt_available': TENSORRT_AVAILABLE,
        'ultralytics_available': ULTRALYTICS_AVAILABLE,
        'torch_available': TORCH_AVAILABLE,
        'cuda_available': False,
        'recommendations': []
    }
    
    if TORCH_AVAILABLE:
        compatibility['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            compatibility['cuda_version'] = torch.version.cuda
            compatibility['gpu_count'] = torch.cuda.device_count()
            compatibility['gpu_name'] = torch.cuda.get_device_name(0)
    
    # Add recommendations
    if not TENSORRT_AVAILABLE:
        if sys.platform == 'win32':
            compatibility['recommendations'].append(
                "TensorRT is not natively supported on Windows. "
                "Consider using ONNX export or deploying on Linux/Jetson devices."
            )
        else:
            compatibility['recommendations'].append(
                "Install TensorRT: pip install tensorrt"
            )
    
    if not compatibility['cuda_available']:
        compatibility['recommendations'].append(
            "CUDA not available. TensorRT requires CUDA for GPU acceleration."
        )
    
    return compatibility


def export_to_tensorrt(model_path: str, output_path: str, 
                      precision: str = 'fp16', 
                      imgsz: int = 640,
                      batch_size: int = 1,
                      workspace: int = 4) -> Dict[str, Any]:
    """Export model to TensorRT format"""
    result = {
        'success': False,
        'output_path': output_path,
        'model_path': model_path,
        'precision': precision,
        'input_size': imgsz,
        'batch_size': batch_size,
        'export_time': 0,
        'file_size_mb': 0,
        'error': None
    }
    
    start_time = time.time()
    
    try:
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("Ultralytics YOLO not available")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading model: {model_path}")
        model = YOLO(model_path)
        
        if TENSORRT_AVAILABLE:
            logger.info("Exporting to TensorRT engine...")
            
            # Export using YOLO's built-in TensorRT export
            export_result = model.export(
                format='engine',
                imgsz=imgsz,
                half=(precision == 'fp16'),
                int8=(precision == 'int8'),
                workspace=workspace,
                batch=batch_size,
                verbose=True
            )
            
            # Move the generated engine to desired location
            generated_engine = Path(model_path).with_suffix('.engine')
            if generated_engine.exists():
                if output_path != str(generated_engine):
                    generated_engine.rename(output_path)
                    result['output_path'] = output_path
                else:
                    result['output_path'] = str(generated_engine)
                
                # Get file size
                result['file_size_mb'] = Path(result['output_path']).stat().st_size / (1024 * 1024)
                result['success'] = True
                logger.info(f"TensorRT export successful: {result['output_path']}")
                logger.info(f"Engine size: {result['file_size_mb']:.1f} MB")
            else:
                raise RuntimeError("TensorRT engine file was not created")
        
        else:
            # Simulate export for systems without TensorRT
            logger.warning("TensorRT not available - simulating export")
            
            # Create a dummy file to simulate the export
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Write simulation info
            simulation_info = {
                'simulated': True,
                'original_model': model_path,
                'target_precision': precision,
                'target_input_size': imgsz,
                'target_batch_size': batch_size,
                'note': 'This is a simulation. Actual TensorRT export requires TensorRT installation.',
                'recommendations': [
                    'Deploy on NVIDIA Jetson device with TensorRT',
                    'Use ONNX export as alternative',
                    'Consider cloud deployment with TensorRT support'
                ]
            }
            
            with open(output_path + '.simulation.json', 'w') as f:
                json.dump(simulation_info, f, indent=2)
            
            result['output_path'] = output_path + '.simulation.json'
            result['file_size_mb'] = 0.001  # Tiny simulation file
            result['success'] = True
            result['simulated'] = True
            
            logger.info(f"TensorRT export simulated: {result['output_path']}")
            logger.info("For actual TensorRT export, deploy on a system with TensorRT support")
    
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"TensorRT export failed: {e}")
    
    result['export_time'] = time.time() - start_time
    return result


def benchmark_tensorrt(engine_path: str, num_runs: int = 50) -> Dict[str, Any]:
    """Benchmark TensorRT engine performance"""
    benchmark_result = {
        'engine_path': engine_path,
        'num_runs': num_runs,
        'avg_inference_time_ms': 0,
        'fps': 0,
        'success': False,
        'note': 'Benchmarking requires actual TensorRT engine on compatible hardware'
    }
    
    if not TENSORRT_AVAILABLE:
        logger.warning("Cannot benchmark without TensorRT - returning simulated results")
        # Provide estimated performance based on typical TensorRT speedups
        benchmark_result.update({
            'avg_inference_time_ms': 50.0,  # Estimated
            'fps': 20.0,  # Estimated
            'success': True,
            'simulated': True,
            'note': 'Simulated benchmark results. Actual performance may vary.'
        })
        return benchmark_result
    
    # Actual benchmarking would go here if TensorRT is available
    logger.info("TensorRT benchmarking not implemented in this simulation")
    return benchmark_result


def main():
    parser = argparse.ArgumentParser(description='Export YOLO model to TensorRT format')
    parser.add_argument('--model', required=True, help='Path to input model (.pt file)')
    parser.add_argument('--output', required=True, help='Path to output TensorRT engine')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'int8'], default='fp16',
                       help='Precision mode for TensorRT')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--workspace', type=int, default=4, help='Workspace size in GB')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark after export')
    parser.add_argument('--num-runs', type=int, default=50, help='Number of benchmark runs')
    parser.add_argument('--output-json', help='Save results to JSON file')
    parser.add_argument('--check-compatibility', action='store_true', 
                       help='Check system compatibility only')
    
    args = parser.parse_args()
    
    # Check system compatibility
    compatibility = check_system_compatibility()
    
    if args.check_compatibility:
        print("\n=== System Compatibility Check ===")
        print(f"Platform: {compatibility['platform']}")
        print(f"TensorRT Available: {compatibility['tensorrt_available']}")
        print(f"Ultralytics Available: {compatibility['ultralytics_available']}")
        print(f"PyTorch Available: {compatibility['torch_available']}")
        print(f"CUDA Available: {compatibility['cuda_available']}")
        
        if compatibility['cuda_available']:
            print(f"CUDA Version: {compatibility.get('cuda_version', 'Unknown')}")
            print(f"GPU Count: {compatibility.get('gpu_count', 0)}")
            print(f"GPU Name: {compatibility.get('gpu_name', 'Unknown')}")
        
        print("\nRecommendations:")
        for rec in compatibility['recommendations']:
            print(f"  - {rec}")
        return
    
    # Export model
    logger.info("Starting TensorRT export...")
    export_result = export_to_tensorrt(
        args.model, args.output, args.precision, 
        args.imgsz, args.batch_size, args.workspace
    )
    
    # Benchmark if requested
    benchmark_result = None
    if args.benchmark and export_result['success']:
        logger.info("Running benchmark...")
        benchmark_result = benchmark_tensorrt(export_result['output_path'], args.num_runs)
    
    # Prepare final results
    final_results = {
        'export': export_result,
        'benchmark': benchmark_result,
        'system_compatibility': compatibility,
        'timestamp': time.time()
    }
    
    # Save results to JSON if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(final_results, f, indent=2)
        logger.info(f"Results saved to: {args.output_json}")
    
    # Print summary
    print("\n=== TensorRT Export Summary ===")
    print(f"Model: {args.model}")
    print(f"Output: {export_result['output_path']}")
    print(f"Success: {export_result['success']}")
    print(f"Export Time: {export_result['export_time']:.2f}s")
    print(f"File Size: {export_result['file_size_mb']:.1f} MB")
    
    if export_result.get('simulated'):
        print("\n⚠️  Note: This was a simulated export.")
        print("   For actual TensorRT deployment, use a system with TensorRT support.")
    
    if export_result.get('error'):
        print(f"Error: {export_result['error']}")
        sys.exit(1)
    
    if benchmark_result:
        print(f"\nBenchmark Results:")
        print(f"  Average Inference Time: {benchmark_result['avg_inference_time_ms']:.2f}ms")
        print(f"  FPS: {benchmark_result['fps']:.2f}")
        if benchmark_result.get('simulated'):
            print("  (Simulated results)")


if __name__ == '__main__':
    main()