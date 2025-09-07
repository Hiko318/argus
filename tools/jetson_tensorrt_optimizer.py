#!/usr/bin/env python3
"""
Jetson TensorRT Model Optimization Pipeline

Comprehensive optimization pipeline for deploying models on NVIDIA Jetson devices:
- Automatic model detection and validation
- ONNX export with optimized configurations
- TensorRT engine generation with device-specific settings
- Performance benchmarking and validation
- Deployment configuration generation

Supported Models:
- YOLOv8 detection models (n, s, m, l, x)
- ReID embedding models
- Custom PyTorch models

Target Devices:
- Jetson Nano (4GB)
- Jetson Xavier NX/AGX (8-32GB)
- Jetson Orin Nano/NX/AGX (8-64GB)

Author: Foresight AI Team
Date: 2024
"""

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum

import torch
import numpy as np

# Optional imports with fallbacks
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    trt = None

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    onnx = None
    ort = None

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False
    cuda = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JetsonDevice(Enum):
    """Supported Jetson devices with their specifications"""
    NANO = {
        'name': 'Jetson Nano',
        'gpu_memory': 4,  # GB
        'max_workspace': 1,  # GB
        'dla_cores': 2,
        'recommended_precision': 'fp16',
        'max_batch_size': 4
    }
    XAVIER_NX = {
        'name': 'Jetson Xavier NX',
        'gpu_memory': 8,
        'max_workspace': 2,
        'dla_cores': 2,
        'recommended_precision': 'fp16',
        'max_batch_size': 8
    }
    XAVIER_AGX = {
        'name': 'Jetson Xavier AGX',
        'gpu_memory': 32,
        'max_workspace': 4,
        'dla_cores': 2,
        'recommended_precision': 'fp16',
        'max_batch_size': 16
    }
    ORIN_NANO = {
        'name': 'Jetson Orin Nano',
        'gpu_memory': 8,
        'max_workspace': 2,
        'dla_cores': 0,
        'recommended_precision': 'fp16',
        'max_batch_size': 8
    }
    ORIN_NX = {
        'name': 'Jetson Orin NX',
        'gpu_memory': 16,
        'max_workspace': 4,
        'dla_cores': 0,
        'recommended_precision': 'fp16',
        'max_batch_size': 16
    }
    ORIN_AGX = {
        'name': 'Jetson Orin AGX',
        'gpu_memory': 64,
        'max_workspace': 8,
        'dla_cores': 0,
        'recommended_precision': 'fp16',
        'max_batch_size': 32
    }

class ModelType(Enum):
    """Supported model types"""
    YOLOV8_DETECTION = 'yolov8_detection'
    REID_EMBEDDING = 'reid_embedding'
    CUSTOM_PYTORCH = 'custom_pytorch'

@dataclass
class OptimizationConfig:
    """Configuration for model optimization"""
    device: JetsonDevice
    precision: str = 'fp16'  # fp32, fp16, int8
    batch_size: int = 1
    input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640)
    workspace_size: int = 2  # GB
    enable_dla: bool = False
    dynamic_shapes: bool = False
    calibration_data: Optional[str] = None
    validate_output: bool = True
    benchmark_iterations: int = 100

class JetsonTensorRTOptimizer:
    """Main optimizer class for Jetson TensorRT deployment"""
    
    def __init__(self, config: OptimizationConfig, output_dir: str = 'optimized_models'):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate environment
        self._validate_environment()
        
        # Device detection
        self.detected_device = self._detect_jetson_device()
        if self.detected_device:
            logger.info(f"Detected Jetson device: {self.detected_device.value['name']}")
        
    def _validate_environment(self):
        """Validate that required packages are available"""
        if not ULTRALYTICS_AVAILABLE:
            logger.warning("Ultralytics not available. YOLOv8 optimization will be limited.")
        
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT not available. Please install TensorRT for Jetson.")
        
        if not ONNX_AVAILABLE:
            logger.warning("ONNX not available. Install with: pip install onnx onnxruntime")
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. Please ensure CUDA is properly installed.")
        
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        
    def _detect_jetson_device(self) -> Optional[JetsonDevice]:
        """Auto-detect Jetson device type"""
        try:
            if Path('/etc/nv_tegra_release').exists():
                with open('/etc/nv_tegra_release', 'r') as f:
                    tegra_info = f.read()
                
                if 'R32' in tegra_info:
                    return JetsonDevice.NANO
                elif 'R34' in tegra_info:
                    # Distinguish between Xavier NX and AGX based on memory
                    gpu_memory = self._get_gpu_memory()
                    return JetsonDevice.XAVIER_AGX if gpu_memory > 16 else JetsonDevice.XAVIER_NX
                elif 'R35' in tegra_info:
                    # Distinguish between Orin variants based on memory
                    gpu_memory = self._get_gpu_memory()
                    if gpu_memory >= 32:
                        return JetsonDevice.ORIN_AGX
                    elif gpu_memory >= 12:
                        return JetsonDevice.ORIN_NX
                    else:
                        return JetsonDevice.ORIN_NANO
        except Exception as e:
            logger.warning(f"Could not detect Jetson device: {e}")
        
        return None
    
    def _get_gpu_memory(self) -> int:
        """Get GPU memory in GB"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory // (1024**3)
        except Exception:
            pass
        return 0
    
    def detect_model_type(self, model_path: str) -> ModelType:
        """Detect the type of model from file path and structure"""
        model_path = Path(model_path)
        
        # Check filename patterns
        if 'yolo' in model_path.name.lower():
            return ModelType.YOLOV8_DETECTION
        elif 'reid' in model_path.name.lower():
            return ModelType.REID_EMBEDDING
        
        # Try to load and inspect model
        try:
            if ULTRALYTICS_AVAILABLE:
                model = YOLO(str(model_path))
                return ModelType.YOLOV8_DETECTION
        except Exception:
            pass
        
        # Default to custom PyTorch
        return ModelType.CUSTOM_PYTORCH
    
    def optimize_model(self, model_path: str, model_name: Optional[str] = None) -> Dict[str, str]:
        """Optimize a single model for Jetson deployment"""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if model_name is None:
            model_name = model_path.stem
        
        logger.info(f"Starting optimization for {model_name}")
        
        # Detect model type
        model_type = self.detect_model_type(str(model_path))
        logger.info(f"Detected model type: {model_type.value}")
        
        # Create output directory for this model
        model_output_dir = self.output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'model_name': model_name,
            'model_type': model_type.value,
            'original_path': str(model_path),
            'optimization_config': asdict(self.config),
            'files': {},
            'benchmarks': {},
            'success': False
        }
        
        try:
            # Step 1: Export to ONNX
            onnx_path = self._export_to_onnx(model_path, model_output_dir, model_type)
            if onnx_path:
                results['files']['onnx'] = str(onnx_path)
                logger.info(f"ONNX export successful: {onnx_path}")
            
            # Step 2: Convert to TensorRT
            if onnx_path:
                trt_path = self._convert_to_tensorrt(onnx_path, model_output_dir, model_name)
                if trt_path:
                    results['files']['tensorrt'] = str(trt_path)
                    logger.info(f"TensorRT conversion successful: {trt_path}")
            
            # Step 3: Benchmark models
            if self.config.validate_output and 'tensorrt' in results['files']:
                benchmarks = self._benchmark_model(
                    str(model_path), 
                    results['files'].get('onnx'),
                    results['files']['tensorrt'],
                    model_type
                )
                results['benchmarks'] = benchmarks
            
            # Step 4: Generate deployment config
            deploy_config = self._generate_deployment_config(results, model_type)
            config_path = model_output_dir / 'deployment_config.json'
            with open(config_path, 'w') as f:
                json.dump(deploy_config, f, indent=2)
            results['files']['deployment_config'] = str(config_path)
            
            results['success'] = True
            logger.info(f"Optimization completed successfully for {model_name}")
            
        except Exception as e:
            logger.error(f"Optimization failed for {model_name}: {e}")
            results['error'] = str(e)
        
        return results
    
    def _export_to_onnx(self, model_path: Path, output_dir: Path, model_type: ModelType) -> Optional[Path]:
        """Export model to ONNX format"""
        onnx_path = output_dir / f"{model_path.stem}.onnx"
        
        try:
            if model_type == ModelType.YOLOV8_DETECTION and ULTRALYTICS_AVAILABLE:
                model = YOLO(str(model_path))
                model.export(
                    format='onnx',
                    imgsz=self.config.input_shape[2:],
                    batch=self.config.batch_size,
                    simplify=True,
                    opset=11,
                    dynamic=self.config.dynamic_shapes
                )
                
                # Move exported file to desired location
                exported_file = model_path.with_suffix('.onnx')
                if exported_file.exists():
                    shutil.move(str(exported_file), str(onnx_path))
                
            elif model_type == ModelType.REID_EMBEDDING:
                # Custom ONNX export for ReID models
                model = torch.load(str(model_path), map_location='cuda')
                model.eval()
                
                dummy_input = torch.randn(self.config.input_shape).cuda()
                
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    } if self.config.dynamic_shapes else None
                )
            
            else:
                # Generic PyTorch model export
                model = torch.load(str(model_path), map_location='cuda')
                model.eval()
                
                dummy_input = torch.randn(self.config.input_shape).cuda()
                
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True
                )
            
            # Validate ONNX model
            if ONNX_AVAILABLE and onnx_path.exists():
                onnx_model = onnx.load(str(onnx_path))
                onnx.checker.check_model(onnx_model)
                logger.info("ONNX model validation passed")
            
            return onnx_path if onnx_path.exists() else None
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return None
    
    def _convert_to_tensorrt(self, onnx_path: Path, output_dir: Path, model_name: str) -> Optional[Path]:
        """Convert ONNX model to TensorRT engine"""
        trt_path = output_dir / f"{model_name}_{self.config.precision}.trt"
        
        try:
            # Build TensorRT engine using trtexec
            cmd = [
                'trtexec',
                f'--onnx={onnx_path}',
                f'--saveEngine={trt_path}',
                f'--workspace={self.config.workspace_size * 1024}',  # Convert GB to MB
                '--verbose'
            ]
            
            # Add precision flags
            if self.config.precision == 'fp16':
                cmd.append('--fp16')
            elif self.config.precision == 'int8':
                cmd.append('--int8')
                if self.config.calibration_data:
                    cmd.append(f'--calib={self.config.calibration_data}')
            
            # Add batch size
            if not self.config.dynamic_shapes:
                cmd.append(f'--explicitBatch')
            
            # Add DLA support if enabled
            if self.config.enable_dla and self.config.device.value['dla_cores'] > 0:
                cmd.extend(['--useDLACore=0', '--allowGPUFallback'])
            
            logger.info(f"Running TensorRT conversion: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode == 0:
                logger.info("TensorRT conversion successful")
                return trt_path if trt_path.exists() else None
            else:
                logger.error(f"TensorRT conversion failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("TensorRT conversion timed out")
            return None
        except Exception as e:
            logger.error(f"TensorRT conversion failed: {e}")
            return None
    
    def _benchmark_model(self, original_path: str, onnx_path: Optional[str], 
                        trt_path: str, model_type: ModelType) -> Dict[str, float]:
        """Benchmark model performance"""
        benchmarks = {}
        
        try:
            # Benchmark TensorRT engine
            if Path(trt_path).exists():
                trt_fps = self._benchmark_tensorrt_engine(trt_path)
                benchmarks['tensorrt_fps'] = trt_fps
                logger.info(f"TensorRT FPS: {trt_fps:.2f}")
            
            # Benchmark ONNX model
            if onnx_path and ONNX_AVAILABLE and Path(onnx_path).exists():
                onnx_fps = self._benchmark_onnx_model(onnx_path)
                benchmarks['onnx_fps'] = onnx_fps
                logger.info(f"ONNX FPS: {onnx_fps:.2f}")
            
            # Benchmark original PyTorch model
            if model_type == ModelType.YOLOV8_DETECTION and ULTRALYTICS_AVAILABLE:
                pytorch_fps = self._benchmark_yolo_model(original_path)
                benchmarks['pytorch_fps'] = pytorch_fps
                logger.info(f"PyTorch FPS: {pytorch_fps:.2f}")
            
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
        
        return benchmarks
    
    def _benchmark_tensorrt_engine(self, engine_path: str) -> float:
        """Benchmark TensorRT engine using trtexec"""
        try:
            cmd = [
                'trtexec',
                f'--loadEngine={engine_path}',
                f'--iterations={self.config.benchmark_iterations}',
                '--warmUp=10',
                '--avgRuns=10'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                # Parse FPS from output
                for line in result.stdout.split('\n'):
                    if 'mean' in line and 'ms' in line:
                        # Extract mean inference time and convert to FPS
                        import re
                        match = re.search(r'mean = ([\d.]+) ms', line)
                        if match:
                            mean_time_ms = float(match.group(1))
                            return 1000.0 / mean_time_ms
            
            return 0.0
            
        except Exception as e:
            logger.error(f"TensorRT benchmarking failed: {e}")
            return 0.0
    
    def _benchmark_onnx_model(self, onnx_path: str) -> float:
        """Benchmark ONNX model using ONNXRuntime"""
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(onnx_path, providers=providers)
            
            # Get input shape
            input_shape = session.get_inputs()[0].shape
            if isinstance(input_shape[0], str):  # Dynamic batch size
                input_shape[0] = self.config.batch_size
            
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            input_name = session.get_inputs()[0].name
            
            # Warmup
            for _ in range(10):
                session.run(None, {input_name: dummy_input})
            
            # Benchmark
            start_time = time.time()
            for _ in range(self.config.benchmark_iterations):
                session.run(None, {input_name: dummy_input})
            end_time = time.time()
            
            avg_time = (end_time - start_time) / self.config.benchmark_iterations
            return 1.0 / avg_time
            
        except Exception as e:
            logger.error(f"ONNX benchmarking failed: {e}")
            return 0.0
    
    def _benchmark_yolo_model(self, model_path: str) -> float:
        """Benchmark YOLOv8 model using Ultralytics"""
        try:
            model = YOLO(model_path)
            
            # Create dummy image
            dummy_image = np.random.randint(
                0, 255, 
                (self.config.input_shape[2], self.config.input_shape[3], 3), 
                dtype=np.uint8
            )
            
            # Warmup
            for _ in range(10):
                model(dummy_image, verbose=False)
            
            # Benchmark
            start_time = time.time()
            for _ in range(self.config.benchmark_iterations):
                model(dummy_image, verbose=False)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / self.config.benchmark_iterations
            return 1.0 / avg_time
            
        except Exception as e:
            logger.error(f"YOLO benchmarking failed: {e}")
            return 0.0
    
    def _generate_deployment_config(self, results: Dict, model_type: ModelType) -> Dict:
        """Generate deployment configuration"""
        config = {
            'model_info': {
                'name': results['model_name'],
                'type': model_type.value,
                'input_shape': self.config.input_shape,
                'precision': self.config.precision,
                'batch_size': self.config.batch_size
            },
            'device_config': {
                'target_device': self.config.device.value['name'],
                'gpu_memory_gb': self.config.device.value['gpu_memory'],
                'workspace_size_gb': self.config.workspace_size,
                'enable_dla': self.config.enable_dla
            },
            'files': results['files'],
            'performance': results.get('benchmarks', {}),
            'deployment_notes': self._generate_deployment_notes(model_type)
        }
        
        return config
    
    def _generate_deployment_notes(self, model_type: ModelType) -> List[str]:
        """Generate deployment notes and recommendations"""
        notes = [
            f"Optimized for {self.config.device.value['name']}",
            f"Precision: {self.config.precision}",
            f"Batch size: {self.config.batch_size}"
        ]
        
        if model_type == ModelType.YOLOV8_DETECTION:
            notes.extend([
                "Use TensorRT engine for best performance",
                "Consider dynamic shapes for variable input sizes",
                "Monitor GPU memory usage during inference"
            ])
        elif model_type == ModelType.REID_EMBEDDING:
            notes.extend([
                "Batch processing recommended for efficiency",
                "Consider feature caching for repeated queries"
            ])
        
        if self.config.precision == 'int8':
            notes.append("INT8 quantization may affect accuracy - validate thoroughly")
        
        return notes
    
    def optimize_directory(self, input_dir: str, pattern: str = '*.pt') -> Dict[str, Dict]:
        """Optimize all models in a directory"""
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        model_files = list(input_path.glob(pattern))
        if not model_files:
            logger.warning(f"No model files found matching pattern: {pattern}")
            return {}
        
        logger.info(f"Found {len(model_files)} model files to optimize")
        
        results = {}
        for model_file in model_files:
            try:
                result = self.optimize_model(str(model_file))
                results[model_file.name] = result
            except Exception as e:
                logger.error(f"Failed to optimize {model_file.name}: {e}")
                results[model_file.name] = {'success': False, 'error': str(e)}
        
        # Generate summary report
        self._generate_summary_report(results)
        
        return results
    
    def _generate_summary_report(self, results: Dict[str, Dict]):
        """Generate optimization summary report"""
        report_path = self.output_dir / 'optimization_summary.json'
        
        summary = {
            'optimization_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device_config': asdict(self.config),
            'total_models': len(results),
            'successful_optimizations': sum(1 for r in results.values() if r.get('success', False)),
            'failed_optimizations': sum(1 for r in results.values() if not r.get('success', False)),
            'results': results
        }
        
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Optimization summary saved to: {report_path}")

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Jetson TensorRT Model Optimization Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize single model for Jetson Orin
  python jetson_tensorrt_optimizer.py --model yolov8n.pt --device orin_agx --precision fp16
  
  # Optimize all models in directory
  python jetson_tensorrt_optimizer.py --input models/ --device xavier_nx --batch-size 4
  
  # Optimize with INT8 quantization
  python jetson_tensorrt_optimizer.py --model yolov8s.pt --precision int8 --calibration-data calibration/
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--model', type=str, help='Path to single model file')
    input_group.add_argument('--input', type=str, help='Input directory containing models')
    
    # Device configuration
    parser.add_argument('--device', type=str, 
                       choices=['nano', 'xavier_nx', 'xavier_agx', 'orin_nano', 'orin_nx', 'orin_agx'],
                       help='Target Jetson device (auto-detect if not specified)')
    
    # Optimization parameters
    parser.add_argument('--precision', type=str, choices=['fp32', 'fp16', 'int8'], 
                       default='fp16', help='Precision mode')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--input-shape', type=str, default='1,3,640,640',
                       help='Input shape as comma-separated values')
    parser.add_argument('--workspace', type=int, default=2, help='Workspace size in GB')
    parser.add_argument('--enable-dla', action='store_true', help='Enable DLA acceleration')
    parser.add_argument('--dynamic-shapes', action='store_true', help='Enable dynamic input shapes')
    parser.add_argument('--calibration-data', type=str, help='Path to calibration data for INT8')
    
    # Output options
    parser.add_argument('--output', type=str, default='optimized_models', help='Output directory')
    parser.add_argument('--pattern', type=str, default='*.pt', help='File pattern for directory input')
    
    # Validation options
    parser.add_argument('--no-validate', action='store_true', help='Skip model validation')
    parser.add_argument('--benchmark-iterations', type=int, default=100, help='Benchmark iterations')
    
    # Logging
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse input shape
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    # Map device string to enum
    device_map = {
        'nano': JetsonDevice.NANO,
        'xavier_nx': JetsonDevice.XAVIER_NX,
        'xavier_agx': JetsonDevice.XAVIER_AGX,
        'orin_nano': JetsonDevice.ORIN_NANO,
        'orin_nx': JetsonDevice.ORIN_NX,
        'orin_agx': JetsonDevice.ORIN_AGX
    }
    
    # Create configuration
    config = OptimizationConfig(
        device=device_map.get(args.device, JetsonDevice.ORIN_NX),  # Default to Orin NX
        precision=args.precision,
        batch_size=args.batch_size,
        input_shape=input_shape,
        workspace_size=args.workspace,
        enable_dla=args.enable_dla,
        dynamic_shapes=args.dynamic_shapes,
        calibration_data=args.calibration_data,
        validate_output=not args.no_validate,
        benchmark_iterations=args.benchmark_iterations
    )
    
    # Create optimizer
    optimizer = JetsonTensorRTOptimizer(config, args.output)
    
    try:
        if args.model:
            # Optimize single model
            result = optimizer.optimize_model(args.model)
            if result['success']:
                logger.info("Optimization completed successfully!")
                print(json.dumps(result, indent=2))
            else:
                logger.error("Optimization failed!")
                sys.exit(1)
        else:
            # Optimize directory
            results = optimizer.optimize_directory(args.input, args.pattern)
            successful = sum(1 for r in results.values() if r.get('success', False))
            total = len(results)
            logger.info(f"Optimization completed: {successful}/{total} models successful")
            
            if successful == 0:
                sys.exit(1)
    
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()