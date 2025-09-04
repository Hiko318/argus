#!/usr/bin/env python3
"""
CUDA and TensorRT Optimization Module

This module provides CUDA and TensorRT optimization capabilities for edge devices,
particularly NVIDIA Jetson platforms.
"""

import os
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    import torch
    import torchvision
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available")

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logging.warning("TensorRT not available")

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX not available")

import numpy as np


class OptimizationLevel(Enum):
    """Optimization level enumeration"""
    NONE = "none"
    BASIC = "basic"
    AGGRESSIVE = "aggressive"
    MAX_PERFORMANCE = "max_performance"


class PrecisionMode(Enum):
    """Precision mode enumeration"""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    MIXED = "mixed"


@dataclass
class DeviceInfo:
    """Device information structure"""
    name: str
    compute_capability: Tuple[int, int]
    total_memory: int
    available_memory: int
    cuda_cores: int
    tensor_cores: bool
    supports_fp16: bool
    supports_int8: bool


@dataclass
class OptimizationResult:
    """Optimization result structure"""
    success: bool
    original_size: int
    optimized_size: int
    compression_ratio: float
    inference_time_ms: float
    throughput_fps: float
    precision_mode: PrecisionMode
    optimization_level: OptimizationLevel
    error_message: Optional[str] = None


class CUDAManager:
    """CUDA device manager"""
    
    def __init__(self):
        self.device_info: Optional[DeviceInfo] = None
        self.cuda_available = False
        self.tensorrt_available = TENSORRT_AVAILABLE
        self._initialize()
        
    def _initialize(self):
        """Initialize CUDA environment"""
        if not TORCH_AVAILABLE:
            logging.warning("PyTorch not available, CUDA functionality disabled")
            return
            
        self.cuda_available = torch.cuda.is_available()
        
        if self.cuda_available:
            self.device_info = self._get_device_info()
            logging.info(f"CUDA initialized: {self.device_info.name}")
        else:
            logging.warning("CUDA not available")
            
    def _get_device_info(self) -> Optional[DeviceInfo]:
        """Get CUDA device information"""
        if not self.cuda_available:
            return None
            
        try:
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            
            # Get memory info
            total_memory = torch.cuda.get_device_properties(device).total_memory
            available_memory = total_memory - torch.cuda.memory_allocated(device)
            
            # Determine capabilities
            major, minor = props.major, props.minor
            supports_fp16 = major >= 6  # Pascal and newer
            supports_int8 = major >= 6  # Pascal and newer
            tensor_cores = major >= 7   # Volta and newer
            
            return DeviceInfo(
                name=props.name,
                compute_capability=(major, minor),
                total_memory=total_memory,
                available_memory=available_memory,
                cuda_cores=props.multi_processor_count * self._cores_per_sm(major, minor),
                tensor_cores=tensor_cores,
                supports_fp16=supports_fp16,
                supports_int8=supports_int8
            )
            
        except Exception as e:
            logging.error(f"Error getting device info: {e}")
            return None
            
    def _cores_per_sm(self, major: int, minor: int) -> int:
        """Get CUDA cores per streaming multiprocessor"""
        # Approximate values for different architectures
        if major == 2:  # Fermi
            return 32
        elif major == 3:  # Kepler
            return 192
        elif major == 5:  # Maxwell
            return 128
        elif major == 6:  # Pascal
            return 64 if minor == 0 else 128
        elif major == 7:  # Volta/Turing
            return 64
        elif major == 8:  # Ampere
            return 64
        else:
            return 64  # Default estimate
            
    def get_optimal_precision(self) -> PrecisionMode:
        """Get optimal precision mode for device"""
        if not self.device_info:
            return PrecisionMode.FP32
            
        if self.device_info.tensor_cores:
            return PrecisionMode.FP16
        elif self.device_info.supports_fp16:
            return PrecisionMode.FP16
        else:
            return PrecisionMode.FP32
            
    def get_memory_usage(self) -> Dict[str, int]:
        """Get current GPU memory usage"""
        if not self.cuda_available:
            return {}
            
        return {
            'allocated': torch.cuda.memory_allocated(),
            'cached': torch.cuda.memory_reserved(),
            'max_allocated': torch.cuda.max_memory_allocated(),
            'max_cached': torch.cuda.max_memory_reserved()
        }
        
    def clear_cache(self):
        """Clear GPU memory cache"""
        if self.cuda_available:
            torch.cuda.empty_cache()
            
    def benchmark_device(self, tensor_size: Tuple[int, ...] = (1, 3, 640, 640), 
                        iterations: int = 100) -> Dict[str, float]:
        """Benchmark device performance"""
        if not self.cuda_available:
            return {}
            
        device = torch.cuda.current_device()
        
        # Create test tensors
        x = torch.randn(tensor_size, device=device, dtype=torch.float32)
        y = torch.randn(tensor_size, device=device, dtype=torch.float32)
        
        # Warm up
        for _ in range(10):
            _ = torch.matmul(x.view(-1, tensor_size[-1]), y.view(-1, tensor_size[-1]).T)
            
        torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        for _ in range(iterations):
            _ = torch.matmul(x.view(-1, tensor_size[-1]), y.view(-1, tensor_size[-1]).T)
        torch.cuda.synchronize()
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / iterations
        
        return {
            'total_time': total_time,
            'average_time': avg_time,
            'operations_per_second': 1.0 / avg_time,
            'tensor_size': tensor_size
        }


class TensorRTOptimizer:
    """TensorRT model optimizer"""
    
    def __init__(self):
        self.logger = trt.Logger(trt.Logger.WARNING) if TENSORRT_AVAILABLE else None
        self.available = TENSORRT_AVAILABLE
        
    def optimize_onnx_model(self, onnx_path: str, output_path: str, 
                           precision: PrecisionMode = PrecisionMode.FP16,
                           max_batch_size: int = 1,
                           max_workspace_size: int = 1 << 30) -> OptimizationResult:
        """Optimize ONNX model with TensorRT"""
        if not self.available:
            return OptimizationResult(
                success=False,
                original_size=0,
                optimized_size=0,
                compression_ratio=0.0,
                inference_time_ms=0.0,
                throughput_fps=0.0,
                precision_mode=precision,
                optimization_level=OptimizationLevel.NONE,
                error_message="TensorRT not available"
            )
            
        try:
            # Get original model size
            original_size = os.path.getsize(onnx_path)
            
            # Create TensorRT builder and network
            builder = trt.Builder(self.logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, self.logger)
            
            # Parse ONNX model
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    errors = []
                    for i in range(parser.num_errors):
                        errors.append(parser.get_error(i))
                    return OptimizationResult(
                        success=False,
                        original_size=original_size,
                        optimized_size=0,
                        compression_ratio=0.0,
                        inference_time_ms=0.0,
                        throughput_fps=0.0,
                        precision_mode=precision,
                        optimization_level=OptimizationLevel.NONE,
                        error_message=f"ONNX parsing errors: {errors}"
                    )
                    
            # Configure builder
            config = builder.create_builder_config()
            config.max_workspace_size = max_workspace_size
            
            # Set precision mode
            if precision == PrecisionMode.FP16:
                config.set_flag(trt.BuilderFlag.FP16)
            elif precision == PrecisionMode.INT8:
                config.set_flag(trt.BuilderFlag.INT8)
                # Note: INT8 calibration would be needed here
                
            # Build engine
            engine = builder.build_engine(network, config)
            if not engine:
                return OptimizationResult(
                    success=False,
                    original_size=original_size,
                    optimized_size=0,
                    compression_ratio=0.0,
                    inference_time_ms=0.0,
                    throughput_fps=0.0,
                    precision_mode=precision,
                    optimization_level=OptimizationLevel.NONE,
                    error_message="Failed to build TensorRT engine"
                )
                
            # Serialize and save engine
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())
                
            optimized_size = os.path.getsize(output_path)
            compression_ratio = original_size / optimized_size if optimized_size > 0 else 0.0
            
            # Benchmark inference time
            inference_time = self._benchmark_engine(engine)
            throughput = 1000.0 / inference_time if inference_time > 0 else 0.0
            
            return OptimizationResult(
                success=True,
                original_size=original_size,
                optimized_size=optimized_size,
                compression_ratio=compression_ratio,
                inference_time_ms=inference_time,
                throughput_fps=throughput,
                precision_mode=precision,
                optimization_level=OptimizationLevel.AGGRESSIVE
            )
            
        except Exception as e:
            return OptimizationResult(
                success=False,
                original_size=0,
                optimized_size=0,
                compression_ratio=0.0,
                inference_time_ms=0.0,
                throughput_fps=0.0,
                precision_mode=precision,
                optimization_level=OptimizationLevel.NONE,
                error_message=str(e)
            )
            
    def _benchmark_engine(self, engine, iterations: int = 100) -> float:
        """Benchmark TensorRT engine inference time"""
        try:
            context = engine.create_execution_context()
            
            # Allocate buffers (simplified for demo)
            # In practice, you'd need to handle input/output shapes properly
            input_size = 1 * 3 * 640 * 640 * 4  # Assuming float32
            output_size = 1 * 1000 * 4  # Assuming 1000 classes
            
            h_input = cuda.pagelocked_empty(input_size // 4, dtype=np.float32)
            h_output = cuda.pagelocked_empty(output_size // 4, dtype=np.float32)
            d_input = cuda.mem_alloc(input_size)
            d_output = cuda.mem_alloc(output_size)
            
            stream = cuda.Stream()
            
            # Warm up
            for _ in range(10):
                cuda.memcpy_htod_async(d_input, h_input, stream)
                context.execute_async_v2([int(d_input), int(d_output)], stream.handle)
                cuda.memcpy_dtoh_async(h_output, d_output, stream)
                stream.synchronize()
                
            # Benchmark
            start_time = time.time()
            for _ in range(iterations):
                cuda.memcpy_htod_async(d_input, h_input, stream)
                context.execute_async_v2([int(d_input), int(d_output)], stream.handle)
                cuda.memcpy_dtoh_async(h_output, d_output, stream)
                stream.synchronize()
            end_time = time.time()
            
            avg_time_ms = ((end_time - start_time) / iterations) * 1000
            return avg_time_ms
            
        except Exception as e:
            logging.error(f"Error benchmarking engine: {e}")
            return 0.0


class EdgeOptimizer:
    """Main edge optimization manager"""
    
    def __init__(self):
        self.cuda_manager = CUDAManager()
        self.tensorrt_optimizer = TensorRTOptimizer() if TENSORRT_AVAILABLE else None
        
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        info = {
            'cuda_available': self.cuda_manager.cuda_available,
            'tensorrt_available': TENSORRT_AVAILABLE,
            'torch_available': TORCH_AVAILABLE,
            'onnx_available': ONNX_AVAILABLE
        }
        
        if self.cuda_manager.device_info:
            info['device'] = {
                'name': self.cuda_manager.device_info.name,
                'compute_capability': self.cuda_manager.device_info.compute_capability,
                'total_memory_gb': self.cuda_manager.device_info.total_memory / (1024**3),
                'cuda_cores': self.cuda_manager.device_info.cuda_cores,
                'tensor_cores': self.cuda_manager.device_info.tensor_cores,
                'supports_fp16': self.cuda_manager.device_info.supports_fp16,
                'supports_int8': self.cuda_manager.device_info.supports_int8
            }
            
        return info
        
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get optimization recommendations for current hardware"""
        recommendations = {
            'precision_mode': 'fp32',
            'optimization_level': 'basic',
            'use_tensorrt': False,
            'batch_size': 1,
            'reasons': []
        }
        
        if not self.cuda_manager.cuda_available:
            recommendations['reasons'].append('CUDA not available, using CPU')
            return recommendations
            
        device_info = self.cuda_manager.device_info
        if not device_info:
            return recommendations
            
        # Determine optimal precision
        if device_info.tensor_cores:
            recommendations['precision_mode'] = 'fp16'
            recommendations['reasons'].append('Tensor cores available, using FP16')
        elif device_info.supports_fp16:
            recommendations['precision_mode'] = 'fp16'
            recommendations['reasons'].append('FP16 supported, recommended for speed')
            
        # Determine optimization level
        if device_info.compute_capability[0] >= 7:  # Volta and newer
            recommendations['optimization_level'] = 'aggressive'
            recommendations['reasons'].append('Modern GPU, aggressive optimization recommended')
        elif device_info.compute_capability[0] >= 6:  # Pascal and newer
            recommendations['optimization_level'] = 'basic'
            recommendations['reasons'].append('Pascal+ GPU, basic optimization recommended')
            
        # TensorRT recommendation
        if TENSORRT_AVAILABLE and device_info.compute_capability[0] >= 6:
            recommendations['use_tensorrt'] = True
            recommendations['reasons'].append('TensorRT available and supported')
            
        # Batch size recommendation
        memory_gb = device_info.total_memory / (1024**3)
        if memory_gb >= 8:
            recommendations['batch_size'] = 4
        elif memory_gb >= 4:
            recommendations['batch_size'] = 2
        else:
            recommendations['batch_size'] = 1
            
        recommendations['reasons'].append(f'Batch size {recommendations["batch_size"]} for {memory_gb:.1f}GB memory')
        
        return recommendations


def demo_cuda_tensorrt():
    """Demo function for testing CUDA and TensorRT optimization"""
    logging.basicConfig(level=logging.INFO)
    
    optimizer = EdgeOptimizer()
    
    # Get system info
    system_info = optimizer.get_system_info()
    print("System Information:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")
        
    # Get recommendations
    recommendations = optimizer.get_optimization_recommendations()
    print("\nOptimization Recommendations:")
    for key, value in recommendations.items():
        if key != 'reasons':
            print(f"  {key}: {value}")
    print("  Reasons:")
    for reason in recommendations['reasons']:
        print(f"    - {reason}")
        
    # Benchmark if CUDA available
    if optimizer.cuda_manager.cuda_available:
        print("\nRunning GPU benchmark...")
        benchmark_results = optimizer.cuda_manager.benchmark_device()
        print(f"Benchmark results: {benchmark_results}")
        
        memory_usage = optimizer.cuda_manager.get_memory_usage()
        print(f"Memory usage: {memory_usage}")
        
    print("\nDemo complete!")


if __name__ == "__main__":
    demo_cuda_tensorrt()