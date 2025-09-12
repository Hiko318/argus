#!/usr/bin/env python3
"""
Edge Device Optimization Module

Optimizes YOLO models for deployment on edge devices including:
- TensorRT optimization for NVIDIA Jetson
- ONNX conversion for cross-platform deployment
- Model quantization for reduced memory usage
- PyTorch Mobile optimization for Android/iOS
- Performance benchmarking and validation

Author: Foresight AI Team
Date: 2024
"""

import logging
import time
import json
import os
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict

import torch
import numpy as np
from ultralytics import YOLO

# Optional imports for optimization
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
    from torch.utils.mobile_optimizer import optimize_for_mobile
    MOBILE_OPTIMIZER_AVAILABLE = True
except ImportError:
    MOBILE_OPTIMIZER_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for model optimization"""
    # Target device
    target_device: str = "cpu"  # cpu, cuda, jetson, android, ios
    
    # Optimization methods
    enable_tensorrt: bool = False
    enable_onnx: bool = True
    enable_quantization: bool = True
    enable_mobile: bool = False
    
    # TensorRT settings
    tensorrt_precision: str = "fp16"  # fp32, fp16, int8
    tensorrt_workspace_size: int = 1 << 30  # 1GB
    
    # Quantization settings
    quantization_method: str = "dynamic"  # dynamic, static, qat
    quantization_backend: str = "fbgemm"  # fbgemm, qnnpack
    
    # Input specifications
    input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640)  # NCHW
    
    # Performance targets
    target_fps: float = 30.0
    max_latency_ms: float = 33.3  # 1000/30 FPS
    
    # Output paths
    output_dir: str = "optimized_models"
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class BenchmarkResult:
    """Benchmark results for optimized models"""
    model_path: str
    optimization_method: str
    device: str
    
    # Performance metrics
    avg_inference_time_ms: float
    fps: float
    memory_usage_mb: float
    model_size_mb: float
    
    # Accuracy metrics (if validation data available)
    accuracy_drop: Optional[float] = None
    
    # System info
    timestamp: str = ""
    system_info: Dict = None
    
    def to_dict(self) -> Dict:
        return asdict(self)

class EdgeOptimizer:
    """Model optimizer for edge devices"""
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize edge optimizer
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Check device capabilities
        self.device_info = self._get_device_info()
        
        logger.info(f"EdgeOptimizer initialized for {config.target_device}")
        logger.info(f"Available optimizations: TensorRT={TENSORRT_AVAILABLE}, "
                   f"ONNX={ONNX_AVAILABLE}, Mobile={MOBILE_OPTIMIZER_AVAILABLE}")
    
    def optimize_model(self, model_path: str) -> Dict[str, str]:
        """
        Optimize model for target device
        
        Args:
            model_path: Path to original YOLO model
            
        Returns:
            Dictionary mapping optimization method to output path
        """
        logger.info(f"Starting optimization for {model_path}")
        
        # Load original model
        model = YOLO(model_path)
        
        optimized_models = {}
        
        # TensorRT optimization
        if self.config.enable_tensorrt and TENSORRT_AVAILABLE:
            try:
                trt_path = self._optimize_tensorrt(model, model_path)
                optimized_models["tensorrt"] = trt_path
                logger.info(f"TensorRT optimization completed: {trt_path}")
            except Exception as e:
                logger.error(f"TensorRT optimization failed: {e}")
        
        # ONNX optimization
        if self.config.enable_onnx and ONNX_AVAILABLE:
            try:
                onnx_path = self._optimize_onnx(model, model_path)
                optimized_models["onnx"] = onnx_path
                logger.info(f"ONNX optimization completed: {onnx_path}")
            except Exception as e:
                logger.error(f"ONNX optimization failed: {e}")
        
        # PyTorch quantization
        if self.config.enable_quantization:
            try:
                quant_path = self._optimize_quantization(model, model_path)
                optimized_models["quantized"] = quant_path
                logger.info(f"Quantization completed: {quant_path}")
            except Exception as e:
                logger.error(f"Quantization failed: {e}")
        
        # Mobile optimization
        if self.config.enable_mobile and MOBILE_OPTIMIZER_AVAILABLE:
            try:
                mobile_path = self._optimize_mobile(model, model_path)
                optimized_models["mobile"] = mobile_path
                logger.info(f"Mobile optimization completed: {mobile_path}")
            except Exception as e:
                logger.error(f"Mobile optimization failed: {e}")
        
        # Save optimization report
        self._save_optimization_report(model_path, optimized_models)
        
        return optimized_models
    
    def benchmark_models(self, model_paths: Dict[str, str], 
                        num_iterations: int = 100) -> List[BenchmarkResult]:
        """
        Benchmark optimized models
        
        Args:
            model_paths: Dictionary mapping method to model path
            num_iterations: Number of benchmark iterations
            
        Returns:
            List of benchmark results
        """
        logger.info(f"Starting benchmark with {num_iterations} iterations")
        
        results = []
        
        # Create dummy input
        dummy_input = torch.randn(self.config.input_shape)
        if torch.cuda.is_available() and self.config.target_device == "cuda":
            dummy_input = dummy_input.cuda()
        
        for method, model_path in model_paths.items():
            logger.info(f"Benchmarking {method} model: {model_path}")
            
            try:
                result = self._benchmark_single_model(
                    model_path, method, dummy_input, num_iterations
                )
                results.append(result)
                
                logger.info(f"{method} - FPS: {result.fps:.1f}, "
                           f"Latency: {result.avg_inference_time_ms:.1f}ms, "
                           f"Memory: {result.memory_usage_mb:.1f}MB")
                
            except Exception as e:
                logger.error(f"Benchmark failed for {method}: {e}")
        
        # Save benchmark results
        self._save_benchmark_results(results)
        
        return results
    
    def _optimize_tensorrt(self, model: YOLO, model_path: str) -> str:
        """Optimize model with TensorRT"""
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")
        
        logger.info("Starting TensorRT optimization")
        
        # Export to TensorRT
        output_path = self.output_dir / f"{Path(model_path).stem}_tensorrt.engine"
        
        # Use YOLO's built-in TensorRT export
        model.export(
            format="engine",
            imgsz=self.config.input_shape[2:],
            half=(self.config.tensorrt_precision == "fp16"),
            int8=(self.config.tensorrt_precision == "int8"),
            workspace=self.config.tensorrt_workspace_size,
            verbose=True
        )
        
        # Move the generated engine file to our output directory
        generated_engine = Path(model_path).with_suffix(".engine")
        if generated_engine.exists():
            generated_engine.rename(output_path)
        
        return str(output_path)
    
    def _optimize_onnx(self, model: YOLO, model_path: str) -> str:
        """Optimize model with ONNX"""
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX not available")
        
        logger.info("Starting ONNX optimization")
        
        # Export to ONNX
        output_path = self.output_dir / f"{Path(model_path).stem}_onnx.onnx"
        
        model.export(
            format="onnx",
            imgsz=self.config.input_shape[2:],
            opset=11,
            simplify=True,
            dynamic=False
        )
        
        # Move the generated ONNX file
        generated_onnx = Path(model_path).with_suffix(".onnx")
        if generated_onnx.exists():
            generated_onnx.rename(output_path)
        
        # Optimize ONNX model
        self._optimize_onnx_model(str(output_path))
        
        return str(output_path)
    
    def _optimize_onnx_model(self, onnx_path: str):
        """Apply ONNX-specific optimizations"""
        try:
            import onnxoptimizer
            
            # Load ONNX model
            model = onnx.load(onnx_path)
            
            # Apply optimizations
            optimized_model = onnxoptimizer.optimize(model)
            
            # Save optimized model
            onnx.save(optimized_model, onnx_path)
            
            logger.info("ONNX model optimized successfully")
            
        except ImportError:
            logger.warning("onnxoptimizer not available, skipping ONNX optimization")
        except Exception as e:
            logger.warning(f"ONNX optimization failed: {e}")
    
    def _optimize_quantization(self, model: YOLO, model_path: str) -> str:
        """Apply quantization to model"""
        logger.info(f"Starting {self.config.quantization_method} quantization")
        
        output_path = self.output_dir / f"{Path(model_path).stem}_quantized.pt"
        
        if self.config.quantization_method == "dynamic":
            return self._dynamic_quantization(model, output_path)
        elif self.config.quantization_method == "static":
            return self._static_quantization(model, output_path)
        else:
            raise ValueError(f"Unsupported quantization method: {self.config.quantization_method}")
    
    def _dynamic_quantization(self, model: YOLO, output_path: Path) -> str:
        """Apply dynamic quantization"""
        # Get the PyTorch model
        torch_model = model.model
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            torch_model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )
        
        # Save quantized model
        torch.save(quantized_model.state_dict(), output_path)
        
        return str(output_path)
    
    def _static_quantization(self, model: YOLO, output_path: Path) -> str:
        """Apply static quantization (requires calibration data)"""
        # This is a simplified implementation
        # In practice, you would need calibration data
        logger.warning("Static quantization requires calibration data - using dynamic instead")
        return self._dynamic_quantization(model, output_path)
    
    def _optimize_mobile(self, model: YOLO, model_path: str) -> str:
        """Optimize model for mobile deployment"""
        if not MOBILE_OPTIMIZER_AVAILABLE:
            raise RuntimeError("Mobile optimizer not available")
        
        logger.info("Starting mobile optimization")
        
        output_path = self.output_dir / f"{Path(model_path).stem}_mobile.ptl"
        
        # Get the PyTorch model
        torch_model = model.model
        torch_model.eval()
        
        # Create example input
        example_input = torch.randn(self.config.input_shape)
        
        # Trace the model
        traced_model = torch.jit.trace(torch_model, example_input)
        
        # Optimize for mobile
        mobile_model = optimize_for_mobile(traced_model)
        
        # Save mobile model
        mobile_model._save_for_lite_interpreter(str(output_path))
        
        return str(output_path)
    
    def _benchmark_single_model(self, model_path: str, method: str, 
                               dummy_input: torch.Tensor, 
                               num_iterations: int) -> BenchmarkResult:
        """Benchmark a single model"""
        # Load model based on method
        if method == "tensorrt":
            # For TensorRT, we would use the engine directly
            # This is a simplified implementation
            model = YOLO(model_path)
        elif method == "onnx":
            # Use ONNX Runtime
            if ONNX_AVAILABLE:
                session = ort.InferenceSession(model_path)
                return self._benchmark_onnx_model(session, dummy_input, num_iterations, model_path, method)
            else:
                raise RuntimeError("ONNX Runtime not available")
        else:
            model = YOLO(model_path)
        
        # Warm up
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input.numpy())
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        inference_times = []
        
        for _ in range(num_iterations):
            iter_start = time.time()
            
            with torch.no_grad():
                _ = model(dummy_input.numpy())
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            iter_end = time.time()
            
            inference_times.append((iter_end - iter_start) * 1000)  # Convert to ms
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        # Calculate metrics
        avg_inference_time = np.mean(inference_times)
        fps = 1000.0 / avg_inference_time
        memory_usage = end_memory - start_memory
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        
        return BenchmarkResult(
            model_path=model_path,
            optimization_method=method,
            device=self.config.target_device,
            avg_inference_time_ms=avg_inference_time,
            fps=fps,
            memory_usage_mb=memory_usage,
            model_size_mb=model_size,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            system_info=self.device_info
        )
    
    def _benchmark_onnx_model(self, session: 'ort.InferenceSession', 
                             dummy_input: torch.Tensor, num_iterations: int,
                             model_path: str, method: str) -> BenchmarkResult:
        """Benchmark ONNX model specifically"""
        input_name = session.get_inputs()[0].name
        input_data = dummy_input.numpy()
        
        # Warm up
        for _ in range(10):
            _ = session.run(None, {input_name: input_data})
        
        # Benchmark
        start_memory = self._get_memory_usage()
        inference_times = []
        
        for _ in range(num_iterations):
            iter_start = time.time()
            _ = session.run(None, {input_name: input_data})
            iter_end = time.time()
            inference_times.append((iter_end - iter_start) * 1000)
        
        end_memory = self._get_memory_usage()
        
        # Calculate metrics
        avg_inference_time = np.mean(inference_times)
        fps = 1000.0 / avg_inference_time
        memory_usage = end_memory - start_memory
        model_size = os.path.getsize(model_path) / (1024 * 1024)
        
        return BenchmarkResult(
            model_path=model_path,
            optimization_method=method,
            device=self.config.target_device,
            avg_inference_time_ms=avg_inference_time,
            fps=fps,
            memory_usage_mb=memory_usage,
            model_size_mb=model_size,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            system_info=self.device_info
        )
    
    def _get_device_info(self) -> Dict:
        """Get device information"""
        info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
        }
        
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory
        else:
            info["cuda_available"] = False
        
        if TENSORRT_AVAILABLE:
            info["tensorrt_version"] = trt.__version__
        
        if ONNX_AVAILABLE:
            info["onnx_version"] = onnx.__version__
            info["onnxruntime_version"] = ort.__version__
        
        return info
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
    
    def _save_optimization_report(self, original_path: str, optimized_models: Dict[str, str]):
        """Save optimization report"""
        report = {
            "original_model": original_path,
            "optimization_config": self.config.to_dict(),
            "optimized_models": optimized_models,
            "device_info": self.device_info,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        report_path = self.output_dir / "optimization_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Optimization report saved: {report_path}")
    
    def _save_benchmark_results(self, results: List[BenchmarkResult]):
        """Save benchmark results"""
        results_data = [result.to_dict() for result in results]
        
        results_path = self.output_dir / "benchmark_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Benchmark results saved: {results_path}")

def create_optimizer(target_device: str = "cpu", **kwargs) -> EdgeOptimizer:
    """Create edge optimizer with default configuration"""
    config = OptimizationConfig(target_device=target_device, **kwargs)
    return EdgeOptimizer(config)

def optimize_for_jetson(model_path: str, output_dir: str = "optimized_models") -> Dict[str, str]:
    """Optimize model specifically for NVIDIA Jetson devices"""
    config = OptimizationConfig(
        target_device="jetson",
        enable_tensorrt=True,
        enable_onnx=True,
        enable_quantization=True,
        tensorrt_precision="fp16",
        output_dir=output_dir
    )
    
    optimizer = EdgeOptimizer(config)
    return optimizer.optimize_model(model_path)

def optimize_for_mobile(model_path: str, output_dir: str = "optimized_models") -> Dict[str, str]:
    """Optimize model for mobile devices (Android/iOS)"""
    config = OptimizationConfig(
        target_device="android",
        enable_mobile=True,
        enable_onnx=True,
        enable_quantization=True,
        quantization_method="dynamic",
        output_dir=output_dir
    )
    
    optimizer = EdgeOptimizer(config)
    return optimizer.optimize_model(model_path)

def demo_optimization():
    """Demo function for edge optimization"""
    logger.info("Starting edge optimization demo")
    
    # Check if model exists
    model_path = "models/yolo11s.pt"
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    # Create optimizer for current device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = OptimizationConfig(
        target_device=device,
        enable_onnx=True,
        enable_quantization=True,
        enable_tensorrt=torch.cuda.is_available() and TENSORRT_AVAILABLE
    )
    
    optimizer = EdgeOptimizer(config)
    
    # Optimize model
    optimized_models = optimizer.optimize_model(model_path)
    
    # Benchmark models
    all_models = {"original": model_path, **optimized_models}
    results = optimizer.benchmark_models(all_models)
    
    # Print results
    logger.info("\n=== OPTIMIZATION RESULTS ===")
    for result in results:
        logger.info(f"{result.optimization_method}: {result.fps:.1f} FPS, "
                   f"{result.avg_inference_time_ms:.1f}ms, {result.model_size_mb:.1f}MB")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_optimization()