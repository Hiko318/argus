#!/usr/bin/env python3
"""
Jetson TensorRT Optimization Example

This example demonstrates how to use the Jetson TensorRT Optimizer
to optimize PyTorch models for deployment on Jetson devices.

Features demonstrated:
- Model discovery and type detection
- ONNX export with optimized configurations
- TensorRT engine generation
- Performance benchmarking
- Deployment configuration generation

Author: Foresight AI Team
Date: 2024
"""

import sys
from pathlib import Path

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))

from jetson_tensorrt_optimizer import (
    JetsonTensorRTOptimizer,
    OptimizationConfig,
    JetsonDevice,
    ModelType
)

def example_single_model_optimization():
    """
    Example: Optimize a single model for Jetson deployment
    """
    print("🚀 Single Model Optimization Example")
    print("=" * 50)
    
    # Configure optimization settings
    config = OptimizationConfig(
        device=JetsonDevice.ORIN_NX,  # Target device
        precision='fp16',              # Use FP16 for better performance
        batch_size=1,                  # Single image inference
        input_shape=(1, 3, 640, 640),  # YOLOv8 input shape
        workspace_size=4,              # 4GB workspace for TensorRT
        enable_dla=True,               # Use Deep Learning Accelerator
        validate_output=True,          # Validate ONNX export
        benchmark_iterations=100       # Benchmark with 100 iterations
    )
    
    # Create optimizer
    output_dir = Path('./optimized_models')
    optimizer = JetsonTensorRTOptimizer(config, str(output_dir))
    
    # Optimize a YOLOv8 model
    model_path = '../models/weights/yolov8n.pt'
    
    try:
        print(f"📦 Optimizing model: {model_path}")
        result = optimizer.optimize_model(model_path, 'yolov8n_jetson')
        
        if result['success']:
            print("✅ Optimization completed successfully!")
            print(f"📁 Model type: {result['model_type']}")
            print(f"📄 ONNX file: {result['files'].get('onnx', 'Not created')}")
            print(f"⚡ TensorRT engine: {result['files'].get('tensorrt', 'Not created')}")
            print(f"📊 Config file: {result['files'].get('deployment_config', 'Not created')}")
            
            # Display performance metrics
            if 'benchmarks' in result:
                benchmarks = result['benchmarks']
                print("\n📈 Performance Metrics:")
                print(f"   TensorRT FPS: {benchmarks.get('tensorrt_fps', 'N/A')}")
                print(f"   ONNX FPS: {benchmarks.get('onnx_fps', 'N/A')}")
                if 'tensorrt_fps' in benchmarks and 'onnx_fps' in benchmarks:
                    speedup = benchmarks['tensorrt_fps'] / benchmarks['onnx_fps']
                    print(f"   Speedup: {speedup:.2f}x")
        else:
            print(f"❌ Optimization failed: {result.get('error', 'Unknown error')}")
            
    except FileNotFoundError:
        print(f"⚠️  Model file not found: {model_path}")
        print("   Please ensure you have trained models in the models directory.")
    except Exception as e:
        print(f"❌ Error during optimization: {e}")

def example_batch_optimization():
    """
    Example: Optimize all models in a directory
    """
    print("\n🔄 Batch Optimization Example")
    print("=" * 50)
    
    # Configure for batch optimization
    config = OptimizationConfig(
        device=JetsonDevice.XAVIER_NX,  # Different target device
        precision='fp16',
        batch_size=1,
        workspace_size=2,               # Smaller workspace for Xavier NX
        enable_dla=True,
        validate_output=False,          # Skip validation for faster processing
        benchmark_iterations=50         # Fewer iterations for batch processing
    )
    
    # Create optimizer
    output_dir = Path('./batch_optimized')
    optimizer = JetsonTensorRTOptimizer(config, str(output_dir))
    
    # Optimize all models in directory
    models_dir = '../models'
    
    try:
        print(f"📂 Scanning directory: {models_dir}")
        results = optimizer.optimize_directory(models_dir)
        
        print(f"\n📊 Batch Optimization Results:")
        print(f"   Total models processed: {len(results)}")
        
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        
        # Show details for each model
        for result in results:
            status = "✅" if result['success'] else "❌"
            print(f"   {status} {result['model_name']} ({result.get('model_type', 'Unknown')})")
            
            if not result['success']:
                print(f"      Error: {result.get('error', 'Unknown error')}")
        
        # Summary report location
        summary_path = output_dir / 'optimization_summary.json'
        if summary_path.exists():
            print(f"\n📋 Detailed summary: {summary_path}")
            
    except Exception as e:
        print(f"❌ Error during batch optimization: {e}")

def example_custom_configuration():
    """
    Example: Custom optimization configuration for specific use cases
    """
    print("\n⚙️  Custom Configuration Example")
    print("=" * 50)
    
    # High-performance configuration for real-time inference
    high_perf_config = OptimizationConfig(
        device=JetsonDevice.ORIN_AGX,   # Most powerful Jetson
        precision='fp16',
        batch_size=4,                   # Batch processing
        input_shape=(4, 3, 640, 640),   # Batch input shape
        workspace_size=8,               # Maximum workspace
        enable_dla=False,               # Disable DLA for maximum GPU performance
        dynamic_shapes=True,            # Enable dynamic batch sizes
        validate_output=True,
        benchmark_iterations=200
    )
    
    # Memory-optimized configuration for edge deployment
    edge_config = OptimizationConfig(
        device=JetsonDevice.NANO,       # Most constrained device
        precision='int8',               # Maximum quantization
        batch_size=1,
        input_shape=(1, 3, 416, 416),   # Smaller input size
        workspace_size=512,             # Minimal workspace (512MB)
        enable_dla=True,                # Use DLA to save GPU memory
        validate_output=True,
        benchmark_iterations=50
    )
    
    print("🏎️  High-Performance Config:")
    print(f"   Device: {high_perf_config.device.value['name']}")
    print(f"   Precision: {high_perf_config.precision}")
    print(f"   Batch size: {high_perf_config.batch_size}")
    print(f"   Workspace: {high_perf_config.workspace_size}GB")
    print(f"   DLA enabled: {high_perf_config.enable_dla}")
    
    print("\n💾 Edge-Optimized Config:")
    print(f"   Device: {edge_config.device.value['name']}")
    print(f"   Precision: {edge_config.precision}")
    print(f"   Batch size: {edge_config.batch_size}")
    print(f"   Workspace: {edge_config.workspace_size}MB")
    print(f"   DLA enabled: {edge_config.enable_dla}")
    
    # Example of choosing configuration based on requirements
    print("\n🎯 Configuration Selection Guide:")
    print("   High-Performance: Real-time multi-stream processing")
    print("   Edge-Optimized: Battery-powered, memory-constrained deployment")

def example_deployment_workflow():
    """
    Example: Complete deployment workflow
    """
    print("\n🚀 Complete Deployment Workflow")
    print("=" * 50)
    
    # Step 1: Model preparation
    print("📋 Step 1: Model Preparation")
    print("   - Ensure models are trained and saved as .pt files")
    print("   - Verify model compatibility with target input shapes")
    print("   - Test models with sample data")
    
    # Step 2: Optimization configuration
    print("\n⚙️  Step 2: Optimization Configuration")
    print("   - Select target Jetson device")
    print("   - Choose precision mode (fp32/fp16/int8)")
    print("   - Configure batch size and input shapes")
    print("   - Set workspace memory limits")
    
    # Step 3: Optimization execution
    print("\n🔧 Step 3: Optimization Execution")
    print("   - Run ONNX export with validation")
    print("   - Generate TensorRT engines")
    print("   - Perform benchmarking")
    print("   - Create deployment configurations")
    
    # Step 4: Validation and testing
    print("\n✅ Step 4: Validation and Testing")
    print("   - Compare optimized vs original model outputs")
    print("   - Verify performance improvements")
    print("   - Test on target hardware if available")
    
    # Step 5: Deployment
    print("\n🚀 Step 5: Deployment")
    print("   - Copy optimized models to Jetson device")
    print("   - Install required runtime dependencies")
    print("   - Configure inference pipeline")
    print("   - Monitor performance in production")
    
    # Example deployment checklist
    print("\n📝 Deployment Checklist:")
    checklist = [
        "✓ TensorRT engines generated successfully",
        "✓ Benchmark results meet performance requirements",
        "✓ Model accuracy validated on test data",
        "✓ Deployment configuration reviewed",
        "✓ Target device compatibility confirmed",
        "✓ Runtime dependencies documented",
        "✓ Monitoring and logging configured"
    ]
    
    for item in checklist:
        print(f"   {item}")

def main():
    """
    Main function to run all examples
    """
    print("🎯 Jetson TensorRT Optimization Examples")
    print("=" * 60)
    print("This script demonstrates various optimization scenarios")
    print("for deploying PyTorch models on NVIDIA Jetson devices.")
    print()
    
    try:
        # Run examples
        example_single_model_optimization()
        example_batch_optimization()
        example_custom_configuration()
        example_deployment_workflow()
        
        print("\n🎉 All examples completed!")
        print("\n📚 Next Steps:")
        print("   1. Review the generated optimized models")
        print("   2. Test on your target Jetson device")
        print("   3. Integrate into your inference pipeline")
        print("   4. Monitor performance in production")
        
        print("\n📖 Documentation:")
        print("   - Check deployment configs for integration details")
        print("   - Review benchmark results for performance insights")
        print("   - Consult TensorRT documentation for advanced tuning")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\n💡 This example requires the jetson_tensorrt_optimizer module.")
        print("   Make sure you have:")
        print("   - PyTorch installed")
        print("   - ONNX and ONNXRuntime (optional for validation)")
        print("   - TensorRT (for actual optimization on Jetson)")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("\n🔍 Troubleshooting:")
        print("   - Check that model files exist")
        print("   - Verify CUDA is available")
        print("   - Ensure sufficient disk space for optimized models")

if __name__ == '__main__':
    main()