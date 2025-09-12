#!/usr/bin/env python3
"""
Foresight SAR System - Main Entry Point

A CLI entry point that checks environment, prints hardware status,
and provides simulation mode for development and CI.
"""

import argparse
import sys
import os
import platform
import subprocess
from pathlib import Path
from typing import Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None
    np = None

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None


def check_environment():
    """Check system environment and dependencies."""
    print("🔍 ARGUS SAR System - Environment Check")
    print("=" * 50)
    
    # System info
    print(f"📱 Platform: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"📁 Working Directory: {os.getcwd()}")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version OK")
    
    # Check dependencies
    deps_ok = True
    
    if TORCH_AVAILABLE:
        print(f"✅ PyTorch: {torch.__version__}")
    else:
        print("❌ PyTorch not available")
        deps_ok = False
    
    if OPENCV_AVAILABLE:
        print(f"✅ OpenCV: {cv2.__version__}")
    else:
        print("❌ OpenCV not available")
        deps_ok = False
    
    if YOLO_AVAILABLE:
        print("✅ Ultralytics YOLO available")
    else:
        print("❌ Ultralytics YOLO not available")
        deps_ok = False
    
    # Check environment file
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file found")
    else:
        print("⚠️  .env file not found (using defaults)")
    
    # Check model files
    models_dir = Path("models")
    if models_dir.exists() and any(models_dir.glob("*.pt")):
        model_files = list(models_dir.glob("*.pt"))
        print(f"✅ Found {len(model_files)} model file(s)")
        for model in model_files[:3]:  # Show first 3
            print(f"   📦 {model.name}")
    else:
        print("⚠️  No model files found in models/ directory")
    
    return deps_ok


def check_hardware():
    """Check hardware capabilities."""
    print("\n🖥️  Hardware Status")
    print("=" * 50)
    
    # CPU info
    try:
        import psutil
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        print(f"🔧 CPU Cores: {cpu_count}")
        print(f"💾 RAM: {memory.total // (1024**3):.1f} GB ({memory.percent}% used)")
    except ImportError:
        print("⚠️  psutil not available for detailed system info")
    
    # GPU info
    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"🎮 CUDA GPUs: {gpu_count}")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory
                print(f"   GPU {i}: {gpu_name} ({gpu_memory // (1024**3):.1f} GB)")
        else:
            print("❌ CUDA not available")
    
    # Camera check
    if OPENCV_AVAILABLE:
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                print("✅ Default camera accessible")
                cap.release()
            else:
                print("❌ Default camera not accessible")
        except Exception as e:
            print(f"❌ Camera check failed: {e}")
    
    # Disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        print(f"💽 Disk Space: {free // (1024**3):.1f} GB free / {total // (1024**3):.1f} GB total")
    except Exception as e:
        print(f"⚠️  Disk space check failed: {e}")


def run_simulation_mode():
    """Run system in simulation mode with test data."""
    print("\n🎭 Starting Simulation Mode")
    print("=" * 50)
    
    # Check for test data
    test_files = [
        "data/samples/test_video.mp4",
        "data/samples/sample_image.jpg",
        "stream.mp4"  # Legacy test file
    ]
    
    available_files = [f for f in test_files if Path(f).exists()]
    
    if not available_files:
        print("⚠️  No test files found. Creating synthetic test data...")
        create_synthetic_test_data()
        available_files = ["data/samples/synthetic_test.mp4"]
    else:
        print(f"✅ Found {len(available_files)} test file(s):")
        for file in available_files:
            print(f"   📹 {file}")
    
    # Set simulation environment variables
    os.environ["FORESIGHT_SIMULATION_MODE"] = "true"
    os.environ["FORESIGHT_TEST_DATA_PATH"] = str(Path(available_files[0]).parent if available_files else "data/samples")
    
    print("\n🚀 Launching SAR system in simulation mode...")
    
    # Import and run the actual SAR service
    try:
        import uvicorn
        from src.backend.sar_service import app
        print("✅ Starting SAR service...")
        uvicorn.run(app, host="0.0.0.0", port=8004, log_level="info")
    except ImportError as e:
        print(f"❌ Could not import SAR service: {e}")
        print("❌ Running legacy server...")
        run_legacy_server()


def create_synthetic_test_data():
    """Create synthetic test data for simulation."""
    if not OPENCV_AVAILABLE:
        print("❌ OpenCV required for synthetic data generation")
        return
    
    # Create data directory
    data_dir = Path("data/samples")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate a simple test video
    output_path = data_dir / "synthetic_test.mp4"
    
    print(f"📹 Generating synthetic test video: {output_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, 10.0, (640, 480))
    
    for i in range(100):  # 10 seconds at 10 FPS
        # Create a simple moving rectangle (simulating a person)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        x = int(50 + (i * 5) % 540)
        y = int(200 + 50 * np.sin(i * 0.1))
        cv2.rectangle(frame, (x, y), (x+50, y+100), (0, 255, 0), -1)
        cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    
    out.release()
    print("✅ Synthetic test data created")


def run_legacy_server():
    """Run the legacy FastAPI server from the original main.py."""
    print("🌐 Starting legacy web server...")
    
    # Import the legacy server components
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    
    app = FastAPI(title="ARGUS SAR System", version="1.0.0")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    def root():
        return {"message": "ARGUS SAR System - Simulation Mode", "mode": "simulation"}
    
    @app.get("/health")
    def health_check():
        return {"status": "healthy", "mode": "simulation"}
    
    print("🚀 Server starting on http://localhost:8004")
    print("📖 API docs available at http://localhost:8004/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8004)


def run_production_mode():
    """Run the full production SAR system."""
    print("\n🚀 Starting Production Mode")
    print("=" * 50)
    
    try:
        import uvicorn
        from src.backend.sar_service import app
        print("✅ Starting SAR service...")
        uvicorn.run(app, host="0.0.0.0", port=8004, log_level="info")
    except ImportError as e:
        print(f"❌ Could not import SAR service: {e}")
        print("💡 Try running: pip install -r requirements.txt")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ARGUS SAR System - AI-Powered Search and Rescue",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --check          # Check environment and hardware
  python main.py --simulate       # Run with test data
  python main.py --test-mode      # Run tests
  python main.py                  # Start production system

For more information, visit: https://github.com/Hiko318/argus
        """
    )
    
    parser.add_argument(
        "--check", 
        action="store_true", 
        help="Check environment and hardware status"
    )
    
    parser.add_argument(
        "--simulate", 
        action="store_true", 
        help="Run in simulation mode with test data"
    )
    
    parser.add_argument(
        "--test-mode", 
        action="store_true", 
        help="Run system tests"
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version="ARGUS SAR System v1.0.0"
    )
    
    args = parser.parse_args()
    
    # Always check environment first
    env_ok = check_environment()
    
    if args.check:
        check_hardware()
        if env_ok:
            print("\n✅ System ready for deployment")
        else:
            print("\n❌ System has dependency issues")
            sys.exit(1)
        return
    
    if args.test_mode:
        print("\n🧪 Running system tests...")
        try:
            subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], check=True)
            print("✅ All tests passed")
        except subprocess.CalledProcessError:
            print("❌ Some tests failed")
            sys.exit(1)
        except FileNotFoundError:
            print("❌ pytest not found. Install with: pip install pytest")
            sys.exit(1)
        return
    
    if args.simulate:
        if not env_ok:
            print("⚠️  Running simulation mode with missing dependencies")
        run_simulation_mode()
        return
    
    # Default: run production mode
    if not env_ok:
        print("❌ Cannot start production mode with missing dependencies")
        print("💡 Run 'python main.py --check' for details")
        sys.exit(1)
    
    check_hardware()
    run_production_mode()


if __name__ == "__main__":
    main()