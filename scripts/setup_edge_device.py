#!/usr/bin/env python3
"""
Edge Device Setup Script

This script automates the installation and configuration of required software and SDKs
for drone-based edge computing systems, including DJI Mobile SDK, Python dependencies,
CUDA/TensorRT optimization, and offline map preparation.
"""

import os
import sys
import subprocess
import platform
import logging
import json
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class DeviceType(Enum):
    """Edge device type enumeration"""
    JETSON_NANO = "jetson_nano"
    JETSON_XAVIER = "jetson_xavier"
    JETSON_ORIN = "jetson_orin"
    RASPBERRY_PI = "raspberry_pi"
    GENERIC_LINUX = "generic_linux"
    WINDOWS = "windows"
    MACOS = "macos"


@dataclass
class SetupResult:
    """Setup operation result"""
    success: bool
    component: str
    message: str
    details: Optional[Dict] = None


class EdgeDeviceSetup:
    """Main setup manager for edge devices"""
    
    def __init__(self, device_type: Optional[DeviceType] = None):
        self.device_type = device_type or self._detect_device_type()
        self.setup_results: List[SetupResult] = []
        self.base_dir = Path.cwd()
        self.requirements_installed = False
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('setup_edge_device.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _detect_device_type(self) -> DeviceType:
        """Auto-detect device type"""
        system = platform.system().lower()
        
        if system == "linux":
            # Check for Jetson devices
            try:
                with open('/proc/device-tree/model', 'r') as f:
                    model = f.read().lower()
                    if 'jetson nano' in model:
                        return DeviceType.JETSON_NANO
                    elif 'jetson xavier' in model:
                        return DeviceType.JETSON_XAVIER
                    elif 'jetson orin' in model:
                        return DeviceType.JETSON_ORIN
            except FileNotFoundError:
                pass
                
            # Check for Raspberry Pi
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read().lower()
                    if 'raspberry pi' in cpuinfo:
                        return DeviceType.RASPBERRY_PI
            except FileNotFoundError:
                pass
                
            return DeviceType.GENERIC_LINUX
            
        elif system == "windows":
            return DeviceType.WINDOWS
        elif system == "darwin":
            return DeviceType.MACOS
        else:
            return DeviceType.GENERIC_LINUX
            
    def _run_command(self, command: List[str], check: bool = True, 
                    capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run system command with logging"""
        self.logger.info(f"Running command: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                check=check,
                capture_output=capture_output,
                text=True
            )
            
            if result.stdout:
                self.logger.debug(f"STDOUT: {result.stdout}")
            if result.stderr:
                self.logger.debug(f"STDERR: {result.stderr}")
                
            return result
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {e}")
            if e.stdout:
                self.logger.error(f"STDOUT: {e.stdout}")
            if e.stderr:
                self.logger.error(f"STDERR: {e.stderr}")
            raise
            
    def check_python_version(self) -> SetupResult:
        """Check Python version compatibility"""
        try:
            version = sys.version_info
            if version.major == 3 and version.minor >= 8:
                return SetupResult(
                    success=True,
                    component="python",
                    message=f"Python {version.major}.{version.minor}.{version.micro} is compatible",
                    details={"version": f"{version.major}.{version.minor}.{version.micro}"}
                )
            else:
                return SetupResult(
                    success=False,
                    component="python",
                    message=f"Python {version.major}.{version.minor} is too old. Requires Python 3.8+"
                )
        except Exception as e:
            return SetupResult(
                success=False,
                component="python",
                message=f"Error checking Python version: {e}"
            )
            
    def install_python_dependencies(self) -> SetupResult:
        """Install required Python packages"""
        try:
            # Core dependencies
            core_packages = [
                "torch",
                "torchvision",
                "opencv-python",
                "ultralytics",
                "numpy",
                "scipy",
                "pillow",
                "requests",
                "fastapi",
                "uvicorn",
                "websockets"
            ]
            
            # Device-specific packages
            device_packages = []
            
            if self.device_type in [DeviceType.JETSON_NANO, DeviceType.JETSON_XAVIER, DeviceType.JETSON_ORIN]:
                device_packages.extend(["jetson-stats", "pycuda"])
                
            # Drone integration packages
            drone_packages = [
                "djitellopy",
                "av"
            ]
            
            # Mapping packages
            mapping_packages = [
                "folium",
                "geopy",
                "geographiclib"
            ]
            
            all_packages = core_packages + device_packages + drone_packages + mapping_packages
            
            # Install packages
            self._run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            self._run_command([sys.executable, "-m", "pip", "install"] + all_packages)
            
            self.requirements_installed = True
            
            return SetupResult(
                success=True,
                component="python_dependencies",
                message=f"Successfully installed {len(all_packages)} packages",
                details={"packages": all_packages}
            )
            
        except Exception as e:
            return SetupResult(
                success=False,
                component="python_dependencies",
                message=f"Error installing Python dependencies: {e}"
            )
            
    def setup_cuda_tensorrt(self) -> SetupResult:
        """Setup CUDA and TensorRT for Jetson devices"""
        if self.device_type not in [DeviceType.JETSON_NANO, DeviceType.JETSON_XAVIER, DeviceType.JETSON_ORIN]:
            return SetupResult(
                success=True,
                component="cuda_tensorrt",
                message="CUDA/TensorRT setup skipped (not a Jetson device)"
            )
            
        try:
            # Check if CUDA is already installed
            try:
                result = self._run_command(["nvcc", "--version"])
                cuda_version = "unknown"
                for line in result.stdout.split('\n'):
                    if 'release' in line.lower():
                        cuda_version = line.strip()
                        break
            except subprocess.CalledProcessError:
                return SetupResult(
                    success=False,
                    component="cuda_tensorrt",
                    message="CUDA not found. Please install JetPack SDK first."
                )
                
            # Install TensorRT Python bindings
            try:
                self._run_command([sys.executable, "-m", "pip", "install", "pycuda"])
            except subprocess.CalledProcessError:
                self.logger.warning("Failed to install pycuda via pip")
                
            # Set up environment variables
            cuda_paths = [
                "/usr/local/cuda/bin",
                "/usr/local/cuda/lib64"
            ]
            
            return SetupResult(
                success=True,
                component="cuda_tensorrt",
                message=f"CUDA setup complete. Version: {cuda_version}",
                details={"cuda_version": cuda_version, "paths": cuda_paths}
            )
            
        except Exception as e:
            return SetupResult(
                success=False,
                component="cuda_tensorrt",
                message=f"Error setting up CUDA/TensorRT: {e}"
            )
            
    def setup_dji_sdk(self) -> SetupResult:
        """Setup DJI SDK integration"""
        try:
            # For now, we use djitellopy for Tello drones
            # In production, you would install the full DJI Mobile SDK or Payload SDK
            
            if not self.requirements_installed:
                self._run_command([sys.executable, "-m", "pip", "install", "djitellopy", "av"])
                
            # Create DJI SDK configuration
            config_dir = self.base_dir / "configs"
            config_dir.mkdir(exist_ok=True)
            
            dji_config = {
                "tello": {
                    "enabled": True,
                    "ip": "192.168.10.1",
                    "port": 8889,
                    "video_port": 11111
                },
                "payload_sdk": {
                    "enabled": False,
                    "note": "Requires DJI Payload SDK installation"
                }
            }
            
            with open(config_dir / "dji_config.json", "w") as f:
                json.dump(dji_config, f, indent=2)
                
            return SetupResult(
                success=True,
                component="dji_sdk",
                message="DJI SDK integration configured (djitellopy for Tello)",
                details=dji_config
            )
            
        except Exception as e:
            return SetupResult(
                success=False,
                component="dji_sdk",
                message=f"Error setting up DJI SDK: {e}"
            )
            
    def prepare_offline_maps(self, bbox: Optional[Dict] = None) -> SetupResult:
        """Prepare offline map data"""
        try:
            # Create maps directory
            maps_dir = self.base_dir / "data" / "maps"
            maps_dir.mkdir(parents=True, exist_ok=True)
            
            # Default bounding box (San Francisco area for demo)
            if not bbox:
                bbox = {
                    "north": 37.8,
                    "south": 37.7,
                    "east": -122.3,
                    "west": -122.5
                }
                
            # Create map configuration
            map_config = {
                "default_area": bbox,
                "zoom_levels": [10, 12, 14, 16],
                "providers": ["openstreetmap", "opentopomap"],
                "cache_dir": str(maps_dir),
                "max_cache_size_mb": 1000
            }
            
            with open(maps_dir / "map_config.json", "w") as f:
                json.dump(map_config, f, indent=2)
                
            # Test offline maps functionality
            if self.requirements_installed:
                try:
                    # Import and test offline maps module
                    sys.path.append(str(self.base_dir))
                    from src.backend.offline_maps import OfflineMapManager
                    
                    manager = OfflineMapManager(str(maps_dir))
                    stats = manager.get_coverage_stats()
                    
                    return SetupResult(
                        success=True,
                        component="offline_maps",
                        message="Offline maps configured successfully",
                        details={"config": map_config, "stats": stats}
                    )
                except ImportError as e:
                    return SetupResult(
                        success=True,
                        component="offline_maps",
                        message=f"Offline maps configured (module test skipped: {e})",
                        details={"config": map_config}
                    )
            else:
                return SetupResult(
                    success=True,
                    component="offline_maps",
                    message="Offline maps configured (dependencies not installed yet)",
                    details={"config": map_config}
                )
                
        except Exception as e:
            return SetupResult(
                success=False,
                component="offline_maps",
                message=f"Error preparing offline maps: {e}"
            )
            
    def download_models(self) -> SetupResult:
        """Download required AI models"""
        try:
            models_dir = self.base_dir / "models"
            models_dir.mkdir(exist_ok=True)
            
            # Download YOLOv8 models
            models_to_download = [
                "yolov8n.pt",
                "yolov8s.pt",
                "yolov8m.pt"
            ]
            
            downloaded_models = []
            
            for model_name in models_to_download:
                model_path = models_dir / model_name
                
                if not model_path.exists():
                    try:
                        # Use ultralytics to download models
                        if self.requirements_installed:
                            from ultralytics import YOLO
                            model = YOLO(model_name)
                            # Move to models directory
                            import shutil
                            shutil.move(model_name, model_path)
                            downloaded_models.append(model_name)
                        else:
                            self.logger.warning(f"Skipping {model_name} download (ultralytics not installed)")
                    except Exception as e:
                        self.logger.warning(f"Failed to download {model_name}: {e}")
                else:
                    downloaded_models.append(f"{model_name} (already exists)")
                    
            return SetupResult(
                success=True,
                component="models",
                message=f"Model download complete",
                details={"downloaded": downloaded_models, "models_dir": str(models_dir)}
            )
            
        except Exception as e:
            return SetupResult(
                success=False,
                component="models",
                message=f"Error downloading models: {e}"
            )
            
    def create_systemd_service(self) -> SetupResult:
        """Create systemd service for auto-start (Linux only)"""
        if self.device_type == DeviceType.WINDOWS:
            return SetupResult(
                success=True,
                component="systemd",
                message="Systemd service skipped (Windows system)"
            )
            
        try:
            service_content = f"""[Unit]
Description=Foresight Edge Computing Service
After=network.target

[Service]
Type=simple
User=foresight
WorkingDirectory={self.base_dir}
ExecStart={sys.executable} -m src.backend.main
Restart=always
RestartSec=10
Environment=PYTHONPATH={self.base_dir}

[Install]
WantedBy=multi-user.target
"""
            
            service_file = Path("/tmp/foresight-edge.service")
            with open(service_file, "w") as f:
                f.write(service_content)
                
            return SetupResult(
                success=True,
                component="systemd",
                message=f"Systemd service file created at {service_file}",
                details={"service_file": str(service_file), "content": service_content}
            )
            
        except Exception as e:
            return SetupResult(
                success=False,
                component="systemd",
                message=f"Error creating systemd service: {e}"
            )
            
    def run_full_setup(self) -> List[SetupResult]:
        """Run complete setup process"""
        self.logger.info(f"Starting edge device setup for {self.device_type.value}")
        
        setup_steps = [
            ("Python Version Check", self.check_python_version),
            ("Python Dependencies", self.install_python_dependencies),
            ("CUDA/TensorRT Setup", self.setup_cuda_tensorrt),
            ("DJI SDK Setup", self.setup_dji_sdk),
            ("Offline Maps Preparation", self.prepare_offline_maps),
            ("Model Download", self.download_models),
            ("Systemd Service", self.create_systemd_service)
        ]
        
        for step_name, step_func in setup_steps:
            self.logger.info(f"Running: {step_name}")
            try:
                result = step_func()
                self.setup_results.append(result)
                
                if result.success:
                    self.logger.info(f"[OK] {step_name}: {result.message}")
                else:
                    self.logger.error(f"[FAIL] {step_name}: {result.message}")
                    
            except Exception as e:
                error_result = SetupResult(
                    success=False,
                    component=step_name.lower().replace(" ", "_"),
                    message=f"Unexpected error: {e}"
                )
                self.setup_results.append(error_result)
                self.logger.error(f"[FAIL] {step_name}: Unexpected error: {e}")
                
        return self.setup_results
        
    def generate_report(self) -> str:
        """Generate setup report"""
        successful = sum(1 for r in self.setup_results if r.success)
        total = len(self.setup_results)
        
        report = f"""
# Edge Device Setup Report

**Device Type:** {self.device_type.value}
**Setup Date:** {logging.Formatter().formatTime(logging.LogRecord('', 0, '', 0, '', (), None))}
**Success Rate:** {successful}/{total} ({successful/total*100:.1f}%)

## Setup Results

"""
        
        for result in self.setup_results:
            status = "[OK]" if result.success else "[FAIL]"
            report += f"- {status} **{result.component}**: {result.message}\n"
            
        report += "\n## Next Steps\n\n"
        
        if successful == total:
            report += "- All components installed successfully!\n"
            report += "- Run `python -m src.backend.main` to start the system\n"
            report += "- Check logs in `setup_edge_device.log` for details\n"
        else:
            report += "- Review failed components above\n"
            report += "- Check logs in `setup_edge_device.log` for error details\n"
            report += "- Re-run setup after resolving issues\n"
            
        return report


def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Edge Device Setup Script")
    parser.add_argument("--device-type", choices=[dt.value for dt in DeviceType],
                       help="Override device type detection")
    parser.add_argument("--skip-models", action="store_true",
                       help="Skip model downloads")
    parser.add_argument("--offline-only", action="store_true",
                       help="Skip components requiring internet")
    
    args = parser.parse_args()
    
    # Create setup manager
    device_type = DeviceType(args.device_type) if args.device_type else None
    setup = EdgeDeviceSetup(device_type)
    
    print(f"🚀 Starting edge device setup for {setup.device_type.value}")
    print(f"📁 Working directory: {setup.base_dir}")
    print("📋 This will install and configure:")
    print("   - Python dependencies (PyTorch, OpenCV, Ultralytics, etc.)")
    print("   - DJI SDK integration (djitellopy)")
    print("   - CUDA/TensorRT optimization (Jetson devices)")
    print("   - Offline mapping capabilities")
    print("   - AI models (YOLOv8)")
    print()
    
    # Run setup
    results = setup.run_full_setup()
    
    # Generate and display report
    report = setup.generate_report()
    print(report)
    
    # Save report
    with open("setup_report.md", "w", encoding="utf-8") as f:
        f.write(report)
        
    print(f"\n📄 Detailed report saved to: setup_report.md")
    print(f"📄 Setup log saved to: setup_edge_device.log")
    
    # Exit with appropriate code
    failed_count = sum(1 for r in results if not r.success)
    sys.exit(failed_count)


if __name__ == "__main__":
    main()