#!/usr/bin/env python3
"""
Field Installer & Setup Wizard for Foresight SAR System

Performs:
- Hardware compatibility check
- Camera intrinsics calibration
- Offline map tiles and DEM selection
- Cryptographic key generation and sign-in setup
"""

import os
import sys
import json
import time
import subprocess
import platform
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet
import base64


class FieldInstaller:
    """Field installation and setup wizard"""
    
    def __init__(self):
        self.config = {}
        self.install_dir = Path.cwd()
        self.config_dir = self.install_dir / "config"
        self.keys_dir = self.config_dir / "keys"
        self.maps_dir = self.install_dir / "maps"
        self.calibration_dir = self.config_dir / "calibration"
        
        # Create directories
        for dir_path in [self.config_dir, self.keys_dir, self.maps_dir, self.calibration_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def print_header(self, title: str):
        """Print formatted section header"""
        print("\n" + "=" * 60)
        print(f" {title}")
        print("=" * 60)
    
    def print_step(self, step: str, status: str = "INFO"):
        """Print formatted step"""
        symbols = {
            "INFO": "ℹ",
            "SUCCESS": "✓",
            "WARNING": "⚠",
            "ERROR": "✗",
            "PROGRESS": "⏳"
        }
        print(f"{symbols.get(status, 'ℹ')} {step}")
    
    def get_user_input(self, prompt: str, default: str = None, options: List[str] = None) -> str:
        """Get user input with validation"""
        while True:
            if default:
                full_prompt = f"{prompt} [{default}]: "
            else:
                full_prompt = f"{prompt}: "
            
            if options:
                full_prompt = f"{prompt} ({'/'.join(options)}): "
            
            response = input(full_prompt).strip()
            
            if not response and default:
                return default
            
            if options and response.lower() not in [opt.lower() for opt in options]:
                print(f"Please choose from: {', '.join(options)}")
                continue
            
            if response:
                return response
            
            if not default:
                print("This field is required.")
                continue
    
    def check_hardware(self) -> Dict[str, Any]:
        """Perform hardware compatibility check"""
        self.print_header("Hardware Compatibility Check")
        
        hardware_info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "python_version": platform.python_version(),
            "opencv_version": cv2.__version__,
            "memory_gb": None,
            "gpu_available": False,
            "camera_available": False,
            "disk_space_gb": None,
            "compatible": True,
            "warnings": []
        }
        
        # Check Python version
        python_version = tuple(map(int, platform.python_version().split('.')))
        if python_version < (3, 8):
            hardware_info["warnings"].append("Python 3.8+ recommended")
            hardware_info["compatible"] = False
        
        # Check memory (Linux/Unix)
        try:
            if platform.system() == "Linux":
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if 'MemTotal' in line:
                            memory_kb = int(line.split()[1])
                            hardware_info["memory_gb"] = round(memory_kb / 1024 / 1024, 1)
                            break
            elif platform.system() == "Windows":
                import psutil
                hardware_info["memory_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
        except Exception:
            hardware_info["warnings"].append("Could not determine memory size")
        
        # Check disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage(".")
            hardware_info["disk_space_gb"] = round(free / (1024**3), 1)
            
            if hardware_info["disk_space_gb"] < 10:
                hardware_info["warnings"].append("Low disk space (< 10GB free)")
        except Exception:
            hardware_info["warnings"].append("Could not determine disk space")
        
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                hardware_info["gpu_available"] = True
                hardware_info["gpu_name"] = torch.cuda.get_device_name(0)
                hardware_info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
        except ImportError:
            hardware_info["warnings"].append("PyTorch not available for GPU detection")
        except Exception:
            pass
        
        # Check camera availability
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                hardware_info["camera_available"] = True
                ret, frame = cap.read()
                if ret:
                    hardware_info["camera_resolution"] = f"{frame.shape[1]}x{frame.shape[0]}"
            cap.release()
        except Exception:
            hardware_info["warnings"].append("Could not access camera")
        
        # Print results
        self.print_step(f"Platform: {hardware_info['platform']}", "INFO")
        self.print_step(f"Python: {hardware_info['python_version']}", "SUCCESS" if python_version >= (3, 8) else "WARNING")
        self.print_step(f"OpenCV: {hardware_info['opencv_version']}", "SUCCESS")
        
        if hardware_info["memory_gb"]:
            status = "SUCCESS" if hardware_info["memory_gb"] >= 8 else "WARNING"
            self.print_step(f"Memory: {hardware_info['memory_gb']} GB", status)
        
        if hardware_info["disk_space_gb"]:
            status = "SUCCESS" if hardware_info["disk_space_gb"] >= 10 else "WARNING"
            self.print_step(f"Disk Space: {hardware_info['disk_space_gb']} GB free", status)
        
        if hardware_info["gpu_available"]:
            self.print_step(f"GPU: {hardware_info.get('gpu_name', 'Available')} ({hardware_info.get('gpu_memory_gb', '?')} GB)", "SUCCESS")
        else:
            self.print_step("GPU: Not available (CPU-only mode)", "WARNING")
        
        if hardware_info["camera_available"]:
            self.print_step(f"Camera: Available ({hardware_info.get('camera_resolution', 'Unknown resolution')})", "SUCCESS")
        else:
            self.print_step("Camera: Not available", "WARNING")
        
        # Print warnings
        for warning in hardware_info["warnings"]:
            self.print_step(warning, "WARNING")
        
        if hardware_info["compatible"]:
            self.print_step("Hardware compatibility check passed", "SUCCESS")
        else:
            self.print_step("Hardware compatibility issues detected", "ERROR")
        
        return hardware_info
    
    def calibrate_camera(self) -> Dict[str, Any]:
        """Perform camera intrinsics calibration"""
        self.print_header("Camera Intrinsics Calibration")
        
        calibration_method = self.get_user_input(
            "Calibration method",
            "auto",
            ["chessboard", "auto", "skip"]
        )
        
        if calibration_method == "skip":
            self.print_step("Camera calibration skipped", "WARNING")
            return {"method": "skipped", "calibrated": False}
        
        if calibration_method == "chessboard":
            return self._calibrate_with_chessboard()
        else:
            return self._auto_calibrate()
    
    def _calibrate_with_chessboard(self) -> Dict[str, Any]:
        """Calibrate camera using chessboard pattern"""
        self.print_step("Starting chessboard calibration", "PROGRESS")
        print("\nInstructions:")
        print("1. Print a chessboard pattern (9x6 squares recommended)")
        print("2. Hold the chessboard in front of the camera")
        print("3. Move it to different positions and angles")
        print("4. Press SPACE to capture, ESC to finish")
        
        # Chessboard parameters
        chessboard_size = (9, 6)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        
        # Arrays to store object points and image points
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.print_step("Could not open camera", "ERROR")
            return {"method": "chessboard", "calibrated": False, "error": "Camera not available"}
        
        captured_frames = 0
        target_frames = 20
        
        while captured_frames < target_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard corners
            ret_corners, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            
            # Draw corners if found
            if ret_corners:
                cv2.drawChessboardCorners(frame, chessboard_size, corners, ret_corners)
                cv2.putText(frame, f"Pattern found! Press SPACE to capture ({captured_frames}/{target_frames})",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Move chessboard into view",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(frame, "SPACE: Capture, ESC: Finish",
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Camera Calibration', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32 and ret_corners:  # SPACE
                # Refine corners
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                objpoints.append(objp)
                imgpoints.append(corners2)
                captured_frames += 1
                self.print_step(f"Captured frame {captured_frames}/{target_frames}", "SUCCESS")
        
        cap.release()
        cv2.destroyAllWindows()
        
        if captured_frames < 10:
            self.print_step(f"Insufficient calibration images ({captured_frames} < 10)", "ERROR")
            return {"method": "chessboard", "calibrated": False, "error": "Insufficient images"}
        
        # Perform calibration
        self.print_step("Computing camera calibration...", "PROGRESS")
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        
        if ret:
            # Calculate reprojection error
            total_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                total_error += error
            
            mean_error = total_error / len(objpoints)
            
            calibration_data = {
                "method": "chessboard",
                "calibrated": True,
                "camera_matrix": camera_matrix.tolist(),
                "distortion_coefficients": dist_coeffs.tolist(),
                "reprojection_error": float(mean_error),
                "calibration_images": captured_frames,
                "image_size": gray.shape[::-1]
            }
            
            # Save calibration
            calib_file = self.calibration_dir / "camera_calibration.json"
            with open(calib_file, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            self.print_step(f"Calibration completed (error: {mean_error:.3f} pixels)", "SUCCESS")
            self.print_step(f"Calibration saved to: {calib_file}", "INFO")
            
            return calibration_data
        else:
            self.print_step("Calibration failed", "ERROR")
            return {"method": "chessboard", "calibrated": False, "error": "Calibration computation failed"}
    
    def _auto_calibrate(self) -> Dict[str, Any]:
        """Perform automatic calibration using default parameters"""
        self.print_step("Using automatic calibration with default parameters", "INFO")
        
        # Get camera resolution
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]
                
                # Estimate camera matrix for typical webcam
                fx = fy = width  # Rough estimate
                cx, cy = width / 2, height / 2
                
                camera_matrix = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ], dtype=np.float32)
                
                # Minimal distortion for auto-calibration
                dist_coeffs = np.zeros((4, 1), dtype=np.float32)
                
                calibration_data = {
                    "method": "auto",
                    "calibrated": True,
                    "camera_matrix": camera_matrix.tolist(),
                    "distortion_coefficients": dist_coeffs.tolist(),
                    "reprojection_error": None,
                    "image_size": [width, height],
                    "note": "Auto-calibration with estimated parameters"
                }
                
                # Save calibration
                calib_file = self.calibration_dir / "camera_calibration.json"
                with open(calib_file, 'w') as f:
                    json.dump(calibration_data, f, indent=2)
                
                self.print_step(f"Auto-calibration completed for {width}x{height}", "SUCCESS")
                cap.release()
                return calibration_data
        
        cap.release()
        self.print_step("Auto-calibration failed", "ERROR")
        return {"method": "auto", "calibrated": False, "error": "Could not access camera"}
    
    def setup_offline_maps(self) -> Dict[str, Any]:
        """Setup offline map tiles and DEM data"""
        self.print_header("Offline Maps & DEM Setup")
        
        maps_config = {
            "enabled": False,
            "tile_sources": [],
            "dem_sources": [],
            "cache_size_mb": 1000
        }
        
        enable_maps = self.get_user_input(
            "Enable offline maps",
            "yes",
            ["yes", "no"]
        ).lower() == "yes"
        
        if not enable_maps:
            self.print_step("Offline maps disabled", "INFO")
            return maps_config
        
        maps_config["enabled"] = True
        
        # Configure tile sources
        self.print_step("Configuring map tile sources...", "INFO")
        
        tile_sources = [
            {
                "name": "OpenStreetMap",
                "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                "max_zoom": 19,
                "attribution": "© OpenStreetMap contributors"
            },
            {
                "name": "Satellite (Esri)",
                "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                "max_zoom": 17,
                "attribution": "© Esri"
            }
        ]
        
        for source in tile_sources:
            use_source = self.get_user_input(
                f"Include {source['name']} tiles",
                "yes",
                ["yes", "no"]
            ).lower() == "yes"
            
            if use_source:
                maps_config["tile_sources"].append(source)
                self.print_step(f"Added tile source: {source['name']}", "SUCCESS")
        
        # Configure DEM sources
        self.print_step("Configuring DEM (elevation) sources...", "INFO")
        
        dem_sources = [
            {
                "name": "SRTM 30m",
                "source": "NASA SRTM",
                "resolution": "30m",
                "coverage": "Global (60°N to 56°S)"
            },
            {
                "name": "ASTER GDEM",
                "source": "NASA/METI",
                "resolution": "30m",
                "coverage": "Global (83°N to 83°S)"
            }
        ]
        
        for source in dem_sources:
            use_source = self.get_user_input(
                f"Include {source['name']} elevation data",
                "yes",
                ["yes", "no"]
            ).lower() == "yes"
            
            if use_source:
                maps_config["dem_sources"].append(source)
                self.print_step(f"Added DEM source: {source['name']}", "SUCCESS")
        
        # Cache size configuration
        cache_size = self.get_user_input(
            "Map cache size (MB)",
            "1000"
        )
        
        try:
            maps_config["cache_size_mb"] = int(cache_size)
        except ValueError:
            maps_config["cache_size_mb"] = 1000
        
        # Create maps directory structure
        (self.maps_dir / "tiles").mkdir(exist_ok=True)
        (self.maps_dir / "dem").mkdir(exist_ok=True)
        
        # Save maps configuration
        maps_file = self.config_dir / "maps_config.json"
        with open(maps_file, 'w') as f:
            json.dump(maps_config, f, indent=2)
        
        self.print_step(f"Maps configuration saved to: {maps_file}", "SUCCESS")
        self.print_step(f"Cache directory: {self.maps_dir}", "INFO")
        
        return maps_config
    
    def generate_keys_and_signin(self) -> Dict[str, Any]:
        """Generate cryptographic keys and setup sign-in"""
        self.print_header("Cryptographic Keys & Sign-in Setup")
        
        # Operator information
        operator_id = self.get_user_input("Operator ID")
        operator_name = self.get_user_input("Operator Name")
        organization = self.get_user_input("Organization", "")
        
        # Generate RSA key pair
        self.print_step("Generating RSA key pair...", "PROGRESS")
        
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        public_key = private_key.public_key()
        
        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Save keys
        private_key_file = self.keys_dir / f"{operator_id}_private.pem"
        public_key_file = self.keys_dir / f"{operator_id}_public.pem"
        
        with open(private_key_file, 'wb') as f:
            f.write(private_pem)
        
        with open(public_key_file, 'wb') as f:
            f.write(public_pem)
        
        # Set restrictive permissions on private key
        os.chmod(private_key_file, 0o600)
        
        # Generate session encryption key
        session_key = Fernet.generate_key()
        session_key_file = self.keys_dir / f"{operator_id}_session.key"
        
        with open(session_key_file, 'wb') as f:
            f.write(session_key)
        
        os.chmod(session_key_file, 0o600)
        
        # Create operator profile
        operator_profile = {
            "operator_id": operator_id,
            "operator_name": operator_name,
            "organization": organization,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "public_key_file": str(public_key_file.name),
            "private_key_file": str(private_key_file.name),
            "session_key_file": str(session_key_file.name),
            "key_algorithm": "RSA-2048",
            "signature_algorithm": "RSA-PSS"
        }
        
        # Save operator profile
        profile_file = self.config_dir / f"operator_{operator_id}.json"
        with open(profile_file, 'w') as f:
            json.dump(operator_profile, f, indent=2)
        
        # Create system configuration
        system_config = {
            "system_id": f"foresight_{int(time.time())}",
            "default_operator": operator_id,
            "evidence_signing": True,
            "encryption_enabled": True,
            "key_directory": str(self.keys_dir),
            "setup_completed": True,
            "setup_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        
        system_config_file = self.config_dir / "system_config.json"
        with open(system_config_file, 'w') as f:
            json.dump(system_config, f, indent=2)
        
        self.print_step(f"RSA key pair generated for {operator_id}", "SUCCESS")
        self.print_step(f"Private key: {private_key_file}", "INFO")
        self.print_step(f"Public key: {public_key_file}", "INFO")
        self.print_step(f"Session key: {session_key_file}", "INFO")
        self.print_step(f"Operator profile: {profile_file}", "INFO")
        self.print_step(f"System config: {system_config_file}", "SUCCESS")
        
        return {
            "operator_profile": operator_profile,
            "system_config": system_config,
            "keys_generated": True
        }
    
    def run_installation(self) -> Dict[str, Any]:
        """Run complete field installation wizard"""
        print("\n" + "=" * 60)
        print(" FORESIGHT SAR FIELD INSTALLER")
        print(" Version 1.0")
        print("=" * 60)
        
        installation_results = {
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hardware_check": None,
            "camera_calibration": None,
            "maps_setup": None,
            "keys_setup": None,
            "completed": False
        }
        
        try:
            # Step 1: Hardware Check
            installation_results["hardware_check"] = self.check_hardware()
            
            if not installation_results["hardware_check"]["compatible"]:
                proceed = self.get_user_input(
                    "Hardware compatibility issues detected. Continue anyway?",
                    "no",
                    ["yes", "no"]
                ).lower() == "yes"
                
                if not proceed:
                    self.print_step("Installation aborted due to hardware issues", "ERROR")
                    return installation_results
            
            # Step 2: Camera Calibration
            installation_results["camera_calibration"] = self.calibrate_camera()
            
            # Step 3: Offline Maps Setup
            installation_results["maps_setup"] = self.setup_offline_maps()
            
            # Step 4: Keys and Sign-in
            installation_results["keys_setup"] = self.generate_keys_and_signin()
            
            installation_results["completed"] = True
            installation_results["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            
            # Save installation log
            install_log = self.config_dir / "installation_log.json"
            with open(install_log, 'w') as f:
                json.dump(installation_results, f, indent=2)
            
            self.print_header("Installation Complete")
            self.print_step("Field installation completed successfully!", "SUCCESS")
            self.print_step(f"Installation log: {install_log}", "INFO")
            self.print_step(f"Configuration directory: {self.config_dir}", "INFO")
            
            print("\nNext steps:")
            print("1. Start the Foresight SAR application")
            print("2. Test camera and detection functionality")
            print("3. Verify evidence packaging and signing")
            print("4. Download offline map data for your area of operations")
            
        except KeyboardInterrupt:
            self.print_step("Installation interrupted by user", "WARNING")
        except Exception as e:
            self.print_step(f"Installation failed: {e}", "ERROR")
            installation_results["error"] = str(e)
        
        return installation_results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Foresight SAR Field Installer")
    parser.add_argument("--step", choices=["hardware", "calibration", "maps", "keys", "all"],
                       default="all", help="Run specific installation step")
    parser.add_argument("--config-dir", help="Configuration directory path")
    
    args = parser.parse_args()
    
    installer = FieldInstaller()
    
    if args.config_dir:
        installer.config_dir = Path(args.config_dir)
        installer.keys_dir = installer.config_dir / "keys"
        installer.calibration_dir = installer.config_dir / "calibration"
    
    if args.step == "all":
        installer.run_installation()
    elif args.step == "hardware":
        installer.check_hardware()
    elif args.step == "calibration":
        installer.calibrate_camera()
    elif args.step == "maps":
        installer.setup_offline_maps()
    elif args.step == "keys":
        installer.generate_keys_and_signin()


if __name__ == "__main__":
    main()