#!/usr/bin/env python3
"""
Camera Calibration Service for DJI O4 and Other Cameras

Provides camera intrinsic parameter calibration and management:
- DJI O4 camera specifications and defaults
- Camera calibration from checkerboard patterns
- Distortion correction and undistortion
- Multi-resolution support for different video modes
- Calibration validation and accuracy assessment

Supports various camera models and provides accurate intrinsics for geolocation.
"""

import cv2
import numpy as np
import json
import yaml
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import math

from .telemetry_service import CameraIntrinsics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CalibrationResult:
    """Camera calibration result with accuracy metrics."""
    
    intrinsics: CameraIntrinsics
    reprojection_error: float  # RMS reprojection error in pixels
    calibration_images: int    # Number of images used
    pattern_size: Tuple[int, int]  # Checkerboard pattern size
    square_size: float         # Physical size of checkerboard squares (mm)
    
    # Calibration quality metrics
    coverage_score: float      # 0-1, how well the calibration covers the image
    symmetry_score: float      # 0-1, symmetry of calibration pattern positions
    
    def is_good_calibration(self) -> bool:
        """Check if calibration meets quality standards."""
        return (self.reprojection_error < 1.0 and 
                self.coverage_score > 0.7 and 
                self.calibration_images >= 10)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['intrinsics'] = asdict(self.intrinsics)
        return result

class DJICameraSpecs:
    """DJI camera specifications and default parameters."""
    
    # DJI O4 Camera Specifications
    DJI_O4_SPECS = {
        "sensor_size_mm": (13.2, 8.8),  # 1-inch sensor
        "resolutions": {
            "4K": (3840, 2160),
            "2.7K": (2720, 1530),
            "FHD": (1920, 1080),
            "HD": (1280, 720)
        },
        "focal_length_mm": 8.8,  # 35mm equivalent: ~24mm
        "fov_degrees": 84,       # Field of view
        "aperture": "f/2.8-f/11",
        "focus_range": "1m to infinity",
        "distortion_model": "brown_conrady"
    }
    
    # Other DJI cameras
    DJI_MINI_3_SPECS = {
        "sensor_size_mm": (9.7, 7.3),
        "resolutions": {
            "4K": (3840, 2160),
            "2.7K": (2720, 1530),
            "FHD": (1920, 1080)
        },
        "focal_length_mm": 6.7,
        "fov_degrees": 83,
        "aperture": "f/1.7",
        "distortion_model": "brown_conrady"
    }
    
    @classmethod
    def get_default_intrinsics(cls, camera_model: str, resolution: str) -> CameraIntrinsics:
        """Get default intrinsics for DJI camera model and resolution.
        
        Args:
            camera_model: "O4", "Mini3", etc.
            resolution: "4K", "2.7K", "FHD", "HD"
            
        Returns:
            Default CameraIntrinsics object
        """
        if camera_model.upper() == "O4":
            specs = cls.DJI_O4_SPECS
        elif camera_model.upper() == "MINI3":
            specs = cls.DJI_MINI_3_SPECS
        else:
            logger.warning(f"Unknown camera model: {camera_model}, using O4 defaults")
            specs = cls.DJI_O4_SPECS
        
        if resolution not in specs["resolutions"]:
            logger.warning(f"Unknown resolution: {resolution}, using 4K")
            resolution = "4K"
        
        width, height = specs["resolutions"][resolution]
        focal_length_mm = specs["focal_length_mm"]
        sensor_width_mm, sensor_height_mm = specs["sensor_size_mm"]
        
        # Calculate focal length in pixels
        fx = (focal_length_mm * width) / sensor_width_mm
        fy = (focal_length_mm * height) / sensor_height_mm
        
        # Principal point (usually near center)
        cx = width / 2.0
        cy = height / 2.0
        
        return CameraIntrinsics(
            cx=cx,
            cy=cy,
            fx=fx,
            fy=fy,
            width=width,
            height=height,
            camera_name=f"DJI_{camera_model}_{resolution}",
            model="pinhole"
        )
    
    @classmethod
    def estimate_distortion_coefficients(cls, camera_model: str) -> List[float]:
        """Estimate typical distortion coefficients for DJI cameras.
        
        Args:
            camera_model: DJI camera model
            
        Returns:
            [k1, k2, p1, p2, k3] distortion coefficients
        """
        if camera_model.upper() == "O4":
            # DJI O4 typically has low distortion due to good lens design
            return [-0.05, 0.02, 0.001, 0.001, -0.005]
        elif camera_model.upper() == "MINI3":
            # Mini 3 may have slightly more distortion
            return [-0.08, 0.04, 0.002, 0.002, -0.01]
        else:
            # Conservative estimates for unknown models
            return [-0.1, 0.05, 0.002, 0.002, -0.01]

class CameraCalibrator:
    """Camera calibration using checkerboard patterns."""
    
    def __init__(self, pattern_size: Tuple[int, int] = (9, 6), 
                 square_size: float = 25.0):
        """
        Initialize camera calibrator.
        
        Args:
            pattern_size: (width, height) of checkerboard interior corners
            square_size: Physical size of checkerboard squares in mm
        """
        self.pattern_size = pattern_size
        self.square_size = square_size
        
        # Prepare object points (3D points in real world space)
        self.objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Storage for calibration data
        self.object_points = []  # 3D points in real world space
        self.image_points = []   # 2D points in image plane
        self.image_size = None
        
    def add_calibration_image(self, image: np.ndarray) -> bool:
        """Add a calibration image with checkerboard pattern.
        
        Args:
            image: Input image (color or grayscale)
            
        Returns:
            True if checkerboard was found and added
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Store image size
        if self.image_size is None:
            self.image_size = gray.shape[::-1]  # (width, height)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
        
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Store the points
            self.object_points.append(self.objp)
            self.image_points.append(corners_refined)
            
            logger.info(f"Added calibration image {len(self.image_points)}, found {len(corners_refined)} corners")
            return True
        else:
            logger.warning("Checkerboard pattern not found in image")
            return False
    
    def calibrate(self) -> Optional[CalibrationResult]:
        """Perform camera calibration.
        
        Returns:
            CalibrationResult or None if calibration failed
        """
        if len(self.object_points) < 5:
            logger.error(f"Need at least 5 calibration images, got {len(self.object_points)}")
            return None
        
        if self.image_size is None:
            logger.error("No image size available")
            return None
        
        try:
            # Perform calibration
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                self.object_points, self.image_points, self.image_size, None, None
            )
            
            if not ret:
                logger.error("Camera calibration failed")
                return None
            
            # Calculate reprojection error
            total_error = 0
            total_points = 0
            
            for i in range(len(self.object_points)):
                projected_points, _ = cv2.projectPoints(
                    self.object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
                )
                error = cv2.norm(self.image_points[i], projected_points, cv2.NORM_L2) / len(projected_points)
                total_error += error * len(projected_points)
                total_points += len(projected_points)
            
            rms_error = total_error / total_points
            
            # Create CameraIntrinsics object
            intrinsics = CameraIntrinsics(
                cx=camera_matrix[0, 2],
                cy=camera_matrix[1, 2],
                fx=camera_matrix[0, 0],
                fy=camera_matrix[1, 1],
                width=self.image_size[0],
                height=self.image_size[1],
                k1=dist_coeffs[0, 0] if len(dist_coeffs) > 0 else 0.0,
                k2=dist_coeffs[0, 1] if len(dist_coeffs) > 1 else 0.0,
                p1=dist_coeffs[0, 2] if len(dist_coeffs) > 2 else 0.0,
                p2=dist_coeffs[0, 3] if len(dist_coeffs) > 3 else 0.0,
                k3=dist_coeffs[0, 4] if len(dist_coeffs) > 4 else 0.0,
                camera_name="calibrated_camera",
                model="brown_conrady"
            )
            
            # Calculate quality metrics
            coverage_score = self._calculate_coverage_score()
            symmetry_score = self._calculate_symmetry_score()
            
            result = CalibrationResult(
                intrinsics=intrinsics,
                reprojection_error=rms_error,
                calibration_images=len(self.object_points),
                pattern_size=self.pattern_size,
                square_size=self.square_size,
                coverage_score=coverage_score,
                symmetry_score=symmetry_score
            )
            
            logger.info(f"Calibration completed: RMS error = {rms_error:.3f} pixels")
            logger.info(f"Coverage score: {coverage_score:.3f}, Symmetry score: {symmetry_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return None
    
    def _calculate_coverage_score(self) -> float:
        """Calculate how well calibration images cover the image area."""
        if not self.image_points or self.image_size is None:
            return 0.0
        
        # Create a grid to track coverage
        grid_size = 10
        coverage_grid = np.zeros((grid_size, grid_size), dtype=bool)
        
        width, height = self.image_size
        
        for image_corners in self.image_points:
            for corner in image_corners:
                x, y = corner[0]
                grid_x = int((x / width) * grid_size)
                grid_y = int((y / height) * grid_size)
                
                grid_x = max(0, min(grid_size - 1, grid_x))
                grid_y = max(0, min(grid_size - 1, grid_y))
                
                coverage_grid[grid_y, grid_x] = True
        
        return np.sum(coverage_grid) / (grid_size * grid_size)
    
    def _calculate_symmetry_score(self) -> float:
        """Calculate symmetry of calibration pattern positions."""
        if not self.image_points or self.image_size is None:
            return 0.0
        
        width, height = self.image_size
        center_x, center_y = width / 2, height / 2
        
        # Calculate average distance from center for each quadrant
        quadrant_counts = [0, 0, 0, 0]
        
        for image_corners in self.image_points:
            for corner in image_corners:
                x, y = corner[0]
                
                if x < center_x and y < center_y:
                    quadrant_counts[0] += 1  # Top-left
                elif x >= center_x and y < center_y:
                    quadrant_counts[1] += 1  # Top-right
                elif x < center_x and y >= center_y:
                    quadrant_counts[2] += 1  # Bottom-left
                else:
                    quadrant_counts[3] += 1  # Bottom-right
        
        if sum(quadrant_counts) == 0:
            return 0.0
        
        # Calculate symmetry as 1 - coefficient of variation
        mean_count = np.mean(quadrant_counts)
        if mean_count == 0:
            return 0.0
        
        std_count = np.std(quadrant_counts)
        cv = std_count / mean_count
        
        return max(0.0, 1.0 - cv)

class CameraCalibrationService:
    """Main service for camera calibration and intrinsics management."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Cache for loaded intrinsics
        self.intrinsics_cache: Dict[str, CameraIntrinsics] = {}
    
    def get_camera_intrinsics(self, camera_model: str, resolution: str = "4K",
                            config_file: Optional[str] = None) -> CameraIntrinsics:
        """Get camera intrinsics from config file or defaults.
        
        Args:
            camera_model: Camera model ("O4", "Mini3", etc.)
            resolution: Video resolution ("4K", "FHD", etc.)
            config_file: Optional path to calibration config file
            
        Returns:
            CameraIntrinsics object
        """
        cache_key = f"{camera_model}_{resolution}_{config_file}"
        
        if cache_key in self.intrinsics_cache:
            return self.intrinsics_cache[cache_key]
        
        # Try to load from config file first
        if config_file and Path(config_file).exists():
            try:
                intrinsics = CameraIntrinsics.from_config(config_file)
                self.intrinsics_cache[cache_key] = intrinsics
                logger.info(f"Loaded camera intrinsics from {config_file}")
                return intrinsics
            except Exception as e:
                logger.warning(f"Failed to load config {config_file}: {e}")
        
        # Fall back to defaults
        intrinsics = DJICameraSpecs.get_default_intrinsics(camera_model, resolution)
        
        # Add estimated distortion coefficients
        distortion = DJICameraSpecs.estimate_distortion_coefficients(camera_model)
        intrinsics.k1, intrinsics.k2, intrinsics.p1, intrinsics.p2, intrinsics.k3 = distortion
        
        self.intrinsics_cache[cache_key] = intrinsics
        logger.info(f"Using default intrinsics for {camera_model} {resolution}")
        
        return intrinsics
    
    def save_calibration(self, result: CalibrationResult, filename: str) -> bool:
        """Save calibration result to file.
        
        Args:
            result: CalibrationResult to save
            filename: Output filename (without extension)
            
        Returns:
            True if saved successfully
        """
        try:
            # Save as YAML (preferred for readability)
            yaml_path = self.config_dir / f"{filename}.yaml"
            with open(yaml_path, 'w') as f:
                yaml.dump(result.to_dict(), f, default_flow_style=False)
            
            # Also save as JSON for compatibility
            json_path = self.config_dir / f"{filename}.json"
            with open(json_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            
            logger.info(f"Saved calibration to {yaml_path} and {json_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
            return False
    
    def load_calibration(self, filename: str) -> Optional[CalibrationResult]:
        """Load calibration result from file.
        
        Args:
            filename: Calibration filename (with or without extension)
            
        Returns:
            CalibrationResult or None if loading failed
        """
        try:
            # Try YAML first
            yaml_path = self.config_dir / f"{filename}.yaml"
            if not yaml_path.exists():
                yaml_path = self.config_dir / filename
            
            if yaml_path.exists() and yaml_path.suffix == '.yaml':
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
            else:
                # Try JSON
                json_path = self.config_dir / f"{filename}.json"
                if not json_path.exists():
                    json_path = self.config_dir / filename
                
                with open(json_path, 'r') as f:
                    data = json.load(f)
            
            # Reconstruct CalibrationResult
            intrinsics_data = data['intrinsics']
            intrinsics = CameraIntrinsics(**intrinsics_data)
            
            result = CalibrationResult(
                intrinsics=intrinsics,
                reprojection_error=data['reprojection_error'],
                calibration_images=data['calibration_images'],
                pattern_size=tuple(data['pattern_size']),
                square_size=data['square_size'],
                coverage_score=data['coverage_score'],
                symmetry_score=data['symmetry_score']
            )
            
            logger.info(f"Loaded calibration from {yaml_path if yaml_path.exists() else json_path}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to load calibration {filename}: {e}")
            return None
    
    def calibrate_from_images(self, image_paths: List[str], 
                            pattern_size: Tuple[int, int] = (9, 6),
                            square_size: float = 25.0) -> Optional[CalibrationResult]:
        """Calibrate camera from a set of checkerboard images.
        
        Args:
            image_paths: List of paths to calibration images
            pattern_size: Checkerboard pattern size (width, height)
            square_size: Physical size of squares in mm
            
        Returns:
            CalibrationResult or None if calibration failed
        """
        calibrator = CameraCalibrator(pattern_size, square_size)
        
        successful_images = 0
        for image_path in image_paths:
            try:
                image = cv2.imread(image_path)
                if image is None:
                    logger.warning(f"Could not load image: {image_path}")
                    continue
                
                if calibrator.add_calibration_image(image):
                    successful_images += 1
                    
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
        
        logger.info(f"Successfully processed {successful_images}/{len(image_paths)} images")
        
        if successful_images < 5:
            logger.error("Need at least 5 successful calibration images")
            return None
        
        return calibrator.calibrate()
    
    def validate_intrinsics(self, intrinsics: CameraIntrinsics) -> Dict[str, Any]:
        """Validate camera intrinsics for reasonableness.
        
        Args:
            intrinsics: CameraIntrinsics to validate
            
        Returns:
            Validation report dictionary
        """
        issues = []
        warnings = []
        
        # Check principal point
        if not (0 < intrinsics.cx < intrinsics.width):
            issues.append(f"Principal point X ({intrinsics.cx}) outside image width ({intrinsics.width})")
        
        if not (0 < intrinsics.cy < intrinsics.height):
            issues.append(f"Principal point Y ({intrinsics.cy}) outside image height ({intrinsics.height})")
        
        # Check if principal point is too far from center
        center_x, center_y = intrinsics.width / 2, intrinsics.height / 2
        px_offset = abs(intrinsics.cx - center_x) / intrinsics.width
        py_offset = abs(intrinsics.cy - center_y) / intrinsics.height
        
        if px_offset > 0.1 or py_offset > 0.1:
            warnings.append(f"Principal point far from center: offset ({px_offset:.3f}, {py_offset:.3f})")
        
        # Check focal lengths
        if intrinsics.fx <= 0 or intrinsics.fy <= 0:
            issues.append(f"Invalid focal lengths: fx={intrinsics.fx}, fy={intrinsics.fy}")
        
        # Check aspect ratio
        aspect_ratio = intrinsics.fx / intrinsics.fy
        if not (0.9 < aspect_ratio < 1.1):
            warnings.append(f"Unusual aspect ratio: {aspect_ratio:.3f}")
        
        # Check field of view
        fov_x = 2 * math.degrees(math.atan(intrinsics.width / (2 * intrinsics.fx)))
        fov_y = 2 * math.degrees(math.atan(intrinsics.height / (2 * intrinsics.fy)))
        
        if fov_x > 120 or fov_y > 120:
            warnings.append(f"Very wide field of view: {fov_x:.1f}° x {fov_y:.1f}°")
        
        if fov_x < 30 or fov_y < 30:
            warnings.append(f"Very narrow field of view: {fov_x:.1f}° x {fov_y:.1f}°")
        
        # Check distortion coefficients
        if abs(intrinsics.k1) > 0.5 or abs(intrinsics.k2) > 0.5:
            warnings.append(f"High radial distortion: k1={intrinsics.k1:.3f}, k2={intrinsics.k2:.3f}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "fov_degrees": (fov_x, fov_y),
            "aspect_ratio": aspect_ratio,
            "principal_point_offset": (px_offset, py_offset)
        }

# Global camera calibration service
camera_calibration_service = CameraCalibrationService()

def get_camera_calibration_service() -> CameraCalibrationService:
    """Get the global camera calibration service."""
    return camera_calibration_service

if __name__ == "__main__":
    # Demo usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Camera Calibration Service Demo")
    parser.add_argument("--camera", default="O4", help="Camera model (O4, Mini3)")
    parser.add_argument("--resolution", default="4K", help="Resolution (4K, FHD, HD)")
    parser.add_argument("--calibrate", nargs="*", help="Calibration image paths")
    parser.add_argument("--save", help="Save calibration with this name")
    parser.add_argument("--load", help="Load calibration from file")
    args = parser.parse_args()
    
    service = get_camera_calibration_service()
    
    if args.calibrate:
        print(f"Calibrating camera from {len(args.calibrate)} images...")
        result = service.calibrate_from_images(args.calibrate)
        
        if result:
            print(f"\nCalibration successful!")
            print(f"  RMS Error: {result.reprojection_error:.3f} pixels")
            print(f"  Coverage: {result.coverage_score:.3f}")
            print(f"  Symmetry: {result.symmetry_score:.3f}")
            print(f"  Quality: {'Good' if result.is_good_calibration() else 'Poor'}")
            
            if args.save:
                service.save_calibration(result, args.save)
                print(f"  Saved as: {args.save}")
        else:
            print("Calibration failed")
    
    elif args.load:
        result = service.load_calibration(args.load)
        if result:
            print(f"Loaded calibration: {args.load}")
            print(f"  RMS Error: {result.reprojection_error:.3f} pixels")
            print(f"  Images used: {result.calibration_images}")
        else:
            print(f"Failed to load: {args.load}")
    
    else:
        # Show default intrinsics
        intrinsics = service.get_camera_intrinsics(args.camera, args.resolution)
        print(f"\nDefault intrinsics for {args.camera} {args.resolution}:")
        print(f"  Resolution: {intrinsics.width} x {intrinsics.height}")
        print(f"  Focal length: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")
        print(f"  Principal point: ({intrinsics.cx:.1f}, {intrinsics.cy:.1f})")
        print(f"  Distortion: k1={intrinsics.k1:.4f}, k2={intrinsics.k2:.4f}")
        
        # Validate intrinsics
        validation = service.validate_intrinsics(intrinsics)
        print(f"\nValidation:")
        print(f"  Valid: {validation['valid']}")
        print(f"  FOV: {validation['fov_degrees'][0]:.1f}° x {validation['fov_degrees'][1]:.1f}°")
        
        if validation['warnings']:
            print(f"  Warnings: {len(validation['warnings'])}")
            for warning in validation['warnings']:
                print(f"    - {warning}")