"""
Camera Calibration Module for Foresight SAR System

This module handles camera calibration data loading, validation, and management
for accurate geolocation calculations. Supports various camera models and
distortion correction parameters.
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from enum import Enum


class CameraModel(Enum):
    """Supported camera models"""
    PINHOLE = "pinhole"
    FISHEYE = "fisheye"
    OMNIDIRECTIONAL = "omnidirectional"
    BROWN_CONRADY = "brown_conrady"


class DistortionModel(Enum):
    """Distortion correction models"""
    RADIAL_TANGENTIAL = "radial_tangential"
    FISHEYE_KB = "fisheye_kb"
    EQUIDISTANT = "equidistant"
    NONE = "none"


@dataclass
class CalibrationData:
    """Camera calibration parameters"""
    # Intrinsic parameters
    camera_matrix: np.ndarray  # 3x3 camera matrix
    distortion_coeffs: np.ndarray  # Distortion coefficients
    
    # Image dimensions
    image_width: int
    image_height: int
    
    # Camera model information
    camera_model: CameraModel = CameraModel.PINHOLE
    distortion_model: DistortionModel = DistortionModel.RADIAL_TANGENTIAL
    
    # Optional parameters
    rectification_matrix: Optional[np.ndarray] = None
    projection_matrix: Optional[np.ndarray] = None
    
    # Metadata
    calibration_date: Optional[str] = None
    camera_serial: Optional[str] = None
    lens_model: Optional[str] = None
    focal_length_mm: Optional[float] = None
    sensor_size_mm: Optional[Tuple[float, float]] = None
    
    # Quality metrics
    reprojection_error: Optional[float] = None
    calibration_flags: Optional[int] = None
    
    def __post_init__(self):
        """Validate calibration data after initialization"""
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate calibration parameters"""
        # Check camera matrix
        if self.camera_matrix.shape != (3, 3):
            raise ValueError(f"Camera matrix must be 3x3, got {self.camera_matrix.shape}")
        
        # Check focal lengths are positive
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        if fx <= 0 or fy <= 0:
            raise ValueError(f"Focal lengths must be positive: fx={fx}, fy={fy}")
        
        # Check principal point is within image bounds
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        if not (0 <= cx <= self.image_width and 0 <= cy <= self.image_height):
            logging.warning(f"Principal point ({cx}, {cy}) outside image bounds ({self.image_width}x{self.image_height})")
        
        # Check distortion coefficients
        if len(self.distortion_coeffs) < 4:
            logging.warning(f"Expected at least 4 distortion coefficients, got {len(self.distortion_coeffs)}")
    
    def get_focal_lengths(self) -> Tuple[float, float]:
        """Get focal lengths in pixels"""
        return self.camera_matrix[0, 0], self.camera_matrix[1, 1]
    
    def get_principal_point(self) -> Tuple[float, float]:
        """Get principal point coordinates"""
        return self.camera_matrix[0, 2], self.camera_matrix[1, 2]
    
    def get_field_of_view(self) -> Tuple[float, float]:
        """Calculate field of view in degrees"""
        fx, fy = self.get_focal_lengths()
        
        # Horizontal and vertical FOV
        fov_x = 2 * np.arctan(self.image_width / (2 * fx)) * 180 / np.pi
        fov_y = 2 * np.arctan(self.image_height / (2 * fy)) * 180 / np.pi
        
        return fov_x, fov_y
    
    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """Undistort image points"""
        if self.distortion_model == DistortionModel.NONE:
            return points
        
        points = np.array(points, dtype=np.float32)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        
        if self.distortion_model == DistortionModel.FISHEYE_KB:
            undistorted = cv2.fisheye.undistortPoints(
                points.reshape(-1, 1, 2),
                self.camera_matrix,
                self.distortion_coeffs
            )
        else:
            undistorted = cv2.undistortPoints(
                points.reshape(-1, 1, 2),
                self.camera_matrix,
                self.distortion_coeffs
            )
        
        return undistorted.reshape(-1, 2)
    
    def project_points(self, object_points: np.ndarray, 
                      rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Project 3D points to image plane"""
        if self.distortion_model == DistortionModel.FISHEYE_KB:
            projected, _ = cv2.fisheye.projectPoints(
                object_points.reshape(-1, 1, 3),
                rvec, tvec,
                self.camera_matrix,
                self.distortion_coeffs
            )
        else:
            projected, _ = cv2.projectPoints(
                object_points.reshape(-1, 1, 3),
                rvec, tvec,
                self.camera_matrix,
                self.distortion_coeffs
            )
        
        return projected.reshape(-1, 2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'camera_matrix': self.camera_matrix.tolist(),
            'distortion_coeffs': self.distortion_coeffs.tolist(),
            'image_width': self.image_width,
            'image_height': self.image_height,
            'camera_model': self.camera_model.value,
            'distortion_model': self.distortion_model.value,
            'rectification_matrix': self.rectification_matrix.tolist() if self.rectification_matrix is not None else None,
            'projection_matrix': self.projection_matrix.tolist() if self.projection_matrix is not None else None,
            'calibration_date': self.calibration_date,
            'camera_serial': self.camera_serial,
            'lens_model': self.lens_model,
            'focal_length_mm': self.focal_length_mm,
            'sensor_size_mm': self.sensor_size_mm,
            'reprojection_error': self.reprojection_error,
            'calibration_flags': self.calibration_flags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationData':
        """Create from dictionary"""
        return cls(
            camera_matrix=np.array(data['camera_matrix']),
            distortion_coeffs=np.array(data['distortion_coeffs']),
            image_width=data['image_width'],
            image_height=data['image_height'],
            camera_model=CameraModel(data.get('camera_model', 'pinhole')),
            distortion_model=DistortionModel(data.get('distortion_model', 'radial_tangential')),
            rectification_matrix=np.array(data['rectification_matrix']) if data.get('rectification_matrix') else None,
            projection_matrix=np.array(data['projection_matrix']) if data.get('projection_matrix') else None,
            calibration_date=data.get('calibration_date'),
            camera_serial=data.get('camera_serial'),
            lens_model=data.get('lens_model'),
            focal_length_mm=data.get('focal_length_mm'),
            sensor_size_mm=tuple(data['sensor_size_mm']) if data.get('sensor_size_mm') else None,
            reprojection_error=data.get('reprojection_error'),
            calibration_flags=data.get('calibration_flags')
        )


class CameraCalibrator:
    """Camera calibration manager"""
    
    def __init__(self, calibration_file: Optional[str] = None):
        """
        Initialize camera calibrator
        
        Args:
            calibration_file: Path to calibration JSON file
        """
        self.calibration_data: Optional[CalibrationData] = None
        self.calibration_file = calibration_file
        
        if calibration_file:
            self.load_calibration(calibration_file)
    
    def load_calibration(self, file_path: str) -> CalibrationData:
        """
        Load calibration data from JSON file
        
        Args:
            file_path: Path to calibration file
            
        Returns:
            Loaded calibration data
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            self.calibration_data = CalibrationData.from_dict(data)
            self.calibration_file = file_path
            
            logging.info(f"Loaded camera calibration from {file_path}")
            return self.calibration_data
            
        except Exception as e:
            logging.error(f"Failed to load calibration from {file_path}: {e}")
            raise
    
    def save_calibration(self, file_path: str, calibration_data: Optional[CalibrationData] = None):
        """
        Save calibration data to JSON file
        
        Args:
            file_path: Output file path
            calibration_data: Calibration data to save (uses current if None)
        """
        data = calibration_data or self.calibration_data
        if data is None:
            raise ValueError("No calibration data to save")
        
        try:
            with open(file_path, 'w') as f:
                json.dump(data.to_dict(), f, indent=2)
            
            logging.info(f"Saved camera calibration to {file_path}")
            
        except Exception as e:
            logging.error(f"Failed to save calibration to {file_path}: {e}")
            raise
    
    def calibrate_camera(self, 
                        object_points: List[np.ndarray],
                        image_points: List[np.ndarray],
                        image_size: Tuple[int, int],
                        camera_model: CameraModel = CameraModel.PINHOLE,
                        flags: Optional[int] = None) -> CalibrationData:
        """
        Perform camera calibration from calibration images
        
        Args:
            object_points: List of 3D calibration points
            image_points: List of corresponding 2D image points
            image_size: Image dimensions (width, height)
            camera_model: Camera model to use
            flags: OpenCV calibration flags
            
        Returns:
            Calibration data
        """
        if not object_points or not image_points:
            raise ValueError("Need calibration points for camera calibration")
        
        if len(object_points) != len(image_points):
            raise ValueError("Number of object and image point sets must match")
        
        # Set default flags
        if flags is None:
            if camera_model == CameraModel.FISHEYE:
                flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + 
                        cv2.fisheye.CALIB_CHECK_COND + 
                        cv2.fisheye.CALIB_FIX_SKEW)
            else:
                flags = (cv2.CALIB_RATIONAL_MODEL + 
                        cv2.CALIB_THIN_PRISM_MODEL + 
                        cv2.CALIB_TILTED_MODEL)
        
        try:
            if camera_model == CameraModel.FISHEYE:
                # Fisheye calibration
                ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.fisheye.calibrate(
                    object_points, image_points, image_size, None, None, flags=flags
                )
                distortion_model = DistortionModel.FISHEYE_KB
            else:
                # Standard calibration
                ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                    object_points, image_points, image_size, None, None, flags=flags
                )
                distortion_model = DistortionModel.RADIAL_TANGENTIAL
            
            # Create calibration data
            self.calibration_data = CalibrationData(
                camera_matrix=camera_matrix,
                distortion_coeffs=distortion_coeffs,
                image_width=image_size[0],
                image_height=image_size[1],
                camera_model=camera_model,
                distortion_model=distortion_model,
                reprojection_error=ret,
                calibration_flags=flags
            )
            
            logging.info(f"Camera calibration completed with RMS error: {ret:.4f}")
            return self.calibration_data
            
        except Exception as e:
            logging.error(f"Camera calibration failed: {e}")
            raise
    
    def validate_calibration(self, 
                           object_points: List[np.ndarray],
                           image_points: List[np.ndarray]) -> Dict[str, float]:
        """
        Validate calibration accuracy
        
        Args:
            object_points: Test 3D points
            image_points: Test 2D points
            
        Returns:
            Validation metrics
        """
        if self.calibration_data is None:
            raise ValueError("No calibration data loaded")
        
        total_error = 0
        total_points = 0
        errors = []
        
        for obj_pts, img_pts in zip(object_points, image_points):
            # Project object points
            rvec = np.zeros(3)
            tvec = np.zeros(3)
            
            projected_pts = self.calibration_data.project_points(obj_pts, rvec, tvec)
            
            # Calculate reprojection error
            error = cv2.norm(img_pts, projected_pts, cv2.NORM_L2) / len(projected_pts)
            errors.append(error)
            total_error += error * len(projected_pts)
            total_points += len(projected_pts)
        
        mean_error = total_error / total_points
        max_error = max(errors)
        min_error = min(errors)
        std_error = np.std(errors)
        
        return {
            'mean_error': mean_error,
            'max_error': max_error,
            'min_error': min_error,
            'std_error': std_error,
            'total_points': total_points
        }
    
    def create_default_calibration(self, 
                                 image_width: int, 
                                 image_height: int,
                                 fov_degrees: float = 60.0) -> CalibrationData:
        """
        Create default calibration for testing
        
        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels
            fov_degrees: Horizontal field of view in degrees
            
        Returns:
            Default calibration data
        """
        # Calculate focal length from FOV
        fov_rad = np.radians(fov_degrees)
        fx = image_width / (2 * np.tan(fov_rad / 2))
        fy = fx  # Assume square pixels
        
        # Principal point at image center
        cx = image_width / 2
        cy = image_height / 2
        
        # Create camera matrix
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        # Minimal distortion
        distortion_coeffs = np.zeros(5)
        
        self.calibration_data = CalibrationData(
            camera_matrix=camera_matrix,
            distortion_coeffs=distortion_coeffs,
            image_width=image_width,
            image_height=image_height,
            camera_model=CameraModel.PINHOLE,
            distortion_model=DistortionModel.NONE
        )
        
        logging.info(f"Created default calibration: {image_width}x{image_height}, FOV={fov_degrees}°")
        return self.calibration_data
    
    def get_calibration_info(self) -> Dict[str, Any]:
        """Get calibration information summary"""
        if self.calibration_data is None:
            return {'status': 'No calibration loaded'}
        
        cal = self.calibration_data
        fx, fy = cal.get_focal_lengths()
        cx, cy = cal.get_principal_point()
        fov_x, fov_y = cal.get_field_of_view()
        
        return {
            'status': 'Calibration loaded',
            'image_size': f"{cal.image_width}x{cal.image_height}",
            'camera_model': cal.camera_model.value,
            'distortion_model': cal.distortion_model.value,
            'focal_lengths': f"fx={fx:.2f}, fy={fy:.2f}",
            'principal_point': f"cx={cx:.2f}, cy={cy:.2f}",
            'field_of_view': f"horizontal={fov_x:.1f}°, vertical={fov_y:.1f}°",
            'reprojection_error': cal.reprojection_error,
            'calibration_file': self.calibration_file
        }