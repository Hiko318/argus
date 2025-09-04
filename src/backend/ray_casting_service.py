#!/usr/bin/env python3
"""
Ray Casting Service for 2D to 3D Projection

Provides ray-casting functionality for converting 2D pixel coordinates to 3D world rays:
- Pinhole camera model implementation
- Pixel to camera coordinate transformation
- Camera to world coordinate transformation using drone attitude
- Ray-plane and ray-terrain intersection
- Support for different coordinate systems and projections

Used for geolocation computation in drone-based detection systems.
"""

import numpy as np
import math
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
import logging

from .telemetry_service import TelemetryData, CameraIntrinsics
from .geolocation_service import CoordinateTransforms

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Ray3D:
    """3D ray representation."""
    
    origin: np.ndarray      # Ray origin point (3D)
    direction: np.ndarray   # Ray direction vector (3D, normalized)
    
    def __post_init__(self):
        """Ensure direction is normalized."""
        self.direction = self.direction / np.linalg.norm(self.direction)
    
    def point_at_distance(self, distance: float) -> np.ndarray:
        """Get point along ray at specified distance.
        
        Args:
            distance: Distance along ray from origin
            
        Returns:
            3D point coordinates
        """
        return self.origin + distance * self.direction
    
    def intersect_plane(self, plane_point: np.ndarray, plane_normal: np.ndarray) -> Optional[np.ndarray]:
        """Intersect ray with a plane.
        
        Args:
            plane_point: Point on the plane
            plane_normal: Plane normal vector (normalized)
            
        Returns:
            Intersection point or None if no intersection
        """
        # Normalize plane normal
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        
        # Check if ray is parallel to plane
        denom = np.dot(self.direction, plane_normal)
        if abs(denom) < 1e-6:
            return None  # Ray is parallel to plane
        
        # Calculate intersection distance
        t = np.dot(plane_point - self.origin, plane_normal) / denom
        
        # Check if intersection is in front of ray origin
        if t < 0:
            return None  # Intersection is behind ray origin
        
        return self.point_at_distance(t)
    
    def intersect_ground_plane(self, ground_elevation: float = 0.0) -> Optional[np.ndarray]:
        """Intersect ray with horizontal ground plane.
        
        Args:
            ground_elevation: Ground elevation (Z coordinate)
            
        Returns:
            Intersection point or None if no intersection
        """
        plane_point = np.array([0, 0, ground_elevation])
        plane_normal = np.array([0, 0, 1])  # Upward normal
        
        return self.intersect_plane(plane_point, plane_normal)

@dataclass
class RaycastResult:
    """Result of ray-casting operation."""
    
    pixel_coords: Tuple[float, float]  # Original pixel coordinates (u, v)
    camera_ray: Ray3D                  # Ray in camera coordinates
    world_ray: Ray3D                   # Ray in world coordinates
    ground_intersection: Optional[np.ndarray]  # Ground intersection point
    
    # Metadata
    camera_intrinsics: CameraIntrinsics
    telemetry: TelemetryData
    
    def get_intersection_distance(self) -> Optional[float]:
        """Get distance from camera to ground intersection.
        
        Returns:
            Distance in meters or None if no intersection
        """
        if self.ground_intersection is None:
            return None
        
        return np.linalg.norm(self.ground_intersection - self.world_ray.origin)
    
    def get_intersection_angles(self) -> Optional[Tuple[float, float]]:
        """Get viewing angles to intersection point.
        
        Returns:
            (elevation_angle, azimuth_angle) in degrees or None
        """
        if self.ground_intersection is None:
            return None
        
        # Vector from camera to intersection
        to_intersection = self.ground_intersection - self.world_ray.origin
        to_intersection_norm = to_intersection / np.linalg.norm(to_intersection)
        
        # Elevation angle (angle from horizontal plane)
        elevation = math.degrees(math.asin(-to_intersection_norm[2]))
        
        # Azimuth angle (angle from north in horizontal plane)
        azimuth = math.degrees(math.atan2(to_intersection_norm[1], to_intersection_norm[0]))
        
        return elevation, azimuth

class PinholeCameraModel:
    """Pinhole camera model for ray-casting."""
    
    def __init__(self, intrinsics: CameraIntrinsics):
        """
        Initialize pinhole camera model.
        
        Args:
            intrinsics: Camera intrinsic parameters
        """
        self.intrinsics = intrinsics
        
        # Create camera matrix
        self.camera_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.cx],
            [0, intrinsics.fy, intrinsics.cy],
            [0, 0, 1]
        ])
        
        # Distortion coefficients
        self.distortion_coeffs = np.array([
            intrinsics.k1, intrinsics.k2, intrinsics.p1, intrinsics.p2, intrinsics.k3
        ])
        
        # Inverse camera matrix for ray-casting
        self.inv_camera_matrix = np.linalg.inv(self.camera_matrix)
    
    def pixel_to_camera_ray(self, u: float, v: float, undistort: bool = True) -> Ray3D:
        """Convert pixel coordinates to camera ray.
        
        Args:
            u: Pixel x-coordinate
            v: Pixel y-coordinate
            undistort: Whether to apply distortion correction
            
        Returns:
            Ray3D in camera coordinates
        """
        # Handle distortion correction
        if undistort and np.any(self.distortion_coeffs != 0):
            # Apply inverse distortion (simplified)
            u_norm = (u - self.intrinsics.cx) / self.intrinsics.fx
            v_norm = (v - self.intrinsics.cy) / self.intrinsics.fy
            
            # Radial distortion correction (iterative)
            r2 = u_norm**2 + v_norm**2
            radial_factor = 1 + self.intrinsics.k1 * r2 + self.intrinsics.k2 * r2**2 + self.intrinsics.k3 * r2**3
            
            # Tangential distortion correction
            u_corrected = u_norm / radial_factor - 2 * self.intrinsics.p1 * u_norm * v_norm - self.intrinsics.p2 * (r2 + 2 * u_norm**2)
            v_corrected = v_norm / radial_factor - self.intrinsics.p1 * (r2 + 2 * v_norm**2) - 2 * self.intrinsics.p2 * u_norm * v_norm
            
            # Convert back to pixel coordinates
            u = u_corrected * self.intrinsics.fx + self.intrinsics.cx
            v = v_corrected * self.intrinsics.fy + self.intrinsics.cy
        
        # Convert to homogeneous coordinates
        pixel_homogeneous = np.array([u, v, 1.0])
        
        # Transform to camera coordinates
        camera_coords = self.inv_camera_matrix @ pixel_homogeneous
        
        # Create ray (origin at camera center, direction towards pixel)
        ray_origin = np.array([0.0, 0.0, 0.0])  # Camera center
        ray_direction = camera_coords / np.linalg.norm(camera_coords)
        
        return Ray3D(origin=ray_origin, direction=ray_direction)
    
    def camera_to_world_ray(self, camera_ray: Ray3D, telemetry: TelemetryData) -> Ray3D:
        """Transform camera ray to world coordinates.
        
        Args:
            camera_ray: Ray in camera coordinates
            telemetry: Drone telemetry data
            
        Returns:
            Ray3D in world coordinates
        """
        # Get rotation matrix from camera to world
        R_world_to_camera = CoordinateTransforms.euler_to_rotation_matrix(
            telemetry.roll, telemetry.pitch, telemetry.yaw
        )
        
        # Apply gimbal rotation if available
        if telemetry.gimbal_pitch is not None:
            R_gimbal = CoordinateTransforms.gimbal_rotation_matrix(
                telemetry.gimbal_pitch, telemetry.gimbal_roll or 0.0, telemetry.gimbal_yaw or 0.0
            )
            R_world_to_camera = R_world_to_camera @ R_gimbal
        
        # Camera to world transformation (inverse of world to camera)
        R_camera_to_world = R_world_to_camera.T
        
        # Transform ray origin (drone position)
        world_origin = np.array([0.0, 0.0, telemetry.altitude])  # Relative to ground
        
        # Transform ray direction
        world_direction = R_camera_to_world @ camera_ray.direction
        
        return Ray3D(origin=world_origin, direction=world_direction)
    
    def validate_pixel_coordinates(self, u: float, v: float) -> bool:
        """Check if pixel coordinates are within image bounds.
        
        Args:
            u: Pixel x-coordinate
            v: Pixel y-coordinate
            
        Returns:
            True if coordinates are valid
        """
        return (0 <= u < self.intrinsics.width and 
                0 <= v < self.intrinsics.height)

class RayCastingService:
    """Main service for ray-casting operations."""
    
    def __init__(self):
        self.camera_models: Dict[str, PinholeCameraModel] = {}
    
    def get_camera_model(self, intrinsics: CameraIntrinsics) -> PinholeCameraModel:
        """Get or create camera model for given intrinsics.
        
        Args:
            intrinsics: Camera intrinsic parameters
            
        Returns:
            PinholeCameraModel instance
        """
        # Use camera name as key for caching
        cache_key = intrinsics.camera_name or "default"
        
        if cache_key not in self.camera_models:
            self.camera_models[cache_key] = PinholeCameraModel(intrinsics)
        
        return self.camera_models[cache_key]
    
    def cast_ray(self, u: float, v: float, intrinsics: CameraIntrinsics, 
                telemetry: TelemetryData, ground_elevation: float = 0.0) -> RaycastResult:
        """Cast ray from pixel coordinates to world coordinates.
        
        Args:
            u: Pixel x-coordinate
            v: Pixel y-coordinate
            intrinsics: Camera intrinsic parameters
            telemetry: Drone telemetry data
            ground_elevation: Ground elevation for intersection
            
        Returns:
            RaycastResult with ray and intersection information
        """
        # Get camera model
        camera_model = self.get_camera_model(intrinsics)
        
        # Validate pixel coordinates
        if not camera_model.validate_pixel_coordinates(u, v):
            logger.warning(f"Pixel coordinates ({u}, {v}) outside image bounds")
        
        # Convert pixel to camera ray
        camera_ray = camera_model.pixel_to_camera_ray(u, v)
        
        # Transform to world coordinates
        world_ray = camera_model.camera_to_world_ray(camera_ray, telemetry)
        
        # Intersect with ground plane
        ground_intersection = world_ray.intersect_ground_plane(ground_elevation)
        
        return RaycastResult(
            pixel_coords=(u, v),
            camera_ray=camera_ray,
            world_ray=world_ray,
            ground_intersection=ground_intersection,
            camera_intrinsics=intrinsics,
            telemetry=telemetry
        )
    
    def cast_detection_ray(self, detection_bbox: Tuple[float, float, float, float],
                          intrinsics: CameraIntrinsics, telemetry: TelemetryData,
                          ground_elevation: float = 0.0) -> RaycastResult:
        """Cast ray from detection bounding box center.
        
        Args:
            detection_bbox: (x1, y1, x2, y2) bounding box coordinates
            intrinsics: Camera intrinsic parameters
            telemetry: Drone telemetry data
            ground_elevation: Ground elevation for intersection
            
        Returns:
            RaycastResult for detection center
        """
        x1, y1, x2, y2 = detection_bbox
        
        # Calculate bounding box center
        center_u = (x1 + x2) / 2.0
        center_v = (y1 + y2) / 2.0
        
        return self.cast_ray(center_u, center_v, intrinsics, telemetry, ground_elevation)
    
    def cast_multiple_rays(self, pixel_coords: List[Tuple[float, float]],
                          intrinsics: CameraIntrinsics, telemetry: TelemetryData,
                          ground_elevation: float = 0.0) -> List[RaycastResult]:
        """Cast multiple rays efficiently.
        
        Args:
            pixel_coords: List of (u, v) pixel coordinates
            intrinsics: Camera intrinsic parameters
            telemetry: Drone telemetry data
            ground_elevation: Ground elevation for intersection
            
        Returns:
            List of RaycastResult objects
        """
        results = []
        
        for u, v in pixel_coords:
            result = self.cast_ray(u, v, intrinsics, telemetry, ground_elevation)
            results.append(result)
        
        return results
    
    def estimate_ray_accuracy(self, result: RaycastResult) -> Dict[str, float]:
        """Estimate accuracy of ray-casting result.
        
        Args:
            result: RaycastResult to analyze
            
        Returns:
            Dictionary with accuracy metrics
        """
        accuracy = {}
        
        # Distance-based accuracy (farther = less accurate)
        if result.ground_intersection is not None:
            distance = result.get_intersection_distance()
            if distance:
                # Assume 1 pixel error translates to angular error
                pixel_error = 1.0  # pixels
                angular_error = pixel_error / result.camera_intrinsics.fx  # radians
                position_error = distance * angular_error  # meters
                
                accuracy['distance_m'] = distance
                accuracy['angular_error_deg'] = math.degrees(angular_error)
                accuracy['position_error_m'] = position_error
                accuracy['relative_error'] = position_error / distance
        
        # Attitude accuracy impact
        attitude_error_deg = 1.0  # Assume 1 degree attitude error
        attitude_error_rad = math.radians(attitude_error_deg)
        
        if result.ground_intersection is not None:
            distance = result.get_intersection_distance()
            if distance:
                attitude_position_error = distance * attitude_error_rad
                accuracy['attitude_error_deg'] = attitude_error_deg
                accuracy['attitude_position_error_m'] = attitude_position_error
        
        # Altitude accuracy impact
        altitude_error = 1.0  # Assume 1 meter altitude error
        accuracy['altitude_error_m'] = altitude_error
        
        # Combined accuracy estimate
        if 'position_error_m' in accuracy and 'attitude_position_error_m' in accuracy:
            total_error = math.sqrt(
                accuracy['position_error_m']**2 + 
                accuracy['attitude_position_error_m']**2 + 
                altitude_error**2
            )
            accuracy['total_error_m'] = total_error
        
        return accuracy
    
    def debug_ray_casting(self, result: RaycastResult) -> Dict[str, Any]:
        """Generate debug information for ray-casting.
        
        Args:
            result: RaycastResult to debug
            
        Returns:
            Debug information dictionary
        """
        debug_info = {
            'pixel_coords': result.pixel_coords,
            'camera_ray_direction': result.camera_ray.direction.tolist(),
            'world_ray_origin': result.world_ray.origin.tolist(),
            'world_ray_direction': result.world_ray.direction.tolist(),
        }
        
        if result.ground_intersection is not None:
            debug_info['ground_intersection'] = result.ground_intersection.tolist()
            debug_info['intersection_distance'] = result.get_intersection_distance()
            
            angles = result.get_intersection_angles()
            if angles:
                debug_info['elevation_angle_deg'] = angles[0]
                debug_info['azimuth_angle_deg'] = angles[1]
        
        # Add telemetry info
        debug_info['telemetry'] = {
            'altitude': result.telemetry.altitude,
            'roll': result.telemetry.roll,
            'pitch': result.telemetry.pitch,
            'yaw': result.telemetry.yaw,
            'gimbal_pitch': result.telemetry.gimbal_pitch
        }
        
        # Add camera info
        debug_info['camera'] = {
            'fx': result.camera_intrinsics.fx,
            'fy': result.camera_intrinsics.fy,
            'cx': result.camera_intrinsics.cx,
            'cy': result.camera_intrinsics.cy,
            'width': result.camera_intrinsics.width,
            'height': result.camera_intrinsics.height
        }
        
        return debug_info

# Global ray-casting service
ray_casting_service = RayCastingService()

def get_ray_casting_service() -> RayCastingService:
    """Get the global ray-casting service."""
    return ray_casting_service

if __name__ == "__main__":
    # Demo usage
    from .camera_calibration import DJICameraSpecs
    
    # Create sample data
    intrinsics = DJICameraSpecs.get_default_intrinsics("O4", "4K")
    
    telemetry = TelemetryData(
        timestamp=0.0,
        latitude=37.7749,
        longitude=-122.4194,
        altitude=100.0,
        roll=0.0,
        pitch=-15.0,  # Camera pointing slightly down
        yaw=45.0,
        gimbal_pitch=-30.0,
        gimbal_roll=0.0,
        gimbal_yaw=0.0
    )
    
    # Test ray-casting
    service = get_ray_casting_service()
    
    # Cast ray from image center
    center_u = intrinsics.width / 2
    center_v = intrinsics.height / 2
    
    result = service.cast_ray(center_u, center_v, intrinsics, telemetry)
    
    print(f"Ray-casting demo:")
    print(f"  Pixel: ({center_u}, {center_v})")
    print(f"  Camera ray direction: {result.camera_ray.direction}")
    print(f"  World ray origin: {result.world_ray.origin}")
    print(f"  World ray direction: {result.world_ray.direction}")
    
    if result.ground_intersection is not None:
        print(f"  Ground intersection: {result.ground_intersection}")
        print(f"  Distance: {result.get_intersection_distance():.1f} m")
        
        angles = result.get_intersection_angles()
        if angles:
            print(f"  Viewing angles: elevation={angles[0]:.1f}°, azimuth={angles[1]:.1f}°")
    
    # Accuracy analysis
    accuracy = service.estimate_ray_accuracy(result)
    print(f"\nAccuracy estimate:")
    for key, value in accuracy.items():
        print(f"  {key}: {value:.3f}")