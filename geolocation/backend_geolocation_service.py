#!/usr/bin/env python3
"""
Geolocation Service for Human Detection

Computes real-world geographic coordinates from 2D detection bounding boxes using:
- Camera intrinsics and pinhole camera model
- Drone telemetry (GPS, altitude, orientation)
- Ray-casting from camera to ground plane
- DEM terrain intersection for accurate elevation
- Coordinate system transformations

Provides accurate geolocation for detected humans in aerial imagery.
"""

import math
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import json

from .telemetry_service import TelemetryData, CameraIntrinsics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GeolocationResult:
    """Result of geolocation computation."""
    
    # Geographic coordinates
    latitude: float   # degrees
    longitude: float  # degrees
    elevation: float  # meters above sea level
    
    # Accuracy estimates
    horizontal_accuracy: float  # meters (estimated error)
    vertical_accuracy: float    # meters (estimated error)
    
    # Source information
    detection_bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    pixel_center: Tuple[float, float]  # (u, v) pixel coordinates
    telemetry_timestamp: float
    
    # Computation metadata
    method: str = "ray_casting"  # "flat_terrain" or "dem_intersection"
    confidence: float = 1.0      # 0-1 confidence in result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "elevation": self.elevation,
            "horizontal_accuracy": self.horizontal_accuracy,
            "vertical_accuracy": self.vertical_accuracy,
            "detection_bbox": self.detection_bbox,
            "pixel_center": self.pixel_center,
            "telemetry_timestamp": self.telemetry_timestamp,
            "method": self.method,
            "confidence": self.confidence
        }

class CoordinateTransforms:
    """Coordinate system transformation utilities."""
    
    @staticmethod
    def euler_to_rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
        """Convert Euler angles to 3x3 rotation matrix.
        
        Args:
            yaw: Rotation around Z-axis (degrees)
            pitch: Rotation around Y-axis (degrees) 
            roll: Rotation around X-axis (degrees)
            
        Returns:
            3x3 rotation matrix (world to camera)
        """
        # Convert to radians
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        roll_rad = math.radians(roll)
        
        # Individual rotation matrices
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
        cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)
        cos_r, sin_r = math.cos(roll_rad), math.sin(roll_rad)
        
        # Combined rotation matrix (ZYX order)
        R = np.array([
            [cos_y * cos_p, cos_y * sin_p * sin_r - sin_y * cos_r, cos_y * sin_p * cos_r + sin_y * sin_r],
            [sin_y * cos_p, sin_y * sin_p * sin_r + cos_y * cos_r, sin_y * sin_p * cos_r - cos_y * sin_r],
            [-sin_p, cos_p * sin_r, cos_p * cos_r]
        ])
        
        return R
    
    @staticmethod
    def apply_gimbal_rotation(camera_ray: np.ndarray, gimbal_yaw: float, 
                            gimbal_pitch: float, gimbal_roll: float) -> np.ndarray:
        """Apply gimbal rotation to camera ray.
        
        Args:
            camera_ray: 3D ray in camera coordinates
            gimbal_yaw: Gimbal yaw angle (degrees)
            gimbal_pitch: Gimbal pitch angle (degrees)
            gimbal_roll: Gimbal roll angle (degrees)
            
        Returns:
            Ray rotated by gimbal angles
        """
        gimbal_rotation = CoordinateTransforms.euler_to_rotation_matrix(
            gimbal_yaw, gimbal_pitch, gimbal_roll
        )
        return gimbal_rotation @ camera_ray
    
    @staticmethod
    def pixel_to_camera_ray(pixel_u: float, pixel_v: float, 
                          intrinsics: CameraIntrinsics) -> np.ndarray:
        """Convert pixel coordinates to normalized camera ray.
        
        Args:
            pixel_u: Pixel x-coordinate
            pixel_v: Pixel y-coordinate
            intrinsics: Camera intrinsic parameters
            
        Returns:
            Normalized 3D ray in camera coordinates
        """
        # Normalize pixel coordinates
        x_norm = (pixel_u - intrinsics.cx) / intrinsics.fx
        y_norm = (pixel_v - intrinsics.cy) / intrinsics.fy
        
        # Create 3D ray (pointing into the scene)
        ray = np.array([x_norm, y_norm, 1.0])
        
        # Normalize to unit vector
        return ray / np.linalg.norm(ray)
    
    @staticmethod
    def camera_to_world_ray(camera_ray: np.ndarray, drone_rotation: np.ndarray) -> np.ndarray:
        """Transform camera ray to world coordinates.
        
        Args:
            camera_ray: 3D ray in camera coordinates
            drone_rotation: 3x3 rotation matrix (camera to world)
            
        Returns:
            3D ray in world coordinates
        """
        # Transform to world coordinates
        world_ray = drone_rotation.T @ camera_ray  # Transpose for camera-to-world
        return world_ray / np.linalg.norm(world_ray)
    
    @staticmethod
    def meters_to_degrees(meters: float, latitude: float) -> Tuple[float, float]:
        """Convert meters to degrees at given latitude.
        
        Args:
            meters: Distance in meters
            latitude: Latitude in degrees
            
        Returns:
            (lat_degrees, lon_degrees) conversion factors
        """
        # Earth radius in meters
        earth_radius = 6378137.0
        
        # Latitude conversion (constant)
        lat_deg_per_meter = 1.0 / (earth_radius * math.pi / 180.0)
        
        # Longitude conversion (varies with latitude)
        lon_deg_per_meter = lat_deg_per_meter / math.cos(math.radians(latitude))
        
        return lat_deg_per_meter * meters, lon_deg_per_meter * meters

class DEMService:
    """Digital Elevation Model service for terrain intersection."""
    
    def __init__(self, dem_data_path: Optional[str] = None):
        self.dem_data_path = dem_data_path
        self.dem_data: Optional[Dict[str, Any]] = None
        self.elevation_cache: Dict[Tuple[float, float], float] = {}
        
        if dem_data_path:
            self.load_dem_data(dem_data_path)
    
    def load_dem_data(self, dem_path: str) -> bool:
        """Load DEM data from file.
        
        Args:
            dem_path: Path to DEM data file
            
        Returns:
            True if loaded successfully
        """
        try:
            dem_file = Path(dem_path)
            if dem_file.exists():
                with open(dem_file, 'r') as f:
                    self.dem_data = json.load(f)
                logger.info(f"Loaded DEM data from {dem_path}")
                return True
            else:
                logger.warning(f"DEM file not found: {dem_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to load DEM data: {e}")
            return False
    
    def get_elevation(self, latitude: float, longitude: float) -> Optional[float]:
        """Get terrain elevation at given coordinates.
        
        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            
        Returns:
            Elevation in meters above sea level, or None if not available
        """
        # Check cache first
        cache_key = (round(latitude, 6), round(longitude, 6))
        if cache_key in self.elevation_cache:
            return self.elevation_cache[cache_key]
        
        if not self.dem_data:
            return None
        
        try:
            # This is a simplified implementation
            # In practice, you would interpolate from DEM grid data
            
            # For now, return a simulated elevation based on coordinates
            # This should be replaced with actual DEM lookup
            elevation = 100.0 + 50.0 * math.sin(latitude * 10) * math.cos(longitude * 10)
            
            # Cache the result
            self.elevation_cache[cache_key] = elevation
            return elevation
            
        except Exception as e:
            logger.error(f"Error getting elevation: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if DEM data is available."""
        return self.dem_data is not None

class GeolocationService:
    """Main geolocation service for computing geographic coordinates."""
    
    def __init__(self, camera_intrinsics: CameraIntrinsics, 
                 dem_service: Optional[DEMService] = None):
        self.camera_intrinsics = camera_intrinsics
        self.dem_service = dem_service or DEMService()
        self.coordinate_transforms = CoordinateTransforms()
        
    def compute_geolocation(self, detection_bbox: Tuple[float, float, float, float],
                          telemetry: TelemetryData,
                          use_dem: bool = True) -> Optional[GeolocationResult]:
        """Compute geographic coordinates for a detection bounding box.
        
        Args:
            detection_bbox: (x1, y1, x2, y2) bounding box in pixels
            telemetry: Drone telemetry data
            use_dem: Whether to use DEM for terrain intersection
            
        Returns:
            GeolocationResult or None if computation failed
        """
        try:
            # Get bounding box center
            x1, y1, x2, y2 = detection_bbox
            pixel_u = (x1 + x2) / 2.0
            pixel_v = (y1 + y2) / 2.0
            
            # Convert pixel to camera ray
            camera_ray = self.coordinate_transforms.pixel_to_camera_ray(
                pixel_u, pixel_v, self.camera_intrinsics
            )
            
            # Apply gimbal rotation
            gimbal_ray = self.coordinate_transforms.apply_gimbal_rotation(
                camera_ray, telemetry.gimbal_yaw, telemetry.gimbal_pitch, telemetry.gimbal_roll
            )
            
            # Get drone rotation matrix
            drone_rotation = self.coordinate_transforms.euler_to_rotation_matrix(
                telemetry.yaw, telemetry.pitch, telemetry.roll
            )
            
            # Transform to world coordinates
            world_ray = self.coordinate_transforms.camera_to_world_ray(
                gimbal_ray, drone_rotation
            )
            
            # Compute ground intersection
            if use_dem and self.dem_service.is_available():
                result = self._intersect_with_dem(world_ray, telemetry)
                method = "dem_intersection"
            else:
                result = self._intersect_with_flat_terrain(world_ray, telemetry)
                method = "flat_terrain"
            
            if result is None:
                return None
            
            lat, lon, elevation, h_accuracy, v_accuracy, confidence = result
            
            return GeolocationResult(
                latitude=lat,
                longitude=lon,
                elevation=elevation,
                horizontal_accuracy=h_accuracy,
                vertical_accuracy=v_accuracy,
                detection_bbox=detection_bbox,
                pixel_center=(pixel_u, pixel_v),
                telemetry_timestamp=telemetry.timestamp,
                method=method,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Geolocation computation failed: {e}")
            return None
    
    def _intersect_with_flat_terrain(self, world_ray: np.ndarray, 
                                   telemetry: TelemetryData) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Intersect ray with flat terrain at drone's altitude.
        
        Args:
            world_ray: 3D ray in world coordinates
            telemetry: Drone telemetry data
            
        Returns:
            (lat, lon, elevation, h_accuracy, v_accuracy, confidence) or None
        """
        try:
            # Drone position in world coordinates
            drone_lat = telemetry.latitude
            drone_lon = telemetry.longitude
            drone_alt_msl = telemetry.altitude_msl
            drone_alt_agl = telemetry.altitude_agl
            
            # Ground elevation (MSL)
            ground_elevation = drone_alt_msl - drone_alt_agl
            
            # Ray direction (normalized)
            ray_dir = world_ray
            
            # Check if ray points downward
            if ray_dir[2] >= 0:  # Z-axis points up
                logger.warning("Ray does not point toward ground")
                return None
            
            # Compute intersection distance
            # Ray equation: P = drone_pos + t * ray_dir
            # Ground plane: Z = ground_elevation
            t = (ground_elevation - drone_alt_msl) / ray_dir[2]
            
            if t <= 0:
                logger.warning("Invalid intersection distance")
                return None
            
            # Intersection point in local coordinates (meters from drone)
            intersection_x = t * ray_dir[0]  # East
            intersection_y = t * ray_dir[1]  # North
            
            # Convert to geographic coordinates
            lat_deg_per_m, lon_deg_per_m = self.coordinate_transforms.meters_to_degrees(
                1.0, drone_lat
            )
            
            target_lat = drone_lat + intersection_y * lat_deg_per_m
            target_lon = drone_lon + intersection_x * lon_deg_per_m
            
            # Estimate accuracy based on altitude and angle
            horizontal_distance = math.sqrt(intersection_x**2 + intersection_y**2)
            angle_from_nadir = math.degrees(math.acos(-ray_dir[2]))
            
            # Accuracy degrades with distance and angle
            h_accuracy = max(2.0, horizontal_distance * 0.02 + angle_from_nadir * 0.1)
            v_accuracy = max(5.0, drone_alt_agl * 0.05)
            
            # Confidence based on geometry
            confidence = max(0.1, 1.0 - angle_from_nadir / 90.0)
            
            return target_lat, target_lon, ground_elevation, h_accuracy, v_accuracy, confidence
            
        except Exception as e:
            logger.error(f"Flat terrain intersection failed: {e}")
            return None
    
    def _intersect_with_dem(self, world_ray: np.ndarray, 
                          telemetry: TelemetryData) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Intersect ray with DEM terrain data.
        
        Args:
            world_ray: 3D ray in world coordinates
            telemetry: Drone telemetry data
            
        Returns:
            (lat, lon, elevation, h_accuracy, v_accuracy, confidence) or None
        """
        try:
            # Start with flat terrain intersection as initial guess
            flat_result = self._intersect_with_flat_terrain(world_ray, telemetry)
            if flat_result is None:
                return None
            
            initial_lat, initial_lon, _, h_acc, v_acc, conf = flat_result
            
            # Iteratively refine using DEM data
            current_lat, current_lon = initial_lat, initial_lon
            
            for iteration in range(5):  # Maximum 5 iterations
                # Get terrain elevation at current position
                terrain_elevation = self.dem_service.get_elevation(current_lat, current_lon)
                if terrain_elevation is None:
                    # Fall back to flat terrain
                    return flat_result
                
                # Recompute intersection with this elevation
                drone_alt_msl = telemetry.altitude_msl
                ray_dir = world_ray
                
                # New intersection distance
                t = (terrain_elevation - drone_alt_msl) / ray_dir[2]
                
                if t <= 0:
                    break
                
                # New intersection point
                intersection_x = t * ray_dir[0]
                intersection_y = t * ray_dir[1]
                
                # Convert to geographic coordinates
                lat_deg_per_m, lon_deg_per_m = self.coordinate_transforms.meters_to_degrees(
                    1.0, telemetry.latitude
                )
                
                new_lat = telemetry.latitude + intersection_y * lat_deg_per_m
                new_lon = telemetry.longitude + intersection_x * lon_deg_per_m
                
                # Check convergence
                lat_diff = abs(new_lat - current_lat)
                lon_diff = abs(new_lon - current_lon)
                
                if lat_diff < 1e-6 and lon_diff < 1e-6:
                    # Converged
                    current_lat, current_lon = new_lat, new_lon
                    break
                
                current_lat, current_lon = new_lat, new_lon
            
            # Final terrain elevation
            final_elevation = self.dem_service.get_elevation(current_lat, current_lon)
            if final_elevation is None:
                final_elevation = terrain_elevation
            
            # Improved accuracy with DEM
            h_accuracy = h_acc * 0.7  # DEM improves horizontal accuracy
            v_accuracy = max(2.0, v_acc * 0.5)  # Significantly improves vertical accuracy
            confidence = min(1.0, conf * 1.2)  # Higher confidence with DEM
            
            return current_lat, current_lon, final_elevation, h_accuracy, v_accuracy, confidence
            
        except Exception as e:
            logger.error(f"DEM intersection failed: {e}")
            # Fall back to flat terrain
            return self._intersect_with_flat_terrain(world_ray, telemetry)
    
    def batch_compute_geolocations(self, detections: List[Tuple[float, float, float, float]],
                                 telemetry: TelemetryData,
                                 use_dem: bool = True) -> List[Optional[GeolocationResult]]:
        """Compute geolocations for multiple detections.
        
        Args:
            detections: List of bounding boxes
            telemetry: Drone telemetry data
            use_dem: Whether to use DEM for terrain intersection
            
        Returns:
            List of GeolocationResult objects (None for failed computations)
        """
        results = []
        for bbox in detections:
            result = self.compute_geolocation(bbox, telemetry, use_dem)
            results.append(result)
        return results
    
    def estimate_accuracy(self, telemetry: TelemetryData, 
                        pixel_coords: Tuple[float, float]) -> Tuple[float, float]:
        """Estimate geolocation accuracy for given conditions.
        
        Args:
            telemetry: Drone telemetry data
            pixel_coords: (u, v) pixel coordinates
            
        Returns:
            (horizontal_accuracy, vertical_accuracy) in meters
        """
        try:
            # Convert pixel to camera ray
            camera_ray = self.coordinate_transforms.pixel_to_camera_ray(
                pixel_coords[0], pixel_coords[1], self.camera_intrinsics
            )
            
            # Compute angle from nadir
            angle_from_nadir = math.degrees(math.acos(abs(camera_ray[2])))
            
            # Base accuracy factors
            altitude_factor = telemetry.altitude_agl / 100.0  # Normalized to 100m
            angle_factor = angle_from_nadir / 45.0  # Normalized to 45 degrees
            
            # Horizontal accuracy (increases with altitude and angle)
            h_accuracy = 2.0 + altitude_factor * 1.0 + angle_factor * 3.0
            
            # Vertical accuracy (mainly depends on altitude)
            v_accuracy = 5.0 + altitude_factor * 2.0
            
            # Apply DEM improvement if available
            if self.dem_service.is_available():
                h_accuracy *= 0.7
                v_accuracy *= 0.5
            
            return h_accuracy, v_accuracy
            
        except Exception as e:
            logger.error(f"Accuracy estimation failed: {e}")
            return 10.0, 20.0  # Conservative defaults

def create_geolocation_service(camera_config_path: Optional[str] = None,
                             dem_data_path: Optional[str] = None) -> GeolocationService:
    """Create a geolocation service with configuration.
    
    Args:
        camera_config_path: Path to camera intrinsics configuration
        dem_data_path: Path to DEM data
        
    Returns:
        Configured GeolocationService
    """
    # Load camera intrinsics
    if camera_config_path:
        intrinsics = CameraIntrinsics.from_config(camera_config_path)
    else:
        intrinsics = CameraIntrinsics.dji_o4_default()
        logger.info("Using default DJI O4 camera intrinsics")
    
    # Load DEM service
    dem_service = DEMService(dem_data_path) if dem_data_path else DEMService()
    
    return GeolocationService(intrinsics, dem_service)

if __name__ == "__main__":
    # Demo usage
    import argparse
    from .telemetry_service import SimulatedTelemetryCollector
    
    parser = argparse.ArgumentParser(description="Geolocation Service Demo")
    parser.add_argument("--camera-config", help="Camera intrinsics configuration file")
    parser.add_argument("--dem-data", help="DEM data file")
    parser.add_argument("--bbox", nargs=4, type=float, default=[100, 100, 200, 200],
                       help="Detection bounding box (x1 y1 x2 y2)")
    args = parser.parse_args()
    
    # Create geolocation service
    service = create_geolocation_service(args.camera_config, args.dem_data)
    
    # Create simulated telemetry
    telemetry_collector = SimulatedTelemetryCollector()
    telemetry_collector.start()
    
    import time
    time.sleep(1.0)  # Wait for telemetry
    
    telemetry = telemetry_collector.get_latest_data()
    if telemetry:
        print(f"Using telemetry: Lat={telemetry.latitude:.6f}, Lon={telemetry.longitude:.6f}, "
              f"Alt={telemetry.altitude_agl:.1f}m")
        
        # Compute geolocation
        bbox = tuple(args.bbox)
        result = service.compute_geolocation(bbox, telemetry, use_dem=True)
        
        if result:
            print(f"\nGeolocation Result:")
            print(f"  Coordinates: {result.latitude:.6f}, {result.longitude:.6f}")
            print(f"  Elevation: {result.elevation:.1f}m")
            print(f"  Accuracy: ±{result.horizontal_accuracy:.1f}m horizontal, ±{result.vertical_accuracy:.1f}m vertical")
            print(f"  Method: {result.method}")
            print(f"  Confidence: {result.confidence:.2f}")
        else:
            print("Geolocation computation failed")
    else:
        print("No telemetry data available")
    
    telemetry_collector.stop()