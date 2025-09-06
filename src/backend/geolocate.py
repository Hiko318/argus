#!/usr/bin/env python3
"""
Geolocation Pipeline for Foresight SAR System

Implements pixel-to-geographic coordinate transformation using:
- Camera intrinsics and distortion correction
- Pinhole camera model
- ENU (East-North-Up) coordinate system
- DEM (Digital Elevation Model) intersection
- Iterative ray-terrain intersection

Author: Foresight SAR Team
Date: 2024-01-15
"""

import json
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from pathlib import Path
import logging
from scipy.optimize import minimize_scalar
from scipy.interpolate import RegularGridInterpolator
import requests
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Point2D:
    """2D pixel coordinates"""
    u: float  # Horizontal pixel coordinate
    v: float  # Vertical pixel coordinate

@dataclass
class Point3D:
    """3D coordinates in various reference frames"""
    x: float
    y: float
    z: float

@dataclass
class GeographicCoordinate:
    """Geographic coordinates (WGS84)"""
    latitude: float   # Degrees
    longitude: float  # Degrees
    altitude: float   # Meters above sea level
    
    def to_dict(self) -> Dict:
        return {
            'lat': self.latitude,
            'lon': self.longitude,
            'alt': self.altitude
        }

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters"""
    fx: float  # Focal length in x (pixels)
    fy: float  # Focal length in y (pixels)
    cx: float  # Principal point x (pixels)
    cy: float  # Principal point y (pixels)
    k1: float  # Radial distortion coefficient 1
    k2: float  # Radial distortion coefficient 2
    k3: float  # Radial distortion coefficient 3
    p1: float  # Tangential distortion coefficient 1
    p2: float  # Tangential distortion coefficient 2
    image_width: int
    image_height: int

@dataclass
class TelemetryData:
    """Aircraft telemetry data"""
    latitude: float      # Degrees
    longitude: float     # Degrees
    altitude: float      # Meters above sea level
    roll: float         # Degrees (bank angle)
    pitch: float        # Degrees (elevation angle)
    yaw: float          # Degrees (heading, 0=North, clockwise)
    gimbal_pitch: float # Degrees (camera pitch relative to aircraft)
    gimbal_yaw: float   # Degrees (camera yaw relative to aircraft)
    gimbal_roll: float  # Degrees (camera roll relative to aircraft)
    timestamp: float    # Unix timestamp

class CoordinateTransforms:
    """Coordinate system transformation utilities"""
    
    @staticmethod
    def wgs84_to_enu(lat: float, lon: float, alt: float, 
                     ref_lat: float, ref_lon: float, ref_alt: float) -> Point3D:
        """Convert WGS84 coordinates to local ENU frame"""
        # WGS84 ellipsoid parameters
        a = 6378137.0  # Semi-major axis
        f = 1 / 298.257223563  # Flattening
        e2 = 2 * f - f * f  # First eccentricity squared
        
        # Convert to radians
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        ref_lat_rad = math.radians(ref_lat)
        ref_lon_rad = math.radians(ref_lon)
        
        # Calculate ECEF coordinates
        def wgs84_to_ecef(lat_r, lon_r, alt_m):
            N = a / math.sqrt(1 - e2 * math.sin(lat_r) ** 2)
            x = (N + alt_m) * math.cos(lat_r) * math.cos(lon_r)
            y = (N + alt_m) * math.cos(lat_r) * math.sin(lon_r)
            z = (N * (1 - e2) + alt_m) * math.sin(lat_r)
            return x, y, z
        
        # Target and reference ECEF coordinates
        x, y, z = wgs84_to_ecef(lat_rad, lon_rad, alt)
        x_ref, y_ref, z_ref = wgs84_to_ecef(ref_lat_rad, ref_lon_rad, ref_alt)
        
        # ECEF to ENU transformation
        dx = x - x_ref
        dy = y - y_ref
        dz = z - z_ref
        
        sin_lat = math.sin(ref_lat_rad)
        cos_lat = math.cos(ref_lat_rad)
        sin_lon = math.sin(ref_lon_rad)
        cos_lon = math.cos(ref_lon_rad)
        
        east = -sin_lon * dx + cos_lon * dy
        north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
        up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
        
        return Point3D(east, north, up)
    
    @staticmethod
    def enu_to_wgs84(east: float, north: float, up: float,
                     ref_lat: float, ref_lon: float, ref_alt: float) -> GeographicCoordinate:
        """Convert ENU coordinates to WGS84"""
        # WGS84 ellipsoid parameters
        a = 6378137.0
        f = 1 / 298.257223563
        e2 = 2 * f - f * f
        
        ref_lat_rad = math.radians(ref_lat)
        ref_lon_rad = math.radians(ref_lon)
        
        # Reference ECEF coordinates
        N_ref = a / math.sqrt(1 - e2 * math.sin(ref_lat_rad) ** 2)
        x_ref = (N_ref + ref_alt) * math.cos(ref_lat_rad) * math.cos(ref_lon_rad)
        y_ref = (N_ref + ref_alt) * math.cos(ref_lat_rad) * math.sin(ref_lon_rad)
        z_ref = (N_ref * (1 - e2) + ref_alt) * math.sin(ref_lat_rad)
        
        # ENU to ECEF transformation
        sin_lat = math.sin(ref_lat_rad)
        cos_lat = math.cos(ref_lat_rad)
        sin_lon = math.sin(ref_lon_rad)
        cos_lon = math.cos(ref_lon_rad)
        
        dx = -sin_lon * east - sin_lat * cos_lon * north + cos_lat * cos_lon * up
        dy = cos_lon * east - sin_lat * sin_lon * north + cos_lat * sin_lon * up
        dz = cos_lat * north + sin_lat * up
        
        x = x_ref + dx
        y = y_ref + dy
        z = z_ref + dz
        
        # ECEF to WGS84
        p = math.sqrt(x * x + y * y)
        theta = math.atan2(z * a, p * (1 - e2) * a)
        
        lat = math.atan2(z + e2 * (1 - e2) * a * math.sin(theta) ** 3,
                        p - e2 * a * math.cos(theta) ** 3)
        lon = math.atan2(y, x)
        
        N = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
        alt = p / math.cos(lat) - N
        
        return GeographicCoordinate(
            latitude=math.degrees(lat),
            longitude=math.degrees(lon),
            altitude=alt
        )
    
    @staticmethod
    def rotation_matrix_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Create rotation matrix from Euler angles (degrees)"""
        r = math.radians(roll)
        p = math.radians(pitch)
        y = math.radians(yaw)
        
        # Roll (rotation around x-axis)
        R_x = np.array([
            [1, 0, 0],
            [0, math.cos(r), -math.sin(r)],
            [0, math.sin(r), math.cos(r)]
        ])
        
        # Pitch (rotation around y-axis)
        R_y = np.array([
            [math.cos(p), 0, math.sin(p)],
            [0, 1, 0],
            [-math.sin(p), 0, math.cos(p)]
        ])
        
        # Yaw (rotation around z-axis)
        R_z = np.array([
            [math.cos(y), -math.sin(y), 0],
            [math.sin(y), math.cos(y), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation (ZYX order)
        return R_z @ R_y @ R_x

class DEMProvider:
    """Digital Elevation Model provider interface"""
    
    def __init__(self, cache_dir: str = "./dem_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def get_elevation(self, latitude: float, longitude: float) -> float:
        """Get elevation at given coordinates (meters above sea level)"""
        # Try cache first
        cache_key = f"{latitude:.6f}_{longitude:.6f}"
        cache_file = self.cache_dir / f"{cache_key}.txt"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return float(f.read().strip())
            except (ValueError, IOError):
                pass
        
        # Fetch from online service (USGS Elevation Point Query Service)
        try:
            elevation = self._fetch_elevation_online(latitude, longitude)
            
            # Cache the result
            with open(cache_file, 'w') as f:
                f.write(str(elevation))
                
            return elevation
        except Exception as e:
            logger.warning(f"Failed to fetch elevation for {latitude}, {longitude}: {e}")
            return 0.0  # Default to sea level
    
    def _fetch_elevation_online(self, latitude: float, longitude: float) -> float:
        """Fetch elevation from USGS Elevation Point Query Service"""
        url = "https://nationalmap.gov/epqs/pqs.php"
        params = {
            'x': longitude,
            'y': latitude,
            'units': 'Meters',
            'output': 'json'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        elevation_query = data.get('USGS_Elevation_Point_Query_Service', {})
        elevation_query_result = elevation_query.get('Elevation_Query', {})
        elevation = elevation_query_result.get('Elevation')
        
        if elevation is None or elevation == -1000000:
            raise ValueError("No elevation data available")
            
        return float(elevation)
    
    def get_elevation_grid(self, lat_min: float, lat_max: float, 
                          lon_min: float, lon_max: float, 
                          resolution: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get elevation grid for a region"""
        lats = np.linspace(lat_min, lat_max, resolution)
        lons = np.linspace(lon_min, lon_max, resolution)
        elevations = np.zeros((resolution, resolution))
        
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                elevations[i, j] = self.get_elevation(lat, lon)
        
        return lats, lons, elevations

class GeolocationPipeline:
    """Main geolocation pipeline class"""
    
    def __init__(self, config_path: str = "config/camera_intrinsics.json"):
        self.config_path = Path(config_path)
        self.camera_profiles = {}
        self.dem_provider = DEMProvider()
        self.load_camera_config()
        
    def load_camera_config(self):
        """Load camera intrinsics configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                
            for profile_name, profile_data in config['camera_profiles'].items():
                camera_matrix = profile_data['camera_matrix']
                distortion = profile_data['distortion']
                
                self.camera_profiles[profile_name] = CameraIntrinsics(
                    fx=camera_matrix['fx'],
                    fy=camera_matrix['fy'],
                    cx=camera_matrix['cx'],
                    cy=camera_matrix['cy'],
                    k1=distortion['k1'],
                    k2=distortion['k2'],
                    k3=distortion['k3'],
                    p1=distortion['p1'],
                    p2=distortion['p2'],
                    image_width=profile_data['image_width_px'],
                    image_height=profile_data['image_height_px']
                )
                
            logger.info(f"Loaded {len(self.camera_profiles)} camera profiles")
            
        except Exception as e:
            logger.error(f"Failed to load camera config: {e}")
            raise
    
    def undistort_pixel(self, pixel: Point2D, intrinsics: CameraIntrinsics) -> Point2D:
        """Remove lens distortion from pixel coordinates"""
        # Normalize coordinates
        x = (pixel.u - intrinsics.cx) / intrinsics.fx
        y = (pixel.v - intrinsics.cy) / intrinsics.fy
        
        # Radial distance
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        
        # Radial distortion correction
        radial_correction = 1 + intrinsics.k1 * r2 + intrinsics.k2 * r4 + intrinsics.k3 * r6
        
        # Tangential distortion correction
        tangential_x = 2 * intrinsics.p1 * x * y + intrinsics.p2 * (r2 + 2 * x * x)
        tangential_y = intrinsics.p1 * (r2 + 2 * y * y) + 2 * intrinsics.p2 * x * y
        
        # Apply corrections
        x_corrected = x * radial_correction + tangential_x
        y_corrected = y * radial_correction + tangential_y
        
        # Convert back to pixel coordinates
        u_corrected = x_corrected * intrinsics.fx + intrinsics.cx
        v_corrected = y_corrected * intrinsics.fy + intrinsics.cy
        
        return Point2D(u_corrected, v_corrected)
    
    def pixel_to_ray(self, pixel: Point2D, intrinsics: CameraIntrinsics) -> Point3D:
        """Convert pixel coordinates to 3D ray in camera frame"""
        # Undistort pixel
        undistorted = self.undistort_pixel(pixel, intrinsics)
        
        # Convert to normalized camera coordinates
        x = (undistorted.u - intrinsics.cx) / intrinsics.fx
        y = (undistorted.v - intrinsics.cy) / intrinsics.fy
        z = 1.0  # Forward direction in camera frame
        
        # Normalize ray direction
        length = math.sqrt(x * x + y * y + z * z)
        
        return Point3D(x / length, y / length, z / length)
    
    def transform_ray_to_world(self, ray: Point3D, telemetry: TelemetryData) -> Point3D:
        """Transform ray from camera frame to world ENU frame"""
        # Camera to body frame transformation (assuming camera aligned with body)
        # Apply gimbal rotations
        gimbal_rotation = CoordinateTransforms.rotation_matrix_from_euler(
            telemetry.gimbal_roll, 
            telemetry.gimbal_pitch, 
            telemetry.gimbal_yaw
        )
        
        # Body to world frame transformation
        body_rotation = CoordinateTransforms.rotation_matrix_from_euler(
            telemetry.roll, 
            telemetry.pitch, 
            telemetry.yaw
        )
        
        # Combined transformation
        combined_rotation = body_rotation @ gimbal_rotation
        
        # Apply rotation to ray
        ray_vector = np.array([ray.x, ray.y, ray.z])
        world_ray = combined_rotation @ ray_vector
        
        return Point3D(world_ray[0], world_ray[1], world_ray[2])
    
    def intersect_ray_with_dem(self, ray_origin: GeographicCoordinate, 
                              ray_direction: Point3D, 
                              max_distance: float = 10000.0,
                              step_size: float = 10.0) -> Optional[GeographicCoordinate]:
        """Find intersection of ray with DEM using iterative approach"""
        
        def elevation_difference(distance: float) -> float:
            """Calculate difference between ray height and terrain height at given distance"""
            # Calculate point along ray
            enu_point = Point3D(
                ray_direction.x * distance,
                ray_direction.y * distance,
                ray_direction.z * distance
            )
            
            # Convert to geographic coordinates
            geo_point = CoordinateTransforms.enu_to_wgs84(
                enu_point.x, enu_point.y, enu_point.z,
                ray_origin.latitude, ray_origin.longitude, ray_origin.altitude
            )
            
            # Get terrain elevation
            terrain_elevation = self.dem_provider.get_elevation(
                geo_point.latitude, geo_point.longitude
            )
            
            # Return difference (positive when ray is above terrain)
            return geo_point.altitude - terrain_elevation
        
        try:
            # Find intersection using optimization
            result = minimize_scalar(
                lambda d: abs(elevation_difference(d)),
                bounds=(0, max_distance),
                method='bounded'
            )
            
            if result.success and abs(result.fun) < 5.0:  # Within 5 meters tolerance
                distance = result.x
                
                # Calculate final intersection point
                enu_point = Point3D(
                    ray_direction.x * distance,
                    ray_direction.y * distance,
                    ray_direction.z * distance
                )
                
                intersection = CoordinateTransforms.enu_to_wgs84(
                    enu_point.x, enu_point.y, enu_point.z,
                    ray_origin.latitude, ray_origin.longitude, ray_origin.altitude
                )
                
                # Use terrain elevation for final altitude
                terrain_elevation = self.dem_provider.get_elevation(
                    intersection.latitude, intersection.longitude
                )
                
                return GeographicCoordinate(
                    latitude=intersection.latitude,
                    longitude=intersection.longitude,
                    altitude=terrain_elevation
                )
                
        except Exception as e:
            logger.warning(f"Ray-DEM intersection failed: {e}")
            
        return None
    
    def project_pixel_to_geo(self, pixel: Point2D, telemetry: TelemetryData, 
                           camera_profile: str = "dji_o4_main") -> Optional[GeographicCoordinate]:
        """Main function: project pixel coordinates to geographic coordinates"""
        try:
            # Get camera intrinsics
            if camera_profile not in self.camera_profiles:
                raise ValueError(f"Unknown camera profile: {camera_profile}")
            
            intrinsics = self.camera_profiles[camera_profile]
            
            # Validate pixel coordinates
            if (pixel.u < 0 or pixel.u >= intrinsics.image_width or 
                pixel.v < 0 or pixel.v >= intrinsics.image_height):
                logger.warning(f"Pixel coordinates out of bounds: {pixel}")
                return None
            
            # Convert pixel to ray in camera frame
            camera_ray = self.pixel_to_ray(pixel, intrinsics)
            
            # Transform ray to world frame
            world_ray = self.transform_ray_to_world(camera_ray, telemetry)
            
            # Ray origin (aircraft position)
            ray_origin = GeographicCoordinate(
                latitude=telemetry.latitude,
                longitude=telemetry.longitude,
                altitude=telemetry.altitude
            )
            
            # Find intersection with terrain
            intersection = self.intersect_ray_with_dem(ray_origin, world_ray)
            
            if intersection:
                logger.debug(f"Projected pixel {pixel} to {intersection}")
                return intersection
            else:
                logger.warning(f"Failed to find terrain intersection for pixel {pixel}")
                return None
                
        except Exception as e:
            logger.error(f"Pixel projection failed: {e}")
            return None
    
    def batch_project_pixels(self, pixels: List[Point2D], telemetry: TelemetryData,
                           camera_profile: str = "dji_o4_main") -> List[Optional[GeographicCoordinate]]:
        """Project multiple pixels to geographic coordinates"""
        results = []
        for pixel in pixels:
            result = self.project_pixel_to_geo(pixel, telemetry, camera_profile)
            results.append(result)
        return results
    
    def get_camera_footprint(self, telemetry: TelemetryData, 
                           camera_profile: str = "dji_o4_main",
                           grid_size: int = 5) -> List[GeographicCoordinate]:
        """Calculate camera footprint on ground"""
        if camera_profile not in self.camera_profiles:
            raise ValueError(f"Unknown camera profile: {camera_profile}")
        
        intrinsics = self.camera_profiles[camera_profile]
        
        # Create grid of pixels across image
        pixels = []
        for i in range(grid_size):
            for j in range(grid_size):
                u = (i / (grid_size - 1)) * (intrinsics.image_width - 1)
                v = (j / (grid_size - 1)) * (intrinsics.image_height - 1)
                pixels.append(Point2D(u, v))
        
        # Project all pixels
        footprint = []
        for pixel in pixels:
            geo_point = self.project_pixel_to_geo(pixel, telemetry, camera_profile)
            if geo_point:
                footprint.append(geo_point)
        
        return footprint

# Example usage and testing functions
def create_synthetic_telemetry() -> TelemetryData:
    """Create synthetic telemetry data for testing"""
    return TelemetryData(
        latitude=37.7749,    # San Francisco
        longitude=-122.4194,
        altitude=100.0,      # 100m above sea level
        roll=0.0,
        pitch=-30.0,         # Looking down 30 degrees
        yaw=45.0,           # Facing northeast
        gimbal_pitch=-15.0,  # Additional 15 degrees down
        gimbal_yaw=0.0,
        gimbal_roll=0.0,
        timestamp=1642262400.0
    )

def run_geolocation_test():
    """Run basic geolocation test"""
    # Initialize pipeline
    pipeline = GeolocationPipeline()
    
    # Create test data
    telemetry = create_synthetic_telemetry()
    test_pixel = Point2D(2000, 1500)  # Center of image
    
    # Project pixel to geographic coordinates
    result = pipeline.project_pixel_to_geo(test_pixel, telemetry)
    
    if result:
        print(f"Pixel {test_pixel.u}, {test_pixel.v} projects to:")
        print(f"  Latitude: {result.latitude:.6f}°")
        print(f"  Longitude: {result.longitude:.6f}°")
        print(f"  Altitude: {result.altitude:.1f}m")
    else:
        print("Projection failed")
    
    # Test camera footprint
    footprint = pipeline.get_camera_footprint(telemetry)
    print(f"\nCamera footprint contains {len(footprint)} points")
    
    return result

if __name__ == "__main__":
    # Run test
    run_geolocation_test()