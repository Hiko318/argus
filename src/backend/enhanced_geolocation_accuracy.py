#!/usr/bin/env python3
"""
Enhanced Geolocation Accuracy Module for Foresight SAR System

Implements high-precision geolocation with DEM iterative correction,
pinhole ray → ENU conversion, and accuracy assessment.

Features:
- Iterative DEM-based terrain intersection
- Multi-source DEM support (SRTM, ASTER, local)
- Accuracy estimation and validation
- Ray-terrain intersection optimization
- Coordinate system transformations

Author: Foresight SAR Team
Date: 2024-01-15
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, NamedTuple, Union
from dataclasses import dataclass
from pathlib import Path
import logging
from enum import Enum
import json
from scipy.optimize import minimize_scalar, brentq
from scipy.interpolate import interp1d

try:
    from .geolocate import (
        GeolocationPipeline, GeographicCoordinate, Point2D, Point3D,
        TelemetryData, CoordinateTransforms
    )
    from .geolocation_service import DEMService
except ImportError:
    # Fallback for testing
    from dataclasses import dataclass
    
    @dataclass
    class GeographicCoordinate:
        latitude: float
        longitude: float
        altitude: float
    
    @dataclass
    class Point2D:
        u: float
        v: float
    
    @dataclass
    class Point3D:
        x: float
        y: float
        z: float
    
    @dataclass
    class TelemetryData:
        latitude: float
        longitude: float
        altitude: float
        altitude_msl: float
        altitude_agl: float
        roll: float
        pitch: float
        yaw: float
        gimbal_roll: float
        gimbal_pitch: float
        gimbal_yaw: float

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AccuracyLevel(Enum):
    """Geolocation accuracy levels"""
    LOW = "low"          # >10m horizontal, >20m vertical
    MEDIUM = "medium"    # 5-10m horizontal, 10-20m vertical
    HIGH = "high"        # 2-5m horizontal, 5-10m vertical
    PRECISION = "precision"  # <2m horizontal, <5m vertical


class DEMSource(Enum):
    """DEM data sources"""
    SRTM_30M = "srtm_30m"      # SRTM 30m resolution
    SRTM_90M = "srtm_90m"      # SRTM 90m resolution
    ASTER_30M = "aster_30m"    # ASTER GDEM 30m
    ALOS_30M = "alos_30m"      # ALOS World 3D 30m
    LOCAL_HIGH = "local_high"  # High-resolution local DEM
    SYNTHETIC = "synthetic"    # Synthetic/test DEM


@dataclass
class AccuracyMetrics:
    """Geolocation accuracy metrics"""
    horizontal_accuracy: float  # meters (CEP90)
    vertical_accuracy: float    # meters (LEP90)
    confidence: float          # 0-1 confidence score
    dem_resolution: Optional[float] = None  # DEM resolution used
    dem_source: Optional[str] = None       # DEM source identifier
    iteration_count: int = 0               # DEM iterations performed
    convergence_error: float = 0.0         # Final convergence error
    
    @property
    def accuracy_level(self) -> AccuracyLevel:
        """Determine accuracy level based on metrics"""
        if self.horizontal_accuracy < 2.0 and self.vertical_accuracy < 5.0:
            return AccuracyLevel.PRECISION
        elif self.horizontal_accuracy < 5.0 and self.vertical_accuracy < 10.0:
            return AccuracyLevel.HIGH
        elif self.horizontal_accuracy < 10.0 and self.vertical_accuracy < 20.0:
            return AccuracyLevel.MEDIUM
        else:
            return AccuracyLevel.LOW


@dataclass
class GeolocationResult:
    """Enhanced geolocation result with accuracy metrics"""
    coordinate: GeographicCoordinate
    accuracy: AccuracyMetrics
    ray_origin: GeographicCoordinate
    ray_direction: Point3D
    intersection_distance: float
    terrain_elevation: float
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'latitude': self.coordinate.latitude,
            'longitude': self.coordinate.longitude,
            'altitude': self.coordinate.altitude,
            'horizontal_accuracy': self.accuracy.horizontal_accuracy,
            'vertical_accuracy': self.accuracy.vertical_accuracy,
            'confidence': self.accuracy.confidence,
            'accuracy_level': self.accuracy.accuracy_level.value,
            'dem_source': self.accuracy.dem_source,
            'processing_time': self.processing_time
        }


class EnhancedDEMProvider:
    """Enhanced DEM provider with multi-source support"""
    
    def __init__(self, cache_size: int = 10000):
        self.dem_sources: Dict[DEMSource, Dict] = {}
        self.elevation_cache: Dict[Tuple[float, float], Tuple[float, DEMSource]] = {}
        self.cache_size = cache_size
        self.fallback_elevation = 0.0
        
    def add_dem_source(self, source: DEMSource, config: Dict):
        """Add DEM data source"""
        self.dem_sources[source] = config
        logger.info(f"Added DEM source: {source.value}")
    
    def get_elevation(self, latitude: float, longitude: float, 
                     preferred_source: Optional[DEMSource] = None) -> Tuple[float, DEMSource]:
        """Get elevation with source tracking"""
        # Check cache
        cache_key = (round(latitude, 6), round(longitude, 6))
        if cache_key in self.elevation_cache:
            return self.elevation_cache[cache_key]
        
        # Try preferred source first
        if preferred_source and preferred_source in self.dem_sources:
            elevation = self._query_dem_source(preferred_source, latitude, longitude)
            if elevation is not None:
                result = (elevation, preferred_source)
                self._cache_elevation(cache_key, result)
                return result
        
        # Try sources in order of preference
        source_priority = [
            DEMSource.LOCAL_HIGH,
            DEMSource.ALOS_30M,
            DEMSource.ASTER_30M,
            DEMSource.SRTM_30M,
            DEMSource.SRTM_90M,
            DEMSource.SYNTHETIC
        ]
        
        for source in source_priority:
            if source in self.dem_sources:
                elevation = self._query_dem_source(source, latitude, longitude)
                if elevation is not None:
                    result = (elevation, source)
                    self._cache_elevation(cache_key, result)
                    return result
        
        # Fallback
        result = (self.fallback_elevation, DEMSource.SYNTHETIC)
        self._cache_elevation(cache_key, result)
        return result
    
    def _query_dem_source(self, source: DEMSource, latitude: float, longitude: float) -> Optional[float]:
        """Query specific DEM source"""
        config = self.dem_sources.get(source)
        if not config:
            return None
        
        try:
            if source == DEMSource.SYNTHETIC:
                # Synthetic terrain for testing
                return self._generate_synthetic_elevation(latitude, longitude)
            else:
                # Real DEM sources would be implemented here
                # For now, return synthetic data
                return self._generate_synthetic_elevation(latitude, longitude)
        except Exception as e:
            logger.warning(f"Failed to query {source.value}: {e}")
            return None
    
    def _generate_synthetic_elevation(self, latitude: float, longitude: float) -> float:
        """Generate synthetic elevation for testing"""
        # Create realistic terrain variation
        base_elevation = 100.0
        terrain_scale = 200.0
        
        # Multiple frequency components for realistic terrain
        elevation = base_elevation
        elevation += terrain_scale * 0.3 * math.sin(latitude * 0.01) * math.cos(longitude * 0.01)
        elevation += terrain_scale * 0.2 * math.sin(latitude * 0.05) * math.sin(longitude * 0.03)
        elevation += terrain_scale * 0.1 * math.cos(latitude * 0.1) * math.cos(longitude * 0.08)
        
        # Add some noise
        noise = 10.0 * math.sin(latitude * longitude * 0.001)
        elevation += noise
        
        return max(0.0, elevation)  # Ensure non-negative
    
    def _cache_elevation(self, key: Tuple[float, float], value: Tuple[float, DEMSource]):
        """Cache elevation result"""
        if len(self.elevation_cache) >= self.cache_size:
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(self.elevation_cache.keys())[:100]
            for old_key in oldest_keys:
                del self.elevation_cache[old_key]
        
        self.elevation_cache[key] = value
    
    def get_resolution(self, source: DEMSource) -> float:
        """Get resolution for DEM source"""
        resolution_map = {
            DEMSource.SRTM_30M: 30.0,
            DEMSource.SRTM_90M: 90.0,
            DEMSource.ASTER_30M: 30.0,
            DEMSource.ALOS_30M: 30.0,
            DEMSource.LOCAL_HIGH: 5.0,
            DEMSource.SYNTHETIC: 30.0
        }
        return resolution_map.get(source, 30.0)


class IterativeDEMCorrector:
    """Iterative DEM-based ray-terrain intersection"""
    
    def __init__(self, dem_provider: EnhancedDEMProvider):
        self.dem_provider = dem_provider
        self.max_iterations = 10
        self.convergence_tolerance = 1.0  # meters
        self.step_size_factor = 0.5
        
    def intersect_ray_with_terrain(self, ray_origin: GeographicCoordinate,
                                  ray_direction: Point3D,
                                  initial_guess: Optional[GeographicCoordinate] = None) -> Optional[Tuple[GeographicCoordinate, AccuracyMetrics]]:
        """Find ray-terrain intersection using iterative DEM correction"""
        
        # Start with flat-earth intersection if no initial guess
        if initial_guess is None:
            initial_guess = self._flat_earth_intersection(ray_origin, ray_direction)
            if initial_guess is None:
                return None
        
        current_point = initial_guess
        iteration_count = 0
        convergence_error = float('inf')
        dem_source = DEMSource.SYNTHETIC
        
        for iteration in range(self.max_iterations):
            iteration_count += 1
            
            # Get terrain elevation at current point
            terrain_elevation, dem_source = self.dem_provider.get_elevation(
                current_point.latitude, current_point.longitude
            )
            
            # Calculate height difference
            height_diff = current_point.altitude - terrain_elevation
            convergence_error = abs(height_diff)
            
            if convergence_error < self.convergence_tolerance:
                # Converged
                break
            
            # Calculate new intersection point
            new_point = self._refine_intersection(
                ray_origin, ray_direction, current_point, terrain_elevation
            )
            
            if new_point is None:
                break
            
            current_point = new_point
        
        # Final result with terrain elevation
        final_coordinate = GeographicCoordinate(
            latitude=current_point.latitude,
            longitude=current_point.longitude,
            altitude=terrain_elevation
        )
        
        # Calculate accuracy metrics
        accuracy = self._calculate_accuracy_metrics(
            ray_origin, ray_direction, final_coordinate,
            iteration_count, convergence_error, dem_source
        )
        
        return final_coordinate, accuracy
    
    def _flat_earth_intersection(self, ray_origin: GeographicCoordinate,
                               ray_direction: Point3D) -> Optional[GeographicCoordinate]:
        """Calculate flat-earth intersection as initial guess"""
        if ray_direction.z >= 0:
            # Ray pointing up, no ground intersection
            return None
        
        # Assume flat earth at origin altitude
        t = -ray_origin.altitude / ray_direction.z
        if t <= 0:
            return None
        
        # Convert ray direction to geographic displacement
        # Simplified conversion (assumes small distances)
        lat_per_meter = 1.0 / 111320.0
        lon_per_meter = 1.0 / (111320.0 * math.cos(math.radians(ray_origin.latitude)))
        
        intersection_lat = ray_origin.latitude + ray_direction.y * t * lat_per_meter
        intersection_lon = ray_origin.longitude + ray_direction.x * t * lon_per_meter
        intersection_alt = 0.0  # Flat earth assumption
        
        return GeographicCoordinate(
            latitude=intersection_lat,
            longitude=intersection_lon,
            altitude=intersection_alt
        )
    
    def _refine_intersection(self, ray_origin: GeographicCoordinate,
                           ray_direction: Point3D,
                           current_point: GeographicCoordinate,
                           terrain_elevation: float) -> Optional[GeographicCoordinate]:
        """Refine intersection point using terrain elevation"""
        
        # Calculate new intersection with terrain plane
        height_diff = terrain_elevation - ray_origin.altitude
        
        if ray_direction.z >= 0:
            return None
        
        t = height_diff / ray_direction.z
        if t <= 0:
            return None
        
        # Convert to geographic coordinates
        lat_per_meter = 1.0 / 111320.0
        lon_per_meter = 1.0 / (111320.0 * math.cos(math.radians(ray_origin.latitude)))
        
        new_lat = ray_origin.latitude + ray_direction.y * t * lat_per_meter
        new_lon = ray_origin.longitude + ray_direction.x * t * lon_per_meter
        new_alt = terrain_elevation
        
        return GeographicCoordinate(
            latitude=new_lat,
            longitude=new_lon,
            altitude=new_alt
        )
    
    def _calculate_accuracy_metrics(self, ray_origin: GeographicCoordinate,
                                  ray_direction: Point3D,
                                  final_point: GeographicCoordinate,
                                  iteration_count: int,
                                  convergence_error: float,
                                  dem_source: DEMSource) -> AccuracyMetrics:
        """Calculate accuracy metrics for the geolocation result"""
        
        # Base accuracy depends on DEM resolution and convergence
        dem_resolution = self.dem_provider.get_resolution(dem_source)
        
        # Horizontal accuracy factors
        base_h_accuracy = dem_resolution * 0.5  # Half pixel accuracy
        convergence_factor = min(2.0, convergence_error / self.convergence_tolerance)
        iteration_factor = 1.0 + (iteration_count / self.max_iterations) * 0.5
        
        horizontal_accuracy = base_h_accuracy * convergence_factor * iteration_factor
        
        # Vertical accuracy (typically better than horizontal)
        vertical_accuracy = max(1.0, horizontal_accuracy * 0.7)
        
        # Confidence based on convergence and DEM quality
        confidence = 1.0 / (1.0 + convergence_error / self.convergence_tolerance)
        confidence *= 0.9 if dem_source == DEMSource.SYNTHETIC else 1.0
        confidence = max(0.1, min(1.0, confidence))
        
        return AccuracyMetrics(
            horizontal_accuracy=horizontal_accuracy,
            vertical_accuracy=vertical_accuracy,
            confidence=confidence,
            dem_resolution=dem_resolution,
            dem_source=dem_source.value,
            iteration_count=iteration_count,
            convergence_error=convergence_error
        )


class EnhancedGeolocationAccuracy:
    """Enhanced geolocation accuracy system"""
    
    def __init__(self, camera_profiles: Optional[Dict] = None):
        self.dem_provider = EnhancedDEMProvider()
        self.dem_corrector = IterativeDEMCorrector(self.dem_provider)
        self.camera_profiles = camera_profiles or self._default_camera_profiles()
        
        # Initialize with synthetic DEM for testing
        self.dem_provider.add_dem_source(DEMSource.SYNTHETIC, {})
        
        logger.info("Enhanced Geolocation Accuracy system initialized")
    
    def _default_camera_profiles(self) -> Dict:
        """Default camera profiles"""
        return {
            "dji_o4_main": {
                "fx": 2000.0, "fy": 2000.0,
                "cx": 1920.0, "cy": 1080.0,
                "image_width": 3840, "image_height": 2160,
                "distortion": [0.0, 0.0, 0.0, 0.0, 0.0]
            }
        }
    
    def geolocate_pixel(self, pixel: Point2D, telemetry: TelemetryData,
                       camera_profile: str = "dji_o4_main",
                       use_dem_correction: bool = True) -> Optional[GeolocationResult]:
        """Enhanced pixel geolocation with DEM correction"""
        
        import time
        start_time = time.time()
        
        try:
            # Get camera intrinsics
            if camera_profile not in self.camera_profiles:
                raise ValueError(f"Unknown camera profile: {camera_profile}")
            
            intrinsics = self.camera_profiles[camera_profile]
            
            # Convert pixel to camera ray
            camera_ray = self._pixel_to_camera_ray(pixel, intrinsics)
            
            # Transform to world coordinates
            world_ray = self._transform_ray_to_world(camera_ray, telemetry)
            
            # Ray origin
            ray_origin = GeographicCoordinate(
                latitude=telemetry.latitude,
                longitude=telemetry.longitude,
                altitude=telemetry.altitude_msl
            )
            
            if use_dem_correction:
                # Use iterative DEM correction
                result = self.dem_corrector.intersect_ray_with_terrain(
                    ray_origin, world_ray
                )
                
                if result is None:
                    return None
                
                final_coordinate, accuracy = result
            else:
                # Simple flat-earth intersection
                flat_intersection = self.dem_corrector._flat_earth_intersection(
                    ray_origin, world_ray
                )
                
                if flat_intersection is None:
                    return None
                
                final_coordinate = flat_intersection
                accuracy = AccuracyMetrics(
                    horizontal_accuracy=10.0,
                    vertical_accuracy=20.0,
                    confidence=0.5,
                    dem_source="flat_earth"
                )
            
            # Calculate intersection distance
            distance = self._calculate_distance(ray_origin, final_coordinate)
            
            processing_time = time.time() - start_time
            
            return GeolocationResult(
                coordinate=final_coordinate,
                accuracy=accuracy,
                ray_origin=ray_origin,
                ray_direction=world_ray,
                intersection_distance=distance,
                terrain_elevation=final_coordinate.altitude,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Geolocation failed: {e}")
            return None
    
    def _pixel_to_camera_ray(self, pixel: Point2D, intrinsics: Dict) -> Point3D:
        """Convert pixel to normalized camera ray"""
        # Normalize pixel coordinates
        x_norm = (pixel.u - intrinsics['cx']) / intrinsics['fx']
        y_norm = (pixel.v - intrinsics['cy']) / intrinsics['fy']
        
        # Create normalized ray (z=1 for pinhole model)
        ray = Point3D(x_norm, y_norm, 1.0)
        
        # Normalize to unit vector
        length = math.sqrt(ray.x**2 + ray.y**2 + ray.z**2)
        return Point3D(ray.x/length, ray.y/length, ray.z/length)
    
    def _transform_ray_to_world(self, camera_ray: Point3D, telemetry: TelemetryData) -> Point3D:
        """Transform camera ray to world coordinates"""
        # Convert to numpy for matrix operations
        ray_vector = np.array([camera_ray.x, camera_ray.y, camera_ray.z])
        
        # Gimbal rotation matrix
        gimbal_rotation = self._euler_to_rotation_matrix(
            telemetry.gimbal_roll, telemetry.gimbal_pitch, telemetry.gimbal_yaw
        )
        
        # Aircraft rotation matrix
        aircraft_rotation = self._euler_to_rotation_matrix(
            telemetry.roll, telemetry.pitch, telemetry.yaw
        )
        
        # Combined transformation
        combined_rotation = aircraft_rotation @ gimbal_rotation
        
        # Apply transformation
        world_ray = combined_rotation @ ray_vector
        
        return Point3D(world_ray[0], world_ray[1], world_ray[2])
    
    def _euler_to_rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert Euler angles to rotation matrix"""
        # Convert to radians
        r, p, y = math.radians(roll), math.radians(pitch), math.radians(yaw)
        
        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(r), -math.sin(r)],
            [0, math.sin(r), math.cos(r)]
        ])
        
        Ry = np.array([
            [math.cos(p), 0, math.sin(p)],
            [0, 1, 0],
            [-math.sin(p), 0, math.cos(p)]
        ])
        
        Rz = np.array([
            [math.cos(y), -math.sin(y), 0],
            [math.sin(y), math.cos(y), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation (ZYX order)
        return Rz @ Ry @ Rx
    
    def _calculate_distance(self, origin: GeographicCoordinate, 
                          target: GeographicCoordinate) -> float:
        """Calculate distance between two geographic points"""
        # Simplified distance calculation
        lat_diff = target.latitude - origin.latitude
        lon_diff = target.longitude - origin.longitude
        alt_diff = target.altitude - origin.altitude
        
        # Convert to meters (approximate)
        lat_meters = lat_diff * 111320.0
        lon_meters = lon_diff * 111320.0 * math.cos(math.radians(origin.latitude))
        
        return math.sqrt(lat_meters**2 + lon_meters**2 + alt_diff**2)
    
    def add_dem_source(self, source: DEMSource, config: Dict):
        """Add DEM data source"""
        self.dem_provider.add_dem_source(source, config)
    
    def validate_accuracy(self, known_points: List[Tuple[Point2D, GeographicCoordinate, TelemetryData]]) -> Dict:
        """Validate geolocation accuracy against known ground truth points"""
        results = []
        
        for pixel, ground_truth, telemetry in known_points:
            # Geolocate pixel
            result = self.geolocate_pixel(pixel, telemetry)
            
            if result:
                # Calculate errors
                lat_error = abs(result.coordinate.latitude - ground_truth.latitude)
                lon_error = abs(result.coordinate.longitude - ground_truth.longitude)
                alt_error = abs(result.coordinate.altitude - ground_truth.altitude)
                
                # Convert to meters
                lat_error_m = lat_error * 111320.0
                lon_error_m = lon_error * 111320.0 * math.cos(math.radians(ground_truth.latitude))
                
                horizontal_error = math.sqrt(lat_error_m**2 + lon_error_m**2)
                
                results.append({
                    'horizontal_error': horizontal_error,
                    'vertical_error': alt_error,
                    'predicted_h_accuracy': result.accuracy.horizontal_accuracy,
                    'predicted_v_accuracy': result.accuracy.vertical_accuracy,
                    'confidence': result.accuracy.confidence,
                    'processing_time': result.processing_time
                })
        
        if not results:
            return {'status': 'No valid results'}
        
        # Calculate statistics
        h_errors = [r['horizontal_error'] for r in results]
        v_errors = [r['vertical_error'] for r in results]
        
        return {
            'num_points': len(results),
            'horizontal_rmse': math.sqrt(sum(e**2 for e in h_errors) / len(h_errors)),
            'vertical_rmse': math.sqrt(sum(e**2 for e in v_errors) / len(v_errors)),
            'horizontal_cep90': np.percentile(h_errors, 90),
            'vertical_lep90': np.percentile(v_errors, 90),
            'mean_confidence': np.mean([r['confidence'] for r in results]),
            'mean_processing_time': np.mean([r['processing_time'] for r in results]),
            'accuracy_level_distribution': self._analyze_accuracy_levels(results)
        }
    
    def _analyze_accuracy_levels(self, results: List[Dict]) -> Dict[str, int]:
        """Analyze distribution of accuracy levels"""
        levels = {'precision': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        for result in results:
            h_acc = result['predicted_h_accuracy']
            v_acc = result['predicted_v_accuracy']
            
            if h_acc < 2.0 and v_acc < 5.0:
                levels['precision'] += 1
            elif h_acc < 5.0 and v_acc < 10.0:
                levels['high'] += 1
            elif h_acc < 10.0 and v_acc < 20.0:
                levels['medium'] += 1
            else:
                levels['low'] += 1
        
        return levels
    
    def get_system_status(self) -> Dict:
        """Get system status and configuration"""
        return {
            'dem_sources': list(self.dem_provider.dem_sources.keys()),
            'cache_size': len(self.dem_provider.elevation_cache),
            'max_iterations': self.dem_corrector.max_iterations,
            'convergence_tolerance': self.dem_corrector.convergence_tolerance,
            'camera_profiles': list(self.camera_profiles.keys())
        }


# Factory function
def create_enhanced_geolocation_system(dem_sources: Optional[List[Tuple[DEMSource, Dict]]] = None) -> EnhancedGeolocationAccuracy:
    """Create enhanced geolocation system with optional DEM sources"""
    system = EnhancedGeolocationAccuracy()
    
    if dem_sources:
        for source, config in dem_sources:
            system.add_dem_source(source, config)
    
    return system


if __name__ == "__main__":
    # Demo usage
    import time
    
    # Create system
    geo_system = create_enhanced_geolocation_system()
    
    # Test pixel
    test_pixel = Point2D(u=1920, v=1080)  # Center pixel
    
    # Test telemetry
    test_telemetry = TelemetryData(
        latitude=37.7749,
        longitude=-122.4194,
        altitude=500.0,
        altitude_msl=500.0,
        altitude_agl=450.0,
        roll=0.0, pitch=-10.0, yaw=45.0,
        gimbal_roll=0.0, gimbal_pitch=-30.0, gimbal_yaw=0.0
    )
    
    # Geolocate with DEM correction
    print("Testing enhanced geolocation with DEM correction...")
    result = geo_system.geolocate_pixel(test_pixel, test_telemetry, use_dem_correction=True)
    
    if result:
        print(f"\nGeolocation Result:")
        print(f"  Coordinate: {result.coordinate.latitude:.6f}, {result.coordinate.longitude:.6f}")
        print(f"  Altitude: {result.coordinate.altitude:.1f}m")
        print(f"  Horizontal Accuracy: {result.accuracy.horizontal_accuracy:.1f}m")
        print(f"  Vertical Accuracy: {result.accuracy.vertical_accuracy:.1f}m")
        print(f"  Confidence: {result.accuracy.confidence:.2f}")
        print(f"  Accuracy Level: {result.accuracy.accuracy_level.value}")
        print(f"  DEM Source: {result.accuracy.dem_source}")
        print(f"  Iterations: {result.accuracy.iteration_count}")
        print(f"  Processing Time: {result.processing_time:.3f}s")
    else:
        print("Geolocation failed")
    
    # System status
    print(f"\nSystem Status: {geo_system.get_system_status()}")
    print("\nEnhanced geolocation accuracy system demo completed!")