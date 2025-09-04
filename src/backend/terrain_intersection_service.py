#!/usr/bin/env python3
"""
Terrain Intersection Service for Ray-Ground Intersection

Provides terrain intersection functionality for ray-casting:
- Flat ground plane intersection
- Digital Elevation Model (DEM) intersection
- Iterative ray-terrain intersection
- Multi-resolution DEM support
- Terrain data caching and management
- Accuracy estimation for different terrain types

Used for accurate geolocation computation in mountainous or varied terrain.
"""

import numpy as np
import math
from typing import Tuple, Optional, List, Dict, Any, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from abc import ABC, abstractmethod

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import transform_bounds, reproject
    from rasterio.crs import CRS
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    logging.warning("rasterio not available, DEM functionality will be limited")

from .ray_casting_service import Ray3D, RaycastResult
from .geolocation_service import CoordinateTransforms

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TerrainPoint:
    """3D terrain point with metadata."""
    
    latitude: float
    longitude: float
    elevation: float
    
    # Metadata
    source: str = "unknown"  # DEM source or "flat"
    accuracy: float = 1.0    # Elevation accuracy in meters
    
    def to_local_coords(self, origin_lat: float, origin_lon: float) -> np.ndarray:
        """Convert to local coordinates relative to origin.
        
        Args:
            origin_lat: Origin latitude
            origin_lon: Origin longitude
            
        Returns:
            Local coordinates [x, y, z] in meters
        """
        x, y = CoordinateTransforms.latlon_to_meters(
            self.latitude, self.longitude, origin_lat, origin_lon
        )
        return np.array([x, y, self.elevation])

@dataclass
class IntersectionResult:
    """Result of ray-terrain intersection."""
    
    intersection_point: Optional[TerrainPoint]  # Geographic coordinates
    local_point: Optional[np.ndarray]          # Local coordinates
    distance: Optional[float]                  # Distance from ray origin
    iterations: int = 0                        # Number of iterations used
    converged: bool = False                    # Whether iteration converged
    
    # Accuracy estimates
    elevation_accuracy: float = 1.0            # Elevation accuracy in meters
    horizontal_accuracy: float = 1.0           # Horizontal accuracy in meters
    
    def is_valid(self) -> bool:
        """Check if intersection is valid."""
        return (self.intersection_point is not None and 
                self.local_point is not None and 
                self.distance is not None and 
                self.distance > 0)

class TerrainModel(ABC):
    """Abstract base class for terrain models."""
    
    @abstractmethod
    def get_elevation(self, latitude: float, longitude: float) -> Optional[float]:
        """Get elevation at given coordinates.
        
        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            
        Returns:
            Elevation in meters or None if not available
        """
        pass
    
    @abstractmethod
    def get_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Get terrain model bounds.
        
        Returns:
            (min_lat, min_lon, max_lat, max_lon) or None
        """
        pass
    
    @abstractmethod
    def get_resolution(self) -> Optional[float]:
        """Get terrain model resolution in meters.
        
        Returns:
            Resolution in meters or None
        """
        pass

class FlatTerrainModel(TerrainModel):
    """Simple flat terrain model."""
    
    def __init__(self, elevation: float = 0.0):
        """
        Initialize flat terrain model.
        
        Args:
            elevation: Constant elevation in meters
        """
        self.elevation = elevation
    
    def get_elevation(self, latitude: float, longitude: float) -> Optional[float]:
        """Get constant elevation."""
        return self.elevation
    
    def get_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Flat terrain has no bounds."""
        return None
    
    def get_resolution(self) -> Optional[float]:
        """Flat terrain has infinite resolution."""
        return 0.0

class DEMTerrainModel(TerrainModel):
    """Digital Elevation Model terrain model."""
    
    def __init__(self, dem_path: str, cache_size: int = 1000):
        """
        Initialize DEM terrain model.
        
        Args:
            dem_path: Path to DEM file (GeoTIFF, etc.)
            cache_size: Number of elevation queries to cache
        """
        self.dem_path = Path(dem_path)
        self.cache_size = cache_size
        self.elevation_cache: Dict[Tuple[float, float], float] = {}
        
        if not RASTERIO_AVAILABLE:
            raise ImportError("rasterio is required for DEM functionality")
        
        if not self.dem_path.exists():
            raise FileNotFoundError(f"DEM file not found: {dem_path}")
        
        # Load DEM metadata
        with rasterio.open(self.dem_path) as dataset:
            self.bounds = dataset.bounds
            self.crs = dataset.crs
            self.transform = dataset.transform
            self.width = dataset.width
            self.height = dataset.height
            self.nodata = dataset.nodata
            
            # Calculate resolution
            self.resolution = abs(self.transform[0])  # Pixel size in CRS units
            
        logger.info(f"Loaded DEM: {self.dem_path}")
        logger.info(f"  Bounds: {self.bounds}")
        logger.info(f"  Resolution: {self.resolution:.1f} m")
        logger.info(f"  Size: {self.width} x {self.height}")
    
    def get_elevation(self, latitude: float, longitude: float) -> Optional[float]:
        """Get elevation from DEM.
        
        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            
        Returns:
            Elevation in meters or None if outside bounds
        """
        # Check cache first
        cache_key = (round(latitude, 6), round(longitude, 6))
        if cache_key in self.elevation_cache:
            return self.elevation_cache[cache_key]
        
        try:
            with rasterio.open(self.dem_path) as dataset:
                # Convert lat/lon to pixel coordinates
                row, col = dataset.index(longitude, latitude)
                
                # Check if within bounds
                if (0 <= row < dataset.height and 0 <= col < dataset.width):
                    # Read elevation value
                    elevation = dataset.read(1, window=((row, row+1), (col, col+1)))[0, 0]
                    
                    # Handle nodata values
                    if elevation == self.nodata or np.isnan(elevation):
                        return None
                    
                    # Cache result
                    if len(self.elevation_cache) < self.cache_size:
                        self.elevation_cache[cache_key] = float(elevation)
                    
                    return float(elevation)
                else:
                    return None
                    
        except Exception as e:
            logger.warning(f"Error reading DEM elevation: {e}")
            return None
    
    def get_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Get DEM bounds in lat/lon."""
        # Convert bounds to lat/lon if needed
        if self.crs != CRS.from_epsg(4326):
            try:
                min_lon, min_lat, max_lon, max_lat = transform_bounds(
                    self.crs, CRS.from_epsg(4326), *self.bounds
                )
                return (min_lat, min_lon, max_lat, max_lon)
            except Exception:
                return None
        else:
            return (self.bounds.bottom, self.bounds.left, 
                   self.bounds.top, self.bounds.right)
    
    def get_resolution(self) -> Optional[float]:
        """Get DEM resolution in meters."""
        return self.resolution
    
    def get_elevation_profile(self, start_lat: float, start_lon: float,
                            end_lat: float, end_lon: float, 
                            num_points: int = 100) -> List[Tuple[float, float, float]]:
        """Get elevation profile along a line.
        
        Args:
            start_lat: Start latitude
            start_lon: Start longitude
            end_lat: End latitude
            end_lon: End longitude
            num_points: Number of points to sample
            
        Returns:
            List of (lat, lon, elevation) tuples
        """
        profile = []
        
        for i in range(num_points):
            t = i / (num_points - 1)
            lat = start_lat + t * (end_lat - start_lat)
            lon = start_lon + t * (end_lon - start_lon)
            
            elevation = self.get_elevation(lat, lon)
            if elevation is not None:
                profile.append((lat, lon, elevation))
        
        return profile

class TerrainIntersectionService:
    """Service for ray-terrain intersection calculations."""
    
    def __init__(self):
        self.terrain_models: Dict[str, TerrainModel] = {}
        self.default_model = FlatTerrainModel(0.0)
    
    def add_terrain_model(self, name: str, model: TerrainModel):
        """Add a terrain model.
        
        Args:
            name: Model name
            model: TerrainModel instance
        """
        self.terrain_models[name] = model
        logger.info(f"Added terrain model: {name}")
    
    def load_dem(self, name: str, dem_path: str) -> bool:
        """Load DEM terrain model.
        
        Args:
            name: Model name
            dem_path: Path to DEM file
            
        Returns:
            True if loaded successfully
        """
        try:
            model = DEMTerrainModel(dem_path)
            self.add_terrain_model(name, model)
            return True
        except Exception as e:
            logger.error(f"Failed to load DEM {dem_path}: {e}")
            return False
    
    def get_terrain_model(self, name: Optional[str] = None) -> TerrainModel:
        """Get terrain model by name.
        
        Args:
            name: Model name or None for default
            
        Returns:
            TerrainModel instance
        """
        if name is None or name not in self.terrain_models:
            return self.default_model
        return self.terrain_models[name]
    
    def intersect_ray_flat(self, ray: Ray3D, ground_elevation: float = 0.0) -> IntersectionResult:
        """Intersect ray with flat ground plane.
        
        Args:
            ray: 3D ray in world coordinates
            ground_elevation: Ground elevation in meters
            
        Returns:
            IntersectionResult
        """
        # Simple plane intersection
        intersection_3d = ray.intersect_ground_plane(ground_elevation)
        
        if intersection_3d is None:
            return IntersectionResult(
                intersection_point=None,
                local_point=None,
                distance=None,
                converged=False
            )
        
        # Calculate distance
        distance = np.linalg.norm(intersection_3d - ray.origin)
        
        # For flat terrain, we need to convert back to lat/lon
        # This requires the drone's GPS position as reference
        terrain_point = TerrainPoint(
            latitude=0.0,  # Will be set by caller
            longitude=0.0,  # Will be set by caller
            elevation=ground_elevation,
            source="flat",
            accuracy=0.1  # Very accurate for flat terrain
        )
        
        return IntersectionResult(
            intersection_point=terrain_point,
            local_point=intersection_3d,
            distance=distance,
            iterations=1,
            converged=True,
            elevation_accuracy=0.1,
            horizontal_accuracy=1.0
        )
    
    def intersect_ray_dem(self, ray: Ray3D, drone_lat: float, drone_lon: float,
                         terrain_model: str = None, max_iterations: int = 10,
                         tolerance: float = 1.0) -> IntersectionResult:
        """Intersect ray with DEM terrain using iterative method.
        
        Args:
            ray: 3D ray in world coordinates
            drone_lat: Drone latitude (reference point)
            drone_lon: Drone longitude (reference point)
            terrain_model: Name of terrain model to use
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance in meters
            
        Returns:
            IntersectionResult
        """
        model = self.get_terrain_model(terrain_model)
        
        # Start with flat ground intersection as initial guess
        initial_elevation = model.get_elevation(drone_lat, drone_lon) or 0.0
        current_intersection = ray.intersect_ground_plane(initial_elevation)
        
        if current_intersection is None:
            return IntersectionResult(
                intersection_point=None,
                local_point=None,
                distance=None,
                converged=False
            )
        
        # Iterative refinement
        for iteration in range(max_iterations):
            # Convert current intersection to lat/lon
            x_local, y_local = current_intersection[0], current_intersection[1]
            lat, lon = CoordinateTransforms.meters_to_latlon(
                x_local, y_local, drone_lat, drone_lon
            )
            
            # Get terrain elevation at this point
            terrain_elevation = model.get_elevation(lat, lon)
            if terrain_elevation is None:
                # Outside DEM bounds, use flat terrain
                terrain_elevation = initial_elevation
            
            # Intersect ray with plane at terrain elevation
            new_intersection = ray.intersect_ground_plane(terrain_elevation)
            if new_intersection is None:
                break
            
            # Check convergence
            if current_intersection is not None:
                displacement = np.linalg.norm(new_intersection - current_intersection)
                if displacement < tolerance:
                    # Converged
                    distance = np.linalg.norm(new_intersection - ray.origin)
                    
                    terrain_point = TerrainPoint(
                        latitude=lat,
                        longitude=lon,
                        elevation=terrain_elevation,
                        source=terrain_model or "dem",
                        accuracy=model.get_resolution() or 1.0
                    )
                    
                    return IntersectionResult(
                        intersection_point=terrain_point,
                        local_point=new_intersection,
                        distance=distance,
                        iterations=iteration + 1,
                        converged=True,
                        elevation_accuracy=model.get_resolution() or 1.0,
                        horizontal_accuracy=model.get_resolution() or 1.0
                    )
            
            current_intersection = new_intersection
        
        # Did not converge, return best estimate
        if current_intersection is not None:
            x_local, y_local = current_intersection[0], current_intersection[1]
            lat, lon = CoordinateTransforms.meters_to_latlon(
                x_local, y_local, drone_lat, drone_lon
            )
            
            terrain_elevation = model.get_elevation(lat, lon) or initial_elevation
            distance = np.linalg.norm(current_intersection - ray.origin)
            
            terrain_point = TerrainPoint(
                latitude=lat,
                longitude=lon,
                elevation=terrain_elevation,
                source=terrain_model or "dem",
                accuracy=(model.get_resolution() or 1.0) * 2  # Lower accuracy
            )
            
            return IntersectionResult(
                intersection_point=terrain_point,
                local_point=current_intersection,
                distance=distance,
                iterations=max_iterations,
                converged=False,
                elevation_accuracy=(model.get_resolution() or 1.0) * 2,
                horizontal_accuracy=(model.get_resolution() or 1.0) * 2
            )
        
        return IntersectionResult(
            intersection_point=None,
            local_point=None,
            distance=None,
            iterations=max_iterations,
            converged=False
        )
    
    def intersect_ray_auto(self, ray: Ray3D, drone_lat: float, drone_lon: float,
                          preferred_model: str = None) -> IntersectionResult:
        """Automatically choose best intersection method.
        
        Args:
            ray: 3D ray in world coordinates
            drone_lat: Drone latitude
            drone_lon: Drone longitude
            preferred_model: Preferred terrain model name
            
        Returns:
            IntersectionResult
        """
        # Try preferred model first
        if preferred_model and preferred_model in self.terrain_models:
            model = self.terrain_models[preferred_model]
            bounds = model.get_bounds()
            
            # Check if drone is within model bounds
            if bounds is None or (
                bounds[0] <= drone_lat <= bounds[2] and 
                bounds[1] <= drone_lon <= bounds[3]
            ):
                return self.intersect_ray_dem(ray, drone_lat, drone_lon, preferred_model)
        
        # Try all available DEM models
        for name, model in self.terrain_models.items():
            if isinstance(model, DEMTerrainModel):
                bounds = model.get_bounds()
                if bounds and (
                    bounds[0] <= drone_lat <= bounds[2] and 
                    bounds[1] <= drone_lon <= bounds[3]
                ):
                    return self.intersect_ray_dem(ray, drone_lat, drone_lon, name)
        
        # Fall back to flat terrain
        logger.info("Using flat terrain intersection (no suitable DEM found)")
        return self.intersect_ray_flat(ray, 0.0)
    
    def estimate_intersection_accuracy(self, result: IntersectionResult, 
                                     ray_accuracy: Dict[str, float]) -> Dict[str, float]:
        """Estimate overall intersection accuracy.
        
        Args:
            result: IntersectionResult to analyze
            ray_accuracy: Ray-casting accuracy from RayCastingService
            
        Returns:
            Combined accuracy metrics
        """
        accuracy = ray_accuracy.copy()
        
        if result.is_valid():
            # Add terrain-specific accuracy
            accuracy['terrain_elevation_accuracy_m'] = result.elevation_accuracy
            accuracy['terrain_horizontal_accuracy_m'] = result.horizontal_accuracy
            accuracy['terrain_iterations'] = result.iterations
            accuracy['terrain_converged'] = result.converged
            
            # Combine with ray-casting accuracy
            if 'total_error_m' in accuracy:
                terrain_error = math.sqrt(
                    result.elevation_accuracy**2 + 
                    result.horizontal_accuracy**2
                )
                
                combined_error = math.sqrt(
                    accuracy['total_error_m']**2 + 
                    terrain_error**2
                )
                
                accuracy['combined_error_m'] = combined_error
                accuracy['terrain_error_m'] = terrain_error
        
        return accuracy
    
    def list_terrain_models(self) -> Dict[str, Dict[str, Any]]:
        """List available terrain models with metadata.
        
        Returns:
            Dictionary of model information
        """
        models = {}
        
        for name, model in self.terrain_models.items():
            info = {
                'type': type(model).__name__,
                'bounds': model.get_bounds(),
                'resolution_m': model.get_resolution()
            }
            
            if isinstance(model, DEMTerrainModel):
                info['dem_path'] = str(model.dem_path)
                info['crs'] = str(model.crs)
                info['size'] = (model.width, model.height)
            
            models[name] = info
        
        return models

# Global terrain intersection service
terrain_intersection_service = TerrainIntersectionService()

def get_terrain_intersection_service() -> TerrainIntersectionService:
    """Get the global terrain intersection service."""
    return terrain_intersection_service

if __name__ == "__main__":
    # Demo usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Terrain Intersection Service Demo")
    parser.add_argument("--dem", help="Path to DEM file")
    parser.add_argument("--lat", type=float, default=37.7749, help="Test latitude")
    parser.add_argument("--lon", type=float, default=-122.4194, help="Test longitude")
    parser.add_argument("--altitude", type=float, default=100.0, help="Drone altitude")
    args = parser.parse_args()
    
    service = get_terrain_intersection_service()
    
    # Load DEM if provided
    if args.dem:
        if service.load_dem("test_dem", args.dem):
            print(f"Loaded DEM: {args.dem}")
        else:
            print(f"Failed to load DEM: {args.dem}")
    
    # Create test ray (pointing down and forward)
    ray_origin = np.array([0.0, 0.0, args.altitude])
    ray_direction = np.array([0.5, 0.0, -1.0])  # Forward and down
    ray_direction = ray_direction / np.linalg.norm(ray_direction)
    
    test_ray = Ray3D(origin=ray_origin, direction=ray_direction)
    
    print(f"\nTesting ray intersection:")
    print(f"  Ray origin: {ray_origin}")
    print(f"  Ray direction: {ray_direction}")
    print(f"  Drone position: {args.lat:.6f}, {args.lon:.6f}")
    
    # Test flat terrain intersection
    flat_result = service.intersect_ray_flat(test_ray, 0.0)
    print(f"\nFlat terrain intersection:")
    if flat_result.is_valid():
        print(f"  Local point: {flat_result.local_point}")
        print(f"  Distance: {flat_result.distance:.1f} m")
    else:
        print(f"  No intersection found")
    
    # Test DEM intersection if available
    if args.dem and "test_dem" in service.terrain_models:
        dem_result = service.intersect_ray_dem(test_ray, args.lat, args.lon, "test_dem")
        print(f"\nDEM terrain intersection:")
        if dem_result.is_valid():
            print(f"  Geographic point: {dem_result.intersection_point.latitude:.6f}, {dem_result.intersection_point.longitude:.6f}")
            print(f"  Elevation: {dem_result.intersection_point.elevation:.1f} m")
            print(f"  Distance: {dem_result.distance:.1f} m")
            print(f"  Iterations: {dem_result.iterations}")
            print(f"  Converged: {dem_result.converged}")
        else:
            print(f"  No intersection found")
    
    # List available models
    models = service.list_terrain_models()
    print(f"\nAvailable terrain models:")
    for name, info in models.items():
        print(f"  {name}: {info['type']}")
        if info['bounds']:
            print(f"    Bounds: {info['bounds']}")
        if info['resolution_m']:
            print(f"    Resolution: {info['resolution_m']:.1f} m")