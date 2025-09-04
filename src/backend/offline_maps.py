#!/usr/bin/env python3
"""
Offline Maps Module

This module handles offline map data including OpenStreetMap tiles, vector maps,
and Digital Elevation Model (DEM) data for fully offline operation.
"""

import os
import json
import logging
import requests
import time
import math
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

try:
    import folium
    from geopy.distance import geodesic
    MAPPING_AVAILABLE = True
except ImportError:
    MAPPING_AVAILABLE = False
    logging.warning("Mapping libraries not available. Install with: pip install folium geopy")

import numpy as np


class MapTileProvider(Enum):
    """Map tile provider enumeration"""
    OPENSTREETMAP = "openstreetmap"
    OPENTOPOMAP = "opentopomap"
    SATELLITE = "satellite"
    TERRAIN = "terrain"


@dataclass
class BoundingBox:
    """Geographic bounding box"""
    north: float
    south: float
    east: float
    west: float
    
    def contains(self, lat: float, lon: float) -> bool:
        """Check if coordinates are within bounding box"""
        return (self.south <= lat <= self.north and 
                self.west <= lon <= self.east)
    
    def area_km2(self) -> float:
        """Calculate approximate area in square kilometers"""
        # Approximate calculation using geodesic distance
        width = geodesic((self.south, self.west), (self.south, self.east)).kilometers
        height = geodesic((self.south, self.west), (self.north, self.west)).kilometers
        return width * height


@dataclass
class TileCoordinate:
    """Map tile coordinate"""
    x: int
    y: int
    zoom: int
    
    def to_filename(self, provider: MapTileProvider) -> str:
        """Generate filename for tile"""
        return f"{provider.value}_z{self.zoom}_x{self.x}_y{self.y}.png"


class OfflineMapManager:
    """Manager for offline map data"""
    
    def __init__(self, cache_dir: str = "data/maps"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.cache_dir / "tiles").mkdir(exist_ok=True)
        (self.cache_dir / "dem").mkdir(exist_ok=True)
        (self.cache_dir / "vector").mkdir(exist_ok=True)
        
        self.tile_providers = {
            MapTileProvider.OPENSTREETMAP: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            MapTileProvider.OPENTOPOMAP: "https://tile.opentopomap.org/{z}/{x}/{y}.png",
            MapTileProvider.SATELLITE: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            MapTileProvider.TERRAIN: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}"
        }
        
        self.downloaded_areas: List[Dict[str, Any]] = []
        self._load_metadata()
        
    def _load_metadata(self):
        """Load metadata about downloaded areas"""
        metadata_file = self.cache_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    self.downloaded_areas = data.get('downloaded_areas', [])
            except Exception as e:
                logging.error(f"Error loading metadata: {e}")
                
    def _save_metadata(self):
        """Save metadata about downloaded areas"""
        metadata_file = self.cache_dir / "metadata.json"
        try:
            with open(metadata_file, 'w') as f:
                json.dump({
                    'downloaded_areas': self.downloaded_areas,
                    'last_updated': time.time()
                }, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving metadata: {e}")
            
    def deg2num(self, lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
        """Convert lat/lon to tile coordinates"""
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        x = int((lon_deg + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (x, y)
        
    def num2deg(self, x: int, y: int, zoom: int) -> Tuple[float, float]:
        """Convert tile coordinates to lat/lon"""
        n = 2.0 ** zoom
        lon_deg = x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        lat_deg = math.degrees(lat_rad)
        return (lat_deg, lon_deg)
        
    def get_tiles_for_bbox(self, bbox: BoundingBox, zoom_levels: List[int]) -> List[TileCoordinate]:
        """Get list of tiles needed for bounding box and zoom levels"""
        tiles = []
        
        for zoom in zoom_levels:
            # Get tile coordinates for corners
            x_min, y_max = self.deg2num(bbox.south, bbox.west, zoom)
            x_max, y_min = self.deg2num(bbox.north, bbox.east, zoom)
            
            # Generate all tiles in range
            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    tiles.append(TileCoordinate(x, y, zoom))
                    
        return tiles
        
    def download_tile(self, tile: TileCoordinate, provider: MapTileProvider, 
                     max_retries: int = 3, delay: float = 1.0) -> bool:
        """Download a single map tile"""
        tile_dir = self.cache_dir / "tiles" / provider.value
        tile_dir.mkdir(parents=True, exist_ok=True)
        
        filename = tile.to_filename(provider)
        filepath = tile_dir / filename
        
        # Skip if already exists
        if filepath.exists():
            return True
            
        url_template = self.tile_providers.get(provider)
        if not url_template:
            logging.error(f"Unknown tile provider: {provider}")
            return False
            
        url = url_template.format(z=tile.zoom, x=tile.x, y=tile.y)
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=30, headers={
                    'User-Agent': 'Foresight Offline Maps/1.0'
                })
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                    
                logging.debug(f"Downloaded tile: {filename}")
                time.sleep(delay)  # Be respectful to tile servers
                return True
                
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed for {filename}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay * (attempt + 1))
                    
        logging.error(f"Failed to download tile after {max_retries} attempts: {filename}")
        return False
        
    def download_area(self, bbox: BoundingBox, zoom_levels: List[int], 
                     providers: List[MapTileProvider], 
                     progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Download map tiles for an area"""
        if not MAPPING_AVAILABLE:
            raise ImportError("Mapping libraries not available")
            
        tiles = self.get_tiles_for_bbox(bbox, zoom_levels)
        total_tiles = len(tiles) * len(providers)
        downloaded = 0
        failed = 0
        
        logging.info(f"Downloading {total_tiles} tiles for area {bbox}")
        
        start_time = time.time()
        
        for provider in providers:
            for i, tile in enumerate(tiles):
                success = self.download_tile(tile, provider)
                
                if success:
                    downloaded += 1
                else:
                    failed += 1
                    
                if progress_callback:
                    progress = (downloaded + failed) / total_tiles
                    progress_callback(progress, downloaded, failed, total_tiles)
                    
        duration = time.time() - start_time
        
        # Save download info
        download_info = {
            'bbox': {
                'north': bbox.north,
                'south': bbox.south,
                'east': bbox.east,
                'west': bbox.west
            },
            'zoom_levels': zoom_levels,
            'providers': [p.value for p in providers],
            'total_tiles': total_tiles,
            'downloaded': downloaded,
            'failed': failed,
            'duration': duration,
            'timestamp': time.time()
        }
        
        self.downloaded_areas.append(download_info)
        self._save_metadata()
        
        logging.info(f"Download complete: {downloaded}/{total_tiles} tiles in {duration:.1f}s")
        
        return download_info
        
    def get_tile_path(self, tile: TileCoordinate, provider: MapTileProvider) -> Optional[Path]:
        """Get local path to tile if it exists"""
        tile_dir = self.cache_dir / "tiles" / provider.value
        filename = tile.to_filename(provider)
        filepath = tile_dir / filename
        
        return filepath if filepath.exists() else None
        
    def is_area_available(self, bbox: BoundingBox, zoom_levels: List[int], 
                         providers: List[MapTileProvider]) -> bool:
        """Check if area is available offline"""
        tiles = self.get_tiles_for_bbox(bbox, zoom_levels)
        
        for provider in providers:
            for tile in tiles:
                if not self.get_tile_path(tile, provider):
                    return False
                    
        return True
        
    def get_coverage_stats(self) -> Dict[str, Any]:
        """Get statistics about offline coverage"""
        total_size = 0
        tile_count = 0
        
        tiles_dir = self.cache_dir / "tiles"
        if tiles_dir.exists():
            for provider_dir in tiles_dir.iterdir():
                if provider_dir.is_dir():
                    for tile_file in provider_dir.glob("*.png"):
                        total_size += tile_file.stat().st_size
                        tile_count += 1
                        
        return {
            'total_tiles': tile_count,
            'total_size_mb': total_size / (1024 * 1024),
            'downloaded_areas': len(self.downloaded_areas),
            'cache_dir': str(self.cache_dir)
        }
        
    def create_offline_map(self, center_lat: float, center_lon: float, 
                          zoom: int = 15, width: int = 800, height: int = 600) -> Optional[str]:
        """Create an offline map HTML file"""
        if not MAPPING_AVAILABLE:
            return None
            
        try:
            # Create folium map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=zoom,
                width=width,
                height=height
            )
            
            # Add downloaded areas as overlays
            for area in self.downloaded_areas:
                bbox_data = area['bbox']
                bounds = [
                    [bbox_data['south'], bbox_data['west']],
                    [bbox_data['north'], bbox_data['east']]
                ]
                
                folium.Rectangle(
                    bounds=bounds,
                    color='blue',
                    fill=False,
                    popup=f"Downloaded: {area['providers']}"
                ).add_to(m)
                
            # Save map
            map_file = self.cache_dir / "offline_map.html"
            m.save(str(map_file))
            
            return str(map_file)
            
        except Exception as e:
            logging.error(f"Error creating offline map: {e}")
            return None


def demo_offline_maps():
    """Demo function for testing offline maps"""
    logging.basicConfig(level=logging.INFO)
    
    if not MAPPING_AVAILABLE:
        print("Mapping libraries not available. Install with: pip install folium geopy")
        return
        
    manager = OfflineMapManager()
    
    # Define a small test area (around San Francisco)
    bbox = BoundingBox(
        north=37.8,
        south=37.7,
        east=-122.3,
        west=-122.5
    )
    
    print(f"Test area: {bbox}")
    print(f"Area size: {bbox.area_km2():.2f} km²")
    
    # Download tiles for zoom levels 10-15
    zoom_levels = [10, 12, 14]
    providers = [MapTileProvider.OPENSTREETMAP]
    
    def progress_callback(progress, downloaded, failed, total):
        print(f"Progress: {progress*100:.1f}% ({downloaded}/{total} downloaded, {failed} failed)")
        
    try:
        result = manager.download_area(bbox, zoom_levels, providers, progress_callback)
        print(f"Download result: {result}")
        
        # Check coverage
        available = manager.is_area_available(bbox, zoom_levels, providers)
        print(f"Area available offline: {available}")
        
        # Get stats
        stats = manager.get_coverage_stats()
        print(f"Coverage stats: {stats}")
        
        # Create offline map
        map_file = manager.create_offline_map(37.75, -122.4)
        if map_file:
            print(f"Offline map created: {map_file}")
            
    except Exception as e:
        print(f"Demo error: {e}")


if __name__ == "__main__":
    demo_offline_maps()