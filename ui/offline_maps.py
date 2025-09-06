#!/usr/bin/env python3
"""
Offline Map Tile Manager for Foresight SAR System

This module handles downloading, caching, and serving offline map tiles
for use in areas with limited or no internet connectivity.
"""

import os
import json
import sqlite3
import requests
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TileInfo:
    """Information about a map tile"""
    z: int  # Zoom level
    x: int  # X coordinate
    y: int  # Y coordinate
    url: str  # Source URL
    file_path: str  # Local file path
    downloaded: bool = False
    file_size: int = 0
    download_time: float = 0.0
    checksum: str = ""

@dataclass
class BoundingBox:
    """Geographic bounding box"""
    north: float
    south: float
    east: float
    west: float

@dataclass
class TileSource:
    """Map tile source configuration"""
    name: str
    url_template: str
    attribution: str
    max_zoom: int = 18
    min_zoom: int = 0
    file_extension: str = "png"
    headers: Dict[str, str] = None

class OfflineMapManager:
    """Manages offline map tiles for SAR operations"""
    
    def __init__(self, cache_dir: str = "map_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self.db_path = self.cache_dir / "tiles.db"
        self._init_database()
        
        # Default tile sources
        self.tile_sources = {
            "osm": TileSource(
                name="OpenStreetMap",
                url_template="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                attribution="© OpenStreetMap contributors"
            ),
            "satellite": TileSource(
                name="Esri Satellite",
                url_template="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attribution="© Esri",
                file_extension="jpg"
            ),
            "topo": TileSource(
                name="OpenTopoMap",
                url_template="https://tile.opentopomap.org/{z}/{x}/{y}.png",
                attribution="© OpenTopoMap contributors"
            )
        }
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Foresight-SAR/1.0 (Emergency Response System)'
        })
    
    def _init_database(self):
        """Initialize SQLite database for tile metadata"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    z INTEGER NOT NULL,
                    x INTEGER NOT NULL,
                    y INTEGER NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size INTEGER DEFAULT 0,
                    checksum TEXT DEFAULT '',
                    download_time REAL DEFAULT 0,
                    created_at REAL DEFAULT (julianday('now')),
                    UNIQUE(source, z, x, y)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tiles_coords 
                ON tiles(source, z, x, y)
            """)
    
    def _deg2num(self, lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
        """Convert lat/lon to tile coordinates"""
        import math
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        x = int((lon_deg + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (x, y)
    
    def _num2deg(self, x: int, y: int, zoom: int) -> Tuple[float, float]:
        """Convert tile coordinates to lat/lon"""
        import math
        n = 2.0 ** zoom
        lon_deg = x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        lat_deg = math.degrees(lat_rad)
        return (lat_deg, lon_deg)
    
    def get_tiles_for_bbox(self, bbox: BoundingBox, zoom_levels: List[int]) -> List[TileInfo]:
        """Get list of tiles needed for a bounding box and zoom levels"""
        tiles = []
        
        for zoom in zoom_levels:
            # Get tile coordinates for corners
            x_min, y_max = self._deg2num(bbox.north, bbox.west, zoom)
            x_max, y_min = self._deg2num(bbox.south, bbox.east, zoom)
            
            # Generate all tiles in the range
            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    tiles.append(TileInfo(
                        z=zoom, x=x, y=y,
                        url="",  # Will be set when downloading
                        file_path=""  # Will be set when downloading
                    ))
        
        return tiles
    
    def download_tile(self, tile: TileInfo, source: str) -> bool:
        """Download a single tile"""
        if source not in self.tile_sources:
            logger.error(f"Unknown tile source: {source}")
            return False
        
        tile_source = self.tile_sources[source]
        
        # Generate URL and file path
        tile.url = tile_source.url_template.format(z=tile.z, x=tile.x, y=tile.y)
        
        # Create directory structure
        tile_dir = self.cache_dir / source / str(tile.z) / str(tile.x)
        tile_dir.mkdir(parents=True, exist_ok=True)
        
        tile.file_path = str(tile_dir / f"{tile.y}.{tile_source.file_extension}")
        
        # Check if tile already exists
        if os.path.exists(tile.file_path):
            tile.downloaded = True
            tile.file_size = os.path.getsize(tile.file_path)
            return True
        
        try:
            start_time = time.time()
            
            # Add custom headers if specified
            headers = tile_source.headers or {}
            
            response = self.session.get(tile.url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Save tile
            with open(tile.file_path, 'wb') as f:
                f.write(response.content)
            
            # Update tile info
            tile.downloaded = True
            tile.file_size = len(response.content)
            tile.download_time = time.time() - start_time
            tile.checksum = hashlib.md5(response.content).hexdigest()
            
            # Save to database
            self._save_tile_to_db(tile, source)
            
            logger.info(f"Downloaded tile {source}/{tile.z}/{tile.x}/{tile.y}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download tile {tile.url}: {e}")
            return False
    
    def _save_tile_to_db(self, tile: TileInfo, source: str):
        """Save tile metadata to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO tiles 
                (source, z, x, y, file_path, file_size, checksum, download_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                source, tile.z, tile.x, tile.y, tile.file_path,
                tile.file_size, tile.checksum, tile.download_time
            ))
    
    def download_area(self, bbox: BoundingBox, zoom_levels: List[int], 
                     source: str = "osm", max_workers: int = 4) -> Dict[str, int]:
        """Download tiles for a specific area"""
        tiles = self.get_tiles_for_bbox(bbox, zoom_levels)
        
        logger.info(f"Downloading {len(tiles)} tiles for {source}")
        
        stats = {
            "total": len(tiles),
            "downloaded": 0,
            "failed": 0,
            "skipped": 0
        }
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit download tasks
            future_to_tile = {
                executor.submit(self.download_tile, tile, source): tile 
                for tile in tiles
            }
            
            # Process completed downloads
            for future in as_completed(future_to_tile):
                tile = future_to_tile[future]
                try:
                    success = future.result()
                    if success:
                        if tile.downloaded:
                            stats["downloaded"] += 1
                        else:
                            stats["skipped"] += 1
                    else:
                        stats["failed"] += 1
                except Exception as e:
                    logger.error(f"Download task failed: {e}")
                    stats["failed"] += 1
        
        logger.info(f"Download complete: {stats}")
        return stats
    
    def get_tile_path(self, z: int, x: int, y: int, source: str = "osm") -> Optional[str]:
        """Get local path for a tile if it exists"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT file_path FROM tiles WHERE source=? AND z=? AND x=? AND y=?",
                (source, z, x, y)
            )
            result = cursor.fetchone()
            
            if result and os.path.exists(result[0]):
                return result[0]
            return None
    
    def get_cache_stats(self) -> Dict[str, any]:
        """Get cache statistics"""
        with sqlite3.connect(self.db_path) as conn:
            # Total tiles
            cursor = conn.execute("SELECT COUNT(*) FROM tiles")
            total_tiles = cursor.fetchone()[0]
            
            # Total size
            cursor = conn.execute("SELECT SUM(file_size) FROM tiles")
            total_size = cursor.fetchone()[0] or 0
            
            # By source
            cursor = conn.execute("""
                SELECT source, COUNT(*), SUM(file_size) 
                FROM tiles GROUP BY source
            """)
            by_source = {row[0]: {"count": row[1], "size": row[2]} 
                        for row in cursor.fetchall()}
            
            # By zoom level
            cursor = conn.execute("""
                SELECT z, COUNT(*), SUM(file_size) 
                FROM tiles GROUP BY z ORDER BY z
            """)
            by_zoom = {row[0]: {"count": row[1], "size": row[2]} 
                      for row in cursor.fetchall()}
        
        return {
            "total_tiles": total_tiles,
            "total_size_mb": total_size / (1024 * 1024),
            "by_source": by_source,
            "by_zoom": by_zoom,
            "cache_dir": str(self.cache_dir)
        }
    
    def cleanup_cache(self, max_age_days: int = 30) -> int:
        """Clean up old tiles from cache"""
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            # Get old tiles
            cursor = conn.execute(
                "SELECT file_path FROM tiles WHERE created_at < ?",
                (cutoff_time,)
            )
            old_files = [row[0] for row in cursor.fetchall()]
            
            # Delete files
            deleted_count = 0
            for file_path in old_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}: {e}")
            
            # Remove from database
            conn.execute(
                "DELETE FROM tiles WHERE created_at < ?",
                (cutoff_time,)
            )
        
        logger.info(f"Cleaned up {deleted_count} old tiles")
        return deleted_count
    
    def export_cache_info(self, output_file: str):
        """Export cache information to JSON file"""
        stats = self.get_cache_stats()
        
        # Add tile sources info
        stats["tile_sources"] = {
            name: asdict(source) for name, source in self.tile_sources.items()
        }
        
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Cache info exported to {output_file}")

def create_sar_area_cache(center_lat: float, center_lon: float, 
                         radius_km: float, zoom_levels: List[int],
                         sources: List[str] = None) -> Dict[str, any]:
    """Create offline cache for a SAR operation area"""
    if sources is None:
        sources = ["osm", "satellite"]
    
    # Calculate bounding box from center and radius
    # Rough conversion: 1 degree ≈ 111 km
    lat_offset = radius_km / 111.0
    lon_offset = radius_km / (111.0 * abs(math.cos(math.radians(center_lat))))
    
    bbox = BoundingBox(
        north=center_lat + lat_offset,
        south=center_lat - lat_offset,
        east=center_lon + lon_offset,
        west=center_lon - lon_offset
    )
    
    manager = OfflineMapManager()
    results = {}
    
    for source in sources:
        logger.info(f"Downloading {source} tiles for SAR area")
        stats = manager.download_area(bbox, zoom_levels, source)
        results[source] = stats
    
    # Export cache info
    cache_info_file = manager.cache_dir / "sar_cache_info.json"
    manager.export_cache_info(str(cache_info_file))
    
    return {
        "bbox": asdict(bbox),
        "zoom_levels": zoom_levels,
        "download_results": results,
        "cache_stats": manager.get_cache_stats()
    }

if __name__ == "__main__":
    import math
    
    # Example: Create cache for a SAR operation area
    # San Francisco Bay Area
    result = create_sar_area_cache(
        center_lat=37.7749,
        center_lon=-122.4194,
        radius_km=10,
        zoom_levels=[10, 11, 12, 13, 14, 15],
        sources=["osm", "satellite"]
    )
    
    print("SAR area cache created:")
    print(json.dumps(result, indent=2))