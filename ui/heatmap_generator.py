#!/usr/bin/env python3
"""
Heatmap Generator for Foresight SAR System

This module generates heatmaps from detection data for tactical visualization
and pattern analysis in search and rescue operations.
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV not available - some heatmap features disabled")

try:
    from PIL import Image, ImageDraw, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available - image export features disabled")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DetectionPoint:
    """Single detection point for heatmap generation"""
    lat: float
    lng: float
    confidence: float
    timestamp: str
    detection_type: str = "person"
    track_id: Optional[int] = None
    locked: bool = False
    metadata: Dict = None

@dataclass
class HeatmapConfig:
    """Configuration for heatmap generation"""
    width: int = 1024
    height: int = 768
    radius: int = 25
    intensity: float = 0.5
    blur_radius: int = 15
    color_scheme: str = "hot"  # hot, cool, rainbow, custom
    background_alpha: float = 0.7
    time_decay: bool = True
    decay_hours: float = 24.0
    confidence_weighting: bool = True
    normalize: bool = True

@dataclass
class BoundingBox:
    """Geographic bounding box for heatmap area"""
    north: float
    south: float
    east: float
    west: float
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center coordinates"""
        return ((self.north + self.south) / 2, (self.east + self.west) / 2)
    
    @property
    def width_deg(self) -> float:
        """Get width in degrees"""
        return abs(self.east - self.west)
    
    @property
    def height_deg(self) -> float:
        """Get height in degrees"""
        return abs(self.north - self.south)

class HeatmapGenerator:
    """Generates heatmaps from detection data"""
    
    def __init__(self, config: HeatmapConfig = None):
        self.config = config or HeatmapConfig()
        
        # Color schemes
        self.color_schemes = {
            "hot": [(0, 0, 0), (128, 0, 0), (255, 0, 0), (255, 128, 0), (255, 255, 0), (255, 255, 255)],
            "cool": [(0, 0, 0), (0, 0, 128), (0, 0, 255), (0, 128, 255), (0, 255, 255), (255, 255, 255)],
            "rainbow": [(0, 0, 0), (128, 0, 128), (0, 0, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)],
            "thermal": [(0, 0, 0), (64, 0, 128), (128, 0, 255), (255, 0, 128), (255, 128, 0), (255, 255, 0)]
        }
    
    def _lat_lng_to_pixel(self, lat: float, lng: float, bbox: BoundingBox) -> Tuple[int, int]:
        """Convert lat/lng to pixel coordinates"""
        # Normalize to 0-1 range
        x_norm = (lng - bbox.west) / bbox.width_deg
        y_norm = (bbox.north - lat) / bbox.height_deg  # Flip Y axis
        
        # Convert to pixel coordinates
        x = int(x_norm * self.config.width)
        y = int(y_norm * self.config.height)
        
        # Clamp to image bounds
        x = max(0, min(self.config.width - 1, x))
        y = max(0, min(self.config.height - 1, y))
        
        return (x, y)
    
    def _calculate_time_weight(self, timestamp: str) -> float:
        """Calculate time-based weight for detection"""
        if not self.config.time_decay:
            return 1.0
        
        try:
            detection_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            current_time = datetime.now(detection_time.tzinfo)
            
            hours_ago = (current_time - detection_time).total_seconds() / 3600
            
            if hours_ago > self.config.decay_hours:
                return 0.1  # Minimum weight for very old detections
            
            # Exponential decay
            decay_factor = np.exp(-hours_ago / (self.config.decay_hours / 3))
            return max(0.1, decay_factor)
            
        except Exception as e:
            logger.warning(f"Failed to parse timestamp {timestamp}: {e}")
            return 1.0
    
    def _create_gaussian_kernel(self, radius: int) -> np.ndarray:
        """Create Gaussian kernel for heatmap blurring"""
        size = radius * 2 + 1
        kernel = np.zeros((size, size))
        
        center = radius
        sigma = radius / 3.0
        
        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x*x + y*y) / (2 * sigma * sigma))
        
        return kernel / np.sum(kernel)
    
    def _apply_color_scheme(self, intensity_map: np.ndarray) -> np.ndarray:
        """Apply color scheme to intensity map"""
        if self.config.color_scheme not in self.color_schemes:
            logger.warning(f"Unknown color scheme: {self.config.color_scheme}, using 'hot'")
            colors = self.color_schemes["hot"]
        else:
            colors = self.color_schemes[self.config.color_scheme]
        
        # Normalize intensity map
        if self.config.normalize and np.max(intensity_map) > 0:
            intensity_map = intensity_map / np.max(intensity_map)
        
        # Create RGB image
        height, width = intensity_map.shape
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Apply color mapping
        for i in range(height):
            for j in range(width):
                intensity = intensity_map[i, j]
                
                if intensity == 0:
                    continue
                
                # Find color interpolation points
                color_index = intensity * (len(colors) - 1)
                lower_idx = int(color_index)
                upper_idx = min(lower_idx + 1, len(colors) - 1)
                
                # Interpolate between colors
                if lower_idx == upper_idx:
                    color = colors[lower_idx]
                else:
                    t = color_index - lower_idx
                    lower_color = np.array(colors[lower_idx])
                    upper_color = np.array(colors[upper_idx])
                    color = lower_color + t * (upper_color - lower_color)
                
                rgb_image[i, j] = color.astype(np.uint8)
        
        return rgb_image
    
    def generate_heatmap(self, detections: List[DetectionPoint], 
                        bbox: BoundingBox) -> np.ndarray:
        """Generate heatmap from detection points"""
        # Create intensity map
        intensity_map = np.zeros((self.config.height, self.config.width), dtype=np.float32)
        
        # Add detection points
        for detection in detections:
            # Skip detections outside bounding box
            if not (bbox.south <= detection.lat <= bbox.north and 
                   bbox.west <= detection.lng <= bbox.east):
                continue
            
            # Convert to pixel coordinates
            x, y = self._lat_lng_to_pixel(detection.lat, detection.lng, bbox)
            
            # Calculate weight
            weight = 1.0
            
            if self.config.confidence_weighting:
                weight *= detection.confidence
            
            if self.config.time_decay:
                weight *= self._calculate_time_weight(detection.timestamp)
            
            # Add intensity with radius
            for dy in range(-self.config.radius, self.config.radius + 1):
                for dx in range(-self.config.radius, self.config.radius + 1):
                    px, py = x + dx, y + dy
                    
                    if 0 <= px < self.config.width and 0 <= py < self.config.height:
                        distance = np.sqrt(dx*dx + dy*dy)
                        if distance <= self.config.radius:
                            # Gaussian falloff
                            falloff = np.exp(-(distance*distance) / (2 * (self.config.radius/3)**2))
                            intensity_map[py, px] += weight * falloff * self.config.intensity
        
        # Apply blur
        if self.config.blur_radius > 0 and CV2_AVAILABLE:
            kernel_size = self.config.blur_radius * 2 + 1
            intensity_map = cv2.GaussianBlur(intensity_map, (kernel_size, kernel_size), 0)
        
        # Apply color scheme
        heatmap_rgb = self._apply_color_scheme(intensity_map)
        
        return heatmap_rgb
    
    def generate_web_heatmap_data(self, detections: List[DetectionPoint]) -> List[List[float]]:
        """Generate heatmap data for web-based visualization (Leaflet.heat)"""
        heatmap_data = []
        
        for detection in detections:
            weight = detection.confidence
            
            if self.config.time_decay:
                weight *= self._calculate_time_weight(detection.timestamp)
            
            if self.config.confidence_weighting:
                weight *= detection.confidence
            
            # Format: [lat, lng, intensity]
            heatmap_data.append([detection.lat, detection.lng, weight])
        
        return heatmap_data
    
    def save_heatmap(self, heatmap: np.ndarray, output_path: str, 
                    bbox: BoundingBox = None, metadata: Dict = None):
        """Save heatmap to file with metadata"""
        if not PIL_AVAILABLE:
            logger.error("PIL not available - cannot save heatmap")
            return
        
        # Convert to PIL Image
        image = Image.fromarray(heatmap)
        
        # Save image
        image.save(output_path)
        
        # Save metadata
        if metadata or bbox:
            metadata_path = Path(output_path).with_suffix('.json')
            
            meta_data = {
                "timestamp": datetime.now().isoformat(),
                "config": asdict(self.config),
                "bbox": asdict(bbox) if bbox else None,
                "image_size": [self.config.width, self.config.height]
            }
            
            if metadata:
                meta_data.update(metadata)
            
            with open(metadata_path, 'w') as f:
                json.dump(meta_data, f, indent=2)
        
        logger.info(f"Heatmap saved to {output_path}")
    
    def create_temporal_heatmap(self, detections: List[DetectionPoint], 
                               bbox: BoundingBox, time_windows: List[int]) -> Dict[str, np.ndarray]:
        """Create multiple heatmaps for different time windows"""
        current_time = datetime.now()
        heatmaps = {}
        
        for hours in time_windows:
            # Filter detections by time window
            cutoff_time = current_time - timedelta(hours=hours)
            
            filtered_detections = []
            for detection in detections:
                try:
                    detection_time = datetime.fromisoformat(detection.timestamp.replace('Z', '+00:00'))
                    if detection_time >= cutoff_time:
                        filtered_detections.append(detection)
                except Exception:
                    # Include detections with invalid timestamps
                    filtered_detections.append(detection)
            
            # Generate heatmap
            heatmap = self.generate_heatmap(filtered_detections, bbox)
            heatmaps[f"{hours}h"] = heatmap
        
        return heatmaps
    
    def analyze_hotspots(self, detections: List[DetectionPoint], 
                        bbox: BoundingBox, min_detections: int = 3) -> List[Dict]:
        """Analyze detection hotspots"""
        if len(detections) < min_detections:
            return []
        
        # Create intensity map
        intensity_map = np.zeros((self.config.height, self.config.width), dtype=np.float32)
        
        # Add detection points
        for detection in detections:
            if not (bbox.south <= detection.lat <= bbox.north and 
                   bbox.west <= detection.lng <= bbox.east):
                continue
            
            x, y = self._lat_lng_to_pixel(detection.lat, detection.lng, bbox)
            intensity_map[y, x] += detection.confidence
        
        # Find local maxima
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available - hotspot analysis limited")
            return []
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(intensity_map, (21, 21), 0)
        
        # Find contours of high-intensity areas
        threshold = np.max(blurred) * 0.3  # 30% of max intensity
        _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
        binary = binary.astype(np.uint8)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hotspots = []
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) < 100:  # Skip small areas
                continue
            
            # Get centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Convert back to lat/lng
                lng = bbox.west + (cx / self.config.width) * bbox.width_deg
                lat = bbox.north - (cy / self.config.height) * bbox.height_deg
                
                # Count detections in this area
                area_detections = 0
                for detection in detections:
                    det_x, det_y = self._lat_lng_to_pixel(detection.lat, detection.lng, bbox)
                    if cv2.pointPolygonTest(contour, (det_x, det_y), False) >= 0:
                        area_detections += 1
                
                if area_detections >= min_detections:
                    hotspots.append({
                        "id": i,
                        "center_lat": lat,
                        "center_lng": lng,
                        "detection_count": area_detections,
                        "intensity": float(blurred[cy, cx]),
                        "area_pixels": int(cv2.contourArea(contour))
                    })
        
        # Sort by detection count
        hotspots.sort(key=lambda x: x["detection_count"], reverse=True)
        
        return hotspots
    
    def export_for_web(self, detections: List[DetectionPoint], 
                      output_dir: str, bbox: BoundingBox = None) -> Dict[str, str]:
        """Export heatmap data for web visualization"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate web heatmap data
        web_data = self.generate_web_heatmap_data(detections)
        
        # Save heatmap data
        heatmap_file = output_path / "heatmap_data.json"
        with open(heatmap_file, 'w') as f:
            json.dump({
                "data": web_data,
                "config": {
                    "radius": self.config.radius,
                    "intensity": self.config.intensity,
                    "blur": self.config.blur_radius
                },
                "bbox": asdict(bbox) if bbox else None,
                "timestamp": datetime.now().isoformat(),
                "total_detections": len(detections)
            }, f, indent=2)
        
        # Generate static heatmap if bbox provided
        static_file = None
        if bbox:
            heatmap_image = self.generate_heatmap(detections, bbox)
            static_file = output_path / "heatmap.png"
            self.save_heatmap(heatmap_image, str(static_file), bbox)
        
        # Analyze hotspots
        hotspots = []
        if bbox:
            hotspots = self.analyze_hotspots(detections, bbox)
        
        hotspots_file = output_path / "hotspots.json"
        with open(hotspots_file, 'w') as f:
            json.dump(hotspots, f, indent=2)
        
        return {
            "heatmap_data": str(heatmap_file),
            "static_heatmap": str(static_file) if static_file else None,
            "hotspots": str(hotspots_file)
        }

def create_sample_detections(center_lat: float, center_lng: float, 
                           count: int = 50, radius_deg: float = 0.01) -> List[DetectionPoint]:
    """Create sample detection data for testing"""
    detections = []
    current_time = datetime.now()
    
    for i in range(count):
        # Random position within radius
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(0, radius_deg)
        
        lat = center_lat + distance * np.cos(angle)
        lng = center_lng + distance * np.sin(angle)
        
        # Random timestamp within last 24 hours
        hours_ago = np.random.uniform(0, 24)
        timestamp = (current_time - timedelta(hours=hours_ago)).isoformat()
        
        detections.append(DetectionPoint(
            lat=lat,
            lng=lng,
            confidence=np.random.uniform(0.5, 1.0),
            timestamp=timestamp,
            detection_type="person",
            track_id=i,
            locked=np.random.choice([True, False], p=[0.2, 0.8])
        ))
    
    return detections

if __name__ == "__main__":
    # Example usage
    center_lat, center_lng = 37.7749, -122.4194  # San Francisco
    
    # Create sample detections
    detections = create_sample_detections(center_lat, center_lng, count=100)
    
    # Define area
    bbox = BoundingBox(
        north=center_lat + 0.01,
        south=center_lat - 0.01,
        east=center_lng + 0.01,
        west=center_lng - 0.01
    )
    
    # Generate heatmap
    config = HeatmapConfig(
        width=800,
        height=600,
        radius=20,
        intensity=0.7,
        color_scheme="thermal"
    )
    
    generator = HeatmapGenerator(config)
    
    # Create heatmap
    heatmap = generator.generate_heatmap(detections, bbox)
    
    # Save results
    output_dir = "heatmap_output"
    results = generator.export_for_web(detections, output_dir, bbox)
    
    print("Heatmap generation complete:")
    print(json.dumps(results, indent=2))