#!/usr/bin/env python3
"""
Geolocation Pipeline for Human Detection

Integrates all geolocation components to provide complete geographic positioning:
- Human detection with bounding boxes
- Drone telemetry collection (GPS, attitude, camera)
- Camera intrinsics and calibration
- Ray-casting from 2D pixels to 3D world rays
- Terrain intersection (flat ground or DEM)
- Geographic coordinate conversion
- Accuracy estimation and validation

Provides real-world coordinates for detected humans in drone footage.
"""

import cv2
import numpy as np
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import json

# Import our geolocation components
from .telemetry_service import (
    TelemetryData, CameraIntrinsics, TelemetryService, 
    get_telemetry_service
)
from .camera_calibration import (
    CameraCalibrationService, get_camera_calibration_service
)
from .ray_casting_service import (
    RayCastingService, RaycastResult, get_ray_casting_service
)
from .terrain_intersection_service import (
    TerrainIntersectionService, IntersectionResult, TerrainPoint,
    get_terrain_intersection_service
)
from .detection_pipeline import DetectionPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DetectionGeolocation:
    """Geolocation result for a single detection."""
    
    # Detection information
    detection_id: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_name: str
    
    # Geolocation results
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    elevation: Optional[float] = None
    
    # Accuracy and metadata
    horizontal_accuracy_m: Optional[float] = None
    elevation_accuracy_m: Optional[float] = None
    total_accuracy_m: Optional[float] = None
    
    # Ray-casting details
    pixel_center: Optional[Tuple[float, float]] = None
    ground_distance_m: Optional[float] = None
    viewing_elevation_deg: Optional[float] = None
    viewing_azimuth_deg: Optional[float] = None
    
    # Processing metadata
    terrain_model: Optional[str] = None
    intersection_iterations: Optional[int] = None
    intersection_converged: Optional[bool] = None
    processing_time_ms: Optional[float] = None
    
    def is_valid(self) -> bool:
        """Check if geolocation is valid."""
        return (self.latitude is not None and 
                self.longitude is not None and 
                self.elevation is not None)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def get_accuracy_summary(self) -> str:
        """Get human-readable accuracy summary."""
        if not self.is_valid():
            return "Invalid geolocation"
        
        if self.total_accuracy_m is not None:
            if self.total_accuracy_m < 2.0:
                return f"High accuracy (±{self.total_accuracy_m:.1f}m)"
            elif self.total_accuracy_m < 5.0:
                return f"Good accuracy (±{self.total_accuracy_m:.1f}m)"
            elif self.total_accuracy_m < 10.0:
                return f"Moderate accuracy (±{self.total_accuracy_m:.1f}m)"
            else:
                return f"Low accuracy (±{self.total_accuracy_m:.1f}m)"
        
        return "Accuracy unknown"

@dataclass
class FrameGeolocationResult:
    """Geolocation results for an entire frame."""
    
    # Frame metadata
    frame_id: int
    timestamp: float
    
    # Telemetry data
    telemetry: TelemetryData
    camera_intrinsics: CameraIntrinsics
    
    # Detection geolocations
    detections: List[DetectionGeolocation]
    
    # Processing statistics
    total_detections: int
    valid_geolocations: int
    processing_time_ms: float
    
    # Terrain and accuracy info
    terrain_model_used: Optional[str] = None
    average_accuracy_m: Optional[float] = None
    
    def get_valid_detections(self) -> List[DetectionGeolocation]:
        """Get only valid geolocations."""
        return [d for d in self.detections if d.is_valid()]
    
    def get_accuracy_stats(self) -> Dict[str, float]:
        """Get accuracy statistics for valid detections."""
        valid_detections = self.get_valid_detections()
        
        if not valid_detections:
            return {}
        
        accuracies = [d.total_accuracy_m for d in valid_detections 
                     if d.total_accuracy_m is not None]
        
        if not accuracies:
            return {}
        
        return {
            'mean_accuracy_m': np.mean(accuracies),
            'median_accuracy_m': np.median(accuracies),
            'min_accuracy_m': np.min(accuracies),
            'max_accuracy_m': np.max(accuracies),
            'std_accuracy_m': np.std(accuracies)
        }

class GeolocationPipeline:
    """Main geolocation pipeline integrating all components."""
    
    def __init__(self, 
                 camera_model: str = "O4",
                 resolution: str = "4K",
                 telemetry_source: str = "simulated",
                 terrain_model: Optional[str] = None,
                 config_dir: str = "configs"):
        """
        Initialize geolocation pipeline.
        
        Args:
            camera_model: Camera model ("O4", "Mini3", etc.)
            resolution: Video resolution ("4K", "FHD", etc.)
            telemetry_source: Telemetry source ("dji", "mavlink", "simulated")
            terrain_model: Terrain model name or None for flat terrain
            config_dir: Configuration directory
        """
        self.camera_model = camera_model
        self.resolution = resolution
        self.telemetry_source = telemetry_source
        self.terrain_model = terrain_model
        self.config_dir = Path(config_dir)
        
        # Initialize services
        from .telemetry_service import initialize_telemetry
        self.telemetry_service = initialize_telemetry({
            "sim_rate": 10.0,
            "enable_dji": telemetry_source == "dji",
            "enable_mavlink": telemetry_source == "mavlink"
        })
        
        # Start telemetry collectors
        for collector in self.telemetry_service.collectors.values():
            collector.start()
        
        self.camera_service = get_camera_calibration_service()
        self.ray_service = get_ray_casting_service()
        self.terrain_service = get_terrain_intersection_service()
        
        # Initialize detection pipeline
        self.detection_pipeline = None
        
        # Get camera intrinsics
        self.camera_intrinsics = self.camera_service.get_camera_intrinsics(
            camera_model, resolution
        )
        
        # Statistics
        self.frame_count = 0
        self.total_detections = 0
        self.valid_geolocations = 0
        self.processing_times = []
        
        logger.info(f"Initialized GeolocationPipeline:")
        logger.info(f"  Camera: {camera_model} {resolution}")
        logger.info(f"  Telemetry: {telemetry_source}")
        logger.info(f"  Terrain: {terrain_model or 'flat'}")
    
    def set_detection_pipeline(self, detection_pipeline: DetectionPipeline):
        """Set the detection pipeline.
        
        Args:
            detection_pipeline: DetectionPipeline instance
        """
        self.detection_pipeline = detection_pipeline
        logger.info("Detection pipeline connected")
    
    def load_dem(self, dem_path: str, name: Optional[str] = None) -> bool:
        """Load DEM for terrain intersection.
        
        Args:
            dem_path: Path to DEM file
            name: DEM name (defaults to filename)
            
        Returns:
            True if loaded successfully
        """
        if name is None:
            name = Path(dem_path).stem
        
        success = self.terrain_service.load_dem(name, dem_path)
        if success:
            self.terrain_model = name
            logger.info(f"Loaded DEM: {name}")
        
        return success
    
    def process_frame(self, frame: np.ndarray, 
                     frame_id: int = 0,
                     timestamp: Optional[float] = None) -> FrameGeolocationResult:
        """Process a single frame for geolocation.
        
        Args:
            frame: Input frame (BGR image)
            frame_id: Frame identifier
            timestamp: Frame timestamp (defaults to current time)
            
        Returns:
            FrameGeolocationResult with all detections and geolocations
        """
        start_time = time.time()
        
        if timestamp is None:
            timestamp = time.time()
        
        # Get current telemetry
        telemetry = self.telemetry_service.get_latest_telemetry()
        if telemetry is None:
            logger.warning("No telemetry data available")
            # Create dummy telemetry for testing
            telemetry = TelemetryData(
                timestamp=timestamp,
                latitude=37.7749,
                longitude=-122.4194,
                altitude=100.0,
                roll=0.0,
                pitch=-15.0,
                yaw=0.0,
                gimbal_pitch=-30.0
            )
        
        # Run detection if pipeline is available
        detections = []
        if self.detection_pipeline is not None:
            detection_frame = self.detection_pipeline.process_frame(frame)
            
            # Convert detection results to our format
            for i, human_detection in enumerate(detection_frame.humans):
                detections.append({
                    'id': human_detection.track_id,
                    'bbox': human_detection.bbox,
                    'confidence': human_detection.confidence,
                    'class_name': 'person'
                })
        else:
            # For testing without detection pipeline
            logger.warning("No detection pipeline available, using dummy detection")
            detections = [{
                'id': 0,
                'bbox': (frame.shape[1]//4, frame.shape[0]//4, 
                        3*frame.shape[1]//4, 3*frame.shape[0]//4),
                'confidence': 0.8,
                'class_name': 'person'
            }]
        
        # Process each detection for geolocation
        detection_geolocations = []
        
        for detection in detections:
            geolocation = self._geolocate_detection(
                detection, telemetry, frame_id
            )
            detection_geolocations.append(geolocation)
        
        # Calculate statistics
        valid_count = sum(1 for d in detection_geolocations if d.is_valid())
        processing_time = (time.time() - start_time) * 1000  # ms
        
        # Update global statistics
        self.frame_count += 1
        self.total_detections += len(detections)
        self.valid_geolocations += valid_count
        self.processing_times.append(processing_time)
        
        # Calculate average accuracy
        valid_detections = [d for d in detection_geolocations if d.is_valid()]
        avg_accuracy = None
        if valid_detections:
            accuracies = [d.total_accuracy_m for d in valid_detections 
                         if d.total_accuracy_m is not None]
            if accuracies:
                avg_accuracy = np.mean(accuracies)
        
        return FrameGeolocationResult(
            frame_id=frame_id,
            timestamp=timestamp,
            telemetry=telemetry,
            camera_intrinsics=self.camera_intrinsics,
            detections=detection_geolocations,
            total_detections=len(detections),
            valid_geolocations=valid_count,
            processing_time_ms=processing_time,
            terrain_model_used=self.terrain_model,
            average_accuracy_m=avg_accuracy
        )
    
    def _geolocate_detection(self, detection: Dict[str, Any], 
                           telemetry: TelemetryData,
                           frame_id: int) -> DetectionGeolocation:
        """Geolocate a single detection.
        
        Args:
            detection: Detection dictionary
            telemetry: Telemetry data
            frame_id: Frame identifier
            
        Returns:
            DetectionGeolocation result
        """
        start_time = time.time()
        
        # Extract detection info
        detection_id = detection['id']
        bbox = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class_name']
        
        # Calculate bounding box center
        x1, y1, x2, y2 = bbox
        center_u = (x1 + x2) / 2.0
        center_v = (y1 + y2) / 2.0
        
        try:
            # Cast ray from detection center
            ray_result = self.ray_service.cast_ray(
                center_u, center_v, self.camera_intrinsics, telemetry
            )
            
            # Intersect with terrain
            if self.terrain_model:
                intersection_result = self.terrain_service.intersect_ray_dem(
                    ray_result.world_ray, telemetry.latitude, telemetry.longitude,
                    self.terrain_model
                )
            else:
                intersection_result = self.terrain_service.intersect_ray_flat(
                    ray_result.world_ray, 0.0
                )
            
            # Extract results
            if intersection_result.is_valid():
                terrain_point = intersection_result.intersection_point
                
                # Convert local coordinates to lat/lon for flat terrain
                if terrain_point.latitude == 0.0 and terrain_point.longitude == 0.0:
                    # Flat terrain case - convert from local coordinates
                    local_point = intersection_result.local_point
                    lat, lon = self._local_to_latlon(
                        local_point[0], local_point[1], 
                        telemetry.latitude, telemetry.longitude
                    )
                    terrain_point.latitude = lat
                    terrain_point.longitude = lon
                
                # Calculate viewing angles
                angles = ray_result.get_intersection_angles()
                elevation_angle = angles[0] if angles else None
                azimuth_angle = angles[1] if angles else None
                
                # Estimate accuracy
                ray_accuracy = self.ray_service.estimate_ray_accuracy(ray_result)
                combined_accuracy = self.terrain_service.estimate_intersection_accuracy(
                    intersection_result, ray_accuracy
                )
                
                processing_time = (time.time() - start_time) * 1000
                
                return DetectionGeolocation(
                    detection_id=detection_id,
                    bbox=bbox,
                    confidence=confidence,
                    class_name=class_name,
                    latitude=terrain_point.latitude,
                    longitude=terrain_point.longitude,
                    elevation=terrain_point.elevation,
                    horizontal_accuracy_m=combined_accuracy.get('terrain_horizontal_accuracy_m'),
                    elevation_accuracy_m=combined_accuracy.get('terrain_elevation_accuracy_m'),
                    total_accuracy_m=combined_accuracy.get('combined_error_m'),
                    pixel_center=(center_u, center_v),
                    ground_distance_m=intersection_result.distance,
                    viewing_elevation_deg=elevation_angle,
                    viewing_azimuth_deg=azimuth_angle,
                    terrain_model=self.terrain_model,
                    intersection_iterations=intersection_result.iterations,
                    intersection_converged=intersection_result.converged,
                    processing_time_ms=processing_time
                )
            
        except Exception as e:
            logger.error(f"Geolocation failed for detection {detection_id}: {e}")
        
        # Return invalid geolocation on failure
        processing_time = (time.time() - start_time) * 1000
        return DetectionGeolocation(
            detection_id=detection_id,
            bbox=bbox,
            confidence=confidence,
            class_name=class_name,
            pixel_center=(center_u, center_v),
            processing_time_ms=processing_time
        )
    
    def _local_to_latlon(self, x: float, y: float, 
                        origin_lat: float, origin_lon: float) -> Tuple[float, float]:
        """Convert local coordinates to lat/lon.
        
        Args:
            x: Local x coordinate (meters)
            y: Local y coordinate (meters)
            origin_lat: Origin latitude
            origin_lon: Origin longitude
            
        Returns:
            (latitude, longitude) in degrees
        """
        # Simple conversion (assumes small distances)
        lat_per_meter = 1.0 / 111320.0  # Approximate
        lon_per_meter = 1.0 / (111320.0 * np.cos(np.radians(origin_lat)))
        
        lat = origin_lat + y * lat_per_meter
        lon = origin_lon + x * lon_per_meter
        
        return lat, lon
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline processing statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.processing_times:
            return {}
        
        return {
            'frames_processed': self.frame_count,
            'total_detections': self.total_detections,
            'valid_geolocations': self.valid_geolocations,
            'geolocation_success_rate': self.valid_geolocations / max(self.total_detections, 1),
            'avg_processing_time_ms': np.mean(self.processing_times),
            'min_processing_time_ms': np.min(self.processing_times),
            'max_processing_time_ms': np.max(self.processing_times),
            'total_processing_time_s': sum(self.processing_times) / 1000,
            'avg_detections_per_frame': self.total_detections / max(self.frame_count, 1)
        }
    
    def save_results(self, results: List[FrameGeolocationResult], 
                    output_path: str) -> bool:
        """Save geolocation results to file.
        
        Args:
            results: List of FrameGeolocationResult
            output_path: Output file path
            
        Returns:
            True if saved successfully
        """
        try:
            output_data = {
                'pipeline_config': {
                    'camera_model': self.camera_model,
                    'resolution': self.resolution,
                    'telemetry_source': self.telemetry_source,
                    'terrain_model': self.terrain_model
                },
                'statistics': self.get_statistics(),
                'results': []
            }
            
            for result in results:
                frame_data = {
                    'frame_id': result.frame_id,
                    'timestamp': result.timestamp,
                    'telemetry': asdict(result.telemetry),
                    'detections': [d.to_dict() for d in result.detections],
                    'processing_time_ms': result.processing_time_ms,
                    'accuracy_stats': result.get_accuracy_stats()
                }
                output_data['results'].append(frame_data)
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"Saved geolocation results to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return False

def create_geolocation_pipeline(**kwargs) -> GeolocationPipeline:
    """Create a geolocation pipeline with default settings.
    
    Args:
        **kwargs: Pipeline configuration options
        
    Returns:
        GeolocationPipeline instance
    """
    return GeolocationPipeline(**kwargs)

if __name__ == "__main__":
    # Demo usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Geolocation Pipeline Demo")
    parser.add_argument("--camera", default="O4", help="Camera model")
    parser.add_argument("--resolution", default="4K", help="Resolution")
    parser.add_argument("--dem", help="DEM file path")
    parser.add_argument("--video", help="Input video file")
    parser.add_argument("--output", default="geolocation_results.json", help="Output file")
    parser.add_argument("--frames", type=int, default=10, help="Number of frames to process")
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = create_geolocation_pipeline(
        camera_model=args.camera,
        resolution=args.resolution,
        telemetry_source="simulated"
    )
    
    # Load DEM if provided
    if args.dem:
        pipeline.load_dem(args.dem)
    
    # Process video or generate test frames
    results = []
    
    if args.video:
        # Process video file
        cap = cv2.VideoCapture(args.video)
        frame_id = 0
        
        while frame_id < args.frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            result = pipeline.process_frame(frame, frame_id)
            results.append(result)
            
            print(f"Frame {frame_id}: {result.valid_geolocations}/{result.total_detections} valid geolocations")
            
            frame_id += 1
        
        cap.release()
    else:
        # Generate test frames
        print("Processing test frames (no video provided)...")
        
        for frame_id in range(args.frames):
            # Create dummy frame
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            
            result = pipeline.process_frame(frame, frame_id)
            results.append(result)
            
            print(f"Frame {frame_id}: {result.valid_geolocations}/{result.total_detections} valid geolocations")
    
    # Save results
    pipeline.save_results(results, args.output)
    
    # Print statistics
    stats = pipeline.get_statistics()
    print(f"\nPipeline Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Print sample results
    if results:
        print(f"\nSample Results (Frame 0):")
        for detection in results[0].detections:
            if detection.is_valid():
                print(f"  Detection {detection.detection_id}:")
                print(f"    Location: {detection.latitude:.6f}, {detection.longitude:.6f}")
                print(f"    Elevation: {detection.elevation:.1f} m")
                print(f"    Accuracy: {detection.get_accuracy_summary()}")