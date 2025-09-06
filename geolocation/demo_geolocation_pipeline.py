#!/usr/bin/env python3
"""
Geolocation Pipeline Demo

Demonstrates the complete human detection and geolocation pipeline:
1. Human detection using YOLOv8
2. Multi-object tracking with SORT/DeepSORT
3. Drone telemetry collection (GPS, attitude, camera)
4. Ray-casting from 2D detections to 3D world coordinates
5. Terrain intersection (flat ground or DEM)
6. Geographic coordinate output (latitude/longitude)
7. Real-time performance monitoring
8. Accuracy estimation and validation

Usage:
    python demo_geolocation_pipeline.py --video input.mp4 --output results.json
    python demo_geolocation_pipeline.py --camera 0 --live  # Live camera
    python demo_geolocation_pipeline.py --benchmark --frames 100  # Benchmark mode
"""

import cv2
import numpy as np
import time
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import our pipeline components
from src.backend.detection_pipeline import DetectionPipeline, create_pipeline
from src.backend.geolocation_pipeline import (
    GeolocationPipeline, FrameGeolocationResult, DetectionGeolocation,
    create_geolocation_pipeline
)
from src.backend.telemetry_service import TelemetryData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GeolocationDemo:
    """Complete geolocation pipeline demonstration."""
    
    def __init__(self, 
                 model_path: str = "models/yolov8n.pt",
                 camera_model: str = "O4",
                 resolution: str = "4K",
                 telemetry_source: str = "simulated",
                 terrain_model: Optional[str] = None,
                 confidence_threshold: float = 0.5,
                 device: str = "cpu"):
        """
        Initialize the geolocation demo.
        
        Args:
            model_path: Path to YOLO model
            camera_model: Camera model ("O4", "Mini3")
            resolution: Video resolution ("4K", "FHD")
            telemetry_source: Telemetry source ("dji", "mavlink", "simulated")
            terrain_model: DEM name or None for flat terrain
            confidence_threshold: Detection confidence threshold
            device: Processing device ("cpu", "cuda")
        """
        self.model_path = model_path
        self.camera_model = camera_model
        self.resolution = resolution
        self.telemetry_source = telemetry_source
        self.terrain_model = terrain_model
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Initialize pipelines
        self.detection_pipeline = None
        self.geolocation_pipeline = None
        
        # Results storage
        self.frame_results: List[FrameGeolocationResult] = []
        
        # Performance tracking
        self.frame_times = []
        self.detection_times = []
        self.geolocation_times = []
        
        # Display settings
        self.show_display = True
        self.save_frames = False
        self.output_dir = Path("output")
        
        logger.info(f"Initialized GeolocationDemo:")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Camera: {camera_model} {resolution}")
        logger.info(f"  Device: {device}")
    
    def initialize_pipelines(self) -> bool:
        """Initialize detection and geolocation pipelines.
        
        Returns:
            True if initialization successful
        """
        try:
            # Initialize detection pipeline
            logger.info("Initializing detection pipeline...")
            self.detection_pipeline = create_pipeline(
                model_path=self.model_path,
                device=self.device,
                confidence_threshold=self.confidence_threshold,
                human_only=True,
                aerial_optimized=True
            )
            
            # Initialize geolocation pipeline
            logger.info("Initializing geolocation pipeline...")
            self.geolocation_pipeline = create_geolocation_pipeline(
                camera_model=self.camera_model,
                resolution=self.resolution,
                telemetry_source=self.telemetry_source,
                terrain_model=self.terrain_model
            )
            
            # Connect pipelines
            self.geolocation_pipeline.set_detection_pipeline(self.detection_pipeline)
            
            logger.info("Pipelines initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            return False
    
    def load_dem(self, dem_path: str) -> bool:
        """Load Digital Elevation Model.
        
        Args:
            dem_path: Path to DEM file
            
        Returns:
            True if loaded successfully
        """
        if self.geolocation_pipeline is None:
            logger.error("Geolocation pipeline not initialized")
            return False
        
        return self.geolocation_pipeline.load_dem(dem_path)
    
    def process_video(self, video_path: str, 
                     max_frames: Optional[int] = None,
                     skip_frames: int = 0) -> List[FrameGeolocationResult]:
        """Process video file for geolocation.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process
            skip_frames: Frames to skip between processing
            
        Returns:
            List of frame results
        """
        if not self.initialize_pipelines():
            return []
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
        
        if max_frames is None:
            max_frames = total_frames
        
        frame_id = 0
        processed_frames = 0
        
        try:
            while processed_frames < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if requested
                if frame_id % (skip_frames + 1) != 0:
                    frame_id += 1
                    continue
                
                # Process frame
                start_time = time.time()
                
                result = self.geolocation_pipeline.process_frame(
                    frame, frame_id, time.time()
                )
                
                frame_time = (time.time() - start_time) * 1000
                self.frame_times.append(frame_time)
                
                self.frame_results.append(result)
                
                # Display progress
                if processed_frames % 10 == 0 or processed_frames < 10:
                    self._print_frame_summary(result, frame_time)
                
                # Show frame if requested
                if self.show_display:
                    display_frame = self._draw_detections(frame, result)
                    cv2.imshow('Geolocation Demo', display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        self.save_frames = not self.save_frames
                        logger.info(f"Frame saving: {'ON' if self.save_frames else 'OFF'}")
                
                # Save frame if requested
                if self.save_frames:
                    self._save_frame(display_frame, frame_id)
                
                frame_id += 1
                processed_frames += 1
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        logger.info(f"Processed {processed_frames} frames")
        return self.frame_results
    
    def process_camera(self, camera_id: int = 0,
                      max_frames: Optional[int] = None) -> List[FrameGeolocationResult]:
        """Process live camera feed.
        
        Args:
            camera_id: Camera device ID
            max_frames: Maximum frames to process
            
        Returns:
            List of frame results
        """
        if not self.initialize_pipelines():
            return []
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error(f"Cannot open camera: {camera_id}")
            return []
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info("Starting live camera processing (press 'q' to quit)")
        
        frame_id = 0
        
        try:
            while max_frames is None or frame_id < max_frames:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read camera frame")
                    continue
                
                # Process frame
                start_time = time.time()
                
                result = self.geolocation_pipeline.process_frame(
                    frame, frame_id, time.time()
                )
                
                frame_time = (time.time() - start_time) * 1000
                self.frame_times.append(frame_time)
                
                self.frame_results.append(result)
                
                # Display frame
                display_frame = self._draw_detections(frame, result)
                cv2.imshow('Live Geolocation Demo', display_frame)
                
                # Show frame info
                if frame_id % 30 == 0:  # Every second at 30 FPS
                    self._print_frame_summary(result, frame_time)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_frames = not self.save_frames
                    logger.info(f"Frame saving: {'ON' if self.save_frames else 'OFF'}")
                
                if self.save_frames:
                    self._save_frame(display_frame, frame_id)
                
                frame_id += 1
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        logger.info(f"Processed {frame_id} camera frames")
        return self.frame_results
    
    def benchmark(self, frames: int = 100) -> Dict[str, Any]:
        """Run benchmark without display.
        
        Args:
            frames: Number of frames to benchmark
            
        Returns:
            Benchmark results
        """
        if not self.initialize_pipelines():
            return {}
        
        logger.info(f"Running benchmark for {frames} frames...")
        
        # Disable display for benchmark
        original_show = self.show_display
        self.show_display = False
        
        # Generate test frames
        test_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        start_time = time.time()
        
        for frame_id in range(frames):
            result = self.geolocation_pipeline.process_frame(
                test_frame, frame_id, time.time()
            )
            self.frame_results.append(result)
            
            if frame_id % 20 == 0:
                progress = (frame_id + 1) / frames * 100
                print(f"\rBenchmark progress: {progress:.1f}%", end="", flush=True)
        
        total_time = time.time() - start_time
        print()  # New line
        
        # Restore display setting
        self.show_display = original_show
        
        # Calculate statistics
        avg_fps = frames / total_time
        pipeline_stats = self.geolocation_pipeline.get_statistics()
        
        benchmark_results = {
            'total_frames': frames,
            'total_time_s': total_time,
            'average_fps': avg_fps,
            'pipeline_stats': pipeline_stats
        }
        
        logger.info(f"Benchmark completed: {avg_fps:.1f} FPS")
        return benchmark_results
    
    def _draw_detections(self, frame: np.ndarray, 
                        result: FrameGeolocationResult) -> np.ndarray:
        """Draw detections and geolocation info on frame.
        
        Args:
            frame: Input frame
            result: Frame geolocation result
            
        Returns:
            Annotated frame
        """
        display_frame = frame.copy()
        
        for detection in result.detections:
            # Draw bounding box
            x1, y1, x2, y2 = [int(x) for x in detection.bbox]
            
            # Color based on geolocation validity
            if detection.is_valid():
                color = (0, 255, 0)  # Green for valid
                thickness = 2
            else:
                color = (0, 0, 255)  # Red for invalid
                thickness = 1
            
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw detection info
            label = f"ID:{detection.detection_id} {detection.confidence:.2f}"
            cv2.putText(display_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw geolocation info if valid
            if detection.is_valid():
                geo_text = f"Lat:{detection.latitude:.6f}"
                cv2.putText(display_frame, geo_text, (x1, y2 + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                geo_text2 = f"Lon:{detection.longitude:.6f}"
                cv2.putText(display_frame, geo_text2, (x1, y2 + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                if detection.total_accuracy_m is not None:
                    acc_text = f"±{detection.total_accuracy_m:.1f}m"
                    cv2.putText(display_frame, acc_text, (x1, y2 + 45),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw frame info
        info_text = f"Frame {result.frame_id}: {result.valid_geolocations}/{result.total_detections} valid"
        cv2.putText(display_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        fps_text = f"Processing: {result.processing_time_ms:.1f}ms"
        cv2.putText(display_frame, fps_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw telemetry info
        telem = result.telemetry
        telem_text = f"GPS: {telem.latitude:.6f}, {telem.longitude:.6f}, {telem.altitude:.1f}m"
        cv2.putText(display_frame, telem_text, (10, display_frame.shape[0] - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        attitude_text = f"Attitude: Y{telem.yaw:.1f}° P{telem.pitch:.1f}° R{telem.roll:.1f}° G{telem.gimbal_pitch:.1f}°"
        cv2.putText(display_frame, attitude_text, (10, display_frame.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        terrain_text = f"Terrain: {result.terrain_model_used or 'flat'}"
        cv2.putText(display_frame, terrain_text, (10, display_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return display_frame
    
    def _print_frame_summary(self, result: FrameGeolocationResult, frame_time: float):
        """Print frame processing summary.
        
        Args:
            result: Frame result
            frame_time: Processing time in ms
        """
        valid_detections = result.get_valid_detections()
        
        print(f"Frame {result.frame_id:4d}: "
              f"{len(valid_detections):2d}/{result.total_detections:2d} valid, "
              f"{frame_time:6.1f}ms")
        
        # Print geolocation details for valid detections
        for detection in valid_detections[:3]:  # Show first 3
            print(f"  ID {detection.detection_id}: "
                  f"{detection.latitude:.6f}, {detection.longitude:.6f} "
                  f"({detection.get_accuracy_summary()})")
    
    def _save_frame(self, frame: np.ndarray, frame_id: int):
        """Save annotated frame to disk.
        
        Args:
            frame: Frame to save
            frame_id: Frame identifier
        """
        self.output_dir.mkdir(exist_ok=True)
        output_path = self.output_dir / f"frame_{frame_id:06d}.jpg"
        cv2.imwrite(str(output_path), frame)
    
    def save_results(self, output_path: str) -> bool:
        """Save all results to file.
        
        Args:
            output_path: Output file path
            
        Returns:
            True if saved successfully
        """
        if self.geolocation_pipeline is None:
            logger.error("No geolocation pipeline available")
            return False
        
        return self.geolocation_pipeline.save_results(self.frame_results, output_path)
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.frame_results:
            return {}
        
        # Calculate frame-level statistics
        total_detections = sum(r.total_detections for r in self.frame_results)
        total_valid = sum(r.valid_geolocations for r in self.frame_results)
        
        processing_times = [r.processing_time_ms for r in self.frame_results]
        
        # Calculate accuracy statistics
        all_accuracies = []
        for result in self.frame_results:
            for detection in result.get_valid_detections():
                if detection.total_accuracy_m is not None:
                    all_accuracies.append(detection.total_accuracy_m)
        
        stats = {
            'frames_processed': len(self.frame_results),
            'total_detections': total_detections,
            'valid_geolocations': total_valid,
            'geolocation_success_rate': total_valid / max(total_detections, 1),
            'avg_processing_time_ms': np.mean(processing_times) if processing_times else 0,
            'avg_fps': 1000 / np.mean(processing_times) if processing_times else 0,
            'avg_detections_per_frame': total_detections / len(self.frame_results)
        }
        
        if all_accuracies:
            stats.update({
                'avg_accuracy_m': np.mean(all_accuracies),
                'median_accuracy_m': np.median(all_accuracies),
                'min_accuracy_m': np.min(all_accuracies),
                'max_accuracy_m': np.max(all_accuracies)
            })
        
        # Add pipeline statistics if available
        if self.geolocation_pipeline:
            pipeline_stats = self.geolocation_pipeline.get_statistics()
            stats['pipeline_stats'] = pipeline_stats
        
        return stats

def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Geolocation Pipeline Demo")
    
    # Input options
    parser.add_argument("--video", help="Input video file")
    parser.add_argument("--camera", type=int, help="Camera device ID")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark mode")
    
    # Processing options
    parser.add_argument("--frames", type=int, default=100, help="Max frames to process")
    parser.add_argument("--skip", type=int, default=0, help="Frames to skip between processing")
    parser.add_argument("--model", default="models/yolov8n.pt", help="YOLO model path")
    parser.add_argument("--device", default="cpu", help="Processing device")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold")
    
    # Camera and geolocation options
    parser.add_argument("--camera-model", default="O4", help="Camera model (O4, Mini3)")
    parser.add_argument("--resolution", default="4K", help="Video resolution (4K, FHD)")
    parser.add_argument("--telemetry", default="simulated", help="Telemetry source")
    parser.add_argument("--dem", help="DEM file path")
    
    # Output options
    parser.add_argument("--output", default="geolocation_results.json", help="Output file")
    parser.add_argument("--no-display", action="store_true", help="Disable video display")
    parser.add_argument("--save-frames", action="store_true", help="Save annotated frames")
    
    args = parser.parse_args()
    
    # Create demo instance
    demo = GeolocationDemo(
        model_path=args.model,
        camera_model=args.camera_model,
        resolution=args.resolution,
        telemetry_source=args.telemetry,
        confidence_threshold=args.confidence,
        device=args.device
    )
    
    # Configure display
    demo.show_display = not args.no_display
    demo.save_frames = args.save_frames
    
    # Load DEM if provided
    if args.dem:
        if not demo.load_dem(args.dem):
            logger.error("Failed to load DEM")
            return
    
    # Run processing
    results = []
    
    if args.benchmark:
        logger.info("Running benchmark mode")
        benchmark_results = demo.benchmark(args.frames)
        print(f"\nBenchmark Results:")
        for key, value in benchmark_results.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    if isinstance(v, float):
                        print(f"    {k}: {v:.3f}")
                    else:
                        print(f"    {k}: {v}")
            elif isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    
    elif args.video:
        logger.info(f"Processing video: {args.video}")
        results = demo.process_video(args.video, args.frames, args.skip)
    
    elif args.camera is not None:
        logger.info(f"Processing camera: {args.camera}")
        results = demo.process_camera(args.camera, args.frames)
    
    else:
        logger.error("No input specified. Use --video, --camera, or --benchmark")
        return
    
    # Save results
    if results:
        if demo.save_results(args.output):
            logger.info(f"Results saved to {args.output}")
        
        # Print summary statistics
        stats = demo.get_summary_statistics()
        print(f"\nProcessing Summary:")
        for key, value in stats.items():
            if key != 'pipeline_stats':
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
        
        # Print sample results
        if results and results[0].detections:
            print(f"\nSample Geolocation Results (Frame 0):")
            for detection in results[0].get_valid_detections()[:3]:
                print(f"  Detection {detection.detection_id}:")
                print(f"    Location: {detection.latitude:.6f}, {detection.longitude:.6f}")
                print(f"    Elevation: {detection.elevation:.1f} m")
                print(f"    Accuracy: {detection.get_accuracy_summary()}")
                print(f"    Distance: {detection.ground_distance_m:.1f} m")
    
    logger.info("Demo completed")

if __name__ == "__main__":
    main()