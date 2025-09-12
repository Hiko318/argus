#!/usr/bin/env python3
"""
Human Detection Pipeline Demo

Demonstrates the complete human detection pipeline with:
- YOLOv8 model fine-tuned for aerial/SAR datasets
- Real-time inference with optimization
- SORT/DeepSORT tracking with ID assignment
- Performance monitoring and optimization
- Video processing and live camera support

Author: Foresight AI Team
Date: 2024
"""

import argparse
import logging
import time
import cv2
import numpy as np
from pathlib import Path
import json
from typing import Optional, Dict, Any

from src.backend.detection_pipeline import DetectionPipeline, create_pipeline
from src.backend.detector import create_detector
from src.backend.tracker import create_tracker

logger = logging.getLogger(__name__)

class HumanDetectionDemo:
    """Demo class for human detection pipeline"""
    
    def __init__(self, model_path: str = "yolo11n.pt", device: Optional[str] = None,
                 confidence_threshold: float = 0.25, tracker_type: str = "sort",
                 enable_tensorrt: bool = False, aerial_optimized: bool = False):
        """
        Initialize demo
        
        Args:
            model_path: Path to YOLO model (use trained model if available)
            device: Device for inference
            confidence_threshold: Detection confidence threshold
            tracker_type: Type of tracker ('sort' or 'deepsort')
            enable_tensorrt: Enable TensorRT optimization
            aerial_optimized: Enable aerial/SAR optimizations
        """
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.tracker_type = tracker_type
        self.enable_tensorrt = enable_tensorrt
        self.aerial_optimized = aerial_optimized
        
        # Check for trained model
        trained_model_path = Path("trained_models")
        if trained_model_path.exists():
            # Look for the most recent trained model
            model_dirs = [d for d in trained_model_path.iterdir() if d.is_dir() and "aerial" in d.name]
            if model_dirs:
                latest_model_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
                trained_model = latest_model_dir / "yolo11n_aerial_trained.pt"
                if trained_model.exists():
                    self.model_path = str(trained_model)
                    logger.info(f"Using trained aerial model: {self.model_path}")
                    self.aerial_optimized = True
        
        # Initialize pipeline
        self.pipeline = None
        self._initialize_pipeline()
        
        # Demo statistics
        self.demo_stats = {
            "total_frames": 0,
            "total_humans_detected": 0,
            "average_fps": 0.0,
            "peak_fps": 0.0,
            "processing_times": []
        }
    
    def _initialize_pipeline(self):
        """Initialize the detection pipeline"""
        try:
            self.pipeline = DetectionPipeline(
                model_path=self.model_path,
                device=self.device,
                confidence_threshold=self.confidence_threshold,
                human_only=True,
                aerial_optimized=self.aerial_optimized,
                enable_tensorrt=self.enable_tensorrt
            )
            logger.info(f"Pipeline initialized successfully with model: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def process_video_file(self, video_path: str, output_path: Optional[str] = None,
                          max_frames: Optional[int] = None, show_preview: bool = False) -> Dict[str, Any]:
        """
        Process video file with human detection and tracking
        
        Args:
            video_path: Path to input video
            output_path: Optional output video path
            max_frames: Maximum frames to process
            show_preview: Show live preview window
            
        Returns:
            Processing results and statistics
        """
        logger.info(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        # Setup output video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Processing loop
        frame_count = 0
        detection_results = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret or (max_frames and frame_count >= max_frames):
                    break
                
                # Process frame
                start_time = time.time()
                detection_frame = self.pipeline.process_frame(frame, frame_count)
                processing_time = time.time() - start_time
                
                # Update statistics
                self.demo_stats["total_frames"] += 1
                self.demo_stats["total_humans_detected"] += detection_frame.total_humans
                self.demo_stats["processing_times"].append(processing_time)
                
                current_fps = 1.0 / processing_time if processing_time > 0 else 0
                self.demo_stats["peak_fps"] = max(self.demo_stats["peak_fps"], current_fps)
                
                detection_results.append(detection_frame)
                
                # Draw annotations
                annotated_frame = self.pipeline.draw_annotations(
                    frame, detection_frame, show_ids=True, show_confidence=True
                )
                
                # Add performance info
                perf_text = f"FPS: {current_fps:.1f} | Humans: {detection_frame.total_humans} | Frame: {frame_count+1}/{total_frames}"
                cv2.putText(annotated_frame, perf_text, (10, height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Write to output video
                if writer:
                    writer.write(annotated_frame)
                
                # Show preview
                if show_preview:
                    cv2.imshow('Human Detection Demo', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
                
                # Progress update
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    avg_fps = np.mean([1.0/t for t in self.demo_stats["processing_times"][-30:]])
                    logger.info(f"Progress: {progress:.1f}% | Avg FPS: {avg_fps:.1f} | Humans detected: {detection_frame.total_humans}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if show_preview:
                cv2.destroyAllWindows()
        
        # Calculate final statistics
        if self.demo_stats["processing_times"]:
            avg_processing_time = np.mean(self.demo_stats["processing_times"])
            self.demo_stats["average_fps"] = 1.0 / avg_processing_time
        
        # Get pipeline statistics
        pipeline_stats = self.pipeline.get_performance_stats()
        
        results = {
            "video_info": {
                "path": video_path,
                "fps": fps,
                "resolution": (width, height),
                "total_frames": total_frames,
                "frames_processed": frame_count
            },
            "detection_results": detection_results,
            "demo_statistics": self.demo_stats,
            "pipeline_statistics": pipeline_stats,
            "performance_summary": {
                "real_time_capable": self.demo_stats["average_fps"] >= 25.0,
                "optimization_needed": self.demo_stats["average_fps"] < 15.0,
                "total_humans_detected": self.demo_stats["total_humans_detected"],
                "detection_rate": self.demo_stats["total_humans_detected"] / frame_count if frame_count > 0 else 0
            }
        }
        
        logger.info(f"Video processing completed: {frame_count} frames, {self.demo_stats['total_humans_detected']} humans detected")
        logger.info(f"Performance: Avg FPS: {self.demo_stats['average_fps']:.1f}, Peak FPS: {self.demo_stats['peak_fps']:.1f}")
        
        return results
    
    def process_camera_feed(self, camera_id: int = 0, duration: Optional[float] = None,
                           show_preview: bool = True) -> Dict[str, Any]:
        """
        Process live camera feed
        
        Args:
            camera_id: Camera device ID
            duration: Duration in seconds (None for infinite)
            show_preview: Show live preview window
            
        Returns:
            Processing results and statistics
        """
        logger.info(f"Starting camera feed processing (Camera ID: {camera_id})")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera: {camera_id}")
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        start_time = time.time()
        frame_count = 0
        detection_results = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check duration limit
                if duration and (time.time() - start_time) > duration:
                    break
                
                # Process frame
                detection_frame = self.pipeline.process_frame(frame, frame_count)
                detection_results.append(detection_frame)
                
                # Update statistics
                self.demo_stats["total_frames"] += 1
                self.demo_stats["total_humans_detected"] += detection_frame.total_humans
                
                # Draw annotations
                annotated_frame = self.pipeline.draw_annotations(
                    frame, detection_frame, show_ids=True, show_confidence=True, show_center=True
                )
                
                # Add real-time performance info
                current_fps = 1000.0 / detection_frame.processing_time_ms
                perf_text = f"LIVE | FPS: {current_fps:.1f} | Humans: {detection_frame.total_humans}"
                cv2.putText(annotated_frame, perf_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show preview
                if show_preview:
                    cv2.imshow('Live Human Detection', annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):  # Reset tracker
                        self.pipeline.reset()
                        logger.info("Tracker reset")
                    elif key == ord('o'):  # Optimize for real-time
                        opt_result = self.pipeline.optimize_for_realtime()
                        logger.info(f"Optimization applied: {opt_result}")
                
                frame_count += 1
                
                # Periodic statistics update
                if frame_count % 60 == 0:  # Every 2 seconds at 30 FPS
                    stats = self.pipeline.get_performance_stats()
                    logger.info(f"Live stats - FPS: {stats['current_fps']:.1f}, Active tracks: {stats['active_tracks']}")
        
        finally:
            cap.release()
            if show_preview:
                cv2.destroyAllWindows()
        
        # Calculate final statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        results = {
            "camera_info": {
                "camera_id": camera_id,
                "duration": total_time,
                "frames_processed": frame_count
            },
            "detection_results": detection_results,
            "performance_summary": {
                "average_fps": avg_fps,
                "total_humans_detected": self.demo_stats["total_humans_detected"],
                "real_time_performance": avg_fps >= 25.0
            },
            "pipeline_statistics": self.pipeline.get_performance_stats()
        }
        
        logger.info(f"Camera processing completed: {frame_count} frames in {total_time:.1f}s (Avg FPS: {avg_fps:.1f})")
        
        return results
    
    def benchmark_performance(self, test_frames: int = 100) -> Dict[str, Any]:
        """
        Benchmark pipeline performance with synthetic frames
        
        Args:
            test_frames: Number of test frames to process
            
        Returns:
            Benchmark results
        """
        logger.info(f"Running performance benchmark with {test_frames} frames")
        
        # Generate synthetic test frames with human-like shapes
        frame_times = []
        detection_counts = []
        
        for i in range(test_frames):
            # Create synthetic frame
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Add human-like rectangles
            num_humans = np.random.randint(0, 5)
            for j in range(num_humans):
                x = np.random.randint(50, 550)
                y = np.random.randint(50, 400)
                w = np.random.randint(30, 80)
                h = np.random.randint(60, 120)
                color = (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255))
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
            
            # Process frame
            start_time = time.time()
            detection_frame = self.pipeline.process_frame(frame, i)
            processing_time = time.time() - start_time
            
            frame_times.append(processing_time)
            detection_counts.append(detection_frame.total_humans)
        
        # Calculate statistics
        avg_processing_time = np.mean(frame_times)
        avg_fps = 1.0 / avg_processing_time
        min_fps = 1.0 / max(frame_times)
        max_fps = 1.0 / min(frame_times)
        
        benchmark_results = {
            "test_configuration": {
                "test_frames": test_frames,
                "model_path": self.model_path,
                "device": self.device,
                "tracker_type": self.tracker_type,
                "tensorrt_enabled": self.enable_tensorrt
            },
            "performance_metrics": {
                "average_fps": avg_fps,
                "min_fps": min_fps,
                "max_fps": max_fps,
                "average_processing_time_ms": avg_processing_time * 1000,
                "frame_time_std_ms": np.std(frame_times) * 1000
            },
            "detection_metrics": {
                "average_detections_per_frame": np.mean(detection_counts),
                "total_detections": sum(detection_counts),
                "detection_rate": sum(detection_counts) / test_frames
            },
            "real_time_assessment": {
                "real_time_capable_30fps": avg_fps >= 30.0,
                "real_time_capable_25fps": avg_fps >= 25.0,
                "optimization_recommended": avg_fps < 15.0
            },
            "pipeline_statistics": self.pipeline.get_performance_stats()
        }
        
        logger.info(f"Benchmark completed - Avg FPS: {avg_fps:.1f}, Min: {min_fps:.1f}, Max: {max_fps:.1f}")
        
        return benchmark_results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save results to JSON file"""
        with open(output_path, 'w') as f:
            # Convert numpy arrays and other non-serializable objects
            serializable_results = self._make_serializable(results)
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Results saved to: {output_path}")
    
    def _make_serializable(self, obj):
        """Convert object to JSON serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj

def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="Human Detection Pipeline Demo")
    parser.add_argument("--model", default="yolo11n.pt", help="Path to YOLO model")
    parser.add_argument("--device", help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--video", help="Path to test video")
    parser.add_argument("--camera", type=int, help="Camera device ID for live feed")
    parser.add_argument("--output", help="Path for output video")
    parser.add_argument("--results", help="Path to save results JSON")
    parser.add_argument("--frames", type=int, help="Max frames to process")
    parser.add_argument("--duration", type=float, help="Duration for camera feed (seconds)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--tracker", choices=["sort", "deepsort"], default="sort", help="Tracker type")
    parser.add_argument("--tensorrt", action="store_true", help="Enable TensorRT optimization")
    parser.add_argument("--aerial", action="store_true", help="Enable aerial optimizations")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--preview", action="store_true", help="Show preview window")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize demo
        demo = HumanDetectionDemo(
            model_path=args.model,
            device=args.device,
            confidence_threshold=args.conf,
            tracker_type=args.tracker,
            enable_tensorrt=args.tensorrt,
            aerial_optimized=args.aerial
        )
        
        results = None
        
        if args.benchmark:
            # Run performance benchmark
            results = demo.benchmark_performance()
            
        elif args.video:
            # Process video file
            results = demo.process_video_file(
                video_path=args.video,
                output_path=args.output,
                max_frames=args.frames,
                show_preview=args.preview
            )
            
        elif args.camera is not None:
            # Process camera feed
            results = demo.process_camera_feed(
                camera_id=args.camera,
                duration=args.duration,
                show_preview=args.preview
            )
            
        else:
            # Default: run benchmark
            logger.info("No input specified, running performance benchmark")
            results = demo.benchmark_performance()
        
        # Save results if requested
        if args.results and results:
            demo.save_results(results, args.results)
        
        # Print summary
        if results:
            if "performance_summary" in results:
                summary = results["performance_summary"]
                logger.info(f"\n=== DEMO SUMMARY ===")
                logger.info(f"Real-time capable: {summary.get('real_time_performance', 'N/A')}")
                logger.info(f"Average FPS: {summary.get('average_fps', 'N/A'):.1f}")
                logger.info(f"Total humans detected: {summary.get('total_humans_detected', 'N/A')}")
            
            if "real_time_assessment" in results:
                assessment = results["real_time_assessment"]
                logger.info(f"30 FPS capable: {assessment.get('real_time_capable_30fps', 'N/A')}")
                logger.info(f"25 FPS capable: {assessment.get('real_time_capable_25fps', 'N/A')}")
                logger.info(f"Optimization needed: {assessment.get('optimization_recommended', 'N/A')}")
        
        logger.info("Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

if __name__ == "__main__":
    main()