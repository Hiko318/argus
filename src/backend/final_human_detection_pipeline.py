#!/usr/bin/env python3
"""
Final Optimized Human Detection Pipeline

Complete human detection pipeline optimized for real-time performance with:
- YOLOv8 model fine-tuned for aerial/SAR datasets
- SORT/DeepSORT tracking with ID assignment
- Performance monitoring and optimization
- Video processing and live camera support
- Edge device optimization (TensorRT/ONNX)

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
from typing import Optional, Dict, Any, List

from src.backend.detection_pipeline import DetectionPipeline, create_pipeline
from src.backend.detector import create_detector
from src.backend.tracker import create_tracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedHumanDetectionPipeline:
    """Optimized human detection pipeline for real-time performance"""
    
    def __init__(self, model_path: str = "yolov8n.pt", device: Optional[str] = None,
                 confidence_threshold: float = 0.5, tracker_type: str = "sort",
                 enable_tensorrt: bool = False, aerial_optimized: bool = False,
                 target_fps: float = 25.0):
        """
        Initialize optimized pipeline
        
        Args:
            model_path: Path to YOLO model (use trained model if available)
            device: Device for inference
            confidence_threshold: Detection confidence threshold (higher = faster)
            tracker_type: Type of tracker ('sort' or 'deepsort')
            enable_tensorrt: Enable TensorRT optimization
            aerial_optimized: Enable aerial/SAR optimizations
            target_fps: Target FPS for real-time performance
        """
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.tracker_type = tracker_type
        self.enable_tensorrt = enable_tensorrt
        self.aerial_optimized = aerial_optimized
        self.target_fps = target_fps
        
        # Performance tracking
        self.frame_times = []
        self.detection_times = []
        self.tracking_times = []
        
        # Initialize pipeline
        self._initialize_pipeline()
        
        logger.info(f"Optimized pipeline initialized with model: {model_path}")
        logger.info(f"Target FPS: {target_fps}, Confidence: {confidence_threshold}")
    
    def _initialize_pipeline(self):
        """Initialize the detection pipeline with optimized settings"""
        try:
            self.pipeline = create_pipeline(
                model_path=self.model_path,
                device=self.device,
                confidence_threshold=self.confidence_threshold,
                iou_threshold=0.4,  # Lower IoU for speed
                tracker_max_disappeared=20,  # Shorter tracking for speed
                tracker_max_distance=80.0,   # Shorter distance for speed
                human_only=True,
                aerial_optimized=self.aerial_optimized,
                enable_tensorrt=self.enable_tensorrt
            )
            
            # Optimize for real-time performance
            self.pipeline.optimize_for_realtime(target_fps=self.target_fps)
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame with timing
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary with detection results and timing info
        """
        start_time = time.time()
        
        # Process frame through pipeline
        detection_start = time.time()
        result = self.pipeline.process_frame(frame)
        detection_time = time.time() - detection_start
        
        total_time = time.time() - start_time
        fps = 1.0 / total_time if total_time > 0 else 0
        
        # Update performance tracking
        self.frame_times.append(total_time)
        self.detection_times.append(detection_time)
        
        # Keep only recent measurements
        if len(self.frame_times) > 100:
            self.frame_times = self.frame_times[-100:]
            self.detection_times = self.detection_times[-100:]
        
        return {
            'detection_result': result,
            'fps': fps,
            'processing_time_ms': total_time * 1000,
            'detection_time_ms': detection_time * 1000,
            'avg_fps': self.get_average_fps(),
            'meets_target': fps >= self.target_fps
        }
    
    def get_average_fps(self) -> float:
        """Get average FPS over recent frames"""
        if not self.frame_times:
            return 0.0
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        if not self.frame_times:
            return {}
        
        fps_values = [1.0 / t for t in self.frame_times if t > 0]
        
        return {
            'avg_fps': sum(fps_values) / len(fps_values) if fps_values else 0,
            'min_fps': min(fps_values) if fps_values else 0,
            'max_fps': max(fps_values) if fps_values else 0,
            'avg_processing_time_ms': sum(self.frame_times) / len(self.frame_times) * 1000,
            'avg_detection_time_ms': sum(self.detection_times) / len(self.detection_times) * 1000,
            'target_fps': self.target_fps,
            'meets_target_pct': sum(1 for fps in fps_values if fps >= self.target_fps) / len(fps_values) * 100 if fps_values else 0,
            'total_frames': len(self.frame_times)
        }
    
    def process_video(self, video_path: str, output_path: Optional[str] = None,
                     max_frames: Optional[int] = None, display: bool = False) -> Dict[str, Any]:
        """
        Process video file with human detection and tracking
        
        Args:
            video_path: Path to input video
            output_path: Path for output video (optional)
            max_frames: Maximum frames to process
            display: Show real-time display
            
        Returns:
            Processing results and statistics
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Setup output video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detections_log = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if max_frames and frame_count >= max_frames:
                    break
                
                # Process frame
                result = self.process_frame(frame)
                detection_result = result['detection_result']
                
                # Draw annotations
                annotated_frame = self.pipeline.draw_annotations(frame, detection_result)
                
                # Add performance info
                fps_text = f"FPS: {result['fps']:.1f} (Avg: {result['avg_fps']:.1f})"
                target_text = f"Target: {self.target_fps} FPS"
                status_text = "✓ REAL-TIME" if result['meets_target'] else "✗ SLOW"
                
                cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, target_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(annotated_frame, status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                           (0, 255, 0) if result['meets_target'] else (0, 0, 255), 2)
                
                # Save frame
                if writer:
                    writer.write(annotated_frame)
                
                # Display frame
                if display:
                    cv2.imshow('Human Detection Pipeline', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Log detection
                detections_log.append({
                    'frame': frame_count,
                    'timestamp': frame_count / fps,
                    'humans_detected': detection_result.total_humans,
                    'fps': result['fps'],
                    'processing_time_ms': result['processing_time_ms']
                })
                
                frame_count += 1
                
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count}/{total_frames if max_frames is None else min(max_frames, total_frames)} frames")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        # Generate final statistics
        stats = self.get_performance_stats()
        stats.update({
            'video_path': video_path,
            'output_path': output_path,
            'frames_processed': frame_count,
            'video_fps': fps,
            'video_resolution': f"{width}x{height}",
            'total_humans_detected': sum(d['humans_detected'] for d in detections_log),
            'detections_log': detections_log
        })
        
        logger.info(f"Video processing completed: {frame_count} frames")
        logger.info(f"Average FPS: {stats['avg_fps']:.1f}, Target: {self.target_fps}")
        logger.info(f"Real-time performance: {stats['meets_target_pct']:.1f}% of frames")
        
        return stats
    
    def benchmark(self, num_frames: int = 100, frame_size: tuple = (640, 640)) -> Dict[str, Any]:
        """
        Run performance benchmark
        
        Args:
            num_frames: Number of frames to test
            frame_size: Size of test frames
            
        Returns:
            Benchmark results
        """
        logger.info(f"Running benchmark with {num_frames} frames of size {frame_size}")
        
        # Generate random test frames
        for i in range(num_frames):
            test_frame = np.random.randint(0, 255, (*frame_size, 3), dtype=np.uint8)
            result = self.process_frame(test_frame)
            
            if (i + 1) % 20 == 0:
                logger.info(f"Benchmark progress: {i+1}/{num_frames} frames, Avg FPS: {result['avg_fps']:.1f}")
        
        stats = self.get_performance_stats()
        logger.info(f"Benchmark completed - Avg FPS: {stats['avg_fps']:.1f}, Target: {self.target_fps}")
        
        return stats

def main():
    parser = argparse.ArgumentParser(description="Optimized Human Detection Pipeline")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLO model")
    parser.add_argument("--device", help="Device to use (cuda/cpu)")
    parser.add_argument("--video", help="Path to test video")
    parser.add_argument("--camera", type=int, help="Camera index for live feed")
    parser.add_argument("--output", help="Path for output video")
    parser.add_argument("--frames", type=int, help="Max frames to process")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--target-fps", type=float, default=25.0, help="Target FPS")
    parser.add_argument("--tensorrt", action="store_true", help="Enable TensorRT optimization")
    parser.add_argument("--aerial", action="store_true", help="Enable aerial optimizations")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--benchmark-frames", type=int, default=100, help="Frames for benchmark")
    parser.add_argument("--display", action="store_true", help="Show real-time display")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Use trained model if available
    model_path = args.model
    if model_path == "yolov8n.pt":
        # Check for trained models
        trained_models_dir = Path("trained_models")
        if trained_models_dir.exists():
            # Find the most recent trained model
            model_dirs = [d for d in trained_models_dir.iterdir() if d.is_dir()]
            if model_dirs:
                latest_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
                trained_model = latest_dir / "yolov8n_aerial_trained.pt"
                if trained_model.exists():
                    model_path = str(trained_model)
                    logger.info(f"Using trained model: {model_path}")
    
    # Initialize pipeline
    pipeline = OptimizedHumanDetectionPipeline(
        model_path=model_path,
        device=args.device,
        confidence_threshold=args.conf,
        tracker_type="sort",
        enable_tensorrt=args.tensorrt,
        aerial_optimized=args.aerial,
        target_fps=args.target_fps
    )
    
    try:
        if args.benchmark:
            # Run benchmark
            stats = pipeline.benchmark(args.benchmark_frames)
            print("\n=== Benchmark Results ===")
            print(f"Average FPS: {stats['avg_fps']:.1f}")
            print(f"Min FPS: {stats['min_fps']:.1f}")
            print(f"Max FPS: {stats['max_fps']:.1f}")
            print(f"Target FPS: {stats['target_fps']:.1f}")
            print(f"Real-time capable: {stats['meets_target_pct']:.1f}% of frames")
            print(f"Average processing time: {stats['avg_processing_time_ms']:.1f} ms")
            
        elif args.video:
            # Process video file
            stats = pipeline.process_video(
                video_path=args.video,
                output_path=args.output,
                max_frames=args.frames,
                display=args.display
            )
            
            # Save results
            results_file = Path(args.output).with_suffix('.json') if args.output else Path('video_results.json')
            with open(results_file, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Results saved to: {results_file}")
            
        elif args.camera is not None:
            # Live camera feed
            cap = cv2.VideoCapture(args.camera)
            if not cap.isOpened():
                raise ValueError(f"Cannot open camera {args.camera}")
            
            logger.info(f"Starting live camera feed from camera {args.camera}")
            logger.info("Press 'q' to quit")
            
            frame_count = 0
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    result = pipeline.process_frame(frame)
                    detection_result = result['detection_result']
                    
                    # Draw annotations
                    annotated_frame = pipeline.draw_annotations(frame, detection_result)
                    
                    # Add performance info
                    fps_text = f"FPS: {result['fps']:.1f} (Avg: {result['avg_fps']:.1f})"
                    status_text = "✓ REAL-TIME" if result['meets_target'] else "✗ SLOW"
                    
                    cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                               (0, 255, 0) if result['meets_target'] else (0, 0, 255), 2)
                    
                    cv2.imshow('Live Human Detection', annotated_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                    frame_count += 1
                    
            finally:
                cap.release()
                cv2.destroyAllWindows()
                
            logger.info(f"Processed {frame_count} frames from live camera")
            
        else:
            # Default: run benchmark
            logger.info("No input specified, running benchmark...")
            stats = pipeline.benchmark(args.benchmark_frames)
            print("\n=== Benchmark Results ===")
            print(f"Average FPS: {stats['avg_fps']:.1f}")
            print(f"Target FPS: {stats['target_fps']:.1f}")
            print(f"Real-time capable: {stats['meets_target_pct']:.1f}% of frames")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
    
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()