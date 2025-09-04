#!/usr/bin/env python3
"""
Human Detection Pipeline Demo

Demonstration script for the human detection and tracking pipeline.
Shows real-time detection capabilities, performance metrics, and visualization.

Usage:
    python -m src.backend.detection_demo --source webcam
    python -m src.backend.detection_demo --source video --input test_video.mp4
    python -m src.backend.detection_demo --source image --input test_image.jpg

Author: Foresight AI Team
Date: 2024
"""

import argparse
import cv2
import logging
import time
import json
from pathlib import Path
from typing import Optional

from src.backend.detection_pipeline import DetectionPipeline, create_pipeline
from src.backend.detector import YOLODetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DetectionDemo:
    """Human detection pipeline demonstration"""
    
    def __init__(self, model_path: str = "models/yolov8s.pt", 
                 confidence: float = 0.5, aerial_mode: bool = False):
        """
        Initialize detection demo
        
        Args:
            model_path: Path to YOLO model
            confidence: Detection confidence threshold
            aerial_mode: Enable aerial optimizations
        """
        self.model_path = model_path
        self.confidence = confidence
        self.aerial_mode = aerial_mode
        
        # Initialize pipeline
        logger.info(f"Initializing detection pipeline - Model: {model_path}, Aerial: {aerial_mode}")
        self.pipeline = DetectionPipeline(
            model_path=model_path,
            confidence_threshold=confidence,
            human_only=True,
            aerial_optimized=aerial_mode,
            enable_tensorrt=False  # Can be enabled for production
        )
        
        # Demo statistics
        self.total_frames = 0
        self.total_humans = 0
        self.start_time = time.time()
        
        logger.info("Detection demo initialized successfully")
    
    def demo_webcam(self, camera_id: int = 0, display: bool = True, 
                   save_output: bool = False, output_path: str = "demo_output.mp4"):
        """
        Run demo with webcam input
        
        Args:
            camera_id: Camera device ID
            display: Whether to display video window
            save_output: Whether to save annotated video
            output_path: Path for output video
        """
        logger.info(f"Starting webcam demo - Camera: {camera_id}")
        
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        logger.info(f"Camera resolution: {width}x{height} @ {fps} FPS")
        
        # Setup video writer if needed
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            logger.info(f"Saving output to: {output_path}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    break
                
                # Process frame
                detection_frame = self.pipeline.process_frame(frame)
                
                # Update statistics
                self.total_frames += 1
                self.total_humans += detection_frame.total_humans
                
                # Draw annotations
                annotated_frame = self.pipeline.draw_annotations(
                    frame, detection_frame, 
                    show_ids=True, show_confidence=True, show_center=True
                )
                
                # Add demo statistics
                self._add_demo_stats(annotated_frame, detection_frame)
                
                # Save frame if requested
                if writer:
                    writer.write(annotated_frame)
                
                # Display frame
                if display:
                    cv2.imshow('Human Detection Demo', annotated_frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("Demo stopped by user")
                        break
                    elif key == ord('s'):
                        # Save screenshot
                        screenshot_path = f"screenshot_{int(time.time())}.jpg"
                        cv2.imwrite(screenshot_path, annotated_frame)
                        logger.info(f"Screenshot saved: {screenshot_path}")
                    elif key == ord('r'):
                        # Reset pipeline
                        self.pipeline.reset()
                        logger.info("Pipeline reset")
                    elif key == ord('o'):
                        # Optimize for real-time
                        result = self.pipeline.optimize_for_realtime()
                        logger.info(f"Optimization result: {result}")
                
                # Print periodic stats
                if self.total_frames % 30 == 0:
                    self._print_stats()
        
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            
            self._print_final_stats()
    
    def demo_video(self, video_path: str, display: bool = True, 
                  save_output: bool = False, output_path: str = "demo_output.mp4",
                  max_frames: Optional[int] = None):
        """
        Run demo with video file input
        
        Args:
            video_path: Path to input video
            display: Whether to display video window
            save_output: Whether to save annotated video
            output_path: Path for output video
            max_frames: Maximum frames to process
        """
        logger.info(f"Starting video demo - Input: {video_path}")
        
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Process video
        results = self.pipeline.process_video(
            video_path=video_path,
            output_path=output_path if save_output else None,
            show_progress=True,
            max_frames=max_frames
        )
        
        logger.info(f"Video processing completed - {len(results)} frames processed")
        
        # Display results if requested
        if display and results:
            self._display_video_results(video_path, results)
        
        # Print final statistics
        self._print_final_stats()
        
        return results
    
    def demo_image(self, image_path: str, display: bool = True, 
                  save_output: bool = False, output_path: str = "demo_output.jpg"):
        """
        Run demo with single image input
        
        Args:
            image_path: Path to input image
            display: Whether to display image window
            save_output: Whether to save annotated image
            output_path: Path for output image
        """
        logger.info(f"Starting image demo - Input: {image_path}")
        
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Process image
        detection_frame = self.pipeline.process_frame(frame)
        
        # Update statistics
        self.total_frames = 1
        self.total_humans = detection_frame.total_humans
        
        # Draw annotations
        annotated_frame = self.pipeline.draw_annotations(
            frame, detection_frame,
            show_ids=True, show_confidence=True, show_center=True
        )
        
        # Add demo statistics
        self._add_demo_stats(annotated_frame, detection_frame)
        
        # Save output if requested
        if save_output:
            cv2.imwrite(output_path, annotated_frame)
            logger.info(f"Annotated image saved: {output_path}")
        
        # Display image
        if display:
            cv2.imshow('Human Detection Demo - Image', annotated_frame)
            logger.info("Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Print results
        logger.info(f"Detection results: {detection_frame.total_humans} humans detected")
        for i, human in enumerate(detection_frame.humans):
            logger.info(f"  Human {i+1}: ID={human.track_id}, Confidence={human.confidence:.2f}, Area={human.area:.0f}px")
        
        return detection_frame
    
    def _add_demo_stats(self, frame, detection_frame):
        """Add demo statistics overlay to frame"""
        # Get performance stats
        stats = self.pipeline.get_performance_stats()
        
        # Demo runtime
        runtime = time.time() - self.start_time
        
        # Create stats text
        stats_lines = [
            f"Demo Runtime: {runtime:.1f}s",
            f"Total Frames: {self.total_frames}",
            f"Total Humans: {self.total_humans}",
            f"Avg FPS: {stats.get('average_fps', 0):.1f}",
            f"Real-time: {'YES' if stats.get('real_time_capable', False) else 'NO'}"
        ]
        
        # Draw stats overlay
        y_offset = frame.shape[0] - 150
        for i, line in enumerate(stats_lines):
            y_pos = y_offset + (i * 25)
            cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def _print_stats(self):
        """Print current statistics"""
        stats = self.pipeline.get_performance_stats()
        runtime = time.time() - self.start_time
        
        logger.info(f"Stats - Frames: {self.total_frames}, Humans: {self.total_humans}, "
                   f"Avg FPS: {stats.get('average_fps', 0):.1f}, Runtime: {runtime:.1f}s")
    
    def _print_final_stats(self):
        """Print final demo statistics"""
        stats = self.pipeline.get_performance_stats()
        runtime = time.time() - self.start_time
        
        logger.info("=== DEMO COMPLETED ===")
        logger.info(f"Total Runtime: {runtime:.2f} seconds")
        logger.info(f"Total Frames Processed: {self.total_frames}")
        logger.info(f"Total Humans Detected: {self.total_humans}")
        logger.info(f"Average FPS: {stats.get('average_fps', 0):.2f}")
        logger.info(f"Average Processing Time: {stats.get('average_processing_time_ms', 0):.2f}ms")
        logger.info(f"Real-time Capable: {'YES' if stats.get('real_time_capable', False) else 'NO'}")
        logger.info(f"Detection Efficiency: {stats.get('detection_efficiency', 0):.2f} humans/frame")
        
        # Save stats to file
        stats_file = f"demo_stats_{int(time.time())}.json"
        with open(stats_file, 'w') as f:
            json.dump({
                "demo_config": {
                    "model_path": self.model_path,
                    "confidence": self.confidence,
                    "aerial_mode": self.aerial_mode
                },
                "results": {
                    "runtime_seconds": runtime,
                    "total_frames": self.total_frames,
                    "total_humans": self.total_humans
                },
                "performance": stats
            }, f, indent=2)
        
        logger.info(f"Demo statistics saved to: {stats_file}")
    
    def _display_video_results(self, video_path: str, results):
        """Display video processing results"""
        logger.info("Displaying video results...")
        
        # Open original video for playback
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        frame_delay = int(1000 / fps)  # Delay in milliseconds
        
        frame_idx = 0
        
        try:
            while frame_idx < len(results):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get detection results for this frame
                detection_frame = results[frame_idx]
                
                # Draw annotations
                annotated_frame = self.pipeline.draw_annotations(
                    frame, detection_frame,
                    show_ids=True, show_confidence=True
                )
                
                # Add frame info
                info_text = f"Frame {frame_idx + 1}/{len(results)} | Press 'q' to quit, SPACE to pause"
                cv2.putText(annotated_frame, info_text, (10, annotated_frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow('Video Results Playback', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(frame_delay) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):  # Spacebar to pause
                    cv2.waitKey(0)  # Wait for any key to continue
                
                frame_idx += 1
        
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="Human Detection Pipeline Demo")
    parser.add_argument("--source", choices=["webcam", "video", "image"], 
                       default="webcam", help="Input source type")
    parser.add_argument("--input", help="Input file path (for video/image sources)")
    parser.add_argument("--model", default="models/yolov8s.pt", help="YOLO model path")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--aerial", action="store_true", help="Enable aerial optimizations")
    parser.add_argument("--no-display", action="store_true", help="Disable video display")
    parser.add_argument("--save-output", action="store_true", help="Save annotated output")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device ID")
    parser.add_argument("--max-frames", type=int, help="Maximum frames to process (video only)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.source in ["video", "image"] and not args.input:
        parser.error(f"--input is required for {args.source} source")
    
    # Set default output paths
    if args.save_output and not args.output:
        if args.source == "webcam":
            args.output = "webcam_demo_output.mp4"
        elif args.source == "video":
            args.output = "video_demo_output.mp4"
        elif args.source == "image":
            args.output = "image_demo_output.jpg"
    
    try:
        # Initialize demo
        demo = DetectionDemo(
            model_path=args.model,
            confidence=args.confidence,
            aerial_mode=args.aerial
        )
        
        # Run appropriate demo
        if args.source == "webcam":
            demo.demo_webcam(
                camera_id=args.camera_id,
                display=not args.no_display,
                save_output=args.save_output,
                output_path=args.output or "webcam_demo_output.mp4"
            )
        
        elif args.source == "video":
            demo.demo_video(
                video_path=args.input,
                display=not args.no_display,
                save_output=args.save_output,
                output_path=args.output or "video_demo_output.mp4",
                max_frames=args.max_frames
            )
        
        elif args.source == "image":
            demo.demo_image(
                image_path=args.input,
                display=not args.no_display,
                save_output=args.save_output,
                output_path=args.output or "image_demo_output.jpg"
            )
        
        logger.info("Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

if __name__ == "__main__":
    main()