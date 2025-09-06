#!/usr/bin/env python3
"""
Detection Pipeline Integration

Combines YOLOv8 detector with multi-object tracker for complete pipeline.
Provides unified interface for detection and tracking.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from datetime import datetime

from src.backend.detector import YOLODetector, DetectionResult, create_detector
from src.backend.tracker import Tracker, Track, Detection, create_tracker, TrackState

logger = logging.getLogger(__name__)

@dataclass
class HumanDetection:
    """Represents a detected human with tracking information"""
    track_id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    center: Tuple[float, float]  # center x, y
    area: float
    timestamp: float
    frame_id: int
    is_new: bool = False
    is_lost: bool = False
    track_length: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def get_center_point(self) -> Tuple[int, int]:
        """Get center point as integers for drawing"""
        return (int(self.center[0]), int(self.center[1]))
    
    def get_bbox_ints(self) -> Tuple[int, int, int, int]:
        """Get bounding box as integers for drawing"""
        return (int(self.bbox[0]), int(self.bbox[1]), int(self.bbox[2]), int(self.bbox[3]))

@dataclass
class DetectionFrame:
    """Represents detection results for a single frame"""
    frame_id: int
    timestamp: float
    humans: List[HumanDetection]
    total_humans: int
    processing_time_ms: float
    frame_shape: Tuple[int, int, int]  # H, W, C
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'frame_id': self.frame_id,
            'timestamp': self.timestamp,
            'humans': [h.to_dict() for h in self.humans],
            'total_humans': self.total_humans,
            'processing_time_ms': self.processing_time_ms,
            'frame_shape': self.frame_shape
        }

class DetectionPipeline:
    """Integrated human detection and tracking pipeline"""
    
    def __init__(self, model_path: str = "yolov8n.pt", device: Optional[str] = None,
                 confidence_threshold: float = 0.25, iou_threshold: float = 0.45,
                 tracker_max_disappeared: int = 30, tracker_max_distance: float = 100.0,
                 human_only: bool = True, aerial_optimized: bool = False, enable_tensorrt: bool = False):
        """
        Initialize human detection pipeline
        
        Args:
            model_path: Path to YOLOv8 model
            device: Device for inference ('cuda', 'cpu', or None for auto)
            confidence_threshold: Detection confidence threshold
            iou_threshold: Detection IoU threshold
            tracker_max_disappeared: Max frames before track deletion
            tracker_max_distance: Max distance for track association
            human_only: Only detect humans (person class)
            aerial_optimized: Enable aerial/SAR optimizations
            enable_tensorrt: Enable TensorRT optimization
        """
        # Initialize detector with human-specific settings
        self.detector = YOLODetector(
            model_path=model_path,
            device=device,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold
        )
        
        self.tracker = create_tracker(
            tracker_type="sort",
            max_disappeared=tracker_max_disappeared,
            max_distance=tracker_max_distance,
            iou_threshold=0.3
        )
        
        # Configuration
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.human_only = human_only
        self.aerial_optimized = aerial_optimized
        self.enable_tensorrt = enable_tensorrt
        
        # Performance tracking
        self.frame_count = 0
        self.total_detection_time = 0.0
        self.total_tracking_time = 0.0
        self.total_processing_time = 0.0
        self.fps_history = []
        self.max_history_size = 100
        self.detection_history = []
        
        # Statistics
        self.total_humans_detected = 0
        self.active_tracks = 0
        self.lost_tracks = 0
        
        logger.info(f"Human detection pipeline initialized - Model: {model_path}, Human-only: {human_only}, Aerial: {aerial_optimized}")
    
    def process_frame(self, frame: np.ndarray, frame_id: int = None, timestamp: float = None) -> DetectionFrame:
        """
        Process a single frame through human detection and tracking pipeline
        
        Args:
            frame: Input frame (BGR format)
            frame_id: Optional frame ID (auto-incremented if None)
            timestamp: Optional timestamp (current time if None)
            
        Returns:
            DetectionFrame containing comprehensive human detection results
        """
        start_time = time.time()
        
        # Set defaults
        if frame_id is None:
            frame_id = self.frame_count
        if timestamp is None:
            timestamp = time.time()
        
        # Run detection
        detection_start = time.time()
        detection_result = self.detector.detect(frame)
        detection_time = time.time() - detection_start
        self.total_detection_time += detection_time
        
        # Convert to tracker format (only humans if human_only is enabled)
        detections = []
        for i in range(len(detection_result)):
            class_id = int(detection_result.classes[i])
            
            # Skip non-human detections if human_only is enabled
            if self.human_only and class_id != 0:  # 0 is person class in COCO
                continue
            
            detection = Detection(
                bbox=detection_result.boxes[i],
                confidence=detection_result.scores[i],
                class_id=class_id,
                class_name=detection_result.class_names[class_id]
            )
            detections.append(detection)
        
        # Run tracking
        tracking_start = time.time()
        tracks = self.tracker.update(detections)
        tracking_time = time.time() - tracking_start
        self.total_tracking_time += tracking_time
        
        # Get current tracks and convert to HumanDetection objects
        humans = []
        
        for track in tracks:
            if track.state == TrackState.CONFIRMED:
                x1, y1, x2, y2 = track.bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                area = (x2 - x1) * (y2 - y1)
                
                human = HumanDetection(
                    track_id=track.track_id,
                    bbox=(x1, y1, x2, y2),
                    confidence=track.confidence,
                    center=(center_x, center_y),
                    area=area,
                    timestamp=timestamp,
                    frame_id=frame_id,
                    is_new=(track.hits == 1),
                    is_lost=(track.time_since_update > 0),
                    track_length=track.hits
                )
                humans.append(human)
        
        total_time = time.time() - start_time
        processing_time_ms = total_time * 1000
        
        # Update statistics
        self.frame_count += 1
        self.total_processing_time += total_time
        self.total_humans_detected += len(detections)
        self.active_tracks = len([h for h in humans if not h.is_lost])
        self.lost_tracks = len([h for h in humans if h.is_lost])
        
        # Update FPS history
        current_fps = 1.0 / total_time if total_time > 0 else 0
        self.fps_history.append(current_fps)
        if len(self.fps_history) > self.max_history_size:
            self.fps_history.pop(0)
        
        # Create detection frame result
        detection_frame = DetectionFrame(
            frame_id=frame_id,
            timestamp=timestamp,
            humans=humans,
            total_humans=len(humans),
            processing_time_ms=processing_time_ms,
            frame_shape=frame.shape
        )
        
        # Store in history
        self.detection_history.append(detection_frame)
        if len(self.detection_history) > self.max_history_size:
            self.detection_history.pop(0)
        
        return detection_frame
    
    def draw_annotations(self, frame: np.ndarray, detection_frame: DetectionFrame, 
                        show_ids: bool = True, show_confidence: bool = True, 
                        show_center: bool = False) -> np.ndarray:
        """
        Draw detection and tracking annotations on frame
        
        Args:
            frame: Input frame (BGR format)
            detection_frame: Detection results
            show_ids: Whether to show track IDs
            show_confidence: Whether to show confidence scores
            show_center: Whether to show center points
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Color scheme for different track states
        colors = {
            'new': (0, 255, 0),      # Green for new tracks
            'active': (255, 0, 0),   # Blue for active tracks
            'lost': (0, 0, 255)      # Red for lost tracks
        }
        
        for human in detection_frame.humans:
            x1, y1, x2, y2 = human.get_bbox_ints()
            
            # Determine color based on track state
            if human.is_new:
                color = colors['new']
                status = "NEW"
            elif human.is_lost:
                color = colors['lost']
                status = "LOST"
            else:
                color = colors['active']
                status = "ACTIVE"
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label_parts = []
            if show_ids:
                label_parts.append(f"ID:{human.track_id}")
            if show_confidence:
                label_parts.append(f"{human.confidence:.2f}")
            label_parts.append(status)
            
            label = " | ".join(label_parts)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw center point if requested
            if show_center:
                center = human.get_center_point()
                cv2.circle(annotated, center, 3, color, -1)
        
        # Draw frame info
        info_text = f"Frame: {detection_frame.frame_id} | Humans: {detection_frame.total_humans} | FPS: {1000/detection_frame.processing_time_ms:.1f}"
        cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics
        
        Returns:
            Dictionary containing performance metrics
        """
        if self.frame_count == 0:
            return {"error": "No frames processed yet"}
        
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        avg_processing_time = self.total_processing_time / self.frame_count * 1000
        
        return {
            "frames_processed": self.frame_count,
            "total_humans_detected": self.total_humans_detected,
            "active_tracks": self.active_tracks,
            "lost_tracks": self.lost_tracks,
            "average_fps": avg_fps,
            "current_fps": self.fps_history[-1] if self.fps_history else 0,
            "average_processing_time_ms": avg_processing_time,
            "total_processing_time_s": self.total_processing_time,
            "detection_efficiency": self.total_humans_detected / self.frame_count if self.frame_count > 0 else 0,
            "real_time_capable": avg_fps >= 25.0,
            "model_info": {
                "human_only": self.human_only,
                "aerial_optimized": self.aerial_optimized,
                "tensorrt_enabled": self.enable_tensorrt
            }
        }
    
    def optimize_for_realtime(self, target_fps: float = 30.0) -> Dict[str, Any]:
        """
        Optimize pipeline settings for real-time performance
        
        Args:
            target_fps: Target FPS for optimization
            
        Returns:
            Dictionary with optimization results
        """
        current_fps = np.mean(self.fps_history) if self.fps_history else 0
        
        optimizations = []
        
        if current_fps < target_fps:
            # Reduce confidence threshold slightly to reduce processing
            if self.confidence_threshold > 0.3:
                old_conf = self.confidence_threshold
                self.confidence_threshold = max(0.3, self.confidence_threshold - 0.1)
                self.detector.set_thresholds(confidence=self.confidence_threshold)
                optimizations.append(f"Reduced confidence threshold: {old_conf:.2f} -> {self.confidence_threshold:.2f}")
            
            # Increase IoU threshold to reduce overlapping detections
            if self.iou_threshold < 0.6:
                old_iou = self.iou_threshold
                self.iou_threshold = min(0.6, self.iou_threshold + 0.05)
                self.detector.set_thresholds(iou=self.iou_threshold)
                optimizations.append(f"Increased IoU threshold: {old_iou:.2f} -> {self.iou_threshold:.2f}")
            
            # Reduce tracker sensitivity
            if self.tracker.max_disappeared > 10:
                old_max_disappeared = self.tracker.max_disappeared
                self.tracker.max_disappeared = max(10, self.tracker.max_disappeared - 5)
                optimizations.append(f"Reduced max disappeared frames: {old_max_disappeared} -> {self.tracker.max_disappeared}")
        
        return {
            "current_fps": current_fps,
            "target_fps": target_fps,
            "optimizations_applied": optimizations,
            "performance_improved": len(optimizations) > 0
        }
    
    def reset(self):
        """
        Reset pipeline state
        """
        self.tracker.reset()
        self.frame_count = 0
        self.total_processing_time = 0.0
        self.detection_history.clear()
        self.fps_history.clear()
        self.total_humans_detected = 0
        self.active_tracks = 0
        self.lost_tracks = 0
        logger.info("Detection pipeline reset")
    
    def process_video(self, video_path: str, output_path: Optional[str] = None, 
                     show_progress: bool = True, max_frames: int = None) -> List[DetectionFrame]:
        """
        Process entire video file with human detection and tracking
        
        Args:
            video_path: Path to input video
            output_path: Optional path for annotated output video
            show_progress: Whether to show processing progress
            max_frames: Maximum number of frames to process (None for all)
            
        Returns:
            List of DetectionFrame results for each frame
        """
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
        
        # Setup output video writer if needed
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        results = []
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret or (max_frames and frame_count >= max_frames):
                    break
                
                # Process frame
                timestamp = frame_count / fps  # Video timestamp
                detection_frame = self.process_frame(frame, frame_count, timestamp)
                results.append(detection_frame)
                
                # Draw annotations if output video is requested
                if writer is not None:
                    annotated_frame = self.draw_annotations(frame, detection_frame)
                    writer.write(annotated_frame)
                
                frame_count += 1
                
                if show_progress and frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    avg_fps = np.mean(self.fps_history[-30:]) if len(self.fps_history) >= 30 else 0
                    logger.info(f"Processing: {progress:.1f}% ({frame_count}/{total_frames}) | Avg FPS: {avg_fps:.1f}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
        
        logger.info(f"Processed {frame_count} frames from {video_path}")
        return results
    

    


def create_pipeline(**kwargs) -> DetectionPipeline:
    """Factory function to create detection pipeline"""
    return DetectionPipeline(**kwargs)

def demo_pipeline():
    """Demo function for testing the complete pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Detection Pipeline Demo")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLOv8 model")
    parser.add_argument("--device", help="Device to use (cuda/cpu)")
    parser.add_argument("--video", help="Path to test video")
    parser.add_argument("--output", help="Path for output video")
    parser.add_argument("--frames", type=int, help="Max frames to process")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create pipeline
    pipeline = DetectionPipeline(
        model_path=args.model,
        device=args.device,
        confidence_threshold=args.conf
    )
    
    if args.video:
        # Process video
        logger.info(f"Processing video: {args.video}")
        results = pipeline.process_video(args.video, args.output, args.frames)
        
        # Print statistics
        stats = pipeline.get_statistics()
        logger.info(f"Pipeline statistics: {stats}")
    else:
        # Process dummy frames
        logger.info("Processing dummy frames")
        
        for i in range(10):
            # Create dummy frame
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Add some simple shapes to detect
            cv2.rectangle(frame, (100 + i*10, 100), (200 + i*10, 200), (255, 255, 255), -1)
            cv2.rectangle(frame, (300 + i*5, 150), (400 + i*5, 250), (128, 128, 128), -1)
            
            result = pipeline.process_frame(frame)
            
            logger.info(f"Frame {i+1}: {result['detections']['count']} detections, "
                       f"{result['tracks']['count']} tracks, "
                       f"FPS: {result['timing']['fps']:.2f}")
        
        # Print final statistics
        stats = pipeline.get_statistics()
        logger.info(f"Final statistics: {stats}")

if __name__ == "__main__":
    demo_pipeline()