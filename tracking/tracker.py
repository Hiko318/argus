#!/usr/bin/env python3
"""
Multi-Object Tracking Module

Implements SORT (Simple Online and Realtime Tracking) and DeepSORT algorithms
for tracking detected humans across video frames with ID assignment.

Features:
- Kalman filter-based motion prediction
- Hungarian algorithm for data association
- Track lifecycle management
- Configurable tracking parameters
- Support for both SORT and DeepSORT modes

Author: Foresight AI Team
Date: 2024
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import time

logger = logging.getLogger(__name__)

class TrackState(Enum):
    """Track state enumeration"""
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    DELETED = "deleted"

@dataclass
class Detection:
    """Detection data structure"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    feature: Optional[np.ndarray] = None  # For DeepSORT
    
    def get_center(self) -> Tuple[float, float]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def get_area(self) -> float:
        """Get area of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

class Track:
    """Individual track for multi-object tracking"""
    
    def __init__(self, detection: Detection, track_id: int, max_age: int = 30):
        """
        Initialize track
        
        Args:
            detection: Initial detection
            track_id: Unique track identifier
            max_age: Maximum frames without detection before deletion
        """
        self.track_id = track_id
        self.max_age = max_age
        self.hits = 1
        self.time_since_update = 0
        self.state = TrackState.TENTATIVE
        self.confidence = detection.confidence
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        
        # Initialize Kalman filter for motion prediction
        self.kf = self._create_kalman_filter()
        self.kf.x[:4] = self._convert_bbox_to_z(detection.bbox)
        
        # Track history
        self.history = [detection.bbox]
        self.feature_history = [detection.feature] if detection.feature is not None else []
        
    def _create_kalman_filter(self) -> KalmanFilter:
        """Create Kalman filter for tracking"""
        # State: [x, y, s, r, dx, dy, ds] where s=scale, r=aspect_ratio
        kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix (constant velocity model)
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement function
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement noise
        kf.R[2:, 2:] *= 10.0
        
        # Process noise
        kf.P[4:, 4:] *= 1000.0
        kf.P *= 10.0
        kf.Q[-1, -1] *= 0.01
        kf.Q[4:, 4:] *= 0.01
        
        return kf
    
    def _convert_bbox_to_z(self, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """Convert bounding box to measurement vector"""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2
        y = y1 + h / 2
        s = w * h  # scale (area)
        r = w / h if h != 0 else 1  # aspect ratio
        return np.array([x, y, s, r]).reshape((4, 1))
    
    def _convert_x_to_bbox(self, x: np.ndarray) -> Tuple[float, float, float, float]:
        """Convert state vector to bounding box"""
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w if w != 0 else 1
        x1 = x[0] - w / 2
        y1 = x[1] - h / 2
        x2 = x1 + w
        y2 = y1 + h
        return (float(x1), float(y1), float(x2), float(y2))
    
    def predict(self):
        """Predict next state using Kalman filter"""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.time_since_update += 1
    
    def update(self, detection: Detection):
        """Update track with new detection"""
        self.time_since_update = 0
        self.hits += 1
        self.confidence = detection.confidence
        
        # Update Kalman filter
        z = self._convert_bbox_to_z(detection.bbox)
        self.kf.update(z)
        
        # Update history
        self.history.append(detection.bbox)
        if detection.feature is not None:
            self.feature_history.append(detection.feature)
        
        # Update state
        if self.state == TrackState.TENTATIVE and self.hits >= 3:
            self.state = TrackState.CONFIRMED
    
    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Get current bounding box prediction"""
        return self._convert_x_to_bbox(self.kf.x)
    
    def mark_missed(self):
        """Mark track as missed (no detection)"""
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.DELETED
        elif self.time_since_update > self.max_age:
            self.state = TrackState.DELETED

def compute_iou(bbox1: Tuple[float, float, float, float], 
                bbox2: Tuple[float, float, float, float]) -> float:
    """Compute Intersection over Union (IoU) of two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def compute_distance(bbox1: Tuple[float, float, float, float],
                    bbox2: Tuple[float, float, float, float]) -> float:
    """Compute Euclidean distance between bbox centers"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    center1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
    center2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
    
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

class Tracker:
    """Multi-object tracker implementing SORT algorithm"""
    
    def __init__(self, max_disappeared: int = 30, max_distance: float = 100.0,
                 min_hits: int = 3, iou_threshold: float = 0.3,
                 use_deep_features: bool = False):
        """
        Initialize tracker
        
        Args:
            max_disappeared: Maximum frames before track deletion
            max_distance: Maximum distance for track association
            min_hits: Minimum hits before track confirmation
            iou_threshold: IoU threshold for data association
            use_deep_features: Enable DeepSORT features
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.use_deep_features = use_deep_features
        
        self.tracks: List[Track] = []
        self.next_id = 1
        self.frame_count = 0
        
        # Statistics
        self.total_tracks_created = 0
        self.total_tracks_deleted = 0
        
        logger.info(f"Tracker initialized - Max disappeared: {max_disappeared}, "
                   f"IoU threshold: {iou_threshold}, DeepSORT: {use_deep_features}")
    
    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detections for current frame
            
        Returns:
            List of active tracks
        """
        self.frame_count += 1
        
        # Predict all tracks
        for track in self.tracks:
            track.predict()
        
        # Associate detections to tracks
        matched_tracks, unmatched_detections, unmatched_tracks = self._associate(
            detections, self.tracks
        )
        
        # Update matched tracks
        for track_idx, detection_idx in matched_tracks:
            self.tracks[track_idx].update(detections[detection_idx])
        
        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            self._create_track(detections[detection_idx])
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # Remove deleted tracks
        self.tracks = [track for track in self.tracks if track.state != TrackState.DELETED]
        
        # Return confirmed tracks
        return [track for track in self.tracks if track.state == TrackState.CONFIRMED]
    
    def _associate(self, detections: List[Detection], tracks: List[Track]
                 ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections to tracks using Hungarian algorithm
        
        Returns:
            Tuple of (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(tracks)))
        
        # Compute cost matrix
        cost_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            for j, detection in enumerate(detections):
                if self.use_deep_features and track.feature_history and detection.feature is not None:
                    # Use feature similarity for DeepSORT
                    feature_cost = self._compute_feature_distance(
                        track.feature_history[-1], detection.feature
                    )
                    iou_cost = 1.0 - compute_iou(track.bbox, detection.bbox)
                    cost_matrix[i, j] = 0.7 * feature_cost + 0.3 * iou_cost
                else:
                    # Use IoU for SORT
                    iou = compute_iou(track.bbox, detection.bbox)
                    cost_matrix[i, j] = 1.0 - iou
        
        # Apply Hungarian algorithm
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches based on threshold
        matched_pairs = []
        for track_idx, detection_idx in zip(track_indices, detection_indices):
            if cost_matrix[track_idx, detection_idx] < (1.0 - self.iou_threshold):
                matched_pairs.append((track_idx, detection_idx))
        
        # Find unmatched detections and tracks
        matched_detection_indices = [pair[1] for pair in matched_pairs]
        matched_track_indices = [pair[0] for pair in matched_pairs]
        
        unmatched_detections = [
            i for i in range(len(detections)) if i not in matched_detection_indices
        ]
        unmatched_tracks = [
            i for i in range(len(tracks)) if i not in matched_track_indices
        ]
        
        return matched_pairs, unmatched_detections, unmatched_tracks
    
    def _compute_feature_distance(self, feature1: np.ndarray, feature2: np.ndarray) -> float:
        """Compute cosine distance between features"""
        norm1 = np.linalg.norm(feature1)
        norm2 = np.linalg.norm(feature2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0
        
        cosine_sim = np.dot(feature1, feature2) / (norm1 * norm2)
        return 1.0 - cosine_sim
    
    def _create_track(self, detection: Detection):
        """Create new track from detection"""
        track = Track(detection, self.next_id, self.max_disappeared)
        self.tracks.append(track)
        self.next_id += 1
        self.total_tracks_created += 1
        
        logger.debug(f"Created new track {track.track_id}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracker statistics"""
        active_tracks = len([t for t in self.tracks if t.state == TrackState.CONFIRMED])
        tentative_tracks = len([t for t in self.tracks if t.state == TrackState.TENTATIVE])
        
        return {
            "frame_count": self.frame_count,
            "active_tracks": active_tracks,
            "tentative_tracks": tentative_tracks,
            "total_tracks_created": self.total_tracks_created,
            "total_tracks_deleted": self.total_tracks_deleted,
            "next_id": self.next_id
        }
    
    def reset(self):
        """Reset tracker state"""
        self.tracks.clear()
        self.next_id = 1
        self.frame_count = 0
        self.total_tracks_created = 0
        self.total_tracks_deleted = 0
        
        logger.info("Tracker reset")

def create_tracker(tracker_type: str = "sort", **kwargs) -> Tracker:
    """
    Factory function to create tracker
    
    Args:
        tracker_type: Type of tracker ('sort' or 'deepsort')
        **kwargs: Additional arguments for tracker
        
    Returns:
        Configured tracker instance
    """
    if tracker_type.lower() == "deepsort":
        kwargs["use_deep_features"] = True
        logger.info("Creating DeepSORT tracker")
    else:
        kwargs["use_deep_features"] = False
        logger.info("Creating SORT tracker")
    
    return Tracker(**kwargs)