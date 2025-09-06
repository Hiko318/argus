"""SORT (Simple Online and Realtime Tracking) implementation.

Provides basic multi-object tracking using Kalman filters and Hungarian algorithm
for data association. Optimized for SAR scenarios with aerial perspectives.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging
from collections import defaultdict

try:
    from scipy.optimize import linear_sum_assignment
    from filterpy.kalman import KalmanFilter
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Install with: pip install scipy filterpy")


@dataclass
class Detection:
    """Detection data structure."""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    features: Optional[np.ndarray] = None  # Optional appearance features
    

@dataclass
class Track:
    """Track data structure."""
    id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    age: int
    hits: int
    time_since_update: int
    state: str  # 'tentative', 'confirmed', 'deleted'
    velocity: Optional[Tuple[float, float]] = None
    features: Optional[np.ndarray] = None
    

class KalmanBoxTracker:
    """Kalman filter for tracking bounding boxes."""
    
    count = 0
    
    def __init__(self, bbox: Tuple[float, float, float, float], class_id: int = 0):
        """Initialize Kalman filter for bounding box tracking."""
        # Define constant velocity model
        # State vector: [x, y, s, r, dx, dy, ds]
        # x, y: center coordinates
        # s: scale (area)
        # r: aspect ratio (width/height)
        # dx, dy, ds: velocities
        
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement function (observe position and scale)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement noise
        self.kf.R[2:, 2:] *= 10.0  # Higher uncertainty for scale measurements
        
        # Process noise
        self.kf.P[4:, 4:] *= 1000.0  # High uncertainty for initial velocities
        self.kf.P *= 10.0
        
        # Process noise covariance
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Convert bbox to state vector
        self.kf.x[:4] = self._bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.class_id = class_id
        
    def update(self, bbox: Tuple[float, float, float, float]):
        """Update the state vector with observed bbox."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._bbox_to_z(bbox))
        
    def predict(self):
        """Advance the state vector and return the predicted bounding box estimate."""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        
        self.time_since_update += 1
        self.history.append(self._z_to_bbox(self.kf.x))
        
        return self.history[-1]
    
    def get_state(self) -> Tuple[float, float, float, float]:
        """Return the current bounding box estimate."""
        return self._z_to_bbox(self.kf.x)
    
    def get_velocity(self) -> Tuple[float, float]:
        """Return the current velocity estimate."""
        return (self.kf.x[4], self.kf.x[5])
    
    def _bbox_to_z(self, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """Convert bounding box to measurement vector."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2.0
        y = y1 + h / 2.0
        s = w * h  # scale (area)
        r = w / h if h != 0 else 1.0  # aspect ratio
        return np.array([x, y, s, r]).reshape((4, 1))
    
    def _z_to_bbox(self, x: np.ndarray) -> Tuple[float, float, float, float]:
        """Convert measurement vector to bounding box."""
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w if w != 0 else 1.0
        x1 = x[0] - w / 2.0
        y1 = x[1] - h / 2.0
        x2 = x[0] + w / 2.0
        y2 = x[1] + h / 2.0
        return (float(x1), float(y1), float(x2), float(y2))


class SORTTracker:
    """SORT tracker implementation."""
    
    def __init__(self, 
                 max_disappeared: int = 3,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3,
                 max_age: int = 30):
        """Initialize SORT tracker.
        
        Args:
            max_disappeared: Maximum frames a track can be unmatched before deletion
            min_hits: Minimum hits before a track is considered confirmed
            iou_threshold: IoU threshold for data association
            max_age: Maximum age of a track before deletion
        """
        self.max_disappeared = max_disappeared
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
        
        # Statistics
        self.total_tracks = 0
        self.active_tracks = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def update(self, detections: List[Detection]) -> List[Track]:
        """Update tracker with new detections.
        
        Args:
            detections: List of detections from current frame
            
        Returns:
            List of active tracks
        """
        self.frame_count += 1
        
        # Get predicted locations of existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        # Convert detections to numpy array
        dets = np.array([[d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence] 
                        for d in detections])
        
        if len(dets) == 0:
            dets = np.empty((0, 5))
        
        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )
        
        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])
        
        # Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            det = detections[i]
            trk = KalmanBoxTracker(det.bbox, det.class_id)
            self.trackers.append(trk)
            self.total_tracks += 1
        
        # Prepare output tracks
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            
            # Determine track state
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                state = 'confirmed'
                ret.append(Track(
                    id=trk.id,
                    bbox=d,
                    confidence=0.8,  # Default confidence for confirmed tracks
                    class_id=trk.class_id,
                    age=trk.age,
                    hits=trk.hits,
                    time_since_update=trk.time_since_update,
                    state=state,
                    velocity=trk.get_velocity()
                ))
            
            i -= 1
            
            # Remove dead tracklets
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        self.active_tracks = len([t for t in ret if t.state == 'confirmed'])
        
        if len(ret) > 0:
            return ret
        return []
    
    def _associate_detections_to_trackers(self, detections: np.ndarray, 
                                        trackers: np.ndarray, 
                                        iou_threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Assign detections to tracked objects using IoU."""
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        
        # Compute IoU matrix
        iou_matrix = self._iou_batch(detections, trackers)
        
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                # Use Hungarian algorithm for optimal assignment
                matched_indices = self._hungarian_assignment(iou_matrix, iou_threshold)
        else:
            matched_indices = np.empty(shape=(0, 2))
        
        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
        
        # Filter out matched with low IoU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
    
    def _hungarian_assignment(self, iou_matrix: np.ndarray, iou_threshold: float) -> np.ndarray:
        """Use Hungarian algorithm for optimal assignment."""
        # Convert IoU to cost (higher IoU = lower cost)
        cost_matrix = 1 - iou_matrix
        
        # Apply threshold
        cost_matrix[iou_matrix < iou_threshold] = 1e6
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Filter valid assignments
        valid_assignments = []
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < 1e6:
                valid_assignments.append([row, col])
        
        return np.array(valid_assignments) if valid_assignments else np.empty((0, 2), dtype=int)
    
    def _iou_batch(self, bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
        """Compute IoU between two sets of bounding boxes."""
        bb_gt = np.expand_dims(bb_gt, 0)
        bb_test = np.expand_dims(bb_test, 1)
        
        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        
        wh = w * h
        o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1]) +
                  (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
        
        return o
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        return {
            'frame_count': self.frame_count,
            'total_tracks': self.total_tracks,
            'active_tracks': self.active_tracks,
            'current_trackers': len(self.trackers)
        }
    
    def reset(self):
        """Reset tracker state."""
        self.trackers.clear()
        self.frame_count = 0
        self.total_tracks = 0
        self.active_tracks = 0
        KalmanBoxTracker.count = 0
        
        self.logger.info("SORT tracker reset")


# Example usage
if __name__ == "__main__":
    # Example usage of SORT tracker
    tracker = SORTTracker(
        max_disappeared=5,
        min_hits=3,
        iou_threshold=0.3
    )
    
    # Simulate some detections
    detections = [
        Detection(bbox=(100, 100, 150, 200), confidence=0.9, class_id=0),
        Detection(bbox=(200, 150, 250, 250), confidence=0.8, class_id=0)
    ]
    
    tracks = tracker.update(detections)
    print(f"Frame 1: {len(tracks)} tracks")
    
    # Simulate movement in next frame
    detections = [
        Detection(bbox=(105, 105, 155, 205), confidence=0.9, class_id=0),
        Detection(bbox=(205, 155, 255, 255), confidence=0.8, class_id=0)
    ]
    
    tracks = tracker.update(detections)
    print(f"Frame 2: {len(tracks)} tracks")
    
    print("SORT tracker example completed successfully!")