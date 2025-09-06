"""
DeepSORT Tracker Implementation for Foresight SAR System

This module implements DeepSORT tracking with appearance features for robust
multi-object tracking in SAR scenarios. Combines motion prediction with
appearance embeddings for improved tracking accuracy.
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine

from .sort_tracker import Detection, Track, KalmanBoxTracker


@dataclass
class AppearanceFeature:
    """Appearance feature for re-identification"""
    embedding: np.ndarray
    timestamp: float
    confidence: float = 1.0
    
    def __post_init__(self):
        # Normalize embedding
        if np.linalg.norm(self.embedding) > 0:
            self.embedding = self.embedding / np.linalg.norm(self.embedding)


@dataclass
class DeepSORTTrack(Track):
    """Extended track with appearance features"""
    features: deque = field(default_factory=lambda: deque(maxlen=100))
    confirmed: bool = False
    time_since_update: int = 0
    hits: int = 0
    hit_streak: int = 0
    age: int = 0
    
    def add_feature(self, feature: AppearanceFeature):
        """Add appearance feature to track"""
        self.features.append(feature)
    
    def get_feature_distance(self, feature: AppearanceFeature, 
                           max_distance: float = 0.2) -> float:
        """Calculate appearance distance to track"""
        if not self.features:
            return max_distance
        
        # Use recent features for comparison
        recent_features = list(self.features)[-10:]
        distances = []
        
        for track_feature in recent_features:
            # Weight by recency and confidence
            age_weight = np.exp(-0.1 * (feature.timestamp - track_feature.timestamp))
            conf_weight = track_feature.confidence
            
            distance = cosine(feature.embedding, track_feature.embedding)
            weighted_distance = distance / (age_weight * conf_weight + 1e-6)
            distances.append(weighted_distance)
        
        return min(distances) if distances else max_distance


class FeatureExtractor:
    """Appearance feature extractor for re-identification"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.input_size = (128, 256)  # Standard ReID input size
        
        # Initialize feature extractor
        self._init_model()
    
    def _init_model(self):
        """Initialize the ReID model"""
        try:
            if self.model_path:
                # Load custom ReID model
                import torch
                self.model = torch.jit.load(self.model_path)
                self.model.eval()
            else:
                # Use simple CNN features as fallback
                self.model = self._create_simple_extractor()
        except Exception as e:
            logging.warning(f"Failed to load ReID model: {e}. Using simple features.")
            self.model = self._create_simple_extractor()
    
    def _create_simple_extractor(self):
        """Create simple feature extractor using OpenCV"""
        # Use ORB features as simple appearance descriptor
        return cv2.ORB_create(nfeatures=500)
    
    def extract_features(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract appearance features from detection"""
        x1, y1, x2, y2 = bbox
        
        # Crop detection region
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(512)  # Return zero vector for invalid crops
        
        # Resize to standard input size
        crop_resized = cv2.resize(crop, self.input_size)
        
        if hasattr(self.model, 'forward'):  # PyTorch model
            return self._extract_deep_features(crop_resized)
        else:  # OpenCV ORB
            return self._extract_orb_features(crop_resized)
    
    def _extract_deep_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features using deep learning model"""
        try:
            import torch
            
            # Preprocess image
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0)
            
            with torch.no_grad():
                features = self.model(image_tensor)
                return features.squeeze().numpy()
        except Exception as e:
            logging.warning(f"Deep feature extraction failed: {e}")
            return self._extract_orb_features(image)
    
    def _extract_orb_features(self, image: np.ndarray) -> np.ndarray:
        """Extract ORB features as fallback"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        keypoints, descriptors = self.model.detectAndCompute(gray, None)
        
        if descriptors is not None:
            # Aggregate descriptors into fixed-size vector
            feature_vector = np.mean(descriptors, axis=0)
            # Pad or truncate to 512 dimensions
            if len(feature_vector) < 512:
                feature_vector = np.pad(feature_vector, (0, 512 - len(feature_vector)))
            else:
                feature_vector = feature_vector[:512]
        else:
            # Use histogram features as last resort
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            feature_vector = np.pad(hist.flatten(), (0, 512 - len(hist.flatten())))
        
        return feature_vector


class DeepSORTTracker:
    """DeepSORT tracker with appearance features"""
    
    def __init__(self, 
                 max_disappeared: int = 30,
                 max_distance: float = 0.3,
                 n_init: int = 3,
                 max_iou_distance: float = 0.7,
                 max_age: int = 70,
                 feature_extractor_path: Optional[str] = None):
        """
        Initialize DeepSORT tracker
        
        Args:
            max_disappeared: Maximum frames a track can be unmatched
            max_distance: Maximum cosine distance for appearance matching
            n_init: Number of consecutive detections before track is confirmed
            max_iou_distance: Maximum IoU distance for association
            max_age: Maximum age of track before deletion
            feature_extractor_path: Path to ReID model
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.n_init = n_init
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        
        self.tracks: List[DeepSORTTrack] = []
        self.next_id = 1
        self.frame_count = 0
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(feature_extractor_path)
        
        # Statistics
        self.stats = {
            'total_tracks': 0,
            'active_tracks': 0,
            'confirmed_tracks': 0,
            'lost_tracks': 0
        }
    
    def update(self, detections: List[Detection], image: np.ndarray) -> List[DeepSORTTrack]:
        """Update tracker with new detections"""
        self.frame_count += 1
        
        # Extract features for all detections
        detection_features = []
        for det in detections:
            bbox = (int(det.x1), int(det.y1), int(det.x2), int(det.y2))
            feature_vec = self.feature_extractor.extract_features(image, bbox)
            feature = AppearanceFeature(
                embedding=feature_vec,
                timestamp=self.frame_count,
                confidence=det.confidence
            )
            detection_features.append(feature)
        
        # Predict existing tracks
        for track in self.tracks:
            track.kalman_filter.predict()
        
        # Associate detections to tracks
        matched, unmatched_dets, unmatched_trks = self._associate(
            detections, detection_features, self.tracks
        )
        
        # Update matched tracks
        for m in matched:
            det_idx, trk_idx = m
            detection = detections[det_idx]
            feature = detection_features[det_idx]
            track = self.tracks[trk_idx]
            
            # Update Kalman filter
            track.kalman_filter.update(detection.to_xyxy())
            
            # Add appearance feature
            track.add_feature(feature)
            
            # Update track state
            track.time_since_update = 0
            track.hits += 1
            track.hit_streak += 1
            
            # Confirm track if enough consecutive hits
            if track.hit_streak >= self.n_init:
                track.confirmed = True
        
        # Create new tracks for unmatched detections
        for i in unmatched_dets:
            detection = detections[i]
            feature = detection_features[i]
            
            # Create new track
            kalman_filter = KalmanBoxTracker(detection.to_xyxy())
            track = DeepSORTTrack(
                id=self.next_id,
                kalman_filter=kalman_filter,
                class_id=detection.class_id,
                confidence=detection.confidence
            )
            track.add_feature(feature)
            track.hits = 1
            track.hit_streak = 1
            track.age = 1
            
            self.tracks.append(track)
            self.next_id += 1
            self.stats['total_tracks'] += 1
        
        # Update unmatched tracks
        for i in unmatched_trks:
            track = self.tracks[i]
            track.time_since_update += 1
            track.hit_streak = 0
        
        # Remove old tracks
        self.tracks = [
            track for track in self.tracks
            if track.time_since_update < self.max_age
        ]
        
        # Update statistics
        self._update_stats()
        
        # Return confirmed tracks
        return [track for track in self.tracks if track.confirmed]
    
    def _associate(self, detections: List[Detection], 
                  features: List[AppearanceFeature],
                  tracks: List[DeepSORTTrack]) -> Tuple[List, List, List]:
        """Associate detections to tracks using appearance and motion"""
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        
        # Calculate cost matrix
        cost_matrix = self._calculate_cost_matrix(detections, features, tracks)
        
        # Solve assignment problem
        if cost_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Filter out high-cost assignments
            matched_indices = []
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] < self.max_distance:
                    matched_indices.append([row, col])
        else:
            matched_indices = []
        
        # Identify unmatched detections and tracks
        unmatched_detections = []
        for d in range(len(detections)):
            if d not in [m[0] for m in matched_indices]:
                unmatched_detections.append(d)
        
        unmatched_tracks = []
        for t in range(len(tracks)):
            if t not in [m[1] for m in matched_indices]:
                unmatched_tracks.append(t)
        
        return matched_indices, unmatched_detections, unmatched_tracks
    
    def _calculate_cost_matrix(self, detections: List[Detection],
                              features: List[AppearanceFeature],
                              tracks: List[DeepSORTTrack]) -> np.ndarray:
        """Calculate cost matrix for assignment"""
        cost_matrix = np.zeros((len(detections), len(tracks)))
        
        for i, (detection, feature) in enumerate(zip(detections, features)):
            for j, track in enumerate(tracks):
                # Motion distance (IoU)
                pred_bbox = track.kalman_filter.get_state()[:4]
                det_bbox = detection.to_xyxy()
                iou_distance = 1 - self._calculate_iou(pred_bbox, det_bbox)
                
                # Appearance distance
                app_distance = track.get_feature_distance(feature, self.max_distance)
                
                # Combined cost (weighted sum)
                if track.confirmed:
                    # For confirmed tracks, use both motion and appearance
                    cost = 0.3 * iou_distance + 0.7 * app_distance
                else:
                    # For unconfirmed tracks, rely more on motion
                    cost = 0.7 * iou_distance + 0.3 * app_distance
                
                # Penalize if IoU distance is too high
                if iou_distance > self.max_iou_distance:
                    cost = self.max_distance + 1
                
                cost_matrix[i, j] = cost
        
        return cost_matrix
    
    def _calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _update_stats(self):
        """Update tracking statistics"""
        self.stats['active_tracks'] = len(self.tracks)
        self.stats['confirmed_tracks'] = len([t for t in self.tracks if t.confirmed])
        self.stats['lost_tracks'] = self.stats['total_tracks'] - self.stats['active_tracks']
    
    def get_tracks(self) -> List[DeepSORTTrack]:
        """Get all active tracks"""
        return self.tracks
    
    def get_confirmed_tracks(self) -> List[DeepSORTTrack]:
        """Get only confirmed tracks"""
        return [track for track in self.tracks if track.confirmed]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics"""
        return {
            **self.stats,
            'frame_count': self.frame_count,
            'tracks_per_frame': self.stats['active_tracks'] / max(self.frame_count, 1)
        }
    
    def reset(self):
        """Reset tracker state"""
        self.tracks.clear()
        self.next_id = 1
        self.frame_count = 0
        self.stats = {
            'total_tracks': 0,
            'active_tracks': 0,
            'confirmed_tracks': 0,
            'lost_tracks': 0
        }