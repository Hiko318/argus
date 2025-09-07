#!/usr/bin/env python3
"""
Enhanced DeepSORT Tracker with Advanced ReID Fusion

This module implements an enhanced version of DeepSORT that provides:
- Improved ReID feature fusion with confidence weighting
- Robust identity continuity through occlusions
- Adaptive appearance model updates
- Multi-scale feature matching
- Temporal consistency validation
- Advanced track lifecycle management

Author: Foresight AI Team
Date: 2024
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine, euclidean
import time
from enum import Enum

from .sort_tracker import Detection, Track, KalmanBoxTracker
from .deepsort_tracker import AppearanceFeature, DeepSORTTrack
from reid.embedder import ReIDEmbedder, EmbeddingConfig, EmbeddingModel


class TrackState(Enum):
    """Enhanced track states"""
    TENTATIVE = "tentative"      # New track, not yet confirmed
    CONFIRMED = "confirmed"      # Confirmed track with sufficient hits
    LOST = "lost"                # Track lost but still searchable
    DELETED = "deleted"          # Track marked for deletion
    OCCLUDED = "occluded"        # Track temporarily occluded
    RECOVERED = "recovered"      # Track recovered after occlusion


class MatchingStrategy(Enum):
    """Matching strategies for different scenarios"""
    MOTION_ONLY = "motion_only"          # IoU-based matching only
    APPEARANCE_ONLY = "appearance_only"  # ReID feature matching only
    HYBRID = "hybrid"                    # Combined motion + appearance
    ADAPTIVE = "adaptive"                # Adaptive based on track state


@dataclass
class EnhancedAppearanceFeature(AppearanceFeature):
    """Enhanced appearance feature with additional metadata"""
    quality_score: float = 0.0           # Feature quality assessment
    occlusion_ratio: float = 0.0         # Estimated occlusion percentage
    motion_blur: float = 0.0             # Motion blur estimation
    illumination_score: float = 0.0      # Illumination quality
    scale_factor: float = 1.0            # Relative scale to reference
    view_angle: float = 0.0              # Estimated viewing angle
    
    def get_weighted_confidence(self) -> float:
        """Calculate weighted confidence based on quality factors"""
        quality_weight = (1.0 - self.occlusion_ratio) * \
                        (1.0 - self.motion_blur) * \
                        self.illumination_score * \
                        min(1.0, self.scale_factor)
        return self.confidence * quality_weight


@dataclass
class EnhancedTrack(DeepSORTTrack):
    """Enhanced track with advanced state management"""
    state: TrackState = TrackState.TENTATIVE
    features: deque = field(default_factory=lambda: deque(maxlen=50))
    feature_history: deque = field(default_factory=lambda: deque(maxlen=200))
    
    # Occlusion handling
    occlusion_count: int = 0
    max_occlusion_frames: int = 30
    last_reliable_feature: Optional[EnhancedAppearanceFeature] = None
    
    # Appearance model
    appearance_model: Optional[np.ndarray] = None
    model_update_count: int = 0
    model_confidence: float = 0.0
    
    # Motion prediction
    velocity_history: deque = field(default_factory=lambda: deque(maxlen=10))
    acceleration_history: deque = field(default_factory=lambda: deque(maxlen=5))
    
    # Quality metrics
    avg_detection_confidence: float = 0.0
    avg_feature_quality: float = 0.0
    consistency_score: float = 0.0
    
    def update_appearance_model(self, feature: EnhancedAppearanceFeature, 
                               learning_rate: float = 0.1):
        """Update appearance model with new feature"""
        if self.appearance_model is None:
            self.appearance_model = feature.embedding.copy()
            self.model_confidence = feature.get_weighted_confidence()
        else:
            # Exponential moving average update
            weight = learning_rate * feature.get_weighted_confidence()
            self.appearance_model = (1 - weight) * self.appearance_model + \
                                  weight * feature.embedding
            self.model_confidence = (1 - learning_rate) * self.model_confidence + \
                                  learning_rate * feature.get_weighted_confidence()
        
        self.model_update_count += 1
    
    def get_appearance_distance(self, feature: EnhancedAppearanceFeature) -> float:
        """Calculate appearance distance with enhanced matching"""
        if self.appearance_model is None:
            return 1.0
        
        # Primary distance using appearance model
        model_distance = cosine(feature.embedding, self.appearance_model)
        
        # Secondary distance using recent features
        recent_distances = []
        for hist_feature in list(self.features)[-5:]:
            if isinstance(hist_feature, EnhancedAppearanceFeature):
                dist = cosine(feature.embedding, hist_feature.embedding)
                weight = hist_feature.get_weighted_confidence()
                recent_distances.append(dist * weight)
        
        if recent_distances:
            recent_distance = np.mean(recent_distances)
            # Combine model and recent distances
            combined_distance = 0.7 * model_distance + 0.3 * recent_distance
        else:
            combined_distance = model_distance
        
        # Apply quality-based weighting
        quality_factor = feature.get_weighted_confidence()
        return combined_distance / (quality_factor + 0.1)
    
    def predict_next_position(self) -> Optional[np.ndarray]:
        """Predict next position using motion history"""
        if len(self.velocity_history) < 2:
            return None
        
        # Calculate average velocity
        velocities = np.array(list(self.velocity_history))
        avg_velocity = np.mean(velocities, axis=0)
        
        # Apply acceleration if available
        if len(self.acceleration_history) > 0:
            accelerations = np.array(list(self.acceleration_history))
            avg_acceleration = np.mean(accelerations, axis=0)
            predicted_velocity = avg_velocity + avg_acceleration
        else:
            predicted_velocity = avg_velocity
        
        # Get current position
        current_pos = self.kalman_filter.get_state()[:4]
        predicted_pos = current_pos + predicted_velocity
        
        return predicted_pos
    
    def update_motion_history(self, new_bbox: np.ndarray):
        """Update motion history with new bounding box"""
        if hasattr(self, '_last_bbox'):
            # Calculate velocity
            velocity = new_bbox - self._last_bbox
            self.velocity_history.append(velocity)
            
            # Calculate acceleration
            if len(self.velocity_history) >= 2:
                prev_velocity = list(self.velocity_history)[-2]
                acceleration = velocity - prev_velocity
                self.acceleration_history.append(acceleration)
        
        self._last_bbox = new_bbox.copy()


class EnhancedDeepSORTTracker:
    """Enhanced DeepSORT tracker with advanced ReID fusion"""
    
    def __init__(self,
                 max_disappeared: int = 30,
                 max_distance: float = 0.3,
                 n_init: int = 3,
                 max_iou_distance: float = 0.7,
                 max_age: int = 70,
                 reid_config: Optional[EmbeddingConfig] = None,
                 matching_strategy: MatchingStrategy = MatchingStrategy.ADAPTIVE):
        """
        Initialize Enhanced DeepSORT tracker
        
        Args:
            max_disappeared: Maximum frames a track can be unmatched
            max_distance: Maximum cosine distance for appearance matching
            n_init: Number of consecutive detections before track confirmation
            max_iou_distance: Maximum IoU distance for association
            max_age: Maximum age of track before deletion
            reid_config: ReID embedder configuration
            matching_strategy: Strategy for detection-track matching
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.n_init = n_init
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.matching_strategy = matching_strategy
        
        # Initialize ReID embedder
        self.reid_embedder = ReIDEmbedder(reid_config or EmbeddingConfig())
        
        # Tracking state
        self.tracks: List[EnhancedTrack] = []
        self.next_id = 1
        self.frame_count = 0
        
        # Adaptive parameters
        self.adaptive_thresholds = {
            'motion_weight': 0.3,
            'appearance_weight': 0.7,
            'quality_threshold': 0.5,
            'occlusion_threshold': 0.3
        }
        
        # Performance metrics
        self.metrics = {
            'total_tracks': 0,
            'active_tracks': 0,
            'confirmed_tracks': 0,
            'recovered_tracks': 0,
            'lost_tracks': 0,
            'avg_track_length': 0.0,
            'identity_switches': 0,
            'processing_time': 0.0
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Enhanced DeepSORT tracker initialized with {matching_strategy.value} matching")
    
    def update(self, detections: List[Detection], image: np.ndarray) -> List[EnhancedTrack]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detections
            image: Current frame image
            
        Returns:
            List of updated tracks
        """
        start_time = time.time()
        self.frame_count += 1
        
        # Extract features for all detections
        detection_features = self._extract_features(detections, image)
        
        # Predict track positions
        self._predict_tracks()
        
        # Perform data association
        matched_pairs, unmatched_dets, unmatched_trks = self._associate_detections(
            detections, detection_features
        )
        
        # Update matched tracks
        self._update_matched_tracks(matched_pairs, detections, detection_features)
        
        # Handle unmatched tracks
        self._handle_unmatched_tracks(unmatched_trks)
        
        # Create new tracks
        self._create_new_tracks(unmatched_dets, detections, detection_features)
        
        # Update track states
        self._update_track_states()
        
        # Clean up old tracks
        self._cleanup_tracks()
        
        # Update metrics
        processing_time = time.time() - start_time
        self._update_metrics(processing_time)
        
        # Return confirmed tracks
        return self.get_confirmed_tracks()
    
    def _extract_features(self, detections: List[Detection], 
                         image: np.ndarray) -> List[EnhancedAppearanceFeature]:
        """Extract enhanced appearance features from detections"""
        features = []
        
        for detection in detections:
            # Extract crop from image
            x1, y1, x2, y2 = detection.to_xyxy().astype(int)
            crop = image[y1:y2, x1:x2]
            
            if crop.size == 0:
                # Create dummy feature for invalid crops
                feature = EnhancedAppearanceFeature(
                    embedding=np.zeros(self.reid_embedder.config.embedding_dim),
                    confidence=0.0,
                    timestamp=time.time()
                )
                features.append(feature)
                continue
            
            # Extract ReID embedding
            embedding = self.reid_embedder.extract_features(crop)
            
            # Assess feature quality
            quality_metrics = self._assess_feature_quality(crop, detection)
            
            # Create enhanced feature
            feature = EnhancedAppearanceFeature(
                embedding=embedding,
                confidence=detection.confidence,
                timestamp=time.time(),
                quality_score=quality_metrics['quality_score'],
                occlusion_ratio=quality_metrics['occlusion_ratio'],
                motion_blur=quality_metrics['motion_blur'],
                illumination_score=quality_metrics['illumination_score'],
                scale_factor=quality_metrics['scale_factor']
            )
            
            features.append(feature)
        
        return features
    
    def _assess_feature_quality(self, crop: np.ndarray, 
                               detection: Detection) -> Dict[str, float]:
        """Assess quality of extracted feature"""
        h, w = crop.shape[:2]
        
        # Quality score based on detection confidence and size
        size_score = min(1.0, (h * w) / (128 * 256))  # Normalize to reference size
        quality_score = detection.confidence * size_score
        
        # Estimate occlusion (simplified - could use more sophisticated methods)
        # Check if crop is at image boundaries
        occlusion_ratio = 0.0
        if h < 50 or w < 30:  # Very small crops likely occluded
            occlusion_ratio = 0.3
        
        # Estimate motion blur using Laplacian variance
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        motion_blur = max(0.0, 1.0 - laplacian_var / 1000.0)  # Normalize
        
        # Illumination score based on histogram
        illumination_score = min(1.0, np.mean(gray) / 128.0)
        
        # Scale factor
        scale_factor = min(2.0, max(0.5, np.sqrt((h * w) / (128 * 64))))
        
        return {
            'quality_score': quality_score,
            'occlusion_ratio': occlusion_ratio,
            'motion_blur': motion_blur,
            'illumination_score': illumination_score,
            'scale_factor': scale_factor
        }
    
    def _predict_tracks(self):
        """Predict track positions using Kalman filters"""
        for track in self.tracks:
            if track.state != TrackState.DELETED:
                track.kalman_filter.predict()
                track.age += 1
    
    def _associate_detections(self, detections: List[Detection],
                            features: List[EnhancedAppearanceFeature]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Associate detections to tracks using adaptive strategy"""
        if len(self.tracks) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
        
        # Calculate cost matrix
        cost_matrix = self._calculate_enhanced_cost_matrix(detections, features)
        
        # Apply Hungarian algorithm
        if cost_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Filter valid assignments
            valid_matches = []
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] < self.max_distance:
                    valid_matches.append([row, col])
            
            matches = np.array(valid_matches) if valid_matches else np.empty((0, 2), dtype=int)
        else:
            matches = np.empty((0, 2), dtype=int)
        
        # Find unmatched detections and tracks
        unmatched_dets = []
        for d in range(len(detections)):
            if len(matches) == 0 or d not in matches[:, 0]:
                unmatched_dets.append(d)
        
        unmatched_trks = []
        for t in range(len(self.tracks)):
            if len(matches) == 0 or t not in matches[:, 1]:
                unmatched_trks.append(t)
        
        return matches, np.array(unmatched_dets), np.array(unmatched_trks)
    
    def _calculate_enhanced_cost_matrix(self, detections: List[Detection],
                                      features: List[EnhancedAppearanceFeature]) -> np.ndarray:
        """Calculate enhanced cost matrix with adaptive weighting"""
        cost_matrix = np.full((len(detections), len(self.tracks)), self.max_distance + 1)
        
        for i, (detection, feature) in enumerate(zip(detections, features)):
            for j, track in enumerate(self.tracks):
                if track.state == TrackState.DELETED:
                    continue
                
                # Calculate motion cost (IoU distance)
                pred_bbox = track.kalman_filter.get_state()[:4]
                det_bbox = detection.to_xyxy()
                iou = self._calculate_iou(pred_bbox, det_bbox)
                motion_cost = 1.0 - iou
                
                # Calculate appearance cost
                appearance_cost = track.get_appearance_distance(feature)
                
                # Adaptive weighting based on track state and feature quality
                motion_weight, appearance_weight = self._get_adaptive_weights(
                    track, feature, motion_cost, appearance_cost
                )
                
                # Combined cost
                combined_cost = motion_weight * motion_cost + appearance_weight * appearance_cost
                
                # Apply penalties for poor quality or high uncertainty
                if feature.quality_score < self.adaptive_thresholds['quality_threshold']:
                    combined_cost *= 1.2
                
                if motion_cost > self.max_iou_distance:
                    combined_cost = self.max_distance + 1
                
                cost_matrix[i, j] = combined_cost
        
        return cost_matrix
    
    def _get_adaptive_weights(self, track: EnhancedTrack, 
                            feature: EnhancedAppearanceFeature,
                            motion_cost: float, appearance_cost: float) -> Tuple[float, float]:
        """Get adaptive weights for motion and appearance costs"""
        if self.matching_strategy == MatchingStrategy.MOTION_ONLY:
            return 1.0, 0.0
        elif self.matching_strategy == MatchingStrategy.APPEARANCE_ONLY:
            return 0.0, 1.0
        elif self.matching_strategy == MatchingStrategy.HYBRID:
            return self.adaptive_thresholds['motion_weight'], self.adaptive_thresholds['appearance_weight']
        else:  # ADAPTIVE
            # Adapt weights based on track state and feature quality
            base_motion_weight = 0.3
            base_appearance_weight = 0.7
            
            # Increase motion weight for new/tentative tracks
            if track.state == TrackState.TENTATIVE:
                base_motion_weight = 0.6
                base_appearance_weight = 0.4
            
            # Increase appearance weight for confirmed tracks with good model
            elif track.state == TrackState.CONFIRMED and track.model_confidence > 0.7:
                base_motion_weight = 0.2
                base_appearance_weight = 0.8
            
            # Adjust based on feature quality
            quality_factor = feature.get_weighted_confidence()
            if quality_factor < 0.5:
                # Poor quality feature - rely more on motion
                base_motion_weight += 0.2
                base_appearance_weight -= 0.2
            
            # Normalize weights
            total_weight = base_motion_weight + base_appearance_weight
            return base_motion_weight / total_weight, base_appearance_weight / total_weight
    
    def _update_matched_tracks(self, matches: np.ndarray, 
                             detections: List[Detection],
                             features: List[EnhancedAppearanceFeature]):
        """Update matched tracks with new detections"""
        for match in matches:
            det_idx, trk_idx = match
            detection = detections[det_idx]
            feature = features[det_idx]
            track = self.tracks[trk_idx]
            
            # Update Kalman filter
            track.kalman_filter.update(detection.to_xyxy())
            
            # Update motion history
            track.update_motion_history(detection.to_xyxy())
            
            # Add feature to track
            track.features.append(feature)
            track.feature_history.append(feature)
            
            # Update appearance model
            if feature.get_weighted_confidence() > 0.5:
                track.update_appearance_model(feature)
                track.last_reliable_feature = feature
            
            # Update track state
            track.hits += 1
            track.hit_streak += 1
            track.time_since_update = 0
            track.confidence = detection.confidence
            
            # Update quality metrics
            track.avg_detection_confidence = (track.avg_detection_confidence * (track.hits - 1) + 
                                            detection.confidence) / track.hits
            track.avg_feature_quality = (track.avg_feature_quality * (track.hits - 1) + 
                                       feature.quality_score) / track.hits
            
            # State transitions
            if track.state == TrackState.TENTATIVE and track.hit_streak >= self.n_init:
                track.state = TrackState.CONFIRMED
                self.logger.debug(f"Track {track.id} confirmed")
            elif track.state == TrackState.LOST:
                track.state = TrackState.RECOVERED
                self.metrics['recovered_tracks'] += 1
                self.logger.debug(f"Track {track.id} recovered")
            elif track.state == TrackState.OCCLUDED:
                track.state = TrackState.CONFIRMED
                track.occlusion_count = 0
    
    def _handle_unmatched_tracks(self, unmatched_trks: np.ndarray):
        """Handle unmatched tracks"""
        for trk_idx in unmatched_trks:
            track = self.tracks[trk_idx]
            track.time_since_update += 1
            track.hit_streak = 0
            
            # State transitions for unmatched tracks
            if track.state == TrackState.CONFIRMED:
                if track.time_since_update > 5:
                    track.state = TrackState.LOST
                elif track.time_since_update > 2:
                    track.state = TrackState.OCCLUDED
                    track.occlusion_count += 1
            elif track.state == TrackState.TENTATIVE:
                if track.time_since_update > 3:
                    track.state = TrackState.DELETED
    
    def _create_new_tracks(self, unmatched_dets: np.ndarray,
                          detections: List[Detection],
                          features: List[EnhancedAppearanceFeature]):
        """Create new tracks for unmatched detections"""
        for det_idx in unmatched_dets:
            detection = detections[det_idx]
            feature = features[det_idx]
            
            # Create Kalman filter
            kalman_filter = KalmanBoxTracker(detection.to_xyxy())
            
            # Create new track
            track = EnhancedTrack(
                id=self.next_id,
                kalman_filter=kalman_filter,
                class_id=detection.class_id,
                confidence=detection.confidence,
                state=TrackState.TENTATIVE
            )
            
            # Initialize with first feature
            track.features.append(feature)
            track.feature_history.append(feature)
            if feature.get_weighted_confidence() > 0.5:
                track.update_appearance_model(feature)
                track.last_reliable_feature = feature
            
            # Initialize motion history
            track.update_motion_history(detection.to_xyxy())
            
            # Initialize quality metrics
            track.avg_detection_confidence = detection.confidence
            track.avg_feature_quality = feature.quality_score
            
            track.hits = 1
            track.hit_streak = 1
            track.age = 1
            
            self.tracks.append(track)
            self.next_id += 1
            self.metrics['total_tracks'] += 1
            
            self.logger.debug(f"Created new track {track.id}")
    
    def _update_track_states(self):
        """Update track states based on current conditions"""
        for track in self.tracks:
            if track.state == TrackState.DELETED:
                continue
            
            # Check for deletion conditions
            if track.time_since_update > self.max_age:
                track.state = TrackState.DELETED
            elif track.state == TrackState.LOST and track.time_since_update > self.max_disappeared:
                track.state = TrackState.DELETED
            elif track.state == TrackState.OCCLUDED and track.occlusion_count > track.max_occlusion_frames:
                track.state = TrackState.LOST
    
    def _cleanup_tracks(self):
        """Remove deleted tracks"""
        active_tracks = []
        for track in self.tracks:
            if track.state != TrackState.DELETED:
                active_tracks.append(track)
            else:
                self.metrics['lost_tracks'] += 1
        
        self.tracks = active_tracks
    
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
    
    def _update_metrics(self, processing_time: float):
        """Update performance metrics"""
        self.metrics['active_tracks'] = len([t for t in self.tracks if t.state != TrackState.DELETED])
        self.metrics['confirmed_tracks'] = len([t for t in self.tracks if t.state == TrackState.CONFIRMED])
        self.metrics['processing_time'] = processing_time
        
        # Calculate average track length
        if self.metrics['total_tracks'] > 0:
            total_length = sum(t.hits for t in self.tracks)
            self.metrics['avg_track_length'] = total_length / self.metrics['total_tracks']
    
    def get_confirmed_tracks(self) -> List[EnhancedTrack]:
        """Get confirmed tracks"""
        return [track for track in self.tracks if track.state == TrackState.CONFIRMED]
    
    def get_all_tracks(self) -> List[EnhancedTrack]:
        """Get all active tracks"""
        return [track for track in self.tracks if track.state != TrackState.DELETED]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get tracking metrics"""
        return {
            **self.metrics,
            'frame_count': self.frame_count,
            'tracks_per_frame': self.metrics['active_tracks'] / max(self.frame_count, 1)
        }
    
    def reset(self):
        """Reset tracker state"""
        self.tracks.clear()
        self.next_id = 1
        self.frame_count = 0
        self.metrics = {
            'total_tracks': 0,
            'active_tracks': 0,
            'confirmed_tracks': 0,
            'recovered_tracks': 0,
            'lost_tracks': 0,
            'avg_track_length': 0.0,
            'identity_switches': 0,
            'processing_time': 0.0
        }
        
        self.logger.info("Enhanced DeepSORT tracker reset")