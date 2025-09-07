"""
Track Manager for Foresight SAR System

This module provides a unified interface for managing multiple tracking
algorithms, suspect locks, and re-identification features.
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
from pathlib import Path

from .sort_tracker import SORTTracker, Detection, Track
from .deepsort_tracker import DeepSORTTracker, DeepSORTTrack
from .suspect_lock import SuspectLockManager, SuspectLock, LockPriority
from reid.embedder import ReIDEmbedder, EmbeddingConfig, EmbeddingModel


class TrackerType(Enum):
    """Available tracker types"""
    SORT = "sort"
    DEEPSORT = "deepsort"
    HYBRID = "hybrid"


@dataclass
class TrackingConfig:
    """Configuration for tracking system"""
    tracker_type: TrackerType = TrackerType.DEEPSORT
    max_disappeared: int = 30
    max_distance: float = 0.3
    iou_threshold: float = 0.3
    confidence_threshold: float = 0.5
    
    # DeepSORT specific
    n_init: int = 3
    max_age: int = 70
    
    # ReID configuration
    reid_model: EmbeddingModel = EmbeddingModel.RESNET50
    reid_model_path: Optional[str] = None
    
    # Suspect lock configuration
    max_locks: int = 10
    auto_lock_threshold: float = 0.8
    lock_timeout: float = 30.0


class TrackManager:
    """Unified tracking manager"""
    
    def __init__(self, config: TrackingConfig = None):
        """
        Initialize track manager
        
        Args:
            config: Tracking configuration
        """
        self.config = config or TrackingConfig()
        
        # Initialize trackers
        self.sort_tracker = None
        self.deepsort_tracker = None
        self.active_tracker = None
        
        # Initialize ReID embedder
        self.reid_embedder = None
        
        # Initialize suspect lock manager
        self.suspect_manager = SuspectLockManager(
            max_locks=self.config.max_locks,
            lost_timeout=self.config.lock_timeout,
            similarity_threshold=self.config.auto_lock_threshold
        )
        
        # Tracking state
        self.frame_count = 0
        self.last_update_time = 0
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'total_tracks': 0,
            'active_tracks': 0,
            'suspect_locks': 0,
            'processing_time': 0.0
        }
        
        # Initialize components
        self._init_trackers()
        self._init_reid_embedder()
    
    def _init_trackers(self):
        """Initialize tracking algorithms"""
        try:
            # Initialize SORT tracker
            self.sort_tracker = SORTTracker(
                max_disappeared=self.config.max_disappeared,
                iou_threshold=self.config.iou_threshold
            )
            
            # Initialize DeepSORT tracker
            if self.config.tracker_type in [TrackerType.DEEPSORT, TrackerType.HYBRID]:
                self.deepsort_tracker = DeepSORTTracker(
                    max_disappeared=self.config.max_disappeared,
                    max_distance=self.config.max_distance,
                    n_init=self.config.n_init,
                    max_age=self.config.max_age,
                    feature_extractor_path=self.config.reid_model_path
                )
            
            # Set active tracker
            if self.config.tracker_type == TrackerType.SORT:
                self.active_tracker = self.sort_tracker
            elif self.config.tracker_type == TrackerType.DEEPSORT:
                self.active_tracker = self.deepsort_tracker
            else:  # HYBRID
                self.active_tracker = self.deepsort_tracker  # Default to DeepSORT
            
            logging.info(f"Initialized {self.config.tracker_type.value} tracker")
            
        except Exception as e:
            logging.error(f"Failed to initialize trackers: {e}")
            # Fallback to SORT
            self.active_tracker = self.sort_tracker
            self.config.tracker_type = TrackerType.SORT
    
    def _init_reid_embedder(self):
        """Initialize ReID embedder"""
        try:
            reid_config = EmbeddingConfig(
                model_type=self.config.reid_model,
                model_path=self.config.reid_model_path,
                normalize=True
            )
            
            self.reid_embedder = ReIDEmbedder(reid_config)
            logging.info(f"Initialized ReID embedder: {self.config.reid_model.value}")
            
        except Exception as e:
            logging.error(f"Failed to initialize ReID embedder: {e}")
            self.reid_embedder = None
    
    def update(self, detections: List[Detection], image: np.ndarray) -> List[Track]:
        """
        Update tracking with new detections
        
        Args:
            detections: List of detections from object detector
            image: Current frame image
            
        Returns:
            List of active tracks
        """
        start_time = time.time()
        self.frame_count += 1
        
        # Filter detections by confidence
        filtered_detections = [
            det for det in detections 
            if det.confidence >= self.config.confidence_threshold
        ]
        
        self.stats['total_detections'] += len(filtered_detections)
        
        # Update tracker
        if self.config.tracker_type == TrackerType.HYBRID:
            tracks = self._hybrid_update(filtered_detections, image)
        else:
            tracks = self.active_tracker.update(filtered_detections, image)
        
        # Update suspect locks
        self.suspect_manager.update_locks(tracks)
        
        # Auto-lock suspects if enabled
        if self.config.auto_lock_threshold > 0:
            new_locks = self.suspect_manager.auto_lock_suspects(tracks)
            if new_locks:
                logging.info(f"Auto-locked {len(new_locks)} suspects")
        
        # Update statistics
        self.stats['active_tracks'] = len(tracks)
        self.stats['suspect_locks'] = len(self.suspect_manager.get_active_locks())
        self.stats['processing_time'] = time.time() - start_time
        self.last_update_time = time.time()
        
        return tracks
    
    def _hybrid_update(self, detections: List[Detection], image: np.ndarray) -> List[Track]:
        """Hybrid tracking using both SORT and DeepSORT"""
        # Use DeepSORT for primary tracking
        deepsort_tracks = self.deepsort_tracker.update(detections, image)
        
        # Use SORT for backup/validation
        sort_tracks = self.sort_tracker.update(detections)
        
        # Merge results (prefer DeepSORT)
        merged_tracks = list(deepsort_tracks)
        
        # Add SORT tracks that don't overlap with DeepSORT
        for sort_track in sort_tracks:
            overlaps = False
            sort_bbox = sort_track.kalman_filter.get_state()[:4]
            
            for deepsort_track in deepsort_tracks:
                deepsort_bbox = deepsort_track.kalman_filter.get_state()[:4]
                iou = self._calculate_iou(sort_bbox, deepsort_bbox)
                
                if iou > 0.5:  # Significant overlap
                    overlaps = True
                    break
            
            if not overlaps:
                merged_tracks.append(sort_track)
        
        return merged_tracks
    
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
    
    def lock_suspect(self, 
                    suspect_id: str,
                    track_id: int,
                    reference_image: Optional[np.ndarray] = None,
                    priority: LockPriority = LockPriority.MEDIUM) -> Optional[SuspectLock]:
        """
        Lock a suspect to a track
        
        Args:
            suspect_id: Unique identifier for suspect
            track_id: Track ID to lock to
            reference_image: Reference image for ReID
            priority: Lock priority
            
        Returns:
            Created suspect lock or None if failed
        """
        # Find track
        tracks = self.get_active_tracks()
        target_track = None
        
        for track in tracks:
            if track.id == track_id:
                target_track = track
                break
        
        if target_track is None:
            logging.error(f"Track {track_id} not found")
            return None
        
        # Create or update suspect profile
        if suspect_id not in self.suspect_manager.suspect_profiles:
            profile = self.suspect_manager.create_suspect_profile(
                suspect_id=suspect_id,
                priority=priority
            )
            
            # Add reference features if image provided
            if reference_image is not None and self.reid_embedder:
                embedding = self.reid_embedder.extract_embedding(reference_image)
                from ..tracking.deepsort_tracker import AppearanceFeature
                feature = AppearanceFeature(
                    embedding=embedding,
                    timestamp=time.time(),
                    confidence=1.0
                )
                profile.reference_features.append(feature)
                profile.reference_images.append(reference_image)
        
        # Create lock
        return self.suspect_manager.lock_suspect(suspect_id, target_track)
    
    def unlock_suspect(self, suspect_id: str) -> bool:
        """Unlock a suspect"""
        return self.suspect_manager.unlock_suspect(suspect_id)
    
    def get_active_tracks(self) -> List[Track]:
        """Get all active tracks"""
        if self.active_tracker:
            return self.active_tracker.get_tracks() if hasattr(self.active_tracker, 'get_tracks') else []
        return []
    
    def get_confirmed_tracks(self) -> List[Track]:
        """Get confirmed tracks (DeepSORT only)"""
        if isinstance(self.active_tracker, DeepSORTTracker):
            return self.active_tracker.get_confirmed_tracks()
        return self.get_active_tracks()
    
    def get_suspect_locks(self) -> List[SuspectLock]:
        """Get all active suspect locks"""
        return self.suspect_manager.get_active_locks()
    
    def get_high_priority_locks(self) -> List[SuspectLock]:
        """Get high priority suspect locks"""
        return self.suspect_manager.get_high_priority_locks()
    
    def search_suspects(self, query_image: np.ndarray, 
                       bbox: Optional[Tuple[int, int, int, int]] = None) -> List[Tuple[str, float]]:
        """
        Search for suspects by appearance
        
        Args:
            query_image: Query image
            bbox: Bounding box for person region
            
        Returns:
            List of (suspect_id, similarity_score) tuples
        """
        if not self.reid_embedder:
            return []
        
        # Extract query embedding
        query_embedding = self.reid_embedder.extract_embedding(query_image, bbox)
        
        # Create appearance feature
        from ..tracking.deepsort_tracker import AppearanceFeature
        query_feature = AppearanceFeature(
            embedding=query_embedding,
            timestamp=time.time(),
            confidence=1.0
        )
        
        # Search suspects
        return self.suspect_manager.search_suspects(query_feature)
    
    def export_tracking_data(self, output_dir: str):
        """Export tracking data and statistics"""
        import json
        from datetime import datetime
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export statistics
        stats_data = {
            'timestamp': timestamp,
            'frame_count': self.frame_count,
            'config': {
                'tracker_type': self.config.tracker_type.value,
                'reid_model': self.config.reid_model.value,
                'max_locks': self.config.max_locks
            },
            'statistics': self.get_statistics()
        }
        
        with open(output_path / f"tracking_stats_{timestamp}.json", 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        # Export suspect data
        suspect_data = {}
        for suspect_id in self.suspect_manager.suspect_profiles.keys():
            suspect_data[suspect_id] = self.suspect_manager.export_suspect_data(suspect_id)
        
        with open(output_path / f"suspect_data_{timestamp}.json", 'w') as f:
            json.dump(suspect_data, f, indent=2)
        
        logging.info(f"Exported tracking data to {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tracking statistics"""
        base_stats = dict(self.stats)
        
        # Add tracker-specific stats
        if self.active_tracker and hasattr(self.active_tracker, 'get_statistics'):
            tracker_stats = self.active_tracker.get_statistics()
            base_stats.update({f"tracker_{k}": v for k, v in tracker_stats.items()})
        
        # Add suspect manager stats
        suspect_stats = self.suspect_manager.get_statistics()
        base_stats.update({f"suspect_{k}": v for k, v in suspect_stats.items()})
        
        # Add ReID stats
        if self.reid_embedder:
            reid_info = self.reid_embedder.get_model_info()
            base_stats.update({f"reid_{k}": v for k, v in reid_info.items()})
        
        return base_stats
    
    def reset(self):
        """Reset tracking state"""
        if self.sort_tracker:
            self.sort_tracker.reset()
        if self.deepsort_tracker:
            self.deepsort_tracker.reset()
        
        self.suspect_manager.clear_all_locks()
        
        self.frame_count = 0
        self.last_update_time = 0
        
        self.stats = {
            'total_detections': 0,
            'total_tracks': 0,
            'active_tracks': 0,
            'suspect_locks': 0,
            'processing_time': 0.0
        }
        
        logging.info("Reset tracking state")
    
    def switch_tracker(self, tracker_type: TrackerType):
        """Switch active tracker type"""
        if tracker_type == self.config.tracker_type:
            return
        
        self.config.tracker_type = tracker_type
        
        if tracker_type == TrackerType.SORT:
            self.active_tracker = self.sort_tracker
        elif tracker_type == TrackerType.DEEPSORT:
            if self.deepsort_tracker is None:
                self._init_trackers()
            self.active_tracker = self.deepsort_tracker
        else:  # HYBRID
            if self.deepsort_tracker is None:
                self._init_trackers()
            self.active_tracker = self.deepsort_tracker
        
        logging.info(f"Switched to {tracker_type.value} tracker")
    
    def save_config(self, path: str):
        """Save tracking configuration"""
        import json
        
        config_dict = {
            'tracker_type': self.config.tracker_type.value,
            'max_disappeared': self.config.max_disappeared,
            'max_distance': self.config.max_distance,
            'iou_threshold': self.config.iou_threshold,
            'confidence_threshold': self.config.confidence_threshold,
            'n_init': self.config.n_init,
            'max_age': self.config.max_age,
            'reid_model': self.config.reid_model.value,
            'reid_model_path': self.config.reid_model_path,
            'max_locks': self.config.max_locks,
            'auto_lock_threshold': self.config.auto_lock_threshold,
            'lock_timeout': self.config.lock_timeout
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_config(cls, path: str) -> 'TrackManager':
        """Load track manager from configuration file"""
        import json
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        config = TrackingConfig(
            tracker_type=TrackerType(config_dict['tracker_type']),
            max_disappeared=config_dict['max_disappeared'],
            max_distance=config_dict['max_distance'],
            iou_threshold=config_dict['iou_threshold'],
            confidence_threshold=config_dict['confidence_threshold'],
            n_init=config_dict['n_init'],
            max_age=config_dict['max_age'],
            reid_model=EmbeddingModel(config_dict['reid_model']),
            reid_model_path=config_dict.get('reid_model_path'),
            max_locks=config_dict['max_locks'],
            auto_lock_threshold=config_dict['auto_lock_threshold'],
            lock_timeout=config_dict['lock_timeout']
        )
        
        return cls(config)