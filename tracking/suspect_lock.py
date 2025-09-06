"""
Suspect Lock System for Foresight SAR

This module implements suspect locking functionality that allows operators to
lock onto specific targets for continuous tracking and monitoring.
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
from collections import deque

from .sort_tracker import Track
from .deepsort_tracker import DeepSORTTrack, AppearanceFeature


class SuspectStatus(Enum):
    """Status of a suspect lock"""
    ACTIVE = "active"
    LOST = "lost"
    CONFIRMED = "confirmed"
    DISMISSED = "dismissed"


class LockPriority(Enum):
    """Priority levels for suspect locks"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class SuspectProfile:
    """Profile information for a suspect"""
    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    reference_features: List[AppearanceFeature] = field(default_factory=list)
    reference_images: List[np.ndarray] = field(default_factory=list)
    priority: LockPriority = LockPriority.MEDIUM
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SuspectLock:
    """A locked suspect target"""
    suspect_id: str
    track_id: int
    profile: SuspectProfile
    status: SuspectStatus = SuspectStatus.ACTIVE
    confidence: float = 1.0
    lock_time: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    position_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    feature_history: deque = field(default_factory=lambda: deque(maxlen=100))
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    
    def update_position(self, bbox: Tuple[float, float, float, float], timestamp: float = None):
        """Update suspect position"""
        if timestamp is None:
            timestamp = time.time()
        
        self.position_history.append({
            'bbox': bbox,
            'timestamp': timestamp,
            'center': ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        })
        self.last_seen = timestamp
    
    def add_feature(self, feature: AppearanceFeature):
        """Add appearance feature"""
        self.feature_history.append(feature)
    
    def add_alert(self, alert_type: str, message: str, severity: str = "info"):
        """Add alert for this suspect"""
        alert = {
            'type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': time.time()
        }
        self.alerts.append(alert)
        logging.info(f"Suspect {self.suspect_id} alert: {message}")
    
    def get_current_position(self) -> Optional[Tuple[float, float, float, float]]:
        """Get current position"""
        if self.position_history:
            return self.position_history[-1]['bbox']
        return None
    
    def get_trajectory(self, duration: float = 60.0) -> List[Tuple[float, float]]:
        """Get trajectory over specified duration (seconds)"""
        current_time = time.time()
        cutoff_time = current_time - duration
        
        trajectory = []
        for pos in self.position_history:
            if pos['timestamp'] >= cutoff_time:
                trajectory.append(pos['center'])
        
        return trajectory
    
    def is_lost(self, timeout: float = 30.0) -> bool:
        """Check if suspect is lost"""
        return (time.time() - self.last_seen) > timeout


class SuspectLockManager:
    """Manager for suspect locks and profiles"""
    
    def __init__(self, 
                 max_locks: int = 10,
                 lost_timeout: float = 30.0,
                 similarity_threshold: float = 0.8):
        """
        Initialize suspect lock manager
        
        Args:
            max_locks: Maximum number of active locks
            lost_timeout: Time before marking suspect as lost
            similarity_threshold: Threshold for automatic suspect matching
        """
        self.max_locks = max_locks
        self.lost_timeout = lost_timeout
        self.similarity_threshold = similarity_threshold
        
        self.active_locks: Dict[str, SuspectLock] = {}
        self.suspect_profiles: Dict[str, SuspectProfile] = {}
        self.track_to_suspect: Dict[int, str] = {}  # Track ID -> Suspect ID
        
        # Statistics
        self.stats = {
            'total_locks_created': 0,
            'active_locks': 0,
            'lost_locks': 0,
            'confirmed_locks': 0
        }
    
    def create_suspect_profile(self, 
                             suspect_id: str,
                             name: Optional[str] = None,
                             description: Optional[str] = None,
                             reference_image: Optional[np.ndarray] = None,
                             priority: LockPriority = LockPriority.MEDIUM) -> SuspectProfile:
        """Create a new suspect profile"""
        profile = SuspectProfile(
            id=suspect_id,
            name=name,
            description=description,
            priority=priority
        )
        
        if reference_image is not None:
            profile.reference_images.append(reference_image)
        
        self.suspect_profiles[suspect_id] = profile
        logging.info(f"Created suspect profile: {suspect_id}")
        
        return profile
    
    def lock_suspect(self, 
                    suspect_id: str,
                    track: Track,
                    confidence: float = 1.0,
                    auto_created: bool = False) -> Optional[SuspectLock]:
        """Lock onto a suspect"""
        # Check if we've reached max locks
        if len(self.active_locks) >= self.max_locks:
            # Remove lowest priority lock if possible
            if not self._make_room_for_lock(suspect_id):
                logging.warning(f"Cannot lock suspect {suspect_id}: max locks reached")
                return None
        
        # Check if suspect already locked
        if suspect_id in self.active_locks:
            logging.warning(f"Suspect {suspect_id} already locked")
            return self.active_locks[suspect_id]
        
        # Get or create profile
        if suspect_id not in self.suspect_profiles:
            if auto_created:
                self.create_suspect_profile(suspect_id)
            else:
                logging.error(f"No profile found for suspect {suspect_id}")
                return None
        
        profile = self.suspect_profiles[suspect_id]
        
        # Create lock
        lock = SuspectLock(
            suspect_id=suspect_id,
            track_id=track.id,
            profile=profile,
            confidence=confidence
        )
        
        # Update position if track has current state
        if hasattr(track, 'kalman_filter'):
            bbox = track.kalman_filter.get_state()[:4]
            lock.update_position(tuple(bbox))
        
        # Store lock
        self.active_locks[suspect_id] = lock
        self.track_to_suspect[track.id] = suspect_id
        
        # Update statistics
        self.stats['total_locks_created'] += 1
        self.stats['active_locks'] = len(self.active_locks)
        
        lock.add_alert("lock_created", f"Suspect locked with confidence {confidence:.2f}")
        logging.info(f"Locked suspect {suspect_id} to track {track.id}")
        
        return lock
    
    def update_locks(self, tracks: List[Track]):
        """Update all active locks with new tracking data"""
        current_time = time.time()
        track_ids = {track.id for track in tracks}
        
        # Update existing locks
        for suspect_id, lock in list(self.active_locks.items()):
            if lock.track_id in track_ids:
                # Find corresponding track
                track = next(t for t in tracks if t.id == lock.track_id)
                
                # Update position
                if hasattr(track, 'kalman_filter'):
                    bbox = track.kalman_filter.get_state()[:4]
                    lock.update_position(tuple(bbox), current_time)
                
                # Add appearance feature if available
                if isinstance(track, DeepSORTTrack) and track.features:
                    lock.add_feature(track.features[-1])
                
                # Update status
                if lock.status == SuspectStatus.LOST:
                    lock.status = SuspectStatus.ACTIVE
                    lock.add_alert("reacquired", "Suspect reacquired")
            else:
                # Track lost
                if lock.status == SuspectStatus.ACTIVE:
                    lock.status = SuspectStatus.LOST
                    lock.add_alert("lost", "Suspect track lost", "warning")
                
                # Check if should be removed
                if lock.is_lost(self.lost_timeout):
                    self._remove_lock(suspect_id)
        
        # Update statistics
        self._update_stats()
    
    def unlock_suspect(self, suspect_id: str) -> bool:
        """Unlock a suspect"""
        if suspect_id not in self.active_locks:
            return False
        
        lock = self.active_locks[suspect_id]
        lock.status = SuspectStatus.DISMISSED
        lock.add_alert("unlocked", "Suspect manually unlocked")
        
        self._remove_lock(suspect_id)
        logging.info(f"Unlocked suspect {suspect_id}")
        
        return True
    
    def get_lock(self, suspect_id: str) -> Optional[SuspectLock]:
        """Get lock by suspect ID"""
        return self.active_locks.get(suspect_id)
    
    def get_lock_by_track(self, track_id: int) -> Optional[SuspectLock]:
        """Get lock by track ID"""
        suspect_id = self.track_to_suspect.get(track_id)
        if suspect_id:
            return self.active_locks.get(suspect_id)
        return None
    
    def get_active_locks(self) -> List[SuspectLock]:
        """Get all active locks"""
        return list(self.active_locks.values())
    
    def get_high_priority_locks(self) -> List[SuspectLock]:
        """Get high priority locks"""
        return [
            lock for lock in self.active_locks.values()
            if lock.profile.priority in [LockPriority.HIGH, LockPriority.CRITICAL]
        ]
    
    def search_suspects(self, 
                       query_feature: AppearanceFeature,
                       max_results: int = 5) -> List[Tuple[str, float]]:
        """Search for suspects by appearance feature"""
        results = []
        
        for suspect_id, profile in self.suspect_profiles.items():
            if not profile.reference_features:
                continue
            
            # Calculate similarity to reference features
            similarities = []
            for ref_feature in profile.reference_features:
                similarity = 1 - np.linalg.norm(query_feature.embedding - ref_feature.embedding)
                similarities.append(similarity)
            
            if similarities:
                max_similarity = max(similarities)
                if max_similarity >= self.similarity_threshold:
                    results.append((suspect_id, max_similarity))
        
        # Sort by similarity and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]
    
    def auto_lock_suspects(self, tracks: List[Track]) -> List[SuspectLock]:
        """Automatically lock suspects based on appearance matching"""
        new_locks = []
        
        for track in tracks:
            # Skip if track already locked
            if track.id in self.track_to_suspect:
                continue
            
            # Only process DeepSORT tracks with features
            if not isinstance(track, DeepSORTTrack) or not track.features:
                continue
            
            # Search for matching suspects
            latest_feature = track.features[-1]
            matches = self.search_suspects(latest_feature)
            
            if matches:
                suspect_id, confidence = matches[0]
                
                # Auto-lock if confidence is high enough
                if confidence >= self.similarity_threshold:
                    lock = self.lock_suspect(
                        suspect_id, track, confidence, auto_created=False
                    )
                    if lock:
                        lock.add_alert(
                            "auto_locked", 
                            f"Automatically locked with {confidence:.2f} confidence"
                        )
                        new_locks.append(lock)
        
        return new_locks
    
    def _make_room_for_lock(self, new_suspect_id: str) -> bool:
        """Make room for a new lock by removing lowest priority lock"""
        if not self.active_locks:
            return True
        
        # Find lowest priority lock
        lowest_priority = min(
            self.active_locks.values(),
            key=lambda lock: (lock.profile.priority.value, lock.lock_time)
        )
        
        # Get priority of new suspect
        new_priority = LockPriority.MEDIUM
        if new_suspect_id in self.suspect_profiles:
            new_priority = self.suspect_profiles[new_suspect_id].priority
        
        # Remove if new suspect has higher priority
        if new_priority.value > lowest_priority.profile.priority.value:
            self._remove_lock(lowest_priority.suspect_id)
            return True
        
        return False
    
    def _remove_lock(self, suspect_id: str):
        """Remove a lock"""
        if suspect_id in self.active_locks:
            lock = self.active_locks[suspect_id]
            
            # Remove from tracking mappings
            if lock.track_id in self.track_to_suspect:
                del self.track_to_suspect[lock.track_id]
            
            del self.active_locks[suspect_id]
    
    def _update_stats(self):
        """Update statistics"""
        self.stats['active_locks'] = len(self.active_locks)
        self.stats['lost_locks'] = len([
            lock for lock in self.active_locks.values()
            if lock.status == SuspectStatus.LOST
        ])
        self.stats['confirmed_locks'] = len([
            lock for lock in self.active_locks.values()
            if lock.status == SuspectStatus.CONFIRMED
        ])
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get lock manager statistics"""
        return {
            **self.stats,
            'total_profiles': len(self.suspect_profiles),
            'high_priority_locks': len(self.get_high_priority_locks())
        }
    
    def export_suspect_data(self, suspect_id: str) -> Optional[Dict[str, Any]]:
        """Export all data for a suspect"""
        if suspect_id not in self.suspect_profiles:
            return None
        
        profile = self.suspect_profiles[suspect_id]
        lock = self.active_locks.get(suspect_id)
        
        data = {
            'profile': {
                'id': profile.id,
                'name': profile.name,
                'description': profile.description,
                'priority': profile.priority.name,
                'created_at': profile.created_at,
                'metadata': profile.metadata
            }
        }
        
        if lock:
            data['lock'] = {
                'track_id': lock.track_id,
                'status': lock.status.value,
                'confidence': lock.confidence,
                'lock_time': lock.lock_time,
                'last_seen': lock.last_seen,
                'trajectory': lock.get_trajectory(),
                'alerts': lock.alerts
            }
        
        return data
    
    def clear_all_locks(self):
        """Clear all active locks"""
        for suspect_id in list(self.active_locks.keys()):
            self.unlock_suspect(suspect_id)
        
        logging.info("Cleared all suspect locks")