"""
Tracking Module for Foresight SAR System

This module provides multi-object tracking capabilities including SORT,
DeepSORT algorithms, and suspect lock functionality for continuous
target monitoring in SAR operations.
"""

from .sort_tracker import SORTTracker, Detection, Track, KalmanBoxTracker
from .deepsort_tracker import DeepSORTTracker, DeepSORTTrack, AppearanceFeature
from .suspect_lock import SuspectLock, SuspectLockManager, SuspectProfile, LockPriority
from .track_manager import TrackManager, TrackingConfig, TrackerType
from .iou_tracker import IOUTracker
from ..reid.embedder import ReIDEmbedder, EmbeddingConfig, EmbeddingModel

__all__ = [
    'SORTTracker',
    'DeepSORTTracker', 
    'SuspectLock',
    'SuspectLockManager',
    'TrackManager',
    'IOUTracker',
    'ReIDEmbedder',
    'Detection',
    'Track',
    'DeepSORTTrack',
    'AppearanceFeature',
    'SuspectProfile',
    'LockPriority',
    'TrackingConfig',
    'TrackerType',
    'EmbeddingConfig',
    'EmbeddingModel',
    'KalmanBoxTracker'
]

__version__ = '1.0.0'