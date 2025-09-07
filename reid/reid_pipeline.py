"""Person re-identification pipeline for SAR operations.

Integrates detection, embedding extraction, and matching for victim identification
and tracking across multiple camera feeds and time periods.

This module provides a high-level interface to the complete Victim-Lock pipeline,
including face and body re-identification, gallery management, and consistency testing.
"""

import time
import logging
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import cv2
from pathlib import Path

# Import Victim-Lock components
from .victim_lock_pipeline import (
    VictimLockPipeline, 
    DetectionInput, 
    ReIDResult, 
    PipelineMode, 
    AlertLevel
)
from .face_embedder import FaceEmbedder, FaceModel
from .body_reid import BodyReIDEmbedder, BodyReIDModel
from .gallery_manager import GalleryManager, MatchResult, MatchConfidence


class ReIDPipeline:
    """Main re-identification pipeline for SAR operations.
    
    This class provides a high-level interface to the Victim-Lock pipeline,
    maintaining backward compatibility while offering enhanced functionality.
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 gallery_dir: str = "data/gallery",
                 face_model: FaceModel = FaceModel.ARCFACE,
                 body_model: BodyReIDModel = BodyReIDModel.OSNET,
                 mode: PipelineMode = PipelineMode.AUTO,
                 device: str = "auto"):
        """
        Initialize the re-identification pipeline.
        
        Args:
            config_path: Path to configuration file (optional)
            gallery_dir: Directory for gallery storage
            face_model: Face embedding model to use
            body_model: Body re-identification model to use
            mode: Pipeline operation mode
            device: Device for inference ('cpu', 'cuda', 'auto')
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize the Victim-Lock pipeline
        self.victim_lock = VictimLockPipeline(
            gallery_dir=gallery_dir,
            face_model=face_model,
            body_model=body_model,
            mode=mode,
            device=device
        )
        
        self.logger.info("ReID pipeline initialized with Victim-Lock backend")
    
    def process_detection(self, detection: DetectionInput) -> ReIDResult:
        """Process a single detection for re-identification.
        
        Args:
            detection: Detection input with image and bounding box
            
        Returns:
            Re-identification result with matches and alert level
        """
        return self.victim_lock.process_detection(detection)
    
    def process_batch(self, detections: List[DetectionInput]) -> List[ReIDResult]:
        """Process multiple detections in batch.
        
        Args:
            detections: List of detection inputs
            
        Returns:
            List of re-identification results
        """
        return self.victim_lock.process_batch(detections)
    
    def add_target_person(self, 
                         name: str,
                         description: str,
                         reference_images: List[np.ndarray],
                         priority: int = 1) -> str:
        """Add a target person to the gallery.
        
        Args:
            name: Person's name or identifier
            description: Description of the person
            reference_images: List of reference images (BGR format)
            priority: Priority level (1=highest, 5=lowest)
            
        Returns:
            Gallery entry ID
        """
        return self.victim_lock.add_target_person(
            name=name,
            description=description,
            reference_images=reference_images,
            priority=priority
        )
    
    def remove_target_person(self, person_id: str) -> bool:
        """Remove a target person from the gallery.
        
        Args:
            person_id: Gallery entry ID
            
        Returns:
            True if successful, False otherwise
        """
        return self.victim_lock.gallery_manager.remove_person(person_id)
    
    def list_target_persons(self) -> List[Dict[str, Any]]:
        """List all target persons in the gallery.
        
        Returns:
            List of person information dictionaries
        """
        return self.victim_lock.gallery_manager.list_persons()
    
    def run_consistency_test(self, 
                           gallery_images: Dict[str, List[np.ndarray]],
                           probe_images: Dict[str, List[np.ndarray]],
                           similarity_thresholds: List[float] = None) -> Dict[str, Any]:
        """Run consistency test: gallery vs probe images.
        
        Args:
            gallery_images: Dict mapping person_id to list of gallery images
            probe_images: Dict mapping person_id to list of probe images
            similarity_thresholds: List of thresholds to test
            
        Returns:
            Test results with TPR, FPR, and other metrics
        """
        return self.victim_lock.run_consistency_test(
            gallery_images=gallery_images,
            probe_images=probe_images,
            similarity_thresholds=similarity_thresholds
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics.
        
        Returns:
            Performance metrics dictionary
        """
        return self.victim_lock.get_performance_stats()
    
    def set_alert_thresholds(self, thresholds: Dict[AlertLevel, float]):
        """Update alert thresholds.
        
        Args:
            thresholds: Dictionary mapping alert levels to similarity thresholds
        """
        self.victim_lock.set_alert_thresholds(thresholds)
    
    def set_quality_thresholds(self, face_threshold: float = None, body_threshold: float = None):
        """Update quality thresholds.
        
        Args:
            face_threshold: Minimum face quality threshold
            body_threshold: Minimum body quality threshold
        """
        self.victim_lock.set_quality_thresholds(face_threshold, body_threshold)
    
    def export_gallery(self, export_path: str) -> bool:
        """Export gallery to file.
        
        Args:
            export_path: Path to export file
            
        Returns:
            True if successful, False otherwise
        """
        return self.victim_lock.export_gallery(export_path)
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.victim_lock.reset_stats()
    
    def cleanup(self):
        """Cleanup resources."""
        self.victim_lock.cleanup()
    
    # Properties for accessing underlying components
    @property
    def face_embedder(self) -> FaceEmbedder:
        """Access to face embedder component."""
        return self.victim_lock.face_embedder
    
    @property
    def body_embedder(self) -> BodyReIDEmbedder:
        """Access to body re-identification component."""
        return self.victim_lock.body_embedder
    
    @property
    def gallery_manager(self) -> GalleryManager:
        """Access to gallery manager component."""
        return self.victim_lock.gallery_manager


def create_default_pipeline(gallery_dir: str = "data/gallery") -> ReIDPipeline:
    """
    Create a default re-identification pipeline
    
    Args:
        gallery_dir: Directory for gallery storage
        
    Returns:
        Configured ReIDPipeline
    """
    return ReIDPipeline(
        gallery_dir=gallery_dir,
        face_model=FaceModel.ARCFACE,
        body_model=BodyReIDModel.OSNET,
        mode=PipelineMode.AUTO,
        device="auto"
    )


def create_sar_pipeline(gallery_dir: str = "data/sar_gallery") -> ReIDPipeline:
    """
    Create a SAR-optimized re-identification pipeline
    
    Args:
        gallery_dir: Directory for gallery storage
        
    Returns:
        SAR-optimized ReIDPipeline
    """
    return ReIDPipeline(
        gallery_dir=gallery_dir,
        face_model=FaceModel.MOBILEFACENET,  # Faster for real-time SAR
        body_model=BodyReIDModel.OSNET,     # Lighter model for SAR
        mode=PipelineMode.REAL_TIME,
        device="auto"
    )


# Re-export main classes for backward compatibility
__all__ = [
    'ReIDPipeline',
    'VictimLockPipeline',
    'DetectionInput',
    'ReIDResult',
    'PipelineMode',
    'AlertLevel',
    'FaceEmbedder',
    'FaceModel',
    'BodyReIDEmbedder', 
    'BodyReIDModel',
    'GalleryManager',
    'MatchResult',
    'MatchConfidence'
]