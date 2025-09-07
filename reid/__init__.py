"""Re-identification module for Victim-Lock.

This module provides a comprehensive re-identification pipeline for SAR operations,
featuring a two-branch approach:
1. Face embedding (ArcFace/FaceNet) when faces are visible
2. Full-body re-identification when faces are occluded or not visible

Key components:
- FaceEmbedder: Face detection and embedding generation
- BodyReIDEmbedder: Full-body re-identification
- ReIDPipeline: Main pipeline orchestrating both approaches
- GalleryManager: Target person gallery management
- VictimLockPipeline: Complete Victim-Lock functionality

Quick Start:
    >>> from reid import VictimLockPipeline, DetectionInput
    >>> import numpy as np
    >>> 
    >>> # Initialize pipeline
    >>> pipeline = VictimLockPipeline(gallery_dir="data/gallery")
    >>> 
    >>> # Add target person
    >>> target_images = [np.array(...)]  # List of reference images
    >>> person_id = pipeline.add_target_person(
    ...     name="John Doe",
    ...     description="Missing person",
    ...     reference_images=target_images
    ... )
    >>> 
    >>> # Process detection
    >>> detection = DetectionInput(
    ...     image=frame,
    ...     detection_bbox=(x, y, w, h),
    ...     detection_confidence=0.9,
    ...     timestamp=time.time(),
    ...     source_id="camera_1"
    ... )
    >>> result = pipeline.process_detection(detection)
    >>> 
    >>> # Check for matches
    >>> if result.matches:
    ...     print(f"Found {len(result.matches)} potential matches")
    ...     for match in result.matches:
    ...         print(f"Person: {match.person_name}, Similarity: {match.similarity:.3f}")
"""

# Import all components
from .face_embedder import FaceEmbedder, FaceModel
from .body_reid import BodyReIDEmbedder, BodyReIDModel
from .gallery_manager import GalleryManager, MatchResult, MatchConfidence
from .victim_lock_pipeline import (
    VictimLockPipeline, 
    DetectionInput, 
    ReIDResult, 
    PipelineMode, 
    AlertLevel
)
from .reid_pipeline import (
    ReIDPipeline,
    create_default_pipeline,
    create_sar_pipeline
)

# Import test utilities
try:
    from .test_victim_lock import VictimLockTester
    _test_available = True
except ImportError:
    _test_available = False

__version__ = "1.0.0"

__all__ = [
    # Main pipelines
    'VictimLockPipeline',
    'ReIDPipeline',
    
    # Face embedding
    'FaceEmbedder',
    'FaceModel',
    
    # Body re-identification
    'BodyReIDEmbedder',
    'BodyReIDModel',
    
    # Gallery management
    'GalleryManager',
    'MatchResult',
    'MatchConfidence',
    
    # Data structures and enums
    'DetectionInput',
    'ReIDResult',
    'PipelineMode',
    'AlertLevel',
    
    # Utility functions
    'create_default_pipeline',
    'create_sar_pipeline',
]

# Add test utilities if available
if _test_available:
    __all__.append('VictimLockTester')


def get_version():
    """Get the current version of the reid module."""
    return __version__


def list_available_models():
    """List all available face and body models."""
    return {
        'face_models': [model.value for model in FaceModel],
        'body_models': [model.value for model in BodyReIDModel]
    }


def create_quick_pipeline(gallery_dir: str = "data/gallery", 
                         device: str = "auto") -> VictimLockPipeline:
    """Create a quick-start pipeline with sensible defaults.
    
    Args:
        gallery_dir: Directory for storing gallery data
        device: Device for inference ('cpu', 'cuda', 'auto')
        
    Returns:
        Configured VictimLockPipeline ready for use
    """
    return VictimLockPipeline(
        gallery_dir=gallery_dir,
        face_model=FaceModel.ARCFACE,
        body_model=BodyReIDModel.OSNET,
        mode=PipelineMode.AUTO,
        device=device
    )