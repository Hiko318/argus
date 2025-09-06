"""Person Re-Identification (ReID) Module for Foresight SAR System.

This module provides person re-identification capabilities for search and rescue operations,
with a focus on privacy-first design and operator oversight. It includes:

- Deep learning-based person embedding generation
- Privacy filtering with face blurring
- Similarity matching and database management
- Operator confirmation workflows
- SAR-optimized configurations

Privacy Features:
- Automatic face blurring in processed images
- Local processing (no cloud dependencies)
- Operator oversight for all matches
- Configurable privacy levels
- Secure embedding storage

Usage:
    from foresight.reid import ReIDPipeline, create_sar_pipeline, PrivacyFilter
    
    # Create SAR-optimized pipeline
    pipeline = create_sar_pipeline()
    pipeline.start()
    
    # Process detection
    result = pipeline.process_detection_sync(detection)
"""

# Core components
from .embedder import ReIDEmbedder, EmbeddingConfig, EmbeddingModel
from .privacy_filter import (
    PrivacyFilter, 
    PrivacyConfig, 
    FaceBlurrer,
    BlurMethod,
    PrivacyLevel,
    create_privacy_filter
)
from .embedding_manager import (
    EmbeddingManager,
    EmbeddingDatabase,
    ReIDEmbedding,
    MatchResult,
    MatchStatus,
    DistanceMetric
)
from .reid_pipeline import (
    ReIDPipeline,
    PipelineConfig,
    DetectionInput,
    ReIDResult,
    PipelineMode,
    AlertLevel,
    create_default_pipeline,
    create_sar_pipeline
)

# Main components for external use
__all__ = [
    # Core pipeline
    'ReIDPipeline',
    'PipelineConfig',
    'DetectionInput', 
    'ReIDResult',
    'PipelineMode',
    'AlertLevel',
    'create_default_pipeline',
    'create_sar_pipeline',
    
    # Embedding components
    'ReIDEmbedder',
    'EmbeddingConfig',
    'EmbeddingModel',
    'EmbeddingManager',
    'EmbeddingDatabase',
    'ReIDEmbedding',
    'MatchResult',
    'MatchStatus',
    'DistanceMetric',
    
    # Privacy components
    'PrivacyFilter',
    'PrivacyConfig',
    'FaceBlurrer',
    'BlurMethod',
    'PrivacyLevel',
    'create_privacy_filter'
]

__version__ = "1.0.0"