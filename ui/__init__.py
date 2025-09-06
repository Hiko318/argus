#!/usr/bin/env python3
"""
Foresight SAR System - UI Module

This module provides user interface components and utilities for the Foresight
Search and Rescue (SAR) system, including offline map management, heatmap
generation, and snapshot packaging for evidence collection.
"""

# Import main classes
from .offline_maps import (
    TileInfo,
    BoundingBox,
    TileSource,
    OfflineMapManager
)

from .heatmap_generator import (
    DetectionPoint,
    HeatmapConfig,
    BoundingBox as HeatmapBoundingBox,
    HeatmapGenerator,
    create_sample_detections
)

from .snapshot_packaging import (
    MissionMetadata,
    DetectionSnapshot,
    TrackingSnapshot,
    TelemetrySnapshot,
    PackagingConfig,
    SnapshotPackager,
    create_sample_mission_data
)

# Version information
__version__ = "1.0.0"
__author__ = "Foresight SAR Team"

# Export all public classes and functions
__all__ = [
    # Offline Maps
    "TileInfo",
    "BoundingBox",
    "TileSource", 
    "OfflineMapManager",
    
    # Heatmap Generation
    "DetectionPoint",
    "HeatmapConfig",
    "HeatmapBoundingBox",
    "HeatmapGenerator",
    "create_sample_detections",
    
    # Snapshot Packaging
    "MissionMetadata",
    "DetectionSnapshot",
    "TrackingSnapshot",
    "TelemetrySnapshot",
    "PackagingConfig",
    "SnapshotPackager",
    "create_sample_mission_data",
    
    # Module info
    "__version__",
    "__author__"
]