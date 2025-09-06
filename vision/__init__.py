"""Vision Module for Foresight SAR System

This module provides computer vision capabilities including object detection,
image processing, and inference pipelines for human detection in aerial imagery.
"""

from .detector import DetectionResult, YOLODetector
from .detection_pipeline import DetectionPipeline
from .detection_service import DetectionService
from .yolo_infer import YOLOInference

__all__ = [
    'DetectionResult',
    'YOLODetector', 
    'DetectionPipeline',
    'DetectionService',
    'YOLOInference'
]

__version__ = '1.0.0'