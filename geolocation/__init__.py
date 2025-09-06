"""Geolocation Module for Foresight SAR System

This module provides geolocation capabilities including camera calibration,
pinhole projection, coordinate transformations, and DEM correction for
accurate geographic positioning of detected objects.
"""

# Camera calibration
from .camera_calibration import (
    CameraCalibrator,
    CalibrationData,
    CameraModel,
    DistortionModel
)

# Projection and pose
from .projection import (
    PinholeProjector,
    CameraPose,
    Ray3D,
    CoordinateSystem
)

# Coordinate transformations
from .coordinate_transform import (
    CoordinateTransformer,
    GeodeticCoordinate,
    UTMCoordinate,
    WGS84Constants
)

# DEM correction
from .dem_correction import (
    DEMCorrector,
    DEMTile,
    DEMMetadata,
    InterpolationMethod,
    DEMFormat
)

# Approximate position
from .approx_pos import ApproximatePosition

# Main service
from .geolocation_service import (
    GeolocationService,
    GeolocationResult,
    GeolocationConfig,
    GeolocationMethod,
    ConfidenceLevel
)

__all__ = [
    # Camera calibration
    'CameraCalibrator',
    'CalibrationData',
    'CameraModel',
    'DistortionModel',
    
    # Projection and pose
    'PinholeProjector',
    'CameraPose',
    'Ray3D',
    'CoordinateSystem',
    
    # Coordinate transformations
    'CoordinateTransformer',
    'GeodeticCoordinate',
    'UTMCoordinate',
    'WGS84Constants',
    
    # DEM correction
    'DEMCorrector',
    'DEMTile',
    'DEMMetadata',
    'InterpolationMethod',
    'DEMFormat',
    
    # Approximate position
    'ApproximatePosition',
    
    # Main service
    'GeolocationService',
    'GeolocationResult',
    'GeolocationConfig',
    'GeolocationMethod',
    'ConfidenceLevel'
]

__version__ = '1.0.0'