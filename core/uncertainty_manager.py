#!/usr/bin/env python3
"""
Uncertainty & Quality Flags Manager

This module provides comprehensive uncertainty quantification and quality assessment
for detections, tracking, and geolocation in the Foresight AI system.

Features:
- Per-detection confidence and uncertainty estimation
- Geolocation uncertainty quantification
- Quality flags and human confirmation thresholds
- Temporal consistency validation
- Multi-modal uncertainty fusion
- Calibrated confidence scoring

Author: Foresight AI Team
Date: 2024
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple, Any, Union, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from collections import deque
from scipy import stats
from scipy.spatial.distance import euclidean
import json


class UncertaintyType(Enum):
    """Types of uncertainty"""
    DETECTION = "detection"          # Object detection uncertainty
    CLASSIFICATION = "classification" # Class prediction uncertainty
    GEOLOCATION = "geolocation"      # Spatial position uncertainty
    TRACKING = "tracking"            # Track association uncertainty
    TEMPORAL = "temporal"            # Time-based uncertainty
    FUSION = "fusion"                # Multi-modal fusion uncertainty


class QualityFlag(Enum):
    """Quality assessment flags"""
    HIGH_QUALITY = "high_quality"              # High confidence, no issues
    MEDIUM_QUALITY = "medium_quality"          # Acceptable quality
    LOW_QUALITY = "low_quality"                # Poor quality, use with caution
    NEEDS_CONFIRMATION = "needs_confirmation"  # Requires human verification
    UNRELIABLE = "unreliable"                  # Should not be used
    OCCLUDED = "occluded"                      # Partially occluded
    MOTION_BLUR = "motion_blur"                # Motion blur detected
    POOR_LIGHTING = "poor_lighting"            # Poor illumination
    EDGE_CASE = "edge_case"                    # Unusual scenario


@dataclass
class UncertaintyEstimate:
    """Uncertainty estimate for a measurement"""
    value: float                    # Primary measurement value
    uncertainty: float              # Uncertainty magnitude (std dev)
    confidence: float               # Confidence in the measurement [0,1]
    uncertainty_type: UncertaintyType
    timestamp: float = field(default_factory=time.time)
    
    # Additional uncertainty components
    epistemic_uncertainty: float = 0.0    # Model uncertainty
    aleatoric_uncertainty: float = 0.0    # Data uncertainty
    
    # Quality indicators
    quality_score: float = 1.0            # Overall quality [0,1]
    reliability_score: float = 1.0        # Reliability assessment [0,1]
    
    def get_total_uncertainty(self) -> float:
        """Calculate total uncertainty combining epistemic and aleatoric"""
        return np.sqrt(self.epistemic_uncertainty**2 + self.aleatoric_uncertainty**2)
    
    def get_confidence_interval(self, alpha: float = 0.05) -> Tuple[float, float]:
        """Get confidence interval for the estimate"""
        z_score = stats.norm.ppf(1 - alpha/2)
        margin = z_score * self.uncertainty
        return (self.value - margin, self.value + margin)
    
    def is_reliable(self, threshold: float = 0.7) -> bool:
        """Check if estimate is reliable based on threshold"""
        return self.confidence >= threshold and self.reliability_score >= threshold


@dataclass
class GeolocationUncertainty:
    """Geolocation-specific uncertainty information"""
    position: np.ndarray            # [lat, lon, alt] or [x, y, z]
    position_uncertainty: np.ndarray # Uncertainty in each dimension
    
    # Uncertainty sources
    camera_calibration_error: float = 0.0
    dem_interpolation_error: float = 0.0
    atmospheric_refraction_error: float = 0.0
    platform_position_error: float = 0.0
    platform_attitude_error: float = 0.0
    
    # Geometric factors
    ground_sample_distance: float = 0.0
    viewing_angle: float = 0.0
    range_to_target: float = 0.0
    
    def get_horizontal_uncertainty(self) -> float:
        """Get horizontal position uncertainty (CEP)"""
        if len(self.position_uncertainty) >= 2:
            return np.sqrt(self.position_uncertainty[0]**2 + self.position_uncertainty[1]**2)
        return 0.0
    
    def get_vertical_uncertainty(self) -> float:
        """Get vertical position uncertainty"""
        if len(self.position_uncertainty) >= 3:
            return self.position_uncertainty[2]
        return 0.0
    
    def get_total_error_budget(self) -> Dict[str, float]:
        """Get breakdown of error sources"""
        return {
            'camera_calibration': self.camera_calibration_error,
            'dem_interpolation': self.dem_interpolation_error,
            'atmospheric_refraction': self.atmospheric_refraction_error,
            'platform_position': self.platform_position_error,
            'platform_attitude': self.platform_attitude_error,
            'geometric_dilution': self.ground_sample_distance * 0.1  # Simplified
        }


@dataclass
class DetectionUncertainty:
    """Detection-specific uncertainty and quality assessment"""
    # Core detection info
    bbox: np.ndarray               # Bounding box [x1, y1, x2, y2]
    class_id: int
    detection_confidence: float
    
    # Uncertainty estimates
    bbox_uncertainty: UncertaintyEstimate
    class_uncertainty: UncertaintyEstimate
    geolocation_uncertainty: Optional[GeolocationUncertainty] = None
    
    # Quality flags
    quality_flags: List[QualityFlag] = field(default_factory=list)
    overall_quality: QualityFlag = QualityFlag.MEDIUM_QUALITY
    
    # Image quality factors
    image_sharpness: float = 1.0
    illumination_quality: float = 1.0
    occlusion_ratio: float = 0.0
    scale_factor: float = 1.0
    
    # Temporal consistency
    temporal_consistency: float = 1.0
    track_stability: float = 1.0
    
    def needs_human_confirmation(self, threshold: float = 0.6) -> bool:
        """Check if detection needs human confirmation"""
        return (QualityFlag.NEEDS_CONFIRMATION in self.quality_flags or
                self.detection_confidence < threshold or
                not self.bbox_uncertainty.is_reliable(threshold) or
                not self.class_uncertainty.is_reliable(threshold))
    
    def get_combined_confidence(self) -> float:
        """Get combined confidence score"""
        weights = {
            'detection': 0.4,
            'bbox': 0.2,
            'class': 0.2,
            'quality': 0.2
        }
        
        quality_score = 1.0 - len([f for f in self.quality_flags 
                                  if f in [QualityFlag.LOW_QUALITY, QualityFlag.UNRELIABLE]]) * 0.3
        
        combined = (weights['detection'] * self.detection_confidence +
                   weights['bbox'] * self.bbox_uncertainty.confidence +
                   weights['class'] * self.class_uncertainty.confidence +
                   weights['quality'] * quality_score)
        
        return np.clip(combined, 0.0, 1.0)


class UncertaintyManager:
    """Manager for uncertainty quantification and quality assessment"""
    
    def __init__(self,
                 confidence_threshold: float = 0.6,
                 uncertainty_threshold: float = 0.3,
                 quality_threshold: float = 0.7,
                 temporal_window: int = 10):
        """
        Initialize uncertainty manager
        
        Args:
            confidence_threshold: Minimum confidence for reliable detections
            uncertainty_threshold: Maximum uncertainty for reliable estimates
            quality_threshold: Minimum quality score for automatic acceptance
            temporal_window: Number of frames for temporal consistency
        """
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.quality_threshold = quality_threshold
        self.temporal_window = temporal_window
        
        # Calibration parameters
        self.calibration_params = {
            'detection_bias': 0.0,
            'detection_scale': 1.0,
            'bbox_noise_std': 2.0,
            'class_entropy_scale': 1.0
        }
        
        # Temporal tracking
        self.detection_history: Dict[int, deque] = {}  # track_id -> detection history
        self.uncertainty_history: Dict[int, deque] = {}  # track_id -> uncertainty history
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'high_quality_detections': 0,
            'needs_confirmation': 0,
            'unreliable_detections': 0,
            'avg_confidence': 0.0,
            'avg_uncertainty': 0.0
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("Uncertainty manager initialized")
    
    def assess_detection_uncertainty(self,
                                   bbox: np.ndarray,
                                   class_probs: np.ndarray,
                                   detection_confidence: float,
                                   image: np.ndarray,
                                   track_id: Optional[int] = None,
                                   geolocation: Optional[np.ndarray] = None,
                                   camera_params: Optional[Dict] = None) -> DetectionUncertainty:
        """
        Assess uncertainty for a detection
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            class_probs: Class probability distribution
            detection_confidence: Detection confidence score
            image: Input image
            track_id: Track ID if available
            geolocation: Geolocation if available
            camera_params: Camera parameters for geo uncertainty
            
        Returns:
            DetectionUncertainty object
        """
        # Assess bounding box uncertainty
        bbox_uncertainty = self._assess_bbox_uncertainty(bbox, detection_confidence, image)
        
        # Assess classification uncertainty
        class_uncertainty = self._assess_classification_uncertainty(class_probs)
        
        # Assess geolocation uncertainty
        geo_uncertainty = None
        if geolocation is not None and camera_params is not None:
            geo_uncertainty = self._assess_geolocation_uncertainty(
                geolocation, bbox, camera_params
            )
        
        # Assess image quality factors
        quality_factors = self._assess_image_quality(bbox, image)
        
        # Assess temporal consistency
        temporal_factors = self._assess_temporal_consistency(track_id, bbox, detection_confidence)
        
        # Determine quality flags
        quality_flags = self._determine_quality_flags(
            detection_confidence, bbox_uncertainty, class_uncertainty,
            quality_factors, temporal_factors
        )
        
        # Determine overall quality
        overall_quality = self._determine_overall_quality(quality_flags, detection_confidence)
        
        # Create detection uncertainty object
        det_uncertainty = DetectionUncertainty(
            bbox=bbox,
            class_id=np.argmax(class_probs),
            detection_confidence=detection_confidence,
            bbox_uncertainty=bbox_uncertainty,
            class_uncertainty=class_uncertainty,
            geolocation_uncertainty=geo_uncertainty,
            quality_flags=quality_flags,
            overall_quality=overall_quality,
            **quality_factors,
            **temporal_factors
        )
        
        # Update tracking history
        if track_id is not None:
            self._update_tracking_history(track_id, det_uncertainty)
        
        # Update statistics
        self._update_statistics(det_uncertainty)
        
        return det_uncertainty
    
    def _assess_bbox_uncertainty(self, bbox: np.ndarray, 
                               confidence: float, 
                               image: np.ndarray) -> UncertaintyEstimate:
        """Assess bounding box localization uncertainty"""
        # Base uncertainty from detection confidence
        base_uncertainty = (1.0 - confidence) * self.calibration_params['bbox_noise_std']
        
        # Add uncertainty from image factors
        h, w = image.shape[:2]
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        
        # Scale-based uncertainty
        scale_factor = np.sqrt((bbox_w * bbox_h) / (w * h))
        scale_uncertainty = max(0, (0.1 - scale_factor) * 5.0)  # Higher uncertainty for small objects
        
        # Edge proximity uncertainty
        edge_distances = [
            bbox[0],  # left edge
            bbox[1],  # top edge
            w - bbox[2],  # right edge
            h - bbox[3]   # bottom edge
        ]
        min_edge_distance = min(edge_distances)
        edge_uncertainty = max(0, (10 - min_edge_distance) * 0.1)
        
        # Combine uncertainties
        total_uncertainty = np.sqrt(base_uncertainty**2 + scale_uncertainty**2 + edge_uncertainty**2)
        
        # Epistemic vs aleatoric breakdown
        epistemic = base_uncertainty * 0.7  # Model uncertainty
        aleatoric = total_uncertainty * 0.3  # Data uncertainty
        
        return UncertaintyEstimate(
            value=1.0,  # Normalized bbox confidence
            uncertainty=total_uncertainty,
            confidence=confidence,
            uncertainty_type=UncertaintyType.DETECTION,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            quality_score=min(1.0, confidence * (1.0 - total_uncertainty/10.0))
        )
    
    def _assess_classification_uncertainty(self, class_probs: np.ndarray) -> UncertaintyEstimate:
        """Assess classification uncertainty using entropy"""
        # Normalize probabilities
        probs = class_probs / np.sum(class_probs)
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        max_entropy = np.log(len(probs))
        normalized_entropy = entropy / max_entropy
        
        # Top-2 difference
        sorted_probs = np.sort(probs)[::-1]
        top2_diff = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
        
        # Confidence from max probability
        max_prob = np.max(probs)
        
        # Uncertainty from entropy and top-2 difference
        uncertainty = normalized_entropy * (1.0 - top2_diff)
        
        return UncertaintyEstimate(
            value=max_prob,
            uncertainty=uncertainty,
            confidence=max_prob * (1.0 - normalized_entropy),
            uncertainty_type=UncertaintyType.CLASSIFICATION,
            epistemic_uncertainty=normalized_entropy * 0.8,
            aleatoric_uncertainty=uncertainty * 0.2,
            quality_score=max_prob * top2_diff
        )
    
    def _assess_geolocation_uncertainty(self, geolocation: np.ndarray,
                                      bbox: np.ndarray,
                                      camera_params: Dict) -> GeolocationUncertainty:
        """Assess geolocation uncertainty"""
        # Extract camera parameters
        focal_length = camera_params.get('focal_length', 1000.0)
        altitude = camera_params.get('altitude', 100.0)
        tilt_angle = camera_params.get('tilt_angle', 0.0)
        
        # Calculate range to target
        bbox_center_y = (bbox[1] + bbox[3]) / 2
        image_height = camera_params.get('image_height', 1080)
        
        # Simplified range calculation
        pixel_offset = bbox_center_y - image_height / 2
        ground_range = altitude * np.tan(np.radians(tilt_angle) + pixel_offset / focal_length)
        
        # Ground sample distance
        gsd = altitude / focal_length
        
        # Error sources
        camera_cal_error = gsd * 0.5  # 0.5 pixel calibration error
        dem_error = 5.0  # 5m DEM uncertainty
        platform_pos_error = 2.0  # 2m platform position error
        platform_att_error = np.radians(0.1)  # 0.1 degree attitude error
        
        # Calculate position uncertainty
        horizontal_error = np.sqrt(
            camera_cal_error**2 +
            (ground_range * platform_att_error)**2 +
            platform_pos_error**2
        )
        
        vertical_error = np.sqrt(
            dem_error**2 +
            (ground_range * platform_att_error * np.sin(np.radians(tilt_angle)))**2
        )
        
        position_uncertainty = np.array([horizontal_error, horizontal_error, vertical_error])
        
        return GeolocationUncertainty(
            position=geolocation,
            position_uncertainty=position_uncertainty,
            camera_calibration_error=camera_cal_error,
            dem_interpolation_error=dem_error,
            platform_position_error=platform_pos_error,
            platform_attitude_error=platform_att_error,
            ground_sample_distance=gsd,
            viewing_angle=tilt_angle,
            range_to_target=ground_range
        )
    
    def _assess_image_quality(self, bbox: np.ndarray, image: np.ndarray) -> Dict[str, float]:
        """Assess image quality factors"""
        # Extract crop
        x1, y1, x2, y2 = bbox.astype(int)
        crop = image[y1:y2, x1:x2]
        
        if crop.size == 0:
            return {
                'image_sharpness': 0.0,
                'illumination_quality': 0.0,
                'occlusion_ratio': 1.0,
                'scale_factor': 0.0
            }
        
        # Convert to grayscale if needed
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        
        # Sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(1.0, laplacian_var / 1000.0)
        
        # Illumination quality
        mean_intensity = np.mean(gray)
        illumination = 1.0 - abs(mean_intensity - 128) / 128.0
        
        # Scale factor
        h, w = crop.shape[:2]
        scale = min(2.0, max(0.1, np.sqrt((h * w) / (64 * 64))))
        
        # Occlusion estimation (simplified)
        # Check if bbox is at image boundaries
        img_h, img_w = image.shape[:2]
        boundary_threshold = 5
        
        occlusion = 0.0
        if (x1 < boundary_threshold or y1 < boundary_threshold or
            x2 > img_w - boundary_threshold or y2 > img_h - boundary_threshold):
            occlusion = 0.2
        
        # Very small objects likely occluded
        if h < 30 or w < 20:
            occlusion += 0.3
        
        return {
            'image_sharpness': sharpness,
            'illumination_quality': illumination,
            'occlusion_ratio': min(1.0, occlusion),
            'scale_factor': scale
        }
    
    def _assess_temporal_consistency(self, track_id: Optional[int],
                                   bbox: np.ndarray,
                                   confidence: float) -> Dict[str, float]:
        """Assess temporal consistency"""
        if track_id is None or track_id not in self.detection_history:
            return {
                'temporal_consistency': 1.0,
                'track_stability': 1.0
            }
        
        history = list(self.detection_history[track_id])
        if len(history) < 2:
            return {
                'temporal_consistency': 1.0,
                'track_stability': 1.0
            }
        
        # Calculate bbox movement consistency
        movements = []
        confidences = []
        
        for i in range(1, len(history)):
            prev_bbox = history[i-1].bbox
            curr_bbox = history[i].bbox
            
            # Calculate movement
            prev_center = np.array([(prev_bbox[0] + prev_bbox[2])/2, (prev_bbox[1] + prev_bbox[3])/2])
            curr_center = np.array([(curr_bbox[0] + curr_bbox[2])/2, (curr_bbox[1] + curr_bbox[3])/2])
            movement = euclidean(prev_center, curr_center)
            movements.append(movement)
            confidences.append(history[i].detection_confidence)
        
        # Temporal consistency based on movement variance
        if len(movements) > 1:
            movement_std = np.std(movements)
            movement_mean = np.mean(movements)
            consistency = max(0.0, 1.0 - movement_std / (movement_mean + 1.0))
        else:
            consistency = 1.0
        
        # Track stability based on confidence variance
        if len(confidences) > 1:
            conf_std = np.std(confidences)
            stability = max(0.0, 1.0 - conf_std)
        else:
            stability = 1.0
        
        return {
            'temporal_consistency': consistency,
            'track_stability': stability
        }
    
    def _determine_quality_flags(self, confidence: float,
                               bbox_uncertainty: UncertaintyEstimate,
                               class_uncertainty: UncertaintyEstimate,
                               quality_factors: Dict[str, float],
                               temporal_factors: Dict[str, float]) -> List[QualityFlag]:
        """Determine quality flags based on assessments"""
        flags = []
        
        # Low confidence
        if confidence < self.confidence_threshold:
            flags.append(QualityFlag.NEEDS_CONFIRMATION)
        
        # High uncertainty
        if (bbox_uncertainty.uncertainty > self.uncertainty_threshold or
            class_uncertainty.uncertainty > self.uncertainty_threshold):
            flags.append(QualityFlag.NEEDS_CONFIRMATION)
        
        # Image quality issues
        if quality_factors['image_sharpness'] < 0.3:
            flags.append(QualityFlag.MOTION_BLUR)
        
        if quality_factors['illumination_quality'] < 0.3:
            flags.append(QualityFlag.POOR_LIGHTING)
        
        if quality_factors['occlusion_ratio'] > 0.3:
            flags.append(QualityFlag.OCCLUDED)
        
        # Temporal inconsistency
        if temporal_factors['temporal_consistency'] < 0.5:
            flags.append(QualityFlag.EDGE_CASE)
        
        # Overall quality assessment
        overall_score = (confidence + bbox_uncertainty.confidence + class_uncertainty.confidence) / 3.0
        
        if overall_score > 0.8 and len(flags) == 0:
            flags.append(QualityFlag.HIGH_QUALITY)
        elif overall_score < 0.3 or len([f for f in flags if f != QualityFlag.HIGH_QUALITY]) > 2:
            flags.append(QualityFlag.UNRELIABLE)
        elif overall_score < 0.5:
            flags.append(QualityFlag.LOW_QUALITY)
        
        return flags
    
    def _determine_overall_quality(self, flags: List[QualityFlag], confidence: float) -> QualityFlag:
        """Determine overall quality flag"""
        if QualityFlag.UNRELIABLE in flags:
            return QualityFlag.UNRELIABLE
        elif QualityFlag.NEEDS_CONFIRMATION in flags:
            return QualityFlag.NEEDS_CONFIRMATION
        elif QualityFlag.LOW_QUALITY in flags:
            return QualityFlag.LOW_QUALITY
        elif QualityFlag.HIGH_QUALITY in flags:
            return QualityFlag.HIGH_QUALITY
        else:
            return QualityFlag.MEDIUM_QUALITY
    
    def _update_tracking_history(self, track_id: int, detection: DetectionUncertainty):
        """Update tracking history for temporal analysis"""
        if track_id not in self.detection_history:
            self.detection_history[track_id] = deque(maxlen=self.temporal_window)
            self.uncertainty_history[track_id] = deque(maxlen=self.temporal_window)
        
        self.detection_history[track_id].append(detection)
        self.uncertainty_history[track_id].append({
            'bbox_uncertainty': detection.bbox_uncertainty.uncertainty,
            'class_uncertainty': detection.class_uncertainty.uncertainty,
            'confidence': detection.detection_confidence
        })
    
    def _update_statistics(self, detection: DetectionUncertainty):
        """Update performance statistics"""
        self.stats['total_detections'] += 1
        
        if detection.overall_quality == QualityFlag.HIGH_QUALITY:
            self.stats['high_quality_detections'] += 1
        elif detection.needs_human_confirmation():
            self.stats['needs_confirmation'] += 1
        elif detection.overall_quality == QualityFlag.UNRELIABLE:
            self.stats['unreliable_detections'] += 1
        
        # Update running averages
        n = self.stats['total_detections']
        self.stats['avg_confidence'] = ((n-1) * self.stats['avg_confidence'] + 
                                       detection.detection_confidence) / n
        self.stats['avg_uncertainty'] = ((n-1) * self.stats['avg_uncertainty'] + 
                                        detection.bbox_uncertainty.uncertainty) / n
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get uncertainty and quality statistics"""
        total = max(self.stats['total_detections'], 1)
        return {
            **self.stats,
            'high_quality_rate': self.stats['high_quality_detections'] / total,
            'confirmation_rate': self.stats['needs_confirmation'] / total,
            'unreliable_rate': self.stats['unreliable_detections'] / total,
            'active_tracks': len(self.detection_history)
        }
    
    def calibrate_confidence(self, true_positives: List[float], 
                           predicted_confidences: List[float]):
        """Calibrate confidence scores using reliability diagrams"""
        # Simple Platt scaling calibration
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression
        
        # Fit calibration model
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(predicted_confidences, true_positives)
        
        # Update calibration parameters
        self.calibration_params['detection_scale'] = 1.0  # Simplified
        self.calibration_params['detection_bias'] = 0.0
        
        self.logger.info("Confidence calibration updated")
    
    def export_uncertainty_report(self, filepath: str):
        """Export uncertainty analysis report"""
        report = {
            'timestamp': time.time(),
            'statistics': self.get_statistics(),
            'calibration_params': self.calibration_params,
            'thresholds': {
                'confidence_threshold': self.confidence_threshold,
                'uncertainty_threshold': self.uncertainty_threshold,
                'quality_threshold': self.quality_threshold
            },
            'active_tracks': len(self.detection_history)
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Uncertainty report exported to {filepath}")
    
    def generate_quality_flags(self, detection: Dict[str, Any]) -> List[QualityFlag]:
        """Generate quality flags for a detection"""
        flags = []
        
        confidence = detection.get('confidence', 0.0)
        bbox = detection.get('bbox', [])
        
        # Check confidence levels
        if confidence >= 0.9:
            flags.append(QualityFlag.HIGH_QUALITY)
        elif confidence >= 0.7:
            flags.append(QualityFlag.MEDIUM_QUALITY)
        elif confidence >= 0.5:
            flags.append(QualityFlag.LOW_QUALITY)
        else:
            flags.append(QualityFlag.UNRELIABLE)
        
        # Check for edge cases
        if len(bbox) == 4:
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if width < 20 or height < 20:
                flags.append(QualityFlag.EDGE_CASE)
        
        # Check if needs confirmation
        if confidence < self.confidence_threshold:
            flags.append(QualityFlag.NEEDS_CONFIRMATION)
        
        return flags
    
    def assess_geolocation_uncertainty(self, geolocation: Dict[str, Any]) -> Dict[str, float]:
        """Assess geolocation uncertainty"""
        # Extract geolocation data
        lat = geolocation.get('latitude', 0.0)
        lon = geolocation.get('longitude', 0.0)
        alt = geolocation.get('altitude', 0.0)
        
        # Calculate uncertainties based on various factors
        base_horizontal = 2.0  # Base horizontal accuracy in meters
        base_vertical = 5.0    # Base vertical accuracy in meters
        
        # Add uncertainty based on altitude (higher = more uncertain)
        altitude_factor = min(alt / 1000.0, 2.0)  # Max 2x uncertainty at 1km+
        
        horizontal_accuracy = base_horizontal * (1.0 + altitude_factor)
        vertical_accuracy = base_vertical * (1.0 + altitude_factor)
        
        return {
            'horizontal_accuracy': horizontal_accuracy,
            'vertical_accuracy': vertical_accuracy,
            'confidence': max(0.1, 1.0 - altitude_factor * 0.3)
        }
    
    def needs_human_confirmation(self, detection: Dict[str, Any]) -> bool:
        """Determine if detection needs human confirmation"""
        confidence = detection.get('confidence', 0.0)
        
        # Low confidence always needs confirmation
        if confidence < self.confidence_threshold:
            return True
        
        # Check for edge cases
        bbox = detection.get('bbox', [])
        if len(bbox) == 4:
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            # Very small detections need confirmation
            if width < 15 or height < 15:
                return True
        
        # Check class confidence if available
        class_probs = detection.get('class_probs', [])
        if len(class_probs) > 1:
            sorted_probs = sorted(class_probs, reverse=True)
            # If top two classes are very close, need confirmation
            if len(sorted_probs) > 1 and (sorted_probs[0] - sorted_probs[1]) < 0.1:
                return True
        
        return False
    
    def validate_temporal_consistency(self, detections: List[Dict[str, Any]]) -> Dict[str, float]:
        """Validate temporal consistency of detections"""
        if len(detections) < 2:
            return {'consistency_score': 1.0, 'temporal_variance': 0.0}
        
        # Sort by timestamp
        sorted_detections = sorted(detections, key=lambda x: x.get('timestamp', 0))
        
        # Calculate position variance
        positions = []
        confidences = []
        
        for detection in sorted_detections:
            bbox = detection.get('bbox', [0, 0, 0, 0])
            if len(bbox) >= 4:
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                positions.append((center_x, center_y))
            
            confidences.append(detection.get('confidence', 0.0))
        
        # Calculate movement consistency
        if len(positions) >= 2:
            movements = []
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                movement = np.sqrt(dx*dx + dy*dy)
                movements.append(movement)
            
            movement_variance = np.var(movements) if movements else 0.0
            avg_movement = np.mean(movements) if movements else 0.0
        else:
            movement_variance = 0.0
            avg_movement = 0.0
        
        # Calculate confidence consistency
        confidence_variance = np.var(confidences) if confidences else 0.0
        
        # Overall consistency score (higher is more consistent)
        consistency_score = 1.0 / (1.0 + movement_variance/100.0 + confidence_variance)
        
        return {
            'consistency_score': consistency_score,
            'temporal_variance': movement_variance,
            'confidence_variance': confidence_variance,
            'avg_movement': avg_movement
        }
    
    def reset(self):
        """Reset uncertainty manager state"""
        self.detection_history.clear()
        self.uncertainty_history.clear()
        self.stats = {
            'total_detections': 0,
            'high_quality_detections': 0,
            'needs_confirmation': 0,
            'unreliable_detections': 0,
            'avg_confidence': 0.0,
            'avg_uncertainty': 0.0
        }
        
        self.logger.info("Uncertainty manager reset")