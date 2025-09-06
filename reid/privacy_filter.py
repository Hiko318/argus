"""Privacy Filter Module for SAR Re-Identification System.

This module implements privacy-preserving features including automatic face
blurring, biometric data protection, and privacy-first processing workflows.
All processing is designed to protect individual privacy while maintaining
operational effectiveness for SAR operations.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe not available. Using OpenCV face detection.")

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    logging.warning("dlib not available. Face detection may be limited.")


class BlurMethod(Enum):
    """Available face blurring methods"""
    GAUSSIAN = "gaussian"
    PIXELATE = "pixelate"
    BLACK_BOX = "black_box"
    MOSAIC = "mosaic"
    ADAPTIVE = "adaptive"


class PrivacyLevel(Enum):
    """Privacy protection levels"""
    MINIMAL = "minimal"      # Basic face blurring
    STANDARD = "standard"    # Face + identifying features
    HIGH = "high"           # Full body anonymization
    MAXIMUM = "maximum"     # Complete person anonymization


@dataclass
class PrivacyConfig:
    """Configuration for privacy filtering"""
    privacy_level: PrivacyLevel = PrivacyLevel.STANDARD
    blur_method: BlurMethod = BlurMethod.GAUSSIAN
    blur_strength: int = 15
    face_detection_confidence: float = 0.5
    enable_face_blur: bool = True
    enable_body_blur: bool = False
    enable_audit_logging: bool = True
    preserve_body_features: bool = True  # For ReID while protecting face
    anonymize_metadata: bool = True


class FaceBlurrer:
    """Face detection and blurring implementation"""
    
    def __init__(self, config: PrivacyConfig = None):
        """
        Initialize face blurrer
        
        Args:
            config: Privacy configuration
        """
        self.config = config or PrivacyConfig()
        self.face_detector = None
        self.face_mesh = None
        
        # Initialize face detection
        self._init_face_detection()
    
    def _init_face_detection(self):
        """Initialize face detection models"""
        try:
            if MEDIAPIPE_AVAILABLE:
                # Use MediaPipe for accurate face detection
                self.face_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=1,  # Full range model
                    min_detection_confidence=self.config.face_detection_confidence
                )
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=10,
                    refine_landmarks=True,
                    min_detection_confidence=self.config.face_detection_confidence
                )
                logging.info("Initialized MediaPipe face detection")
            else:
                # Fallback to OpenCV Haar cascades
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_detector = cv2.CascadeClassifier(cascade_path)
                logging.info("Initialized OpenCV face detection")
                
        except Exception as e:
            logging.error(f"Failed to initialize face detection: {e}")
            # Create dummy detector that returns no faces
            self.face_detector = None
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        if self.face_detector is None:
            return []
        
        try:
            if MEDIAPIPE_AVAILABLE and hasattr(self.face_detector, 'process'):
                # MediaPipe detection
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.face_detector.process(rgb_image)
                
                faces = []
                if results.detections:
                    h, w = image.shape[:2]
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        faces.append((x, y, width, height))
                
                return faces
            else:
                # OpenCV detection
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                return [(x, y, w, h) for x, y, w, h in faces]
                
        except Exception as e:
            logging.error(f"Face detection failed: {e}")
            return []
    
    def blur_face(self, image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Blur a single face region
        
        Args:
            image: Input image
            face_bbox: Face bounding box (x, y, w, h)
            
        Returns:
            Image with blurred face
        """
        x, y, w, h = face_bbox
        
        # Expand bounding box slightly for better coverage
        padding = max(w, h) // 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        # Extract face region
        face_region = image[y:y+h, x:x+w].copy()
        
        # Apply blurring based on method
        if self.config.blur_method == BlurMethod.GAUSSIAN:
            kernel_size = self.config.blur_strength * 2 + 1
            blurred = cv2.GaussianBlur(face_region, (kernel_size, kernel_size), 0)
        
        elif self.config.blur_method == BlurMethod.PIXELATE:
            # Pixelate effect
            pixel_size = max(1, self.config.blur_strength // 3)
            small = cv2.resize(face_region, (w//pixel_size, h//pixel_size), interpolation=cv2.INTER_LINEAR)
            blurred = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        elif self.config.blur_method == BlurMethod.BLACK_BOX:
            # Black rectangle
            blurred = np.zeros_like(face_region)
        
        elif self.config.blur_method == BlurMethod.MOSAIC:
            # Mosaic effect
            tile_size = max(1, self.config.blur_strength // 2)
            for i in range(0, h, tile_size):
                for j in range(0, w, tile_size):
                    tile = face_region[i:i+tile_size, j:j+tile_size]
                    if tile.size > 0:
                        avg_color = np.mean(tile, axis=(0, 1))
                        face_region[i:i+tile_size, j:j+tile_size] = avg_color
            blurred = face_region
        
        else:  # ADAPTIVE
            # Adaptive blur based on face size
            kernel_size = max(5, min(51, w // 10))
            if kernel_size % 2 == 0:
                kernel_size += 1
            blurred = cv2.GaussianBlur(face_region, (kernel_size, kernel_size), 0)
        
        # Replace face region in original image
        result = image.copy()
        result[y:y+h, x:x+w] = blurred
        
        return result
    
    def blur_all_faces(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Detect and blur all faces in image
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (blurred_image, num_faces_blurred)
        """
        if not self.config.enable_face_blur:
            return image, 0
        
        faces = self.detect_faces(image)
        result = image.copy()
        
        for face_bbox in faces:
            result = self.blur_face(result, face_bbox)
        
        return result, len(faces)


class PrivacyFilter:
    """Main privacy filtering class for SAR operations"""
    
    def __init__(self, config: PrivacyConfig = None):
        """
        Initialize privacy filter
        
        Args:
            config: Privacy configuration
        """
        self.config = config or PrivacyConfig()
        self.face_blurrer = FaceBlurrer(self.config)
        self.audit_log = []
        
        logging.info(f"Privacy filter initialized with level: {self.config.privacy_level.value}")
    
    def process_image(self, image: np.ndarray, metadata: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process image with privacy filtering
        
        Args:
            image: Input image
            metadata: Optional metadata dictionary
            
        Returns:
            Tuple of (processed_image, processed_metadata)
        """
        result = image.copy()
        processed_metadata = metadata.copy() if metadata else {}
        
        # Apply face blurring
        if self.config.enable_face_blur:
            result, num_faces = self.face_blurrer.blur_all_faces(result)
            processed_metadata['faces_blurred'] = num_faces
        
        # Apply body blurring if enabled
        if self.config.enable_body_blur:
            result = self._blur_body_regions(result)
        
        # Anonymize metadata
        if self.config.anonymize_metadata:
            processed_metadata = self._anonymize_metadata(processed_metadata)
        
        # Add privacy processing info
        processed_metadata['privacy_level'] = self.config.privacy_level.value
        processed_metadata['privacy_processed'] = True
        
        # Audit logging
        if self.config.enable_audit_logging:
            self._log_privacy_operation(processed_metadata)
        
        return result, processed_metadata
    
    def _blur_body_regions(self, image: np.ndarray) -> np.ndarray:
        """
        Blur body regions while preserving features needed for ReID
        
        Args:
            image: Input image
            
        Returns:
            Image with body regions blurred
        """
        # This is a placeholder for body region detection and blurring
        # In a full implementation, this would use pose estimation
        # to identify and selectively blur body parts
        
        if self.config.privacy_level == PrivacyLevel.HIGH:
            # High privacy: blur entire person except clothing patterns
            kernel_size = 15
            blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            
            # Preserve some texture for ReID (this is a simplified approach)
            # In practice, you'd use more sophisticated methods
            alpha = 0.7  # Blend factor
            result = cv2.addWeighted(image, 1-alpha, blurred, alpha, 0)
            return result
        
        return image
    
    def _anonymize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove or anonymize sensitive metadata
        
        Args:
            metadata: Original metadata
            
        Returns:
            Anonymized metadata
        """
        # Remove potentially identifying information
        sensitive_keys = [
            'gps_coordinates', 'location', 'address', 'operator_id',
            'device_id', 'serial_number', 'user_name', 'timestamp_precise'
        ]
        
        anonymized = {}
        for key, value in metadata.items():
            if key not in sensitive_keys:
                anonymized[key] = value
            else:
                # Replace with anonymized version
                if key == 'timestamp_precise':
                    # Round timestamp to nearest hour
                    if isinstance(value, (int, float)):
                        anonymized['timestamp_hour'] = int(value // 3600) * 3600
                elif key in ['gps_coordinates', 'location']:
                    # Replace with general area
                    anonymized['general_area'] = 'SAR_OPERATION_ZONE'
        
        return anonymized
    
    def _log_privacy_operation(self, metadata: Dict[str, Any]):
        """
        Log privacy operation for audit trail
        
        Args:
            metadata: Processing metadata
        """
        import time
        
        log_entry = {
            'timestamp': time.time(),
            'privacy_level': self.config.privacy_level.value,
            'faces_processed': metadata.get('faces_blurred', 0),
            'body_blur_applied': self.config.enable_body_blur,
            'metadata_anonymized': self.config.anonymize_metadata
        }
        
        self.audit_log.append(log_entry)
        
        # Keep only last 1000 entries
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """
        Get privacy operation audit log
        
        Returns:
            List of audit log entries
        """
        return self.audit_log.copy()
    
    def export_audit_log(self, filepath: str):
        """
        Export audit log to file
        
        Args:
            filepath: Output file path
        """
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self.audit_log, f, indent=2)
        
        logging.info(f"Privacy audit log exported to {filepath}")


def create_privacy_filter(privacy_level: str = "standard") -> PrivacyFilter:
    """
    Factory function to create privacy filter with preset configurations
    
    Args:
        privacy_level: Privacy level ("minimal", "standard", "high", "maximum")
        
    Returns:
        Configured PrivacyFilter instance
    """
    level_map = {
        "minimal": PrivacyLevel.MINIMAL,
        "standard": PrivacyLevel.STANDARD,
        "high": PrivacyLevel.HIGH,
        "maximum": PrivacyLevel.MAXIMUM
    }
    
    config = PrivacyConfig(
        privacy_level=level_map.get(privacy_level.lower(), PrivacyLevel.STANDARD)
    )
    
    # Adjust settings based on privacy level
    if config.privacy_level == PrivacyLevel.MINIMAL:
        config.blur_strength = 10
        config.enable_body_blur = False
        config.anonymize_metadata = False
    elif config.privacy_level == PrivacyLevel.HIGH:
        config.blur_strength = 20
        config.enable_body_blur = True
        config.blur_method = BlurMethod.ADAPTIVE
    elif config.privacy_level == PrivacyLevel.MAXIMUM:
        config.blur_strength = 25
        config.enable_body_blur = True
        config.blur_method = BlurMethod.BLACK_BOX
        config.preserve_body_features = False
    
    return PrivacyFilter(config)