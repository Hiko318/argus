#!/usr/bin/env python3
"""
Thermal & Low-Light Support Module

This module provides thermal and RGB stream fusion capabilities for enhanced detection
and tracking in challenging lighting conditions. Supports multi-modal sensor fusion,
thermal-specific preprocessing, and adaptive fusion strategies.

Features:
- RGB + Thermal stream fusion
- Thermal-specific preprocessing
- Adaptive fusion strategies
- Low-light enhancement
- Temperature-based filtering
- Multi-modal model training support
- Real-time thermal calibration
- Thermal-RGB alignment

Author: Foresight AI Team
Date: 2024
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import time
import json
from collections import deque
import threading
from scipy import ndimage
from scipy.spatial.transform import Rotation
try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    logging.warning("Albumentations not available. Some thermal augmentations disabled.")
try:
    from albumentations.pytorch import ToTensorV2
except ImportError:
    ToTensorV2 = None


class FusionStrategy(Enum):
    """Thermal-RGB fusion strategies"""
    EARLY_FUSION = "early_fusion"          # Concatenate at input level
    LATE_FUSION = "late_fusion"            # Fuse at feature/decision level
    ADAPTIVE_FUSION = "adaptive_fusion"    # Dynamic fusion based on conditions
    ATTENTION_FUSION = "attention_fusion"  # Attention-based fusion
    WEIGHTED_FUSION = "weighted_fusion"    # Weighted combination


class ThermalMode(Enum):
    """Thermal camera operating modes"""
    ABSOLUTE_TEMP = "absolute_temp"        # Absolute temperature values
    RELATIVE_TEMP = "relative_temp"        # Relative temperature differences
    CONTRAST_ENHANCED = "contrast_enhanced" # Enhanced thermal contrast
    PSEUDO_COLOR = "pseudo_color"          # False color thermal imaging


class LightingCondition(Enum):
    """Lighting condition classifications"""
    DAYLIGHT = "daylight"                  # Normal daylight conditions
    TWILIGHT = "twilight"                  # Dawn/dusk conditions
    NIGHT = "night"                        # Night conditions
    INDOOR = "indoor"                      # Indoor lighting
    ARTIFICIAL = "artificial"              # Artificial lighting
    MIXED = "mixed"                        # Mixed lighting conditions


@dataclass
class ThermalCalibration:
    """Thermal camera calibration parameters"""
    # Temperature calibration
    temp_offset: float = 0.0               # Temperature offset in Celsius
    temp_scale: float = 1.0                # Temperature scaling factor
    
    # Spatial calibration
    thermal_to_rgb_transform: np.ndarray = field(default_factory=lambda: np.eye(3))
    distortion_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5))
    
    # Temporal calibration
    temporal_offset: float = 0.0           # Temporal offset in seconds
    
    # Quality parameters
    noise_threshold: float = 0.1           # Thermal noise threshold
    dead_pixel_mask: Optional[np.ndarray] = None
    
    def save(self, filepath: str):
        """Save calibration to file"""
        data = {
            'temp_offset': self.temp_offset,
            'temp_scale': self.temp_scale,
            'thermal_to_rgb_transform': self.thermal_to_rgb_transform.tolist(),
            'distortion_coeffs': self.distortion_coeffs.tolist(),
            'temporal_offset': self.temporal_offset,
            'noise_threshold': self.noise_threshold,
            'dead_pixel_mask': self.dead_pixel_mask.tolist() if self.dead_pixel_mask is not None else None
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ThermalCalibration':
        """Load calibration from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        calibration = cls(
            temp_offset=data['temp_offset'],
            temp_scale=data['temp_scale'],
            thermal_to_rgb_transform=np.array(data['thermal_to_rgb_transform']),
            distortion_coeffs=np.array(data['distortion_coeffs']),
            temporal_offset=data['temporal_offset'],
            noise_threshold=data['noise_threshold']
        )
        
        if data['dead_pixel_mask'] is not None:
            calibration.dead_pixel_mask = np.array(data['dead_pixel_mask'])
        
        return calibration


@dataclass
class FusionConfig:
    """Configuration for thermal-RGB fusion"""
    # Fusion strategy
    strategy: FusionStrategy = FusionStrategy.ADAPTIVE_FUSION
    
    # Input configuration
    rgb_resolution: Tuple[int, int] = (1920, 1080)
    thermal_resolution: Tuple[int, int] = (640, 480)
    target_resolution: Tuple[int, int] = (1920, 1080)
    
    # Thermal processing
    thermal_mode: ThermalMode = ThermalMode.CONTRAST_ENHANCED
    temperature_range: Tuple[float, float] = (-20.0, 100.0)  # Celsius
    
    # Fusion parameters
    rgb_weight: float = 0.7                # RGB stream weight
    thermal_weight: float = 0.3            # Thermal stream weight
    adaptive_threshold: float = 0.5        # Threshold for adaptive fusion
    
    # Processing options
    enable_alignment: bool = True          # Enable spatial alignment
    enable_temporal_sync: bool = True      # Enable temporal synchronization
    enable_enhancement: bool = True        # Enable low-light enhancement
    
    # Quality control
    min_thermal_quality: float = 0.3       # Minimum thermal image quality
    max_temperature_diff: float = 50.0     # Maximum temperature difference
    

class ThermalPreprocessor:
    """Thermal image preprocessing and enhancement"""
    
    def __init__(self, config: FusionConfig, calibration: Optional[ThermalCalibration] = None):
        """Initialize thermal preprocessor"""
        self.config = config
        self.calibration = calibration or ThermalCalibration()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Temperature statistics for adaptive processing
        self.temp_history = deque(maxlen=100)
        
        # Preprocessing pipelines
        self._setup_preprocessing_pipelines()
    
    def _setup_preprocessing_pipelines(self):
        """Setup preprocessing pipelines for different thermal modes"""
        self.preprocessing_pipelines = {
            ThermalMode.ABSOLUTE_TEMP: self._process_absolute_temp,
            ThermalMode.RELATIVE_TEMP: self._process_relative_temp,
            ThermalMode.CONTRAST_ENHANCED: self._process_contrast_enhanced,
            ThermalMode.PSEUDO_COLOR: self._process_pseudo_color
        }
    
    def process_thermal_frame(self, thermal_frame: np.ndarray, 
                            timestamp: Optional[float] = None) -> np.ndarray:
        """Process thermal frame according to configuration"""
        # Apply calibration
        calibrated_frame = self._apply_calibration(thermal_frame)
        
        # Apply dead pixel correction
        if self.calibration.dead_pixel_mask is not None:
            calibrated_frame = self._correct_dead_pixels(calibrated_frame)
        
        # Apply thermal mode processing
        processed_frame = self.preprocessing_pipelines[self.config.thermal_mode](calibrated_frame)
        
        # Resize to target resolution
        if processed_frame.shape[:2] != self.config.target_resolution[::-1]:
            processed_frame = cv2.resize(processed_frame, self.config.target_resolution)
        
        # Update temperature statistics
        if timestamp is not None:
            self._update_temperature_stats(calibrated_frame, timestamp)
        
        return processed_frame
    
    def preprocess_thermal(self, thermal_frame: np.ndarray) -> np.ndarray:
        """Alias for process_thermal_frame for test compatibility"""
        return self.process_thermal_frame(thermal_frame)
    
    def _apply_calibration(self, thermal_frame: np.ndarray) -> np.ndarray:
        """Apply thermal calibration"""
        # Apply temperature calibration
        calibrated = thermal_frame * self.calibration.temp_scale + self.calibration.temp_offset
        
        # Apply spatial calibration if needed
        if not np.allclose(self.calibration.thermal_to_rgb_transform, np.eye(3)):
            h, w = thermal_frame.shape[:2]
            calibrated = cv2.warpPerspective(
                calibrated, 
                self.calibration.thermal_to_rgb_transform, 
                (w, h)
            )
        
        return calibrated
    
    def _correct_dead_pixels(self, thermal_frame: np.ndarray) -> np.ndarray:
        """Correct dead pixels in thermal image"""
        corrected = thermal_frame.copy()
        
        # Find dead pixels
        dead_pixels = self.calibration.dead_pixel_mask > 0
        
        if np.any(dead_pixels):
            # Use inpainting to fill dead pixels
            mask = dead_pixels.astype(np.uint8) * 255
            corrected = cv2.inpaint(corrected.astype(np.float32), mask, 3, cv2.INPAINT_TELEA)
        
        return corrected
    
    def _process_absolute_temp(self, thermal_frame: np.ndarray) -> np.ndarray:
        """Process thermal frame as absolute temperature"""
        # Normalize to temperature range
        min_temp, max_temp = self.config.temperature_range
        normalized = np.clip((thermal_frame - min_temp) / (max_temp - min_temp), 0, 1)
        
        # Convert to 8-bit
        return (normalized * 255).astype(np.uint8)
    
    def _process_relative_temp(self, thermal_frame: np.ndarray) -> np.ndarray:
        """Process thermal frame as relative temperature differences"""
        # Calculate relative to frame mean
        frame_mean = np.mean(thermal_frame)
        relative = thermal_frame - frame_mean
        
        # Normalize relative differences
        std_dev = np.std(relative)
        if std_dev > 0:
            normalized = np.clip((relative / (3 * std_dev)) + 0.5, 0, 1)
        else:
            normalized = np.ones_like(relative) * 0.5
        
        return (normalized * 255).astype(np.uint8)
    
    def _process_contrast_enhanced(self, thermal_frame: np.ndarray) -> np.ndarray:
        """Process thermal frame with contrast enhancement"""
        # Apply histogram equalization
        normalized = self._process_absolute_temp(thermal_frame)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(normalized)
        
        # Apply additional sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def _process_pseudo_color(self, thermal_frame: np.ndarray) -> np.ndarray:
        """Process thermal frame with pseudo-color mapping"""
        # Normalize to 8-bit
        normalized = self._process_absolute_temp(thermal_frame)
        
        # Apply colormap
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        
        return colored
    
    def _update_temperature_stats(self, thermal_frame: np.ndarray, timestamp: float):
        """Update temperature statistics for adaptive processing"""
        stats = {
            'timestamp': timestamp,
            'mean_temp': np.mean(thermal_frame),
            'std_temp': np.std(thermal_frame),
            'min_temp': np.min(thermal_frame),
            'max_temp': np.max(thermal_frame)
        }
        
        self.temp_history.append(stats)
    
    def get_temperature_stats(self) -> Dict[str, float]:
        """Get current temperature statistics"""
        if not self.temp_history:
            return {}
        
        recent_stats = list(self.temp_history)[-10:]  # Last 10 frames
        
        return {
            'avg_mean_temp': np.mean([s['mean_temp'] for s in recent_stats]),
            'avg_std_temp': np.mean([s['std_temp'] for s in recent_stats]),
            'temp_range': np.max([s['max_temp'] for s in recent_stats]) - 
                         np.min([s['min_temp'] for s in recent_stats]),
            'temp_stability': np.std([s['mean_temp'] for s in recent_stats])
        }


class LightingAnalyzer:
    """Analyze lighting conditions for adaptive fusion"""
    
    def __init__(self):
        """Initialize lighting analyzer"""
        self.logger = logging.getLogger(__name__)
        
        # Lighting history for temporal analysis
        self.lighting_history = deque(maxlen=50)
    
    def analyze_lighting_condition(self, rgb_frame: np.ndarray, 
                                 thermal_frame: Optional[np.ndarray] = None) -> LightingCondition:
        """Analyze current lighting condition"""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY) if len(rgb_frame.shape) == 3 else rgb_frame
        
        # Calculate lighting metrics
        mean_brightness = np.mean(gray)
        brightness_std = np.std(gray)
        
        # Calculate histogram features
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_normalized = hist / np.sum(hist)
        
        # Low light indicators
        low_light_ratio = np.sum(hist_normalized[:64])  # Ratio of dark pixels
        high_light_ratio = np.sum(hist_normalized[192:])  # Ratio of bright pixels
        
        # Contrast measure
        contrast = brightness_std / (mean_brightness + 1e-6)
        
        # Classify lighting condition
        if mean_brightness < 50 and low_light_ratio > 0.6:
            condition = LightingCondition.NIGHT
        elif mean_brightness < 100 and low_light_ratio > 0.4:
            condition = LightingCondition.TWILIGHT
        elif high_light_ratio > 0.3 and contrast > 0.8:
            condition = LightingCondition.ARTIFICIAL
        elif contrast < 0.3:
            condition = LightingCondition.INDOOR
        elif self._detect_mixed_lighting(hist_normalized):
            condition = LightingCondition.MIXED
        else:
            condition = LightingCondition.DAYLIGHT
        
        # Update history
        self.lighting_history.append({
            'condition': condition,
            'mean_brightness': mean_brightness,
            'contrast': contrast,
            'timestamp': time.time()
        })
        
        return condition
    
    def analyze_lighting(self, rgb_frame: np.ndarray) -> Dict[str, Any]:
        """Analyze lighting conditions and return detailed info"""
        # Convert to grayscale
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate brightness metrics
        mean_brightness = np.mean(gray)
        brightness_std = np.std(gray)
        
        # Get lighting condition
        lighting_condition = self.analyze_lighting_condition(rgb_frame)
        
        # Determine if enhancement is needed
        enhancement_needed = (
            lighting_condition in [LightingCondition.NIGHT, LightingCondition.TWILIGHT] or
            mean_brightness < 80
        )
        
        return {
            'brightness_level': float(mean_brightness),
            'lighting_condition': lighting_condition.value,
            'enhancement_needed': enhancement_needed,
            'contrast': float(brightness_std / (mean_brightness + 1e-6))
        }
    
    def _detect_mixed_lighting(self, hist_normalized: np.ndarray) -> bool:
        """Detect mixed lighting conditions"""
        # Look for multiple peaks in histogram
        peaks = []
        for i in range(1, len(hist_normalized) - 1):
            if (hist_normalized[i] > hist_normalized[i-1] and 
                hist_normalized[i] > hist_normalized[i+1] and 
                hist_normalized[i] > 0.01):
                peaks.append(i)
        
        return len(peaks) >= 3
    
    def get_lighting_stability(self) -> float:
        """Get lighting stability measure"""
        if len(self.lighting_history) < 5:
            return 1.0
        
        recent_conditions = [entry['condition'] for entry in list(self.lighting_history)[-10:]]
        unique_conditions = len(set(recent_conditions))
        
        return 1.0 / unique_conditions  # Higher value = more stable


class ThermalRGBFusion:
    """Main thermal-RGB fusion module"""
    
    def __init__(self, config: FusionConfig, 
                 calibration: Optional[ThermalCalibration] = None):
        """Initialize thermal-RGB fusion"""
        self.config = config
        self.calibration = calibration or ThermalCalibration()
        
        # Initialize components
        self.thermal_preprocessor = ThermalPreprocessor(config, calibration)
        self.lighting_analyzer = LightingAnalyzer()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Fusion models (would be loaded from trained models)
        self.fusion_models = self._initialize_fusion_models()
        
        # Temporal synchronization
        self.rgb_buffer = deque(maxlen=10)
        self.thermal_buffer = deque(maxlen=10)
        
        # Performance tracking
        self.fusion_stats = {
            'total_frames': 0,
            'fusion_time': 0.0,
            'alignment_time': 0.0
        }
    
    def _initialize_fusion_models(self) -> Dict[str, Any]:
        """Initialize fusion models (placeholder for actual model loading)"""
        # In practice, these would be loaded PyTorch/TensorFlow models
        return {
            'early_fusion': None,
            'late_fusion': None,
            'attention_fusion': None,
            'adaptive_fusion': None
        }
    
    def process_frame_pair(self, rgb_frame: np.ndarray, 
                          thermal_frame: np.ndarray,
                          rgb_timestamp: Optional[float] = None,
                          thermal_timestamp: Optional[float] = None) -> Dict[str, Any]:
        """Process RGB-thermal frame pair"""
        start_time = time.time()
        
        # Add frames to buffers for temporal synchronization
        if self.config.enable_temporal_sync:
            self._add_to_buffers(rgb_frame, thermal_frame, rgb_timestamp, thermal_timestamp)
            
            # Get synchronized frames
            sync_rgb, sync_thermal = self._get_synchronized_frames()
            if sync_rgb is None or sync_thermal is None:
                return {'fused_frame': rgb_frame, 'fusion_info': {'status': 'waiting_for_sync'}}
        else:
            sync_rgb, sync_thermal = rgb_frame, thermal_frame
        
        # Preprocess thermal frame
        processed_thermal = self.thermal_preprocessor.process_thermal_frame(
            sync_thermal, thermal_timestamp
        )
        
        # Analyze lighting conditions
        lighting_condition = self.lighting_analyzer.analyze_lighting_condition(
            sync_rgb, sync_thermal
        )
        
        # Perform spatial alignment if enabled
        if self.config.enable_alignment:
            aligned_thermal = self._align_thermal_to_rgb(processed_thermal, sync_rgb)
        else:
            aligned_thermal = processed_thermal
        
        # Perform fusion based on strategy
        fused_frame, fusion_info = self._perform_fusion(
            sync_rgb, aligned_thermal, lighting_condition
        )
        
        # Apply low-light enhancement if needed
        if (self.config.enable_enhancement and 
            lighting_condition in [LightingCondition.NIGHT, LightingCondition.TWILIGHT]):
            fused_frame = self._enhance_low_light(fused_frame)
        
        # Update statistics
        self.fusion_stats['total_frames'] += 1
        self.fusion_stats['fusion_time'] += time.time() - start_time
        
        # Prepare result
        result = {
            'fused_frame': fused_frame,
            'fusion_info': {
                'strategy': self.config.strategy.value,
                'lighting_condition': lighting_condition.value,
                'thermal_quality': self._assess_thermal_quality(sync_thermal),
                'alignment_offset': fusion_info.get('alignment_offset', (0, 0)),
                'fusion_weights': fusion_info.get('fusion_weights', {}),
                'processing_time': time.time() - start_time
            }
        }
        
        return result
    
    def _add_to_buffers(self, rgb_frame: np.ndarray, thermal_frame: np.ndarray,
                       rgb_timestamp: Optional[float], thermal_timestamp: Optional[float]):
        """Add frames to temporal buffers"""
        current_time = time.time()
        
        self.rgb_buffer.append({
            'frame': rgb_frame,
            'timestamp': rgb_timestamp or current_time
        })
        
        self.thermal_buffer.append({
            'frame': thermal_frame,
            'timestamp': thermal_timestamp or current_time
        })
    
    def _get_synchronized_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get temporally synchronized frames"""
        if len(self.rgb_buffer) == 0 or len(self.thermal_buffer) == 0:
            return None, None
        
        # Find best temporal match
        best_rgb = None
        best_thermal = None
        min_time_diff = float('inf')
        
        for rgb_entry in self.rgb_buffer:
            for thermal_entry in self.thermal_buffer:
                time_diff = abs(rgb_entry['timestamp'] - thermal_entry['timestamp'])
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    best_rgb = rgb_entry['frame']
                    best_thermal = thermal_entry['frame']
        
        # Only return if synchronization is good enough
        if min_time_diff < 0.1:  # 100ms threshold
            return best_rgb, best_thermal
        else:
            return None, None
    
    def _align_thermal_to_rgb(self, thermal_frame: np.ndarray, 
                            rgb_frame: np.ndarray) -> np.ndarray:
        """Align thermal frame to RGB frame"""
        start_time = time.time()
        
        # Convert RGB to grayscale for alignment
        rgb_gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        
        # Convert thermal to grayscale if it's colored
        if len(thermal_frame.shape) == 3:
            thermal_gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)
        else:
            thermal_gray = thermal_frame
        
        # Resize thermal to match RGB if needed
        if thermal_gray.shape != rgb_gray.shape:
            thermal_gray = cv2.resize(thermal_gray, (rgb_gray.shape[1], rgb_gray.shape[0]))
        
        # Use feature-based alignment
        try:
            # Detect features
            orb = cv2.ORB_create(nfeatures=500)
            kp1, des1 = orb.detectAndCompute(rgb_gray, None)
            kp2, des2 = orb.detectAndCompute(thermal_gray, None)
            
            if des1 is not None and des2 is not None and len(des1) > 10 and len(des2) > 10:
                # Match features
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                
                if len(matches) > 10:
                    # Extract matched points
                    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    
                    # Find homography
                    M, mask = cv2.findHomography(src_pts, dst_pts, 
                                                cv2.RANSAC, 5.0)
                    
                    if M is not None:
                        # Apply transformation
                        h, w = rgb_frame.shape[:2]
                        aligned = cv2.warpPerspective(thermal_frame, M, (w, h))
                        
                        self.fusion_stats['alignment_time'] += time.time() - start_time
                        return aligned
        
        except Exception as e:
            self.logger.warning(f"Feature-based alignment failed: {e}")
        
        # Fallback to simple resize
        if thermal_frame.shape[:2] != rgb_frame.shape[:2]:
            aligned = cv2.resize(thermal_frame, (rgb_frame.shape[1], rgb_frame.shape[0]))
        else:
            aligned = thermal_frame
        
        self.fusion_stats['alignment_time'] += time.time() - start_time
        return aligned
    
    def _perform_fusion(self, rgb_frame: np.ndarray, thermal_frame: np.ndarray,
                       lighting_condition: LightingCondition) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform fusion based on configured strategy"""
        fusion_info = {}
        
        if self.config.strategy == FusionStrategy.EARLY_FUSION:
            fused_frame, info = self._early_fusion(rgb_frame, thermal_frame)
        elif self.config.strategy == FusionStrategy.LATE_FUSION:
            fused_frame, info = self._late_fusion(rgb_frame, thermal_frame)
        elif self.config.strategy == FusionStrategy.ADAPTIVE_FUSION:
            fused_frame, info = self._adaptive_fusion(rgb_frame, thermal_frame, lighting_condition)
        elif self.config.strategy == FusionStrategy.ATTENTION_FUSION:
            fused_frame, info = self._attention_fusion(rgb_frame, thermal_frame)
        elif self.config.strategy == FusionStrategy.WEIGHTED_FUSION:
            fused_frame, info = self._weighted_fusion(rgb_frame, thermal_frame, lighting_condition)
        else:
            fused_frame, info = self._weighted_fusion(rgb_frame, thermal_frame, lighting_condition)
        
        fusion_info.update(info)
        return fused_frame, fusion_info
    
    def _early_fusion(self, rgb_frame: np.ndarray, 
                     thermal_frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Early fusion - concatenate channels"""
        # Ensure thermal is single channel
        if len(thermal_frame.shape) == 3:
            thermal_gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)
        else:
            thermal_gray = thermal_frame
        
        # Concatenate as 4-channel image (RGB + Thermal)
        fused = np.dstack([rgb_frame, thermal_gray])
        
        # For display, convert back to 3-channel
        display_frame = rgb_frame.copy()
        
        return display_frame, {
            'fusion_weights': {'rgb': 1.0, 'thermal': 0.0},  # For display
            'channels': 4
        }
    
    def _late_fusion(self, rgb_frame: np.ndarray, 
                    thermal_frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Late fusion - process separately then combine"""
        # This would typically involve running separate models on each modality
        # For now, implement as weighted combination
        return self._weighted_fusion(rgb_frame, thermal_frame, LightingCondition.MIXED)
    
    def _adaptive_fusion(self, rgb_frame: np.ndarray, thermal_frame: np.ndarray,
                        lighting_condition: LightingCondition) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Adaptive fusion based on lighting conditions"""
        # Adjust weights based on lighting condition
        if lighting_condition == LightingCondition.NIGHT:
            rgb_weight = 0.3
            thermal_weight = 0.7
        elif lighting_condition == LightingCondition.TWILIGHT:
            rgb_weight = 0.5
            thermal_weight = 0.5
        elif lighting_condition == LightingCondition.DAYLIGHT:
            rgb_weight = 0.8
            thermal_weight = 0.2
        else:
            rgb_weight = self.config.rgb_weight
            thermal_weight = self.config.thermal_weight
        
        # Normalize weights
        total_weight = rgb_weight + thermal_weight
        rgb_weight /= total_weight
        thermal_weight /= total_weight
        
        return self._weighted_fusion_with_weights(rgb_frame, thermal_frame, rgb_weight, thermal_weight)
    
    def _attention_fusion(self, rgb_frame: np.ndarray, 
                         thermal_frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Attention-based fusion (simplified implementation)"""
        # Calculate attention maps based on gradient magnitude
        rgb_gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        
        if len(thermal_frame.shape) == 3:
            thermal_gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)
        else:
            thermal_gray = thermal_frame
        
        # Calculate gradients
        rgb_grad = cv2.Laplacian(rgb_gray, cv2.CV_64F)
        thermal_grad = cv2.Laplacian(thermal_gray, cv2.CV_64F)
        
        # Create attention maps
        rgb_attention = np.abs(rgb_grad)
        thermal_attention = np.abs(thermal_grad)
        
        # Normalize attention maps
        total_attention = rgb_attention + thermal_attention + 1e-6
        rgb_attention_norm = rgb_attention / total_attention
        thermal_attention_norm = thermal_attention / total_attention
        
        # Apply attention to each channel
        fused = np.zeros_like(rgb_frame, dtype=np.float32)
        
        for c in range(3):
            fused[:, :, c] = (rgb_frame[:, :, c] * rgb_attention_norm + 
                             thermal_gray * thermal_attention_norm)
        
        fused = np.clip(fused, 0, 255).astype(np.uint8)
        
        return fused, {
            'fusion_weights': {
                'rgb_avg_attention': np.mean(rgb_attention_norm),
                'thermal_avg_attention': np.mean(thermal_attention_norm)
            }
        }
    
    def _weighted_fusion(self, rgb_frame: np.ndarray, thermal_frame: np.ndarray,
                        lighting_condition: LightingCondition) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Weighted fusion with configured weights"""
        return self._weighted_fusion_with_weights(
            rgb_frame, thermal_frame, 
            self.config.rgb_weight, self.config.thermal_weight
        )
    
    def _weighted_fusion_with_weights(self, rgb_frame: np.ndarray, thermal_frame: np.ndarray,
                                    rgb_weight: float, thermal_weight: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Weighted fusion with specified weights"""
        # Ensure thermal is 3-channel for blending
        if len(thermal_frame.shape) == 2:
            thermal_3ch = cv2.cvtColor(thermal_frame, cv2.COLOR_GRAY2BGR)
        else:
            thermal_3ch = thermal_frame
        
        # Weighted combination
        fused = cv2.addWeighted(rgb_frame, rgb_weight, thermal_3ch, thermal_weight, 0)
        
        return fused, {
            'fusion_weights': {'rgb': rgb_weight, 'thermal': thermal_weight}
        }
    
    def _enhance_low_light(self, frame: np.ndarray) -> np.ndarray:
        """Enhance low-light conditions"""
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge channels and convert back
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Apply gamma correction for further enhancement
        gamma = 1.2
        enhanced = np.power(enhanced / 255.0, 1.0 / gamma) * 255.0
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced
    
    def _assess_thermal_quality(self, thermal_frame: np.ndarray) -> float:
        """Assess thermal image quality"""
        # Calculate various quality metrics
        
        # 1. Signal-to-noise ratio estimate
        mean_val = np.mean(thermal_frame)
        std_val = np.std(thermal_frame)
        snr = mean_val / (std_val + 1e-6)
        
        # 2. Gradient magnitude (sharpness)
        if len(thermal_frame.shape) == 3:
            gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = thermal_frame
        
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        sharpness = np.mean(gradient_magnitude)
        
        # 3. Dynamic range
        dynamic_range = np.max(thermal_frame) - np.min(thermal_frame)
        
        # Combine metrics (normalized to 0-1)
        snr_norm = np.clip(snr / 10.0, 0, 1)
        sharpness_norm = np.clip(sharpness / 50.0, 0, 1)
        range_norm = np.clip(dynamic_range / 100.0, 0, 1)
        
        quality = (snr_norm + sharpness_norm + range_norm) / 3.0
        
        return quality
    
    def get_fusion_statistics(self) -> Dict[str, Any]:
        """Get fusion performance statistics"""
        stats = self.fusion_stats.copy()
        
        if stats['total_frames'] > 0:
            stats['avg_fusion_time'] = stats['fusion_time'] / stats['total_frames']
            stats['avg_alignment_time'] = stats['alignment_time'] / stats['total_frames']
        
        # Add thermal statistics
        stats['thermal_stats'] = self.thermal_preprocessor.get_temperature_stats()
        
        # Add lighting statistics
        stats['lighting_stability'] = self.lighting_analyzer.get_lighting_stability()
        
        return stats
    
    def fuse_streams(self, rgb_frame: np.ndarray, thermal_frame: np.ndarray) -> np.ndarray:
        """Fuse RGB and thermal streams"""
        result = self.process_frame_pair(rgb_frame, thermal_frame)
        return result['fused_frame']
    
    def filter_by_temperature(self, thermal_frame: np.ndarray, 
                             temperature_data: Optional[np.ndarray] = None,
                             min_temp: float = 0.0, max_temp: float = 100.0) -> np.ndarray:
        """Filter thermal frame by temperature range"""
        # Use provided temperature data or apply calibration
        if temperature_data is not None:
            temp_values = temperature_data
        else:
            temp_values = self.thermal_preprocessor._apply_calibration(thermal_frame)
        
        # Create mask for temperature range
        mask = (temp_values >= min_temp) & (temp_values <= max_temp)
        
        # Apply mask
        filtered = thermal_frame.copy()
        filtered[~mask] = 0
        
        return filtered
    
    def enhance_low_light(self, frame: np.ndarray) -> np.ndarray:
        """Enhance low-light conditions (public method)"""
        return self._enhance_low_light(frame)
    
    def save_calibration(self, filepath: str):
        """Save thermal calibration"""
        self.calibration.save(filepath)
    
    def load_calibration(self, filepath: str):
        """Load thermal calibration"""
        self.calibration = ThermalCalibration.load(filepath)
        self.thermal_preprocessor.calibration = self.calibration


# Example usage and testing
if __name__ == '__main__':
    # Example configuration
    config = FusionConfig(
        strategy=FusionStrategy.ADAPTIVE_FUSION,
        rgb_resolution=(1920, 1080),
        thermal_resolution=(640, 480),
        thermal_mode=ThermalMode.CONTRAST_ENHANCED,
        enable_alignment=True,
        enable_enhancement=True
    )
    
    # Initialize fusion system
    fusion_system = ThermalRGBFusion(config)
    
    # Example processing (with dummy data)
    rgb_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    thermal_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    
    result = fusion_system.process_frame_pair(rgb_frame, thermal_frame)
    
    print(f"Fusion completed: {result['fusion_info']}")
    print(f"Fusion statistics: {fusion_system.get_fusion_statistics()}")