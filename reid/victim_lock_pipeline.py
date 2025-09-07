"""Victim-Lock Re-identification Pipeline.

Complete pipeline for victim identification and tracking using
two-branch approach: face embedding + full-body re-id.
"""

import time
import logging
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import cv2
from pathlib import Path

from .face_embedder import FaceEmbedder, FaceModel
from .body_reid import BodyReIDEmbedder, BodyReIDModel
from .gallery_manager import GalleryManager, MatchResult, MatchConfidence


class PipelineMode(Enum):
    """Pipeline operation modes"""
    FACE_ONLY = "face_only"
    BODY_ONLY = "body_only"
    COMBINED = "combined"
    AUTO = "auto"  # Automatically choose best approach


class AlertLevel(Enum):
    """Alert levels for matches"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DetectionInput:
    """Input detection for re-identification"""
    image: np.ndarray
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int = 0  # 0 for person
    track_id: Optional[int] = None
    timestamp: float = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}
    
    def crop_image(self) -> np.ndarray:
        """Crop detection from image"""
        x1, y1, x2, y2 = self.bbox
        return self.image[y1:y2, x1:x2]


@dataclass
class ReIDResult:
    """Re-identification result"""
    detection: DetectionInput
    matches: List[MatchResult]
    best_match: Optional[MatchResult]
    alert_level: AlertLevel
    processing_time: float
    pipeline_mode: PipelineMode
    embeddings: Dict[str, np.ndarray]
    quality_scores: Dict[str, float]
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class VictimLockPipeline:
    """Complete Victim-Lock re-identification pipeline"""
    
    def __init__(self, 
                 gallery_dir: str = "data/gallery",
                 face_model: FaceModel = FaceModel.ARCFACE,
                 body_model: BodyReIDModel = BodyReIDModel.OSNET,
                 mode: PipelineMode = PipelineMode.AUTO,
                 device: str = "auto"):
        
        self.logger = logging.getLogger(__name__)
        self.mode = mode
        
        # Initialize components
        self.face_embedder = FaceEmbedder(
            model_type=face_model,
            device=device
        )
        
        self.body_embedder = BodyReIDEmbedder(
            model_type=body_model,
            device=device
        )
        
        self.gallery_manager = GalleryManager(gallery_dir=gallery_dir)
        
        # Quality thresholds
        self.min_face_size = 32  # Minimum face size in pixels
        self.min_body_size = 64  # Minimum body size in pixels
        self.face_quality_threshold = 0.5
        self.body_quality_threshold = 0.3
        
        # Alert thresholds
        self.alert_thresholds = {
            AlertLevel.CRITICAL: 0.9,
            AlertLevel.HIGH: 0.75,
            AlertLevel.MEDIUM: 0.6,
            AlertLevel.LOW: 0.4
        }
        
        # Performance tracking
        self.stats = {
            'total_processed': 0,
            'face_extractions': 0,
            'body_extractions': 0,
            'matches_found': 0,
            'processing_times': [],
            'quality_scores': {'face': [], 'body': []}
        }
        
        self.logger.info(f"VictimLock pipeline initialized with {face_model.value} + {body_model.value}")
    
    def process_detection(self, detection: DetectionInput) -> ReIDResult:
        """Process a single detection for re-identification
        
        Args:
            detection: Detection input with image and bounding box
            
        Returns:
            Re-identification result with matches and alert level
        """
        start_time = time.time()
        
        # Crop detection
        cropped_image = detection.crop_image()
        
        if cropped_image.size == 0:
            self.logger.warning("Empty crop detected, skipping")
            return self._create_empty_result(detection, start_time)
        
        # Determine processing mode
        processing_mode = self._determine_mode(cropped_image)
        
        # Extract embeddings
        embeddings = {}
        quality_scores = {}
        
        face_embedding = None
        body_embedding = None
        
        # Face embedding extraction
        if processing_mode in [PipelineMode.FACE_ONLY, PipelineMode.COMBINED, PipelineMode.AUTO]:
            try:
                face_result = self.face_embedder.extract_embedding(cropped_image)
                if face_result['success'] and face_result['quality_score'] >= self.face_quality_threshold:
                    face_embedding = face_result['embedding']
                    embeddings['face'] = face_embedding
                    quality_scores['face'] = face_result['quality_score']
                    self.stats['face_extractions'] += 1
                    self.stats['quality_scores']['face'].append(face_result['quality_score'])
            except Exception as e:
                self.logger.warning(f"Face embedding extraction failed: {e}")
        
        # Body embedding extraction
        if processing_mode in [PipelineMode.BODY_ONLY, PipelineMode.COMBINED, PipelineMode.AUTO]:
            try:
                body_result = self.body_embedder.extract_embedding(cropped_image)
                if body_result['success'] and body_result['quality_score'] >= self.body_quality_threshold:
                    body_embedding = body_result['embedding']
                    embeddings['body'] = body_embedding
                    quality_scores['body'] = body_result['quality_score']
                    self.stats['body_extractions'] += 1
                    self.stats['quality_scores']['body'].append(body_result['quality_score'])
            except Exception as e:
                self.logger.warning(f"Body embedding extraction failed: {e}")
        
        # Search gallery for matches
        matches = []
        if face_embedding is not None or body_embedding is not None:
            try:
                matches = self.gallery_manager.search(
                    face_embedding=face_embedding,
                    body_embedding=body_embedding,
                    top_k=5,
                    min_similarity=0.3
                )
                
                if matches:
                    self.stats['matches_found'] += 1
                    
            except Exception as e:
                self.logger.error(f"Gallery search failed: {e}")
        
        # Determine best match and alert level
        best_match = matches[0] if matches else None
        alert_level = self._determine_alert_level(best_match)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        self.stats['processing_times'].append(processing_time)
        self.stats['total_processed'] += 1
        
        # Create result
        result = ReIDResult(
            detection=detection,
            matches=matches,
            best_match=best_match,
            alert_level=alert_level,
            processing_time=processing_time,
            pipeline_mode=processing_mode,
            embeddings=embeddings,
            quality_scores=quality_scores
        )
        
        # Log significant matches
        if alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]:
            self.logger.warning(
                f"High-confidence match detected: {best_match.gallery_name} "
                f"(similarity: {best_match.similarity_score:.3f}, alert: {alert_level.value})"
            )
        
        return result
    
    def process_batch(self, detections: List[DetectionInput]) -> List[ReIDResult]:
        """Process multiple detections in batch
        
        Args:
            detections: List of detection inputs
            
        Returns:
            List of re-identification results
        """
        results = []
        
        for detection in detections:
            try:
                result = self.process_detection(detection)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process detection: {e}")
                # Add empty result to maintain order
                results.append(self._create_empty_result(detection, time.time()))
        
        return results
    
    def _determine_mode(self, image: np.ndarray) -> PipelineMode:
        """Determine optimal processing mode based on image characteristics"""
        if self.mode != PipelineMode.AUTO:
            return self.mode
        
        h, w = image.shape[:2]
        
        # Check image size and aspect ratio
        if h < self.min_body_size or w < self.min_body_size:
            return PipelineMode.FACE_ONLY
        
        # Check if image is likely a face crop (square-ish, small)
        aspect_ratio = w / h
        if 0.7 <= aspect_ratio <= 1.3 and max(h, w) < 200:
            return PipelineMode.FACE_ONLY
        
        # Check if image is likely a full body (tall aspect ratio)
        if aspect_ratio < 0.6 and h > 150:
            return PipelineMode.COMBINED
        
        # Default to combined approach
        return PipelineMode.COMBINED
    
    def _determine_alert_level(self, match: Optional[MatchResult]) -> AlertLevel:
        """Determine alert level based on match confidence"""
        if match is None:
            return AlertLevel.NONE
        
        similarity = match.similarity_score
        
        for level, threshold in self.alert_thresholds.items():
            if similarity >= threshold:
                return level
        
        return AlertLevel.NONE
    
    def _create_empty_result(self, detection: DetectionInput, start_time: float) -> ReIDResult:
        """Create empty result for failed processing"""
        return ReIDResult(
            detection=detection,
            matches=[],
            best_match=None,
            alert_level=AlertLevel.NONE,
            processing_time=time.time() - start_time,
            pipeline_mode=self.mode,
            embeddings={},
            quality_scores={}
        )
    
    def add_target_person(self, 
                         name: str,
                         description: str,
                         reference_images: List[np.ndarray],
                         priority: int = 1) -> str:
        """Add a target person to the gallery
        
        Args:
            name: Person's name or identifier
            description: Description of the person
            reference_images: List of reference images (BGR format)
            priority: Priority level (1=highest, 5=lowest)
            
        Returns:
            Gallery entry ID
        """
        if not reference_images:
            raise ValueError("At least one reference image must be provided")
        
        # Extract embeddings from all reference images
        face_embeddings = []
        body_embeddings = []
        
        for img in reference_images:
            # Face embedding
            try:
                face_result = self.face_embedder.extract_embedding(img)
                if face_result['success']:
                    face_embeddings.append(face_result['embedding'])
            except Exception as e:
                self.logger.warning(f"Face embedding extraction failed for {name}: {e}")
            
            # Body embedding
            try:
                body_result = self.body_embedder.extract_embedding(img)
                if body_result['success']:
                    body_embeddings.append(body_result['embedding'])
            except Exception as e:
                self.logger.warning(f"Body embedding extraction failed for {name}: {e}")
        
        # Average embeddings if multiple images
        face_embedding = None
        if face_embeddings:
            face_embedding = np.mean(face_embeddings, axis=0)
        
        body_embedding = None
        if body_embeddings:
            body_embedding = np.mean(body_embeddings, axis=0)
        
        if face_embedding is None and body_embedding is None:
            raise ValueError(f"Failed to extract any embeddings for {name}")
        
        # Add to gallery
        gallery_id = self.gallery_manager.add_person(
            name=name,
            description=description,
            face_embedding=face_embedding,
            body_embedding=body_embedding,
            reference_image=reference_images[0],  # Use first image as reference
            metadata={
                'num_reference_images': len(reference_images),
                'has_face_embedding': face_embedding is not None,
                'has_body_embedding': body_embedding is not None
            },
            priority=priority
        )
        
        self.logger.info(
            f"Added target person: {name} (ID: {gallery_id}, "
            f"face: {face_embedding is not None}, body: {body_embedding is not None})"
        )
        
        return gallery_id
    
    def run_consistency_test(self, 
                            gallery_images: Dict[str, List[np.ndarray]],
                            probe_images: Dict[str, List[np.ndarray]],
                            similarity_thresholds: List[float] = None) -> Dict[str, Any]:
        """Run consistency test: gallery vs probe images
        
        Args:
            gallery_images: Dict mapping person_id to list of gallery images
            probe_images: Dict mapping person_id to list of probe images
            similarity_thresholds: List of thresholds to test
            
        Returns:
            Test results with TPR, FPR, and other metrics
        """
        if similarity_thresholds is None:
            similarity_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        self.logger.info("Running consistency test...")
        
        # Clear existing gallery for test
        original_gallery = dict(self.gallery_manager.gallery)
        self.gallery_manager.clear_gallery()
        
        try:
            # Add gallery images
            gallery_ids = {}
            for person_id, images in gallery_images.items():
                gallery_id = self.add_target_person(
                    name=f"test_person_{person_id}",
                    description=f"Test person {person_id}",
                    reference_images=images,
                    priority=1
                )
                gallery_ids[person_id] = gallery_id
            
            # Test probe images
            results = {}
            for threshold in similarity_thresholds:
                tp = 0  # True positives
                fp = 0  # False positives
                tn = 0  # True negatives
                fn = 0  # False negatives
                
                for probe_person_id, probe_imgs in probe_images.items():
                    for probe_img in probe_imgs:
                        # Create detection input
                        h, w = probe_img.shape[:2]
                        detection = DetectionInput(
                            image=probe_img,
                            bbox=(0, 0, w, h),
                            confidence=1.0
                        )
                        
                        # Process detection
                        result = self.process_detection(detection)
                        
                        # Check if match found above threshold
                        match_found = False
                        matched_person = None
                        
                        if result.best_match and result.best_match.similarity_score >= threshold:
                            match_found = True
                            # Find which person was matched
                            for pid, gid in gallery_ids.items():
                                if result.best_match.gallery_id == gid:
                                    matched_person = pid
                                    break
                        
                        # Classify result
                        if probe_person_id in gallery_ids:
                            # This person should be in gallery
                            if match_found and matched_person == probe_person_id:
                                tp += 1
                            elif match_found and matched_person != probe_person_id:
                                fp += 1
                            else:
                                fn += 1
                        else:
                            # This person should NOT be in gallery
                            if match_found:
                                fp += 1
                            else:
                                tn += 1
                
                # Calculate metrics
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Recall)
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                f1_score = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0
                
                results[threshold] = {
                    'threshold': threshold,
                    'tp': tp,
                    'fp': fp,
                    'tn': tn,
                    'fn': fn,
                    'tpr': tpr,
                    'fpr': fpr,
                    'precision': precision,
                    'f1_score': f1_score
                }
        
        finally:
            # Restore original gallery
            self.gallery_manager.clear_gallery()
            for entry in original_gallery.values():
                self.gallery_manager.gallery[entry.id] = entry
        
        self.logger.info("Consistency test completed")
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics"""
        stats = dict(self.stats)
        
        # Calculate averages
        if self.stats['processing_times']:
            stats['avg_processing_time'] = np.mean(self.stats['processing_times'])
            stats['p95_processing_time'] = np.percentile(self.stats['processing_times'], 95)
        
        if self.stats['quality_scores']['face']:
            stats['avg_face_quality'] = np.mean(self.stats['quality_scores']['face'])
        
        if self.stats['quality_scores']['body']:
            stats['avg_body_quality'] = np.mean(self.stats['quality_scores']['body'])
        
        # Add gallery stats
        stats['gallery_stats'] = self.gallery_manager.get_statistics()
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.stats = {
            'total_processed': 0,
            'face_extractions': 0,
            'body_extractions': 0,
            'matches_found': 0,
            'processing_times': [],
            'quality_scores': {'face': [], 'body': []}
        }
    
    def set_alert_thresholds(self, thresholds: Dict[AlertLevel, float]):
        """Update alert thresholds"""
        self.alert_thresholds.update(thresholds)
        self.logger.info(f"Updated alert thresholds: {thresholds}")
    
    def set_quality_thresholds(self, face_threshold: float = None, body_threshold: float = None):
        """Update quality thresholds"""
        if face_threshold is not None:
            self.face_quality_threshold = face_threshold
        if body_threshold is not None:
            self.body_quality_threshold = body_threshold
        
        self.logger.info(
            f"Updated quality thresholds - face: {self.face_quality_threshold}, "
            f"body: {self.body_quality_threshold}"
        )
    
    def export_gallery(self, export_path: str) -> bool:
        """Export gallery to file"""
        try:
            import json
            
            export_data = {
                'persons': self.gallery_manager.list_persons(),
                'statistics': self.gallery_manager.get_statistics(),
                'export_timestamp': time.time()
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Gallery exported to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export gallery: {e}")
            return False
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self.face_embedder, 'cleanup'):
                self.face_embedder.cleanup()
            if hasattr(self.body_embedder, 'cleanup'):
                self.body_embedder.cleanup()
            self.logger.info("Pipeline cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")