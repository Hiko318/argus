"""Person Re-Identification Pipeline for SAR Operations.

This module provides the main pipeline for person re-identification in SAR operations,
integrating embedding generation, privacy filtering, similarity matching, and operator
confirmation workflows. It ensures privacy-first operation with operator oversight.
"""

import cv2
import numpy as np
import time
import hashlib
import logging
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
from queue import Queue, Empty

from .embedder import ReIDEmbedder, EmbeddingConfig
from .privacy_filter import PrivacyFilter, PrivacyConfig, create_privacy_filter
from .embedding_manager import EmbeddingManager, ReIDEmbedding, MatchResult, MatchStatus


class PipelineMode(Enum):
    """Pipeline operation modes"""
    REAL_TIME = "real_time"  # Real-time processing
    BATCH = "batch"  # Batch processing
    INTERACTIVE = "interactive"  # Interactive with operator confirmation


class AlertLevel(Enum):
    """Alert levels for matches"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DetectionInput:
    """Input for re-identification pipeline"""
    image: np.ndarray
    detection_bbox: Tuple[int, int, int, int]  # x, y, w, h
    detection_confidence: float
    timestamp: float
    source_id: str
    metadata: Dict[str, Any]
    
    def get_cropped_image(self) -> np.ndarray:
        """Get cropped person image from detection"""
        x, y, w, h = self.detection_bbox
        return self.image[y:y+h, x:x+w]


@dataclass
class ReIDResult:
    """Result from re-identification pipeline"""
    input_id: str
    embedding_id: str
    matches: List[MatchResult]
    alert_level: AlertLevel
    requires_confirmation: bool
    processed_image: Optional[np.ndarray]  # Privacy-filtered image
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['alert_level'] = self.alert_level.value
        result['matches'] = [match.to_dict() for match in self.matches]
        # Don't serialize image data
        result['processed_image'] = None
        return result


@dataclass
class PipelineConfig:
    """Configuration for re-identification pipeline"""
    # Embedding configuration
    embedding_config: EmbeddingConfig
    
    # Privacy configuration
    privacy_config: PrivacyConfig
    
    # Matching thresholds
    similarity_threshold: float = 0.7
    high_confidence_threshold: float = 0.85
    critical_threshold: float = 0.95
    
    # Pipeline behavior
    mode: PipelineMode = PipelineMode.INTERACTIVE
    max_matches: int = 5
    require_confirmation_above: float = 0.8
    auto_confirm_below: float = 0.6
    
    # Performance settings
    max_queue_size: int = 100
    processing_timeout: float = 30.0
    
    # Database settings
    database_path: str = "reid_embeddings.db"
    cleanup_interval_hours: int = 24
    max_embedding_age_days: int = 30
    
    # Logging
    log_level: str = "INFO"
    log_matches: bool = True
    log_rejections: bool = True


class ReIDPipeline:
    """Main re-identification pipeline"""
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize re-identification pipeline
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        
        # Initialize components
        self.embedder = ReIDEmbedder(config.embedding_config)
        self.privacy_filter = PrivacyFilter(config.privacy_config)
        self.embedding_manager = EmbeddingManager(config.database_path)
        
        # Processing queue for async operation
        self.input_queue = Queue(maxsize=config.max_queue_size)
        self.result_queue = Queue()
        
        # Threading
        self.processing_thread = None
        self.cleanup_thread = None
        self.running = False
        
        # Callbacks
        self.match_callback: Optional[Callable[[ReIDResult], None]] = None
        self.confirmation_callback: Optional[Callable[[MatchResult], bool]] = None
        
        # Statistics
        self.stats = {
            'processed_count': 0,
            'match_count': 0,
            'confirmation_count': 0,
            'rejection_count': 0,
            'error_count': 0,
            'start_time': time.time()
        }
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.log_level.upper()))
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("ReID pipeline initialized")
    
    def start(self):
        """Start the pipeline processing"""
        if self.running:
            self.logger.warning("Pipeline already running")
            return
        
        self.running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        self.logger.info("ReID pipeline started")
    
    def stop(self):
        """Stop the pipeline processing"""
        if not self.running:
            return
        
        self.running = False
        
        # Wait for threads to finish
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5.0)
        
        self.logger.info("ReID pipeline stopped")
    
    def process_detection(self, detection: DetectionInput) -> Optional[str]:
        """
        Process a person detection
        
        Args:
            detection: Detection input
            
        Returns:
            Processing ID if queued successfully, None otherwise
        """
        try:
            # Generate processing ID
            processing_id = hashlib.md5(
                f"{detection.source_id}_{detection.timestamp}_{time.time()}".encode()
            ).hexdigest()
            
            # Add to processing queue
            detection_with_id = (processing_id, detection)
            
            if self.config.mode == PipelineMode.REAL_TIME:
                # Non-blocking for real-time
                try:
                    self.input_queue.put_nowait(detection_with_id)
                    return processing_id
                except:
                    self.logger.warning("Input queue full, dropping detection")
                    return None
            else:
                # Blocking for batch/interactive
                self.input_queue.put(detection_with_id, timeout=self.config.processing_timeout)
                return processing_id
                
        except Exception as e:
            self.logger.error(f"Failed to queue detection: {e}")
            self.stats['error_count'] += 1
            return None
    
    def get_result(self, timeout: float = 1.0) -> Optional[ReIDResult]:
        """
        Get processing result
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            ReIDResult if available, None otherwise
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def process_detection_sync(self, detection: DetectionInput) -> Optional[ReIDResult]:
        """
        Process detection synchronously
        
        Args:
            detection: Detection input
            
        Returns:
            ReIDResult if successful, None otherwise
        """
        processing_id = hashlib.md5(
            f"{detection.source_id}_{detection.timestamp}_{time.time()}".encode()
        ).hexdigest()
        
        return self._process_single_detection(processing_id, detection)
    
    def _processing_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                # Get detection from queue
                processing_id, detection = self.input_queue.get(timeout=1.0)
                
                # Process detection
                result = self._process_single_detection(processing_id, detection)
                
                if result:
                    # Add to result queue
                    self.result_queue.put(result)
                    
                    # Call match callback if configured
                    if self.match_callback and result.matches:
                        try:
                            self.match_callback(result)
                        except Exception as e:
                            self.logger.error(f"Match callback failed: {e}")
                
                # Mark task as done
                self.input_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Processing loop error: {e}")
                self.stats['error_count'] += 1
    
    def _process_single_detection(self, processing_id: str, detection: DetectionInput) -> Optional[ReIDResult]:
        """
        Process a single detection
        
        Args:
            processing_id: Processing ID
            detection: Detection input
            
        Returns:
            ReIDResult if successful, None otherwise
        """
        start_time = time.time()
        
        try:
            # Get cropped person image
            person_image = detection.get_cropped_image()
            
            if person_image.size == 0:
                self.logger.warning(f"Empty person image for detection {processing_id}")
                return None
            
            # Apply privacy filtering
            filtered_image = self.privacy_filter.process_image(person_image)
            
            # Generate embedding
            embedding_vector = self.embedder.extract_embedding(person_image)
            
            if embedding_vector is None:
                self.logger.warning(f"Failed to extract embedding for detection {processing_id}")
                return None
            
            # Create embedding object
            embedding_id = hashlib.md5(
                f"{processing_id}_{detection.timestamp}".encode()
            ).hexdigest()
            
            # Calculate image hash for deduplication
            image_hash = hashlib.md5(person_image.tobytes()).hexdigest()
            
            reid_embedding = ReIDEmbedding(
                embedding_id=embedding_id,
                person_id=None,  # Will be assigned after confirmation
                embedding_vector=embedding_vector,
                confidence=detection.detection_confidence,
                timestamp=detection.timestamp,
                metadata={
                    'source_id': detection.source_id,
                    'bbox': detection.detection_bbox,
                    'processing_id': processing_id,
                    **detection.metadata
                },
                source_image_hash=image_hash,
                privacy_level=self.config.privacy_config.privacy_level.value
            )
            
            # Store embedding
            self.embedding_manager.add_embedding(reid_embedding)
            
            # Find similar embeddings
            similar_embeddings = self.embedding_manager.find_similar_embeddings(
                embedding_vector,
                threshold=self.config.similarity_threshold,
                max_results=self.config.max_matches
            )
            
            # Create match results
            matches = []
            for matched_embedding, similarity_score in similar_embeddings:
                match = self.embedding_manager.create_match_result(
                    embedding_id,
                    matched_embedding,
                    similarity_score
                )
                matches.append(match)
            
            # Determine alert level
            alert_level = self._determine_alert_level(matches)
            
            # Determine if confirmation is required
            requires_confirmation = self._requires_confirmation(matches)
            
            # Handle automatic confirmation/rejection
            if not requires_confirmation:
                for match in matches:
                    if match.similarity_score >= self.config.auto_confirm_below:
                        self.embedding_manager.confirm_match(match)
                        self.stats['confirmation_count'] += 1
                    else:
                        self.embedding_manager.reject_match(match)
                        self.stats['rejection_count'] += 1
            
            # Create result
            processing_time = time.time() - start_time
            
            result = ReIDResult(
                input_id=processing_id,
                embedding_id=embedding_id,
                matches=matches,
                alert_level=alert_level,
                requires_confirmation=requires_confirmation,
                processed_image=filtered_image,
                confidence=detection.detection_confidence,
                processing_time=processing_time,
                metadata={
                    'detection_metadata': detection.metadata,
                    'embedding_dimension': len(embedding_vector),
                    'privacy_applied': True,
                    'similar_count': len(similar_embeddings)
                }
            )
            
            # Update statistics
            self.stats['processed_count'] += 1
            if matches:
                self.stats['match_count'] += 1
            
            # Log result
            if self.config.log_matches and matches:
                self.logger.info(
                    f"ReID match found: {len(matches)} matches, "
                    f"best similarity: {max(m.similarity_score for m in matches):.3f}, "
                    f"alert: {alert_level.value}"
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process detection {processing_id}: {e}")
            self.stats['error_count'] += 1
            return None
    
    def _determine_alert_level(self, matches: List[MatchResult]) -> AlertLevel:
        """
        Determine alert level based on matches
        
        Args:
            matches: List of match results
            
        Returns:
            Alert level
        """
        if not matches:
            return AlertLevel.LOW
        
        best_similarity = max(match.similarity_score for match in matches)
        
        if best_similarity >= self.config.critical_threshold:
            return AlertLevel.CRITICAL
        elif best_similarity >= self.config.high_confidence_threshold:
            return AlertLevel.HIGH
        elif best_similarity >= self.config.similarity_threshold:
            return AlertLevel.MEDIUM
        else:
            return AlertLevel.LOW
    
    def _requires_confirmation(self, matches: List[MatchResult]) -> bool:
        """
        Determine if matches require operator confirmation
        
        Args:
            matches: List of match results
            
        Returns:
            True if confirmation required
        """
        if not matches:
            return False
        
        if self.config.mode != PipelineMode.INTERACTIVE:
            return False
        
        best_similarity = max(match.similarity_score for match in matches)
        return best_similarity >= self.config.require_confirmation_above
    
    def confirm_match(self, match_result: MatchResult, person_id: str = None) -> bool:
        """
        Confirm a match result
        
        Args:
            match_result: Match result to confirm
            person_id: Person ID to assign (optional)
            
        Returns:
            True if successful
        """
        try:
            # Update match status
            confirmed = self.embedding_manager.confirm_match(match_result)
            
            if confirmed:
                self.stats['confirmation_count'] += 1
                
                # Assign person ID if provided
                if person_id:
                    # Update embedding with person ID
                    embedding = self.embedding_manager.database.get_embedding(
                        match_result.query_embedding_id
                    )
                    if embedding:
                        embedding.person_id = person_id
                        self.embedding_manager.database.store_embedding(embedding)
                
                if self.config.log_matches:
                    self.logger.info(f"Match confirmed: {match_result.matched_embedding_id}")
            
            return confirmed
            
        except Exception as e:
            self.logger.error(f"Failed to confirm match: {e}")
            return False
    
    def reject_match(self, match_result: MatchResult) -> bool:
        """
        Reject a match result
        
        Args:
            match_result: Match result to reject
            
        Returns:
            True if successful
        """
        try:
            rejected = self.embedding_manager.reject_match(match_result)
            
            if rejected:
                self.stats['rejection_count'] += 1
                
                if self.config.log_rejections:
                    self.logger.info(f"Match rejected: {match_result.matched_embedding_id}")
            
            return rejected
            
        except Exception as e:
            self.logger.error(f"Failed to reject match: {e}")
            return False
    
    def _cleanup_loop(self):
        """Cleanup loop for old embeddings"""
        cleanup_interval = self.config.cleanup_interval_hours * 3600
        
        while self.running:
            try:
                time.sleep(cleanup_interval)
                
                if not self.running:
                    break
                
                # Cleanup old embeddings
                deleted_count = self.embedding_manager.database.cleanup_old_embeddings(
                    self.config.max_embedding_age_days
                )
                
                if deleted_count > 0:
                    self.logger.info(f"Cleaned up {deleted_count} old embeddings")
                
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get pipeline statistics
        
        Returns:
            Statistics dictionary
        """
        runtime = time.time() - self.stats['start_time']
        
        stats = self.stats.copy()
        stats.update({
            'runtime_seconds': runtime,
            'processing_rate': self.stats['processed_count'] / max(runtime, 1),
            'match_rate': self.stats['match_count'] / max(self.stats['processed_count'], 1),
            'queue_size': self.input_queue.qsize(),
            'result_queue_size': self.result_queue.qsize(),
            'running': self.running
        })
        
        # Add embedding manager statistics
        stats.update(self.embedding_manager.get_statistics())
        
        return stats
    
    def set_match_callback(self, callback: Callable[[ReIDResult], None]):
        """
        Set callback for match results
        
        Args:
            callback: Callback function
        """
        self.match_callback = callback
    
    def set_confirmation_callback(self, callback: Callable[[MatchResult], bool]):
        """
        Set callback for match confirmation
        
        Args:
            callback: Callback function that returns True to confirm, False to reject
        """
        self.confirmation_callback = callback


def create_default_pipeline(database_path: str = "reid_embeddings.db") -> ReIDPipeline:
    """
    Create a default re-identification pipeline
    
    Args:
        database_path: Path to embedding database
        
    Returns:
        Configured ReIDPipeline
    """
    # Default embedding config
    embedding_config = EmbeddingConfig(
        model_type="resnet50",
        device="auto",
        batch_size=1,
        input_size=(224, 224),
        normalize=True
    )
    
    # Default privacy config (high privacy)
    privacy_config = create_privacy_filter("high")
    
    # Default pipeline config
    pipeline_config = PipelineConfig(
        embedding_config=embedding_config,
        privacy_config=privacy_config,
        database_path=database_path,
        mode=PipelineMode.INTERACTIVE,
        similarity_threshold=0.7,
        require_confirmation_above=0.8
    )
    
    return ReIDPipeline(pipeline_config)


def create_sar_pipeline(database_path: str = "sar_reid_embeddings.db") -> ReIDPipeline:
    """
    Create a SAR-optimized re-identification pipeline
    
    Args:
        database_path: Path to embedding database
        
    Returns:
        SAR-optimized ReIDPipeline
    """
    # SAR-optimized embedding config
    embedding_config = EmbeddingConfig(
        model_type="mobilenet_v3",  # Faster for real-time SAR
        device="auto",
        batch_size=1,
        input_size=(224, 224),
        normalize=True,
        use_onnx=True  # Optimized inference
    )
    
    # SAR privacy config (standard privacy for operational needs)
    privacy_config = create_privacy_filter("standard")
    
    # SAR pipeline config
    pipeline_config = PipelineConfig(
        embedding_config=embedding_config,
        privacy_config=privacy_config,
        database_path=database_path,
        mode=PipelineMode.REAL_TIME,
        similarity_threshold=0.65,  # Lower threshold for SAR
        high_confidence_threshold=0.8,
        critical_threshold=0.9,
        require_confirmation_above=0.75,
        auto_confirm_below=0.5,
        max_matches=10,
        log_level="INFO"
    )
    
    return ReIDPipeline(pipeline_config)