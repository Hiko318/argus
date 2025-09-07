#!/usr/bin/env python3
"""
Privacy-by-Default Manager

This module implements comprehensive privacy protection features for the Foresight AI system,
including face blurring, identity masking, secure data handling, and privacy-compliant
logging and export functionality.

Features:
- Face detection and blurring (default ON)
- Persistent ID masking and hashing
- Privacy-compliant logging
- Secure data export
- GDPR/privacy regulation compliance
- Configurable privacy levels
- Audit trail management
- Data retention policies
- Anonymization techniques

Author: Foresight AI Team
Date: 2024
"""

import numpy as np
import cv2
import hashlib
import hmac
import secrets
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
from collections import defaultdict, deque
import base64
from datetime import datetime, timedelta
import sqlite3
import pickle
try:
    from cryptography.fernet import Fernet
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    logging.warning("Cryptography not available. Encryption features disabled.")
try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
except ImportError:
    hashes = None
    PBKDF2HMAC = None


class PrivacyLevel(Enum):
    """Privacy protection levels"""
    MINIMAL = "minimal"          # Basic privacy protection
    STANDARD = "standard"        # Standard privacy protection (default)
    HIGH = "high"                # High privacy protection
    MAXIMUM = "maximum"          # Maximum privacy protection


class BlurMethod(Enum):
    """Face blurring methods"""
    GAUSSIAN = "gaussian"        # Gaussian blur
    PIXELATE = "pixelate"        # Pixelation
    BLACK_BOX = "black_box"      # Black rectangle
    MOSAIC = "mosaic"            # Mosaic effect
    ADAPTIVE = "adaptive"        # Adaptive based on face size


class DataType(Enum):
    """Types of data for privacy handling"""
    FACE_IMAGE = "face_image"
    FULL_BODY = "full_body"
    TRACKING_ID = "tracking_id"
    GEOLOCATION = "geolocation"
    METADATA = "metadata"
    LOG_ENTRY = "log_entry"
    EXPORT_DATA = "export_data"


class RetentionPolicy(Enum):
    """Data retention policies"""
    IMMEDIATE = "immediate"      # Delete immediately after processing
    SESSION = "session"          # Keep for current session only
    DAILY = "daily"              # Keep for 24 hours
    WEEKLY = "weekly"            # Keep for 7 days
    MONTHLY = "monthly"          # Keep for 30 days
    PERMANENT = "permanent"      # Keep permanently (with user consent)


@dataclass
class PrivacyConfig:
    """Privacy configuration settings"""
    # General privacy settings
    privacy_level: PrivacyLevel = PrivacyLevel.STANDARD
    
    # Face blurring settings
    enable_face_blur: bool = True              # Default ON
    blur_method: BlurMethod = BlurMethod.GAUSSIAN
    blur_intensity: float = 15.0               # Blur kernel size/intensity
    min_face_size: int = 30                    # Minimum face size to blur
    
    # ID masking settings
    enable_id_masking: bool = True             # Mask persistent IDs
    hash_algorithm: str = "sha256"             # Hashing algorithm
    use_salt: bool = True                      # Use salt for hashing
    
    # Data retention
    default_retention: RetentionPolicy = RetentionPolicy.DAILY
    retention_policies: Dict[DataType, RetentionPolicy] = field(default_factory=lambda: {
        DataType.FACE_IMAGE: RetentionPolicy.IMMEDIATE,
        DataType.TRACKING_ID: RetentionPolicy.SESSION,
        DataType.GEOLOCATION: RetentionPolicy.DAILY,
        DataType.METADATA: RetentionPolicy.WEEKLY,
        DataType.LOG_ENTRY: RetentionPolicy.MONTHLY,
        DataType.EXPORT_DATA: RetentionPolicy.PERMANENT
    })
    
    # Logging settings
    enable_privacy_logging: bool = True        # Log privacy actions
    log_access_attempts: bool = True           # Log data access attempts
    anonymize_logs: bool = True                # Anonymize log entries
    
    # Export settings
    require_consent: bool = True               # Require explicit consent for exports
    encrypt_exports: bool = True               # Encrypt exported data
    watermark_exports: bool = True             # Add watermarks to exported images
    
    # Compliance settings
    gdpr_compliance: bool = True               # GDPR compliance mode
    ccpa_compliance: bool = True               # CCPA compliance mode
    audit_trail: bool = True                   # Maintain audit trail
    

@dataclass
class PrivacyAuditEntry:
    """Privacy audit trail entry"""
    timestamp: float
    action: str
    data_type: DataType
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    consent_given: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'timestamp': self.timestamp,
            'action': self.action,
            'data_type': self.data_type.value,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'details': self.details,
            'ip_address': self.ip_address,
            'consent_given': self.consent_given
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PrivacyAuditEntry':
        """Create from dictionary"""
        return cls(
            timestamp=data['timestamp'],
            action=data['action'],
            data_type=DataType(data['data_type']),
            user_id=data.get('user_id'),
            session_id=data.get('session_id'),
            details=data.get('details', {}),
            ip_address=data.get('ip_address'),
            consent_given=data.get('consent_given', False)
        )


class FaceDetector:
    """Face detection for privacy protection"""
    
    def __init__(self):
        """Initialize face detector"""
        self.logger = logging.getLogger(__name__)
        
        # Initialize face detection models
        self._initialize_detectors()
        
        # Detection cache for performance
        self.detection_cache = {}
        self.cache_max_size = 100
    
    def _initialize_detectors(self):
        """Initialize face detection models"""
        try:
            # Primary: OpenCV DNN face detector
            self.dnn_net = cv2.dnn.readNetFromTensorflow(
                'models/opencv_face_detector_uint8.pb',
                'models/opencv_face_detector.pbtxt'
            )
            self.use_dnn = True
        except:
            self.use_dnn = False
            self.logger.warning("DNN face detector not available, using Haar cascades")
        
        # Fallback: Haar cascade detector
        try:
            self.haar_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.use_haar = True
        except:
            self.use_haar = False
            self.logger.error("No face detection method available")
    
    def detect_faces(self, image: np.ndarray, min_confidence: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image"""
        # Check cache first
        image_hash = hashlib.md5(image.tobytes()).hexdigest()
        if image_hash in self.detection_cache:
            return self.detection_cache[image_hash]
        
        faces = []
        
        if self.use_dnn:
            faces = self._detect_faces_dnn(image, min_confidence)
        elif self.use_haar:
            faces = self._detect_faces_haar(image)
        
        # Cache result
        if len(self.detection_cache) >= self.cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.detection_cache))
            del self.detection_cache[oldest_key]
        
        self.detection_cache[image_hash] = faces
        
        return faces
    
    def _detect_faces_dnn(self, image: np.ndarray, min_confidence: float) -> List[Tuple[int, int, int, int]]:
        """Detect faces using DNN"""
        h, w = image.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
        
        # Set input to the network
        self.dnn_net.setInput(blob)
        
        # Run forward pass
        detections = self.dnn_net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > min_confidence:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                faces.append((x1, y1, x2 - x1, y2 - y1))
        
        return faces
    
    def _detect_faces_haar(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using Haar cascades"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = self.haar_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return [(x, y, w, h) for (x, y, w, h) in faces]


class IDMasker:
    """Identity masking and hashing utilities"""
    
    def __init__(self, config: PrivacyConfig):
        """Initialize ID masker"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Generate or load salt for hashing
        self.salt = self._get_or_generate_salt()
        
        # ID mapping for session consistency
        self.id_mapping = {}
        self.reverse_mapping = {}
        
        # Thread lock for thread safety
        self.lock = threading.Lock()
    
    def _get_or_generate_salt(self) -> bytes:
        """Get existing salt or generate new one"""
        salt_file = Path('privacy_salt.key')
        
        if salt_file.exists():
            with open(salt_file, 'rb') as f:
                return f.read()
        else:
            salt = secrets.token_bytes(32)
            with open(salt_file, 'wb') as f:
                f.write(salt)
            return salt
    
    def mask_id(self, original_id: str, session_id: Optional[str] = None) -> str:
        """Mask/hash an ID for privacy"""
        with self.lock:
            # Check if already mapped in this session
            mapping_key = f"{session_id}:{original_id}" if session_id else original_id
            
            if mapping_key in self.id_mapping:
                return self.id_mapping[mapping_key]
            
            # Generate masked ID
            if self.config.use_salt:
                data_to_hash = f"{original_id}{session_id or ''}".encode() + self.salt
            else:
                data_to_hash = f"{original_id}{session_id or ''}".encode()
            
            if self.config.hash_algorithm == "sha256":
                hash_obj = hashlib.sha256(data_to_hash)
            elif self.config.hash_algorithm == "sha512":
                hash_obj = hashlib.sha512(data_to_hash)
            elif self.config.hash_algorithm == "md5":
                hash_obj = hashlib.md5(data_to_hash)
            else:
                hash_obj = hashlib.sha256(data_to_hash)
            
            masked_id = hash_obj.hexdigest()[:16]  # Use first 16 characters
            
            # Store mapping
            self.id_mapping[mapping_key] = masked_id
            self.reverse_mapping[masked_id] = original_id
            
            return masked_id
    
    def unmask_id(self, masked_id: str) -> Optional[str]:
        """Unmask an ID (if mapping exists)"""
        with self.lock:
            return self.reverse_mapping.get(masked_id)
    
    def clear_session_mappings(self, session_id: str):
        """Clear ID mappings for a specific session"""
        with self.lock:
            keys_to_remove = [k for k in self.id_mapping.keys() if k.startswith(f"{session_id}:")]
            
            for key in keys_to_remove:
                masked_id = self.id_mapping[key]
                del self.id_mapping[key]
                if masked_id in self.reverse_mapping:
                    del self.reverse_mapping[masked_id]
    
    def get_mapping_stats(self) -> Dict[str, int]:
        """Get ID mapping statistics"""
        with self.lock:
            return {
                'total_mappings': len(self.id_mapping),
                'unique_masked_ids': len(self.reverse_mapping)
            }


class DataRetentionManager:
    """Manage data retention policies"""
    
    def __init__(self, config: PrivacyConfig, db_path: str = "privacy_retention.db"):
        """Initialize data retention manager"""
        self.config = config
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._initialize_database()
        
        # Background cleanup thread
        self.cleanup_thread = None
        self.cleanup_interval = 3600  # 1 hour
        self.running = False
    
    def _initialize_database(self):
        """Initialize retention database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_type TEXT NOT NULL,
                data_id TEXT NOT NULL,
                created_timestamp REAL NOT NULL,
                retention_policy TEXT NOT NULL,
                expiry_timestamp REAL NOT NULL,
                file_path TEXT,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_expiry ON data_entries(expiry_timestamp)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_data_type ON data_entries(data_type)
        ''')
        
        conn.commit()
        conn.close()
    
    def register_data(self, data_type: DataType, data_id: str, 
                     file_path: Optional[str] = None, 
                     metadata: Optional[Dict[str, Any]] = None,
                     custom_retention: Optional[RetentionPolicy] = None) -> str:
        """Register data for retention management"""
        retention_policy = custom_retention or self.config.retention_policies.get(
            data_type, self.config.default_retention
        )
        
        created_timestamp = time.time()
        expiry_timestamp = self._calculate_expiry(created_timestamp, retention_policy)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO data_entries 
            (data_type, data_id, created_timestamp, retention_policy, expiry_timestamp, file_path, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            data_type.value,
            data_id,
            created_timestamp,
            retention_policy.value,
            expiry_timestamp,
            file_path,
            json.dumps(metadata) if metadata else None
        ))
        
        entry_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        self.logger.debug(f"Registered data entry {entry_id} for {data_type.value}")
        return str(entry_id)
    
    def _calculate_expiry(self, created_timestamp: float, retention_policy: RetentionPolicy) -> float:
        """Calculate expiry timestamp based on retention policy"""
        if retention_policy == RetentionPolicy.IMMEDIATE:
            return created_timestamp  # Expire immediately
        elif retention_policy == RetentionPolicy.SESSION:
            return created_timestamp + 86400  # 24 hours (max session)
        elif retention_policy == RetentionPolicy.DAILY:
            return created_timestamp + 86400  # 24 hours
        elif retention_policy == RetentionPolicy.WEEKLY:
            return created_timestamp + 604800  # 7 days
        elif retention_policy == RetentionPolicy.MONTHLY:
            return created_timestamp + 2592000  # 30 days
        elif retention_policy == RetentionPolicy.PERMANENT:
            return created_timestamp + 3153600000  # 100 years (effectively permanent)
        else:
            return created_timestamp + 86400  # Default to daily
    
    def cleanup_expired_data(self) -> Dict[str, int]:
        """Clean up expired data"""
        current_timestamp = time.time()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find expired entries
        cursor.execute('''
            SELECT id, data_type, data_id, file_path FROM data_entries 
            WHERE expiry_timestamp <= ?
        ''', (current_timestamp,))
        
        expired_entries = cursor.fetchall()
        
        cleanup_stats = defaultdict(int)
        
        for entry_id, data_type, data_id, file_path in expired_entries:
            try:
                # Delete file if exists
                if file_path and Path(file_path).exists():
                    Path(file_path).unlink()
                    cleanup_stats['files_deleted'] += 1
                
                # Remove from database
                cursor.execute('DELETE FROM data_entries WHERE id = ?', (entry_id,))
                cleanup_stats['entries_removed'] += 1
                cleanup_stats[f'{data_type}_removed'] += 1
                
                self.logger.debug(f"Cleaned up expired data entry {entry_id} ({data_type})")
                
            except Exception as e:
                self.logger.error(f"Error cleaning up entry {entry_id}: {e}")
                cleanup_stats['errors'] += 1
        
        conn.commit()
        conn.close()
        
        if cleanup_stats['entries_removed'] > 0:
            self.logger.info(f"Cleaned up {cleanup_stats['entries_removed']} expired data entries")
        
        return dict(cleanup_stats)
    
    def start_background_cleanup(self):
        """Start background cleanup thread"""
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
            self.running = True
            self.cleanup_thread = threading.Thread(target=self._background_cleanup_loop)
            self.cleanup_thread.daemon = True
            self.cleanup_thread.start()
            self.logger.info("Started background data cleanup")
    
    def stop_background_cleanup(self):
        """Stop background cleanup thread"""
        self.running = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
            self.logger.info("Stopped background data cleanup")
    
    def _background_cleanup_loop(self):
        """Background cleanup loop"""
        while self.running:
            try:
                self.cleanup_expired_data()
                time.sleep(self.cleanup_interval)
            except Exception as e:
                self.logger.error(f"Error in background cleanup: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def get_retention_stats(self) -> Dict[str, Any]:
        """Get data retention statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total entries
        cursor.execute('SELECT COUNT(*) FROM data_entries')
        total_entries = cursor.fetchone()[0]
        
        # Entries by type
        cursor.execute('''
            SELECT data_type, COUNT(*) FROM data_entries GROUP BY data_type
        ''')
        entries_by_type = dict(cursor.fetchall())
        
        # Expired entries
        current_timestamp = time.time()
        cursor.execute('''
            SELECT COUNT(*) FROM data_entries WHERE expiry_timestamp <= ?
        ''', (current_timestamp,))
        expired_entries = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_entries': total_entries,
            'entries_by_type': entries_by_type,
            'expired_entries': expired_entries,
            'cleanup_interval': self.cleanup_interval
        }


class PrivacyAuditLogger:
    """Privacy audit trail logger"""
    
    def __init__(self, config: PrivacyConfig, log_file: str = "privacy_audit.log"):
        """Initialize privacy audit logger"""
        self.config = config
        self.log_file = log_file
        
        # Setup audit logging
        self.audit_logger = logging.getLogger('privacy_audit')
        self.audit_logger.setLevel(logging.INFO)
        
        # File handler for audit logs
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.audit_logger.addHandler(handler)
        
        # In-memory audit trail
        self.audit_entries = deque(maxlen=1000)
        
        # Thread lock
        self.lock = threading.Lock()
    
    def log_privacy_action(self, action: str, data_type: DataType, 
                          user_id: Optional[str] = None,
                          session_id: Optional[str] = None,
                          details: Optional[Dict[str, Any]] = None,
                          ip_address: Optional[str] = None,
                          consent_given: bool = False):
        """Log a privacy-related action"""
        if not self.config.enable_privacy_logging:
            return
        
        entry = PrivacyAuditEntry(
            timestamp=time.time(),
            action=action,
            data_type=data_type,
            user_id=user_id,
            session_id=session_id,
            details=details or {},
            ip_address=ip_address,
            consent_given=consent_given
        )
        
        with self.lock:
            self.audit_entries.append(entry)
        
        # Log to file
        log_message = self._format_audit_entry(entry)
        self.audit_logger.info(log_message)
    
    def _format_audit_entry(self, entry: PrivacyAuditEntry) -> str:
        """Format audit entry for logging"""
        # Anonymize if required
        if self.config.anonymize_logs:
            user_id = "[ANONYMIZED]" if entry.user_id else None
            ip_address = "[ANONYMIZED]" if entry.ip_address else None
        else:
            user_id = entry.user_id
            ip_address = entry.ip_address
        
        return json.dumps({
            'timestamp': entry.timestamp,
            'action': entry.action,
            'data_type': entry.data_type.value,
            'user_id': user_id,
            'session_id': entry.session_id,
            'ip_address': ip_address,
            'consent_given': entry.consent_given,
            'details': entry.details
        })
    
    def get_audit_trail(self, start_time: Optional[float] = None,
                       end_time: Optional[float] = None,
                       action_filter: Optional[str] = None,
                       data_type_filter: Optional[DataType] = None) -> List[PrivacyAuditEntry]:
        """Get filtered audit trail"""
        with self.lock:
            entries = list(self.audit_entries)
        
        # Apply filters
        if start_time:
            entries = [e for e in entries if e.timestamp >= start_time]
        
        if end_time:
            entries = [e for e in entries if e.timestamp <= end_time]
        
        if action_filter:
            entries = [e for e in entries if action_filter.lower() in e.action.lower()]
        
        if data_type_filter:
            entries = [e for e in entries if e.data_type == data_type_filter]
        
        return entries
    
    def export_audit_trail(self, output_file: str, 
                          start_time: Optional[float] = None,
                          end_time: Optional[float] = None) -> str:
        """Export audit trail to file"""
        entries = self.get_audit_trail(start_time, end_time)
        
        export_data = {
            'export_timestamp': time.time(),
            'total_entries': len(entries),
            'entries': [entry.to_dict() for entry in entries]
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        # Log the export action
        self.log_privacy_action(
            action="audit_trail_exported",
            data_type=DataType.LOG_ENTRY,
            details={'output_file': output_file, 'entry_count': len(entries)}
        )
        
        return output_file


class PrivacyManager:
    """Main privacy management system"""
    
    def __init__(self, config: Optional[PrivacyConfig] = None):
        """Initialize privacy manager"""
        self.config = config or PrivacyConfig()
        
        # Initialize components
        self.face_detector = FaceDetector()
        self.id_masker = IDMasker(self.config)
        self.retention_manager = DataRetentionManager(self.config)
        self.audit_logger = PrivacyAuditLogger(self.config)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Session management
        self.current_session_id = None
        
        # Consent tracking
        self.consent_records = {}
        
        # Start background services
        self.retention_manager.start_background_cleanup()
        
        self.logger.info(f"Privacy manager initialized with {self.config.privacy_level.value} privacy level")
    
    def start_session(self, session_id: Optional[str] = None, 
                     user_id: Optional[str] = None) -> str:
        """Start a new privacy session"""
        if session_id is None:
            session_id = secrets.token_hex(16)
        
        self.current_session_id = session_id
        
        # Log session start
        self.audit_logger.log_privacy_action(
            action="session_started",
            data_type=DataType.METADATA,
            user_id=user_id,
            session_id=session_id
        )
        
        self.logger.info(f"Started privacy session: {session_id}")
        return session_id
    
    def end_session(self, session_id: Optional[str] = None):
        """End privacy session and cleanup"""
        session_id = session_id or self.current_session_id
        
        if session_id:
            # Clear session ID mappings
            self.id_masker.clear_session_mappings(session_id)
            
            # Log session end
            self.audit_logger.log_privacy_action(
                action="session_ended",
                data_type=DataType.METADATA,
                session_id=session_id
            )
            
            self.logger.info(f"Ended privacy session: {session_id}")
            
            if session_id == self.current_session_id:
                self.current_session_id = None
    
    def process_image_for_privacy(self, image: np.ndarray, 
                                 image_id: Optional[str] = None,
                                 save_original: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process image for privacy protection"""
        start_time = time.time()
        
        # Generate image ID if not provided
        if image_id is None:
            image_id = hashlib.md5(image.tobytes()).hexdigest()[:16]
        
        privacy_info = {
            'image_id': image_id,
            'faces_detected': 0,
            'faces_blurred': 0,
            'processing_time': 0.0,
            'privacy_level': self.config.privacy_level.value
        }
        
        processed_image = image.copy()
        
        # Face detection and blurring
        if self.config.enable_face_blur:
            faces = self.face_detector.detect_faces(image)
            privacy_info['faces_detected'] = len(faces)
            
            faces_blurred = 0
            for (x, y, w, h) in faces:
                # Check minimum face size
                if w >= self.config.min_face_size and h >= self.config.min_face_size:
                    processed_image = self._blur_face_region(
                        processed_image, (x, y, w, h)
                    )
                    faces_blurred += 1
            
            privacy_info['faces_blurred'] = faces_blurred
            
            # Log face blurring action
            if faces_blurred > 0:
                self.audit_logger.log_privacy_action(
                    action="faces_blurred",
                    data_type=DataType.FACE_IMAGE,
                    session_id=self.current_session_id,
                    details={'faces_blurred': faces_blurred, 'image_id': image_id}
                )
        
        # Register for data retention
        if save_original:
            self.retention_manager.register_data(
                data_type=DataType.FACE_IMAGE,
                data_id=image_id,
                metadata=privacy_info
            )
        
        privacy_info['processing_time'] = time.time() - start_time
        
        return processed_image, privacy_info
    
    def _blur_face_region(self, image: np.ndarray, face_rect: Tuple[int, int, int, int]) -> np.ndarray:
        """Blur a face region in the image"""
        x, y, w, h = face_rect
        
        # Extract face region
        face_region = image[y:y+h, x:x+w]
        
        if self.config.blur_method == BlurMethod.GAUSSIAN:
            # Gaussian blur
            kernel_size = int(self.config.blur_intensity)
            if kernel_size % 2 == 0:
                kernel_size += 1
            blurred_face = cv2.GaussianBlur(face_region, (kernel_size, kernel_size), 0)
        
        elif self.config.blur_method == BlurMethod.PIXELATE:
            # Pixelation effect
            pixel_size = int(self.config.blur_intensity)
            small = cv2.resize(face_region, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
            blurred_face = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        elif self.config.blur_method == BlurMethod.BLACK_BOX:
            # Black rectangle
            blurred_face = np.zeros_like(face_region)
        
        elif self.config.blur_method == BlurMethod.MOSAIC:
            # Mosaic effect
            tile_size = max(1, int(self.config.blur_intensity / 2))
            h_tiles, w_tiles = h // tile_size, w // tile_size
            
            blurred_face = face_region.copy()
            for i in range(h_tiles):
                for j in range(w_tiles):
                    y1, y2 = i * tile_size, (i + 1) * tile_size
                    x1, x2 = j * tile_size, (j + 1) * tile_size
                    
                    if y2 <= h and x2 <= w:
                        tile = face_region[y1:y2, x1:x2]
                        avg_color = np.mean(tile, axis=(0, 1))
                        blurred_face[y1:y2, x1:x2] = avg_color
        
        elif self.config.blur_method == BlurMethod.ADAPTIVE:
            # Adaptive blur based on face size
            blur_strength = max(5, min(25, w // 10))
            if blur_strength % 2 == 0:
                blur_strength += 1
            blurred_face = cv2.GaussianBlur(face_region, (blur_strength, blur_strength), 0)
        
        else:
            # Default to Gaussian blur
            blurred_face = cv2.GaussianBlur(face_region, (15, 15), 0)
        
        # Replace face region in original image
        result_image = image.copy()
        result_image[y:y+h, x:x+w] = blurred_face
        
        return result_image
    
    def mask_tracking_id(self, tracking_id: str) -> str:
        """Mask a tracking ID for privacy"""
        if not self.config.enable_id_masking:
            return tracking_id
        
        masked_id = self.id_masker.mask_id(tracking_id, self.current_session_id)
        
        # Log ID masking
        self.audit_logger.log_privacy_action(
            action="id_masked",
            data_type=DataType.TRACKING_ID,
            session_id=self.current_session_id,
            details={'original_length': len(tracking_id), 'masked_id': masked_id}
        )
        
        return masked_id
    
    def anonymize_geolocation(self, lat: float, lon: float, 
                            precision_meters: float = 100.0) -> Tuple[float, float]:
        """Anonymize geolocation by reducing precision"""
        # Calculate precision in degrees (approximate)
        lat_precision = precision_meters / 111000  # ~111km per degree latitude
        lon_precision = precision_meters / (111000 * np.cos(np.radians(lat)))
        
        # Round to precision
        anonymized_lat = round(lat / lat_precision) * lat_precision
        anonymized_lon = round(lon / lon_precision) * lon_precision
        
        # Log geolocation anonymization
        self.audit_logger.log_privacy_action(
            action="geolocation_anonymized",
            data_type=DataType.GEOLOCATION,
            session_id=self.current_session_id,
            details={'precision_meters': precision_meters}
        )
        
        return anonymized_lat, anonymized_lon
    
    def record_consent(self, user_id: str, consent_type: str, 
                      granted: bool, details: Optional[Dict[str, Any]] = None):
        """Record user consent"""
        consent_record = {
            'timestamp': time.time(),
            'user_id': user_id,
            'consent_type': consent_type,
            'granted': granted,
            'details': details or {},
            'session_id': self.current_session_id
        }
        
        consent_key = f"{user_id}:{consent_type}"
        self.consent_records[consent_key] = consent_record
        
        # Log consent action
        self.audit_logger.log_privacy_action(
            action="consent_recorded",
            data_type=DataType.METADATA,
            user_id=user_id,
            session_id=self.current_session_id,
            consent_given=granted,
            details={'consent_type': consent_type, 'granted': granted}
        )
    
    def check_consent(self, user_id: str, consent_type: str) -> bool:
        """Check if user has given consent"""
        consent_key = f"{user_id}:{consent_type}"
        consent_record = self.consent_records.get(consent_key)
        
        if consent_record:
            return consent_record['granted']
        
        # Default based on privacy level
        if self.config.privacy_level in [PrivacyLevel.HIGH, PrivacyLevel.MAXIMUM]:
            return False  # Require explicit consent
        else:
            return True   # Assume consent for lower privacy levels
    
    def export_data(self, user_id: str, output_file: str, 
                   data_types: Optional[List[DataType]] = None,
                   encrypt: bool = True) -> str:
        """Export user data with privacy protection"""
        # Check consent for data export
        if self.config.require_consent and not self.check_consent(user_id, "data_export"):
            raise PermissionError("User consent required for data export")
        
        # Collect data to export
        export_data = {
            'export_timestamp': time.time(),
            'user_id': user_id,
            'privacy_level': self.config.privacy_level.value,
            'data': {}
        }
        
        # Add audit trail
        if not data_types or DataType.LOG_ENTRY in data_types:
            audit_entries = self.audit_logger.get_audit_trail()
            user_entries = [e for e in audit_entries if e.user_id == user_id]
            export_data['data']['audit_trail'] = [e.to_dict() for e in user_entries]
        
        # Add other data types as needed
        # (This would be extended based on actual data storage)
        
        # Save export data
        if encrypt and self.config.encrypt_exports:
            # Encrypt the export
            encrypted_file = self._encrypt_export_file(export_data, output_file)
            final_output = encrypted_file
        else:
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            final_output = output_file
        
        # Register export for retention
        self.retention_manager.register_data(
            data_type=DataType.EXPORT_DATA,
            data_id=f"export_{user_id}_{int(time.time())}",
            file_path=final_output,
            custom_retention=RetentionPolicy.PERMANENT
        )
        
        # Log export action
        self.audit_logger.log_privacy_action(
            action="data_exported",
            data_type=DataType.EXPORT_DATA,
            user_id=user_id,
            session_id=self.current_session_id,
            consent_given=True,
            details={'output_file': final_output, 'encrypted': encrypt}
        )
        
        return final_output
    
    def _encrypt_export_file(self, data: Dict[str, Any], output_file: str) -> str:
        """Encrypt export file"""
        if not CRYPTOGRAPHY_AVAILABLE:
            # Fallback: save as plain JSON with warning
            logging.warning("Cryptography not available. Saving as plain JSON.")
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            return output_file
            
        # Generate encryption key from password (in practice, use proper key management)
        password = b"foresight_privacy_export"  # This should be user-provided
        salt = secrets.token_bytes(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        
        # Encrypt data
        fernet = Fernet(key)
        json_data = json.dumps(data).encode()
        encrypted_data = fernet.encrypt(json_data)
        
        # Save encrypted file
        encrypted_file = output_file + '.encrypted'
        with open(encrypted_file, 'wb') as f:
            f.write(salt + encrypted_data)
        
        return encrypted_file
    
    def get_privacy_statistics(self) -> Dict[str, Any]:
        """Get comprehensive privacy statistics"""
        return {
            'config': {
                'privacy_level': self.config.privacy_level.value,
                'face_blur_enabled': self.config.enable_face_blur,
                'id_masking_enabled': self.config.enable_id_masking,
                'audit_logging_enabled': self.config.enable_privacy_logging
            },
            'session': {
                'current_session_id': self.current_session_id,
                'id_mapping_stats': self.id_masker.get_mapping_stats()
            },
            'retention': self.retention_manager.get_retention_stats(),
            'consent_records': len(self.consent_records),
            'face_detection_cache_size': len(self.face_detector.detection_cache)
        }
    
    def shutdown(self):
        """Shutdown privacy manager and cleanup"""
        # End current session
        if self.current_session_id:
            self.end_session()
        
        # Stop background services
        self.retention_manager.stop_background_cleanup()
        
        # Final cleanup
        self.retention_manager.cleanup_expired_data()
        
        self.logger.info("Privacy manager shutdown complete")


# Example usage and testing
if __name__ == '__main__':
    # Example configuration
    config = PrivacyConfig(
        privacy_level=PrivacyLevel.HIGH,
        enable_face_blur=True,
        blur_method=BlurMethod.GAUSSIAN,
        enable_id_masking=True,
        require_consent=True
    )
    
    # Initialize privacy manager
    privacy_manager = PrivacyManager(config)
    
    # Start session
    session_id = privacy_manager.start_session(user_id="test_user")
    
    # Record consent
    privacy_manager.record_consent("test_user", "data_processing", True)
    
    # Process image (with dummy data)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    processed_image, privacy_info = privacy_manager.process_image_for_privacy(test_image)
    
    # Mask tracking ID
    original_id = "track_12345"
    masked_id = privacy_manager.mask_tracking_id(original_id)
    
    # Anonymize geolocation
    lat, lon = 37.7749, -122.4194  # San Francisco
    anon_lat, anon_lon = privacy_manager.anonymize_geolocation(lat, lon, 500.0)
    
    # Get statistics
    stats = privacy_manager.get_privacy_statistics()
    
    print(f"Privacy processing completed:")
    print(f"- Session ID: {session_id}")
    print(f"- Privacy info: {privacy_info}")
    print(f"- Masked ID: {original_id} -> {masked_id}")
    print(f"- Anonymized location: ({lat}, {lon}) -> ({anon_lat}, {anon_lon})")
    print(f"- Statistics: {stats}")
    
    # End session
    privacy_manager.end_session()
    
    # Shutdown
    privacy_manager.shutdown()