#!/usr/bin/env python3
"""
Suspect Lock Module for Foresight SAR

Provides suspect/victim locking functionality with reference image upload,
live matching, and audit logging.

Endpoints:
- POST /suspect - Upload reference image(s), returns target_id
- GET /suspect/{id}/matches - Returns live matches with locations
- PUT /suspect/{id}/lock - Lock/unlock suspect
- DELETE /suspect/{id} - Remove suspect from tracking
"""

import os
import uuid
import json
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ReferenceImage:
    """Reference image for suspect identification"""
    image_id: str
    file_path: str
    upload_timestamp: datetime
    image_hash: str
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None


@dataclass
class SuspectTarget:
    """Suspect target with reference images and tracking info"""
    target_id: str
    name: Optional[str]
    description: Optional[str]
    reference_images: List[ReferenceImage]
    created_timestamp: datetime
    is_locked: bool = False
    lock_timestamp: Optional[datetime] = None
    locked_by: Optional[str] = None
    priority: str = "medium"  # low, medium, high, critical
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class SuspectMatch:
    """Live match result for a suspect"""
    match_id: str
    target_id: str
    confidence: float
    timestamp: datetime
    location: Optional[Dict] = None  # {"lat": float, "lon": float, "alt": float}
    bounding_box: Optional[Dict] = None  # {"x": int, "y": int, "w": int, "h": int}
    frame_id: Optional[str] = None
    camera_id: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    verified: bool = False
    verified_by: Optional[str] = None
    verification_timestamp: Optional[datetime] = None


@dataclass
class AuditLogEntry:
    """Audit log entry for suspect operations"""
    entry_id: str
    timestamp: datetime
    action: str  # create, lock, unlock, match, verify, delete
    target_id: str
    user_id: Optional[str]
    details: Dict
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class FaceEmbeddingExtractor:
    """Extract face embeddings for suspect identification"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()
    
    def _load_model(self):
        """Load face embedding model"""
        try:
            if self.model_path and os.path.exists(self.model_path):
                # Load trained face embedding model
                checkpoint = torch.load(self.model_path, map_location=self.device)
                # Assuming model architecture is available
                # self.model = FaceEmbeddingModel(**checkpoint['config'])
                # self.model.load_state_dict(checkpoint['model_state_dict'])
                # self.model.eval()
                print(f"Loaded face embedding model from {self.model_path}")
            else:
                print("Using mock face embedding model for development")
                self.model = None
        except Exception as e:
            print(f"Error loading face embedding model: {e}")
            self.model = None
    
    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """Extract face embedding from image"""
        if self.model is None:
            # Mock embedding for development
            return np.random.rand(512).astype(np.float32)
        
        try:
            # Preprocess image
            face_crop = self._preprocess_face(image)
            if face_crop is None:
                return None
            
            # Extract embedding
            with torch.no_grad():
                face_tensor = torch.from_numpy(face_crop).unsqueeze(0).to(self.device)
                embedding = self.model(face_tensor)
                embedding = F.normalize(embedding, p=2, dim=1)
                return embedding.cpu().numpy().flatten()
        
        except Exception as e:
            print(f"Error extracting face embedding: {e}")
            return None
    
    def _preprocess_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess face image for embedding extraction"""
        try:
            # Detect face using OpenCV
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return None
            
            # Use largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Crop and resize face
            face_crop = image[y:y+h, x:x+w]
            face_crop = cv2.resize(face_crop, (112, 112))
            
            # Normalize to [-1, 1]
            face_crop = face_crop.astype(np.float32) / 255.0
            face_crop = (face_crop - 0.5) / 0.5
            
            # Convert to CHW format
            face_crop = np.transpose(face_crop, (2, 0, 1))
            
            return face_crop
        
        except Exception as e:
            print(f"Error preprocessing face: {e}")
            return None


class SuspectLockManager:
    """Manages suspect targets and live matching"""
    
    def __init__(self, storage_dir: str = "data/suspects", model_path: Optional[str] = None):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.targets_file = self.storage_dir / "targets.json"
        self.matches_file = self.storage_dir / "matches.json"
        self.audit_log_file = self.storage_dir / "audit_log.json"
        
        self.face_extractor = FaceEmbeddingExtractor(model_path)
        
        # In-memory storage for fast access
        self.targets: Dict[str, SuspectTarget] = {}
        self.matches: Dict[str, List[SuspectMatch]] = {}  # target_id -> matches
        self.audit_log: List[AuditLogEntry] = []
        
        self._load_data()
    
    def _load_data(self):
        """Load existing data from storage"""
        try:
            # Load targets
            if self.targets_file.exists():
                with open(self.targets_file, 'r') as f:
                    targets_data = json.load(f)
                    for target_data in targets_data:
                        target = self._dict_to_target(target_data)
                        self.targets[target.target_id] = target
            
            # Load matches
            if self.matches_file.exists():
                with open(self.matches_file, 'r') as f:
                    matches_data = json.load(f)
                    for target_id, target_matches in matches_data.items():
                        self.matches[target_id] = [
                            self._dict_to_match(match_data) for match_data in target_matches
                        ]
            
            # Load audit log
            if self.audit_log_file.exists():
                with open(self.audit_log_file, 'r') as f:
                    audit_data = json.load(f)
                    self.audit_log = [
                        self._dict_to_audit_entry(entry_data) for entry_data in audit_data
                    ]
        
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def _save_data(self):
        """Save data to storage"""
        try:
            # Save targets
            targets_data = [self._target_to_dict(target) for target in self.targets.values()]
            with open(self.targets_file, 'w') as f:
                json.dump(targets_data, f, indent=2, default=str)
            
            # Save matches
            matches_data = {
                target_id: [self._match_to_dict(match) for match in matches]
                for target_id, matches in self.matches.items()
            }
            with open(self.matches_file, 'w') as f:
                json.dump(matches_data, f, indent=2, default=str)
            
            # Save audit log
            audit_data = [self._audit_entry_to_dict(entry) for entry in self.audit_log]
            with open(self.audit_log_file, 'w') as f:
                json.dump(audit_data, f, indent=2, default=str)
        
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def create_suspect_target(self, 
                            reference_images: List[str], 
                            name: Optional[str] = None,
                            description: Optional[str] = None,
                            priority: str = "medium",
                            tags: List[str] = None,
                            user_id: Optional[str] = None) -> str:
        """Create new suspect target with reference images"""
        target_id = str(uuid.uuid4())
        
        # Process reference images
        ref_images = []
        for img_path in reference_images:
            try:
                # Load and process image
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                # Calculate image hash
                with open(img_path, 'rb') as f:
                    image_hash = hashlib.sha256(f.read()).hexdigest()
                
                # Extract face embedding
                embedding = self.face_extractor.extract_embedding(image)
                
                # Create reference image
                ref_image = ReferenceImage(
                    image_id=str(uuid.uuid4()),
                    file_path=img_path,
                    upload_timestamp=datetime.now(timezone.utc),
                    image_hash=image_hash,
                    embedding=embedding
                )
                ref_images.append(ref_image)
            
            except Exception as e:
                print(f"Error processing reference image {img_path}: {e}")
        
        if not ref_images:
            raise ValueError("No valid reference images provided")
        
        # Create suspect target
        target = SuspectTarget(
            target_id=target_id,
            name=name,
            description=description,
            reference_images=ref_images,
            created_timestamp=datetime.now(timezone.utc),
            priority=priority,
            tags=tags or []
        )
        
        self.targets[target_id] = target
        self.matches[target_id] = []
        
        # Log audit entry
        self._log_audit_entry(
            action="create",
            target_id=target_id,
            user_id=user_id,
            details={
                "name": name,
                "description": description,
                "priority": priority,
                "reference_images_count": len(ref_images)
            }
        )
        
        self._save_data()
        return target_id
    
    def get_suspect_matches(self, target_id: str, limit: int = 100) -> List[SuspectMatch]:
        """Get live matches for a suspect target"""
        if target_id not in self.matches:
            return []
        
        # Return most recent matches first
        matches = sorted(self.matches[target_id], key=lambda x: x.timestamp, reverse=True)
        return matches[:limit]
    
    def lock_suspect(self, target_id: str, locked: bool, user_id: Optional[str] = None) -> bool:
        """Lock or unlock a suspect target"""
        if target_id not in self.targets:
            return False
        
        target = self.targets[target_id]
        target.is_locked = locked
        target.lock_timestamp = datetime.now(timezone.utc) if locked else None
        target.locked_by = user_id if locked else None
        
        # Log audit entry
        self._log_audit_entry(
            action="lock" if locked else "unlock",
            target_id=target_id,
            user_id=user_id,
            details={"locked": locked}
        )
        
        self._save_data()
        return True
    
    def delete_suspect_target(self, target_id: str, user_id: Optional[str] = None) -> bool:
        """Delete a suspect target"""
        if target_id not in self.targets:
            return False
        
        # Log audit entry before deletion
        self._log_audit_entry(
            action="delete",
            target_id=target_id,
            user_id=user_id,
            details={"target_name": self.targets[target_id].name}
        )
        
        # Remove target and matches
        del self.targets[target_id]
        if target_id in self.matches:
            del self.matches[target_id]
        
        self._save_data()
        return True
    
    def add_live_match(self, 
                      target_id: str,
                      confidence: float,
                      location: Optional[Dict] = None,
                      bounding_box: Optional[Dict] = None,
                      frame_id: Optional[str] = None,
                      camera_id: Optional[str] = None,
                      embedding: Optional[np.ndarray] = None) -> str:
        """Add a live match for a suspect target"""
        if target_id not in self.targets:
            raise ValueError(f"Target {target_id} not found")
        
        match_id = str(uuid.uuid4())
        match = SuspectMatch(
            match_id=match_id,
            target_id=target_id,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc),
            location=location,
            bounding_box=bounding_box,
            frame_id=frame_id,
            camera_id=camera_id,
            embedding=embedding
        )
        
        if target_id not in self.matches:
            self.matches[target_id] = []
        
        self.matches[target_id].append(match)
        
        # Keep only recent matches (last 1000)
        if len(self.matches[target_id]) > 1000:
            self.matches[target_id] = self.matches[target_id][-1000:]
        
        # Log audit entry for high-confidence matches
        if confidence > 0.8:
            self._log_audit_entry(
                action="match",
                target_id=target_id,
                user_id=None,
                details={
                    "match_id": match_id,
                    "confidence": confidence,
                    "location": location,
                    "camera_id": camera_id
                }
            )
        
        self._save_data()
        return match_id
    
    def verify_match(self, target_id: str, match_id: str, verified: bool, user_id: Optional[str] = None) -> bool:
        """Verify or reject a suspect match"""
        if target_id not in self.matches:
            return False
        
        for match in self.matches[target_id]:
            if match.match_id == match_id:
                match.verified = verified
                match.verified_by = user_id
                match.verification_timestamp = datetime.now(timezone.utc)
                
                # Log audit entry
                self._log_audit_entry(
                    action="verify",
                    target_id=target_id,
                    user_id=user_id,
                    details={
                        "match_id": match_id,
                        "verified": verified,
                        "confidence": match.confidence
                    }
                )
                
                self._save_data()
                return True
        
        return False
    
    def search_suspects(self, image: np.ndarray, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Search for suspects matching the given image"""
        query_embedding = self.face_extractor.extract_embedding(image)
        if query_embedding is None:
            return []
        
        matches = []
        for target_id, target in self.targets.items():
            if not target.is_locked:  # Only search unlocked targets
                continue
            
            best_similarity = 0.0
            for ref_image in target.reference_images:
                if ref_image.embedding is not None:
                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        ref_image.embedding.reshape(1, -1)
                    )[0][0]
                    best_similarity = max(best_similarity, similarity)
            
            if best_similarity >= threshold:
                matches.append((target_id, best_similarity))
        
        # Sort by similarity (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def get_audit_log(self, target_id: Optional[str] = None, limit: int = 100) -> List[AuditLogEntry]:
        """Get audit log entries"""
        entries = self.audit_log
        
        if target_id:
            entries = [entry for entry in entries if entry.target_id == target_id]
        
        # Return most recent entries first
        entries = sorted(entries, key=lambda x: x.timestamp, reverse=True)
        return entries[:limit]
    
    def _log_audit_entry(self, action: str, target_id: str, user_id: Optional[str], details: Dict):
        """Log an audit entry"""
        entry = AuditLogEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            action=action,
            target_id=target_id,
            user_id=user_id,
            details=details
        )
        
        self.audit_log.append(entry)
        
        # Keep only recent entries (last 10000)
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-10000:]
    
    def _target_to_dict(self, target: SuspectTarget) -> Dict:
        """Convert SuspectTarget to dictionary"""
        data = asdict(target)
        # Convert numpy arrays to lists
        for ref_img in data['reference_images']:
            if ref_img['embedding'] is not None:
                ref_img['embedding'] = ref_img['embedding'].tolist()
        return data
    
    def _dict_to_target(self, data: Dict) -> SuspectTarget:
        """Convert dictionary to SuspectTarget"""
        # Convert embedding lists back to numpy arrays
        for ref_img in data['reference_images']:
            if ref_img['embedding'] is not None:
                ref_img['embedding'] = np.array(ref_img['embedding'])
            ref_img['upload_timestamp'] = datetime.fromisoformat(ref_img['upload_timestamp'])
        
        data['created_timestamp'] = datetime.fromisoformat(data['created_timestamp'])
        if data['lock_timestamp']:
            data['lock_timestamp'] = datetime.fromisoformat(data['lock_timestamp'])
        
        # Convert reference images
        ref_images = [ReferenceImage(**ref_img) for ref_img in data['reference_images']]
        data['reference_images'] = ref_images
        
        return SuspectTarget(**data)
    
    def _match_to_dict(self, match: SuspectMatch) -> Dict:
        """Convert SuspectMatch to dictionary"""
        data = asdict(match)
        if data['embedding'] is not None:
            data['embedding'] = data['embedding'].tolist()
        return data
    
    def _dict_to_match(self, data: Dict) -> SuspectMatch:
        """Convert dictionary to SuspectMatch"""
        if data['embedding'] is not None:
            data['embedding'] = np.array(data['embedding'])
        
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data['verification_timestamp']:
            data['verification_timestamp'] = datetime.fromisoformat(data['verification_timestamp'])
        
        return SuspectMatch(**data)
    
    def _audit_entry_to_dict(self, entry: AuditLogEntry) -> Dict:
        """Convert AuditLogEntry to dictionary"""
        return asdict(entry)
    
    def _dict_to_audit_entry(self, data: Dict) -> AuditLogEntry:
        """Convert dictionary to AuditLogEntry"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return AuditLogEntry(**data)


# Example usage
if __name__ == "__main__":
    # Initialize suspect lock manager
    manager = SuspectLockManager()
    
    # Create a suspect target
    target_id = manager.create_suspect_target(
        reference_images=["path/to/suspect1.jpg", "path/to/suspect2.jpg"],
        name="John Doe",
        description="Suspected individual from incident #123",
        priority="high",
        tags=["incident_123", "high_priority"],
        user_id="operator_001"
    )
    
    print(f"Created suspect target: {target_id}")
    
    # Lock the suspect
    manager.lock_suspect(target_id, True, "operator_001")
    
    # Simulate adding a live match
    match_id = manager.add_live_match(
        target_id=target_id,
        confidence=0.85,
        location={"lat": 40.7128, "lon": -74.0060, "alt": 100},
        bounding_box={"x": 100, "y": 50, "w": 80, "h": 120},
        frame_id="frame_001",
        camera_id="cam_001"
    )
    
    print(f"Added live match: {match_id}")
    
    # Get matches for the suspect
    matches = manager.get_suspect_matches(target_id)
    print(f"Found {len(matches)} matches for suspect {target_id}")
    
    # Get audit log
    audit_entries = manager.get_audit_log(target_id)
    print(f"Found {len(audit_entries)} audit entries for suspect {target_id}")