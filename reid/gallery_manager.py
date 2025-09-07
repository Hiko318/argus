"""Gallery management for victim identification and tracking.

Manages target person galleries for re-identification matching,
supporting both face and body embeddings with persistence and search capabilities.
"""

import json
import pickle
import sqlite3
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import numpy as np
import cv2


class EmbeddingType(Enum):
    """Types of embeddings stored in gallery"""
    FACE = "face"
    BODY = "body"
    COMBINED = "combined"


class MatchConfidence(Enum):
    """Match confidence levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class GalleryEntry:
    """Single entry in the gallery"""
    id: str
    name: str
    description: str
    embedding_type: EmbeddingType
    face_embedding: Optional[np.ndarray] = None
    body_embedding: Optional[np.ndarray] = None
    reference_image: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    created_at: float = None
    updated_at: float = None
    priority: int = 1  # 1=highest, 5=lowest
    active: bool = True
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.updated_at is None:
            self.updated_at = self.created_at
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        if self.face_embedding is not None:
            data['face_embedding'] = self.face_embedding.tolist()
        if self.body_embedding is not None:
            data['body_embedding'] = self.body_embedding.tolist()
        if self.reference_image is not None:
            # Store image as base64 or save separately
            data['reference_image'] = None  # Handle separately
        data['embedding_type'] = self.embedding_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GalleryEntry':
        """Create from dictionary"""
        # Convert lists back to numpy arrays
        if data.get('face_embedding') is not None:
            data['face_embedding'] = np.array(data['face_embedding'])
        if data.get('body_embedding') is not None:
            data['body_embedding'] = np.array(data['body_embedding'])
        if 'embedding_type' in data:
            data['embedding_type'] = EmbeddingType(data['embedding_type'])
        return cls(**data)


@dataclass
class MatchResult:
    """Result of gallery matching"""
    gallery_id: str
    gallery_name: str
    similarity_score: float
    confidence: MatchConfidence
    embedding_type: EmbeddingType
    match_details: Dict[str, Any]
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class GalleryManager:
    """Manages gallery of target persons for re-identification"""
    
    def __init__(self, gallery_dir: str = "data/gallery", use_database: bool = True):
        self.gallery_dir = Path(gallery_dir)
        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_database = use_database
        self.db_path = self.gallery_dir / "gallery.db"
        self.images_dir = self.gallery_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # In-memory gallery for fast access
        self.gallery: Dict[str, GalleryEntry] = {}
        
        # Similarity thresholds
        self.face_threshold = 0.6
        self.body_threshold = 0.5
        self.combined_threshold = 0.55
        
        # Initialize storage
        if self.use_database:
            self._init_database()
        
        # Load existing gallery
        self.load_gallery()
    
    def _init_database(self):
        """Initialize SQLite database for gallery storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create gallery table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS gallery (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    embedding_type TEXT NOT NULL,
                    face_embedding BLOB,
                    body_embedding BLOB,
                    reference_image_path TEXT,
                    metadata TEXT,
                    created_at REAL,
                    updated_at REAL,
                    priority INTEGER DEFAULT 1,
                    active BOOLEAN DEFAULT 1
                )
            """)
            
            # Create matches table for history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gallery_id TEXT,
                    similarity_score REAL,
                    confidence TEXT,
                    embedding_type TEXT,
                    match_details TEXT,
                    timestamp REAL,
                    FOREIGN KEY (gallery_id) REFERENCES gallery (id)
                )
            """)
            
            # Create indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_gallery_active ON gallery (active)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_gallery_priority ON gallery (priority)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_timestamp ON matches (timestamp)")
            
            conn.commit()
            conn.close()
            
            self.logger.info("Gallery database initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
    
    def add_person(self, 
                   name: str,
                   description: str,
                   face_embedding: Optional[np.ndarray] = None,
                   body_embedding: Optional[np.ndarray] = None,
                   reference_image: Optional[np.ndarray] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   priority: int = 1) -> str:
        """Add a person to the gallery
        
        Args:
            name: Person's name or identifier
            description: Description of the person
            face_embedding: Face embedding vector
            body_embedding: Body embedding vector
            reference_image: Reference image (BGR format)
            metadata: Additional metadata
            priority: Priority level (1=highest, 5=lowest)
            
        Returns:
            Gallery entry ID
        """
        # Generate unique ID
        entry_id = self._generate_id(name)
        
        # Determine embedding type
        if face_embedding is not None and body_embedding is not None:
            embedding_type = EmbeddingType.COMBINED
        elif face_embedding is not None:
            embedding_type = EmbeddingType.FACE
        elif body_embedding is not None:
            embedding_type = EmbeddingType.BODY
        else:
            raise ValueError("At least one embedding (face or body) must be provided")
        
        # Save reference image if provided
        image_path = None
        if reference_image is not None:
            image_path = self._save_reference_image(entry_id, reference_image)
        
        # Create gallery entry
        entry = GalleryEntry(
            id=entry_id,
            name=name,
            description=description,
            embedding_type=embedding_type,
            face_embedding=face_embedding,
            body_embedding=body_embedding,
            reference_image=reference_image,
            metadata=metadata or {},
            priority=priority
        )
        
        # Add to in-memory gallery
        self.gallery[entry_id] = entry
        
        # Save to persistent storage
        if self.use_database:
            self._save_to_database(entry, image_path)
        else:
            self._save_to_file(entry)
        
        self.logger.info(f"Added person to gallery: {name} (ID: {entry_id})")
        return entry_id
    
    def _generate_id(self, name: str) -> str:
        """Generate unique ID for gallery entry"""
        timestamp = str(int(time.time() * 1000))
        name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
        return f"{name_hash}_{timestamp}"
    
    def _save_reference_image(self, entry_id: str, image: np.ndarray) -> str:
        """Save reference image to disk"""
        image_path = self.images_dir / f"{entry_id}.jpg"
        cv2.imwrite(str(image_path), image)
        return str(image_path)
    
    def _save_to_database(self, entry: GalleryEntry, image_path: Optional[str] = None):
        """Save entry to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Serialize embeddings
            face_blob = pickle.dumps(entry.face_embedding) if entry.face_embedding is not None else None
            body_blob = pickle.dumps(entry.body_embedding) if entry.body_embedding is not None else None
            metadata_json = json.dumps(entry.metadata)
            
            cursor.execute("""
                INSERT OR REPLACE INTO gallery 
                (id, name, description, embedding_type, face_embedding, body_embedding, 
                 reference_image_path, metadata, created_at, updated_at, priority, active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.id, entry.name, entry.description, entry.embedding_type.value,
                face_blob, body_blob, image_path, metadata_json,
                entry.created_at, entry.updated_at, entry.priority, entry.active
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to save to database: {e}")
    
    def _save_to_file(self, entry: GalleryEntry):
        """Save entry to JSON file (fallback)"""
        try:
            file_path = self.gallery_dir / f"{entry.id}.json"
            with open(file_path, 'w') as f:
                json.dump(entry.to_dict(), f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save to file: {e}")
    
    def load_gallery(self):
        """Load gallery from persistent storage"""
        if self.use_database and self.db_path.exists():
            self._load_from_database()
        else:
            self._load_from_files()
        
        self.logger.info(f"Loaded {len(self.gallery)} entries from gallery")
    
    def _load_from_database(self):
        """Load gallery from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, name, description, embedding_type, face_embedding, body_embedding,
                       reference_image_path, metadata, created_at, updated_at, priority, active
                FROM gallery WHERE active = 1
            """)
            
            for row in cursor.fetchall():
                (
                    entry_id, name, description, embedding_type, face_blob, body_blob,
                    image_path, metadata_json, created_at, updated_at, priority, active
                ) = row
                
                # Deserialize embeddings
                face_embedding = pickle.loads(face_blob) if face_blob else None
                body_embedding = pickle.loads(body_blob) if body_blob else None
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                # Load reference image if exists
                reference_image = None
                if image_path and Path(image_path).exists():
                    reference_image = cv2.imread(image_path)
                
                entry = GalleryEntry(
                    id=entry_id,
                    name=name,
                    description=description,
                    embedding_type=EmbeddingType(embedding_type),
                    face_embedding=face_embedding,
                    body_embedding=body_embedding,
                    reference_image=reference_image,
                    metadata=metadata,
                    created_at=created_at,
                    updated_at=updated_at,
                    priority=priority,
                    active=bool(active)
                )
                
                self.gallery[entry_id] = entry
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to load from database: {e}")
    
    def _load_from_files(self):
        """Load gallery from JSON files (fallback)"""
        try:
            for json_file in self.gallery_dir.glob("*.json"):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                entry = GalleryEntry.from_dict(data)
                self.gallery[entry.id] = entry
                
        except Exception as e:
            self.logger.error(f"Failed to load from files: {e}")
    
    def search(self, 
               face_embedding: Optional[np.ndarray] = None,
               body_embedding: Optional[np.ndarray] = None,
               top_k: int = 5,
               min_similarity: float = 0.3) -> List[MatchResult]:
        """Search gallery for matching persons
        
        Args:
            face_embedding: Query face embedding
            body_embedding: Query body embedding
            top_k: Number of top matches to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of match results sorted by similarity
        """
        if face_embedding is None and body_embedding is None:
            raise ValueError("At least one embedding (face or body) must be provided")
        
        matches = []
        
        for entry in self.gallery.values():
            if not entry.active:
                continue
            
            similarity_score = 0.0
            embedding_type = None
            match_details = {}
            
            # Face matching
            if face_embedding is not None and entry.face_embedding is not None:
                face_sim = self._compute_similarity(face_embedding, entry.face_embedding)
                match_details['face_similarity'] = face_sim
                
                if body_embedding is not None and entry.body_embedding is not None:
                    # Combined matching
                    body_sim = self._compute_similarity(body_embedding, entry.body_embedding)
                    match_details['body_similarity'] = body_sim
                    
                    # Weighted combination (face has higher weight)
                    similarity_score = 0.7 * face_sim + 0.3 * body_sim
                    embedding_type = EmbeddingType.COMBINED
                else:
                    # Face only
                    similarity_score = face_sim
                    embedding_type = EmbeddingType.FACE
            
            # Body matching only
            elif body_embedding is not None and entry.body_embedding is not None:
                body_sim = self._compute_similarity(body_embedding, entry.body_embedding)
                match_details['body_similarity'] = body_sim
                similarity_score = body_sim
                embedding_type = EmbeddingType.BODY
            
            else:
                # No compatible embeddings
                continue
            
            # Check minimum similarity
            if similarity_score < min_similarity:
                continue
            
            # Determine confidence level
            confidence = self._determine_confidence(similarity_score, embedding_type)
            
            match_result = MatchResult(
                gallery_id=entry.id,
                gallery_name=entry.name,
                similarity_score=similarity_score,
                confidence=confidence,
                embedding_type=embedding_type,
                match_details=match_details
            )
            
            matches.append(match_result)
        
        # Sort by similarity score (descending) and priority
        matches.sort(key=lambda x: (x.similarity_score, -self.gallery[x.gallery_id].priority), reverse=True)
        
        # Return top-k matches
        top_matches = matches[:top_k]
        
        # Log matches to database
        if self.use_database:
            for match in top_matches:
                self._log_match(match)
        
        return top_matches
    
    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        try:
            # Normalize embeddings
            emb1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
            emb2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
            
            # Compute cosine similarity
            similarity = np.dot(emb1_norm, emb2_norm)
            
            # Convert to 0-1 range
            similarity = (similarity + 1) / 2
            
            return float(np.clip(similarity, 0, 1))
            
        except Exception as e:
            self.logger.error(f"Similarity computation failed: {e}")
            return 0.0
    
    def _determine_confidence(self, similarity: float, embedding_type: EmbeddingType) -> MatchConfidence:
        """Determine confidence level based on similarity and embedding type"""
        # Adjust thresholds based on embedding type
        if embedding_type == EmbeddingType.FACE:
            thresholds = [0.9, 0.75, 0.6]
        elif embedding_type == EmbeddingType.BODY:
            thresholds = [0.85, 0.7, 0.5]
        else:  # COMBINED
            thresholds = [0.88, 0.72, 0.55]
        
        if similarity >= thresholds[0]:
            return MatchConfidence.VERY_HIGH
        elif similarity >= thresholds[1]:
            return MatchConfidence.HIGH
        elif similarity >= thresholds[2]:
            return MatchConfidence.MEDIUM
        else:
            return MatchConfidence.LOW
    
    def _log_match(self, match: MatchResult):
        """Log match result to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO matches 
                (gallery_id, similarity_score, confidence, embedding_type, match_details, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                match.gallery_id, match.similarity_score, match.confidence.value,
                match.embedding_type.value, json.dumps(match.match_details), match.timestamp
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to log match: {e}")
    
    def get_person(self, person_id: str) -> Optional[GalleryEntry]:
        """Get person by ID"""
        return self.gallery.get(person_id)
    
    def update_person(self, person_id: str, **kwargs) -> bool:
        """Update person information"""
        if person_id not in self.gallery:
            return False
        
        entry = self.gallery[person_id]
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(entry, key):
                setattr(entry, key, value)
        
        entry.updated_at = time.time()
        
        # Save to persistent storage
        if self.use_database:
            self._save_to_database(entry)
        else:
            self._save_to_file(entry)
        
        return True
    
    def remove_person(self, person_id: str) -> bool:
        """Remove person from gallery (soft delete)"""
        if person_id not in self.gallery:
            return False
        
        # Soft delete - mark as inactive
        self.gallery[person_id].active = False
        self.gallery[person_id].updated_at = time.time()
        
        # Update in persistent storage
        if self.use_database:
            self._save_to_database(self.gallery[person_id])
        
        # Remove from in-memory gallery
        del self.gallery[person_id]
        
        self.logger.info(f"Removed person from gallery: {person_id}")
        return True
    
    def list_persons(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """List all persons in gallery"""
        persons = []
        for entry in self.gallery.values():
            if active_only and not entry.active:
                continue
            
            person_info = {
                'id': entry.id,
                'name': entry.name,
                'description': entry.description,
                'embedding_type': entry.embedding_type.value,
                'priority': entry.priority,
                'created_at': entry.created_at,
                'updated_at': entry.updated_at,
                'has_face_embedding': entry.face_embedding is not None,
                'has_body_embedding': entry.body_embedding is not None,
                'has_reference_image': entry.reference_image is not None
            }
            persons.append(person_info)
        
        # Sort by priority and creation time
        persons.sort(key=lambda x: (x['priority'], -x['created_at']))
        return persons
    
    def get_match_history(self, person_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get match history from database"""
        if not self.use_database:
            return []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if person_id:
                cursor.execute("""
                    SELECT m.*, g.name 
                    FROM matches m 
                    JOIN gallery g ON m.gallery_id = g.id 
                    WHERE m.gallery_id = ? 
                    ORDER BY m.timestamp DESC 
                    LIMIT ?
                """, (person_id, limit))
            else:
                cursor.execute("""
                    SELECT m.*, g.name 
                    FROM matches m 
                    JOIN gallery g ON m.gallery_id = g.id 
                    ORDER BY m.timestamp DESC 
                    LIMIT ?
                """, (limit,))
            
            matches = []
            for row in cursor.fetchall():
                match_data = {
                    'id': row[0],
                    'gallery_id': row[1],
                    'gallery_name': row[7],
                    'similarity_score': row[2],
                    'confidence': row[3],
                    'embedding_type': row[4],
                    'match_details': json.loads(row[5]) if row[5] else {},
                    'timestamp': row[6]
                }
                matches.append(match_data)
            
            conn.close()
            return matches
            
        except Exception as e:
            self.logger.error(f"Failed to get match history: {e}")
            return []
    
    def clear_gallery(self):
        """Clear all entries from gallery"""
        self.gallery.clear()
        
        if self.use_database:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("UPDATE gallery SET active = 0")
                conn.commit()
                conn.close()
            except Exception as e:
                self.logger.error(f"Failed to clear database: {e}")
        
        self.logger.info("Gallery cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get gallery statistics"""
        stats = {
            'total_persons': len(self.gallery),
            'face_only': sum(1 for e in self.gallery.values() if e.embedding_type == EmbeddingType.FACE),
            'body_only': sum(1 for e in self.gallery.values() if e.embedding_type == EmbeddingType.BODY),
            'combined': sum(1 for e in self.gallery.values() if e.embedding_type == EmbeddingType.COMBINED),
            'priority_distribution': {},
            'storage_type': 'database' if self.use_database else 'files'
        }
        
        # Priority distribution
        for entry in self.gallery.values():
            priority = entry.priority
            stats['priority_distribution'][priority] = stats['priority_distribution'].get(priority, 0) + 1
        
        return stats