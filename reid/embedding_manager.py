"""Embedding Manager for Person Re-Identification.

This module manages person embeddings for re-identification, including storage,
retrieval, similarity matching, and database operations. It provides a secure
and efficient way to handle biometric embeddings while maintaining privacy.
"""

import numpy as np
import sqlite3
import json
import hashlib
import time
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging
import threading
from contextlib import contextmanager

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Using basic similarity search.")


class DistanceMetric(Enum):
    """Available distance metrics for similarity"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    HAMMING = "hamming"


class MatchStatus(Enum):
    """Status of embedding matches"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class ReIDEmbedding:
    """Person re-identification embedding"""
    embedding_id: str
    person_id: Optional[str]
    embedding_vector: np.ndarray
    confidence: float
    timestamp: float
    metadata: Dict[str, Any]
    source_image_hash: Optional[str] = None
    privacy_level: str = "standard"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'embedding_id': self.embedding_id,
            'person_id': self.person_id,
            'embedding_vector': self.embedding_vector.tolist(),
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
            'source_image_hash': self.source_image_hash,
            'privacy_level': self.privacy_level
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReIDEmbedding':
        """Create from dictionary"""
        data['embedding_vector'] = np.array(data['embedding_vector'])
        return cls(**data)


@dataclass
class MatchResult:
    """Result of embedding similarity match"""
    query_embedding_id: str
    matched_embedding_id: str
    similarity_score: float
    distance: float
    confidence: float
    status: MatchStatus
    timestamp: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['status'] = self.status.value
        return result


class EmbeddingDatabase:
    """Database for storing and managing embeddings"""
    
    def __init__(self, db_path: str = "embeddings.db"):
        """
        Initialize embedding database
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        
        # Initialize database
        self._init_database()
        
        logging.info(f"Embedding database initialized: {self.db_path}")
    
    def _init_database(self):
        """Initialize database tables"""
        with self._get_connection() as conn:
            # Embeddings table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    embedding_id TEXT PRIMARY KEY,
                    person_id TEXT,
                    embedding_vector BLOB,
                    confidence REAL,
                    timestamp REAL,
                    metadata TEXT,
                    source_image_hash TEXT,
                    privacy_level TEXT DEFAULT 'standard',
                    created_at REAL DEFAULT (strftime('%s', 'now')),
                    updated_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """)
            
            # Matches table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS matches (
                    match_id TEXT PRIMARY KEY,
                    query_embedding_id TEXT,
                    matched_embedding_id TEXT,
                    similarity_score REAL,
                    distance REAL,
                    confidence REAL,
                    status TEXT DEFAULT 'pending',
                    timestamp REAL,
                    metadata TEXT,
                    created_at REAL DEFAULT (strftime('%s', 'now')),
                    FOREIGN KEY (query_embedding_id) REFERENCES embeddings (embedding_id),
                    FOREIGN KEY (matched_embedding_id) REFERENCES embeddings (embedding_id)
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_person_id ON embeddings (person_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON embeddings (timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_match_status ON matches (status)")
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper locking"""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
    
    def store_embedding(self, embedding: ReIDEmbedding) -> bool:
        """
        Store embedding in database
        
        Args:
            embedding: Embedding to store
            
        Returns:
            True if successful
        """
        try:
            with self._get_connection() as conn:
                # Serialize embedding vector
                vector_blob = embedding.embedding_vector.tobytes()
                metadata_json = json.dumps(embedding.metadata)
                
                conn.execute("""
                    INSERT OR REPLACE INTO embeddings 
                    (embedding_id, person_id, embedding_vector, confidence, 
                     timestamp, metadata, source_image_hash, privacy_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    embedding.embedding_id,
                    embedding.person_id,
                    vector_blob,
                    embedding.confidence,
                    embedding.timestamp,
                    metadata_json,
                    embedding.source_image_hash,
                    embedding.privacy_level
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logging.error(f"Failed to store embedding {embedding.embedding_id}: {e}")
            return False
    
    def get_embedding(self, embedding_id: str) -> Optional[ReIDEmbedding]:
        """
        Retrieve embedding by ID
        
        Args:
            embedding_id: Embedding ID
            
        Returns:
            ReIDEmbedding if found, None otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM embeddings WHERE embedding_id = ?",
                    (embedding_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_embedding(row)
                return None
                
        except Exception as e:
            logging.error(f"Failed to get embedding {embedding_id}: {e}")
            return None
    
    def get_embeddings_by_person(self, person_id: str) -> List[ReIDEmbedding]:
        """
        Get all embeddings for a person
        
        Args:
            person_id: Person ID
            
        Returns:
            List of embeddings
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM embeddings WHERE person_id = ? ORDER BY timestamp DESC",
                    (person_id,)
                )
                
                return [self._row_to_embedding(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logging.error(f"Failed to get embeddings for person {person_id}: {e}")
            return []
    
    def get_all_embeddings(self, limit: Optional[int] = None) -> List[ReIDEmbedding]:
        """
        Get all embeddings
        
        Args:
            limit: Maximum number of embeddings to return
            
        Returns:
            List of embeddings
        """
        try:
            with self._get_connection() as conn:
                query = "SELECT * FROM embeddings ORDER BY timestamp DESC"
                if limit:
                    query += f" LIMIT {limit}"
                
                cursor = conn.execute(query)
                return [self._row_to_embedding(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logging.error(f"Failed to get all embeddings: {e}")
            return []
    
    def delete_embedding(self, embedding_id: str) -> bool:
        """
        Delete embedding
        
        Args:
            embedding_id: Embedding ID
            
        Returns:
            True if successful
        """
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM embeddings WHERE embedding_id = ?", (embedding_id,))
                conn.execute("DELETE FROM matches WHERE query_embedding_id = ? OR matched_embedding_id = ?", 
                           (embedding_id, embedding_id))
                conn.commit()
                return True
                
        except Exception as e:
            logging.error(f"Failed to delete embedding {embedding_id}: {e}")
            return False
    
    def store_match(self, match: MatchResult) -> bool:
        """
        Store match result
        
        Args:
            match: Match result to store
            
        Returns:
            True if successful
        """
        try:
            with self._get_connection() as conn:
                match_id = hashlib.md5(
                    f"{match.query_embedding_id}_{match.matched_embedding_id}_{match.timestamp}".encode()
                ).hexdigest()
                
                metadata_json = json.dumps(match.metadata)
                
                conn.execute("""
                    INSERT OR REPLACE INTO matches 
                    (match_id, query_embedding_id, matched_embedding_id, 
                     similarity_score, distance, confidence, status, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    match_id,
                    match.query_embedding_id,
                    match.matched_embedding_id,
                    match.similarity_score,
                    match.distance,
                    match.confidence,
                    match.status.value,
                    match.timestamp,
                    metadata_json
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logging.error(f"Failed to store match: {e}")
            return False
    
    def _row_to_embedding(self, row) -> ReIDEmbedding:
        """Convert database row to ReIDEmbedding"""
        # Deserialize embedding vector
        vector_data = np.frombuffer(row['embedding_vector'], dtype=np.float32)
        metadata = json.loads(row['metadata']) if row['metadata'] else {}
        
        return ReIDEmbedding(
            embedding_id=row['embedding_id'],
            person_id=row['person_id'],
            embedding_vector=vector_data,
            confidence=row['confidence'],
            timestamp=row['timestamp'],
            metadata=metadata,
            source_image_hash=row['source_image_hash'],
            privacy_level=row['privacy_level']
        )
    
    def cleanup_old_embeddings(self, max_age_days: int = 30) -> int:
        """
        Clean up old embeddings
        
        Args:
            max_age_days: Maximum age in days
            
        Returns:
            Number of embeddings deleted
        """
        try:
            cutoff_time = time.time() - (max_age_days * 24 * 3600)
            
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM embeddings WHERE timestamp < ?",
                    (cutoff_time,)
                )
                deleted_count = cursor.rowcount
                
                # Also clean up orphaned matches
                conn.execute("""
                    DELETE FROM matches WHERE 
                    query_embedding_id NOT IN (SELECT embedding_id FROM embeddings) OR
                    matched_embedding_id NOT IN (SELECT embedding_id FROM embeddings)
                """)
                
                conn.commit()
                return deleted_count
                
        except Exception as e:
            logging.error(f"Failed to cleanup old embeddings: {e}")
            return 0


class EmbeddingManager:
    """Main embedding manager for re-identification"""
    
    def __init__(self, db_path: str = "embeddings.db", use_faiss: bool = True):
        """
        Initialize embedding manager
        
        Args:
            db_path: Path to database file
            use_faiss: Whether to use FAISS for fast similarity search
        """
        self.database = EmbeddingDatabase(db_path)
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.faiss_index = None
        self.embedding_ids = []
        
        # Initialize FAISS index if available
        if self.use_faiss:
            self._init_faiss_index()
        
        logging.info(f"Embedding manager initialized (FAISS: {self.use_faiss})")
    
    def _init_faiss_index(self):
        """Initialize FAISS index for fast similarity search"""
        try:
            # Load existing embeddings
            embeddings = self.database.get_all_embeddings()
            
            if embeddings:
                # Get embedding dimension from first embedding
                dim = len(embeddings[0].embedding_vector)
                
                # Create FAISS index (cosine similarity)
                self.faiss_index = faiss.IndexFlatIP(dim)
                
                # Add embeddings to index
                vectors = np.array([emb.embedding_vector for emb in embeddings])
                # Normalize for cosine similarity
                faiss.normalize_L2(vectors)
                
                self.faiss_index.add(vectors)
                self.embedding_ids = [emb.embedding_id for emb in embeddings]
                
                logging.info(f"FAISS index initialized with {len(embeddings)} embeddings")
            else:
                # Create empty index (will be initialized when first embedding is added)
                self.faiss_index = None
                
        except Exception as e:
            logging.error(f"Failed to initialize FAISS index: {e}")
            self.use_faiss = False
    
    def add_embedding(self, embedding: ReIDEmbedding) -> bool:
        """
        Add new embedding
        
        Args:
            embedding: Embedding to add
            
        Returns:
            True if successful
        """
        # Store in database
        if not self.database.store_embedding(embedding):
            return False
        
        # Add to FAISS index if available
        if self.use_faiss:
            try:
                if self.faiss_index is None:
                    # Initialize index with first embedding
                    dim = len(embedding.embedding_vector)
                    self.faiss_index = faiss.IndexFlatIP(dim)
                
                # Normalize and add to index
                vector = embedding.embedding_vector.copy().reshape(1, -1)
                faiss.normalize_L2(vector)
                
                self.faiss_index.add(vector)
                self.embedding_ids.append(embedding.embedding_id)
                
            except Exception as e:
                logging.error(f"Failed to add embedding to FAISS index: {e}")
        
        return True
    
    def find_similar_embeddings(
        self, 
        query_embedding: np.ndarray, 
        threshold: float = 0.7,
        max_results: int = 10,
        metric: DistanceMetric = DistanceMetric.COSINE
    ) -> List[Tuple[ReIDEmbedding, float]]:
        """
        Find similar embeddings
        
        Args:
            query_embedding: Query embedding vector
            threshold: Similarity threshold
            max_results: Maximum number of results
            metric: Distance metric to use
            
        Returns:
            List of (embedding, similarity_score) tuples
        """
        if self.use_faiss and self.faiss_index is not None:
            return self._find_similar_faiss(query_embedding, threshold, max_results)
        else:
            return self._find_similar_basic(query_embedding, threshold, max_results, metric)
    
    def _find_similar_faiss(self, query_embedding: np.ndarray, threshold: float, max_results: int) -> List[Tuple[ReIDEmbedding, float]]:
        """Find similar embeddings using FAISS"""
        try:
            # Normalize query embedding
            query_vector = query_embedding.copy().reshape(1, -1)
            faiss.normalize_L2(query_vector)
            
            # Search
            similarities, indices = self.faiss_index.search(query_vector, max_results)
            
            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if similarity >= threshold and idx < len(self.embedding_ids):
                    embedding_id = self.embedding_ids[idx]
                    embedding = self.database.get_embedding(embedding_id)
                    if embedding:
                        results.append((embedding, float(similarity)))
            
            return results
            
        except Exception as e:
            logging.error(f"FAISS search failed: {e}")
            return []
    
    def _find_similar_basic(self, query_embedding: np.ndarray, threshold: float, max_results: int, metric: DistanceMetric) -> List[Tuple[ReIDEmbedding, float]]:
        """Find similar embeddings using basic similarity computation"""
        embeddings = self.database.get_all_embeddings()
        results = []
        
        for embedding in embeddings:
            similarity = self._compute_similarity(query_embedding, embedding.embedding_vector, metric)
            
            if similarity >= threshold:
                results.append((embedding, similarity))
        
        # Sort by similarity and limit results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]
    
    def _compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray, metric: DistanceMetric) -> float:
        """Compute similarity between two vectors"""
        if metric == DistanceMetric.COSINE:
            # Cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)
        
        elif metric == DistanceMetric.EUCLIDEAN:
            # Convert Euclidean distance to similarity
            distance = np.linalg.norm(vec1 - vec2)
            return 1.0 / (1.0 + distance)
        
        elif metric == DistanceMetric.MANHATTAN:
            # Convert Manhattan distance to similarity
            distance = np.sum(np.abs(vec1 - vec2))
            return 1.0 / (1.0 + distance)
        
        else:
            # Default to cosine
            return self._compute_similarity(vec1, vec2, DistanceMetric.COSINE)
    
    def create_match_result(
        self, 
        query_embedding_id: str, 
        matched_embedding: ReIDEmbedding, 
        similarity_score: float,
        confidence: float = None
    ) -> MatchResult:
        """
        Create match result
        
        Args:
            query_embedding_id: Query embedding ID
            matched_embedding: Matched embedding
            similarity_score: Similarity score
            confidence: Confidence score (optional)
            
        Returns:
            MatchResult object
        """
        if confidence is None:
            confidence = similarity_score
        
        return MatchResult(
            query_embedding_id=query_embedding_id,
            matched_embedding_id=matched_embedding.embedding_id,
            similarity_score=similarity_score,
            distance=1.0 - similarity_score,
            confidence=confidence,
            status=MatchStatus.PENDING,
            timestamp=time.time(),
            metadata={
                'matched_person_id': matched_embedding.person_id,
                'matched_confidence': matched_embedding.confidence,
                'privacy_level': matched_embedding.privacy_level
            }
        )
    
    def confirm_match(self, match: MatchResult) -> bool:
        """
        Confirm a match result
        
        Args:
            match: Match result to confirm
            
        Returns:
            True if successful
        """
        match.status = MatchStatus.CONFIRMED
        return self.database.store_match(match)
    
    def reject_match(self, match: MatchResult) -> bool:
        """
        Reject a match result
        
        Args:
            match: Match result to reject
            
        Returns:
            True if successful
        """
        match.status = MatchStatus.REJECTED
        return self.database.store_match(match)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get embedding database statistics
        
        Returns:
            Statistics dictionary
        """
        try:
            with self.database._get_connection() as conn:
                # Count embeddings
                cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
                total_embeddings = cursor.fetchone()[0]
                
                # Count unique persons
                cursor = conn.execute("SELECT COUNT(DISTINCT person_id) FROM embeddings WHERE person_id IS NOT NULL")
                unique_persons = cursor.fetchone()[0]
                
                # Count matches by status
                cursor = conn.execute("SELECT status, COUNT(*) FROM matches GROUP BY status")
                match_counts = dict(cursor.fetchall())
                
                return {
                    'total_embeddings': total_embeddings,
                    'unique_persons': unique_persons,
                    'match_counts': match_counts,
                    'faiss_enabled': self.use_faiss,
                    'faiss_index_size': len(self.embedding_ids) if self.use_faiss else 0
                }
                
        except Exception as e:
            logging.error(f"Failed to get statistics: {e}")
            return {}