#!/usr/bin/env python3
"""
Re-identification (Re-ID) Module for SAR Application
Foresight SAR Application - Person/Object Re-identification

This module provides functionality for:
- Computing embeddings from images for re-identification
- Matching embeddings to find similar persons/objects
- Managing a database of known targets
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import List, Tuple, Dict, Optional, Union
import logging
from pathlib import Path
import pickle
from datetime import datetime
import hashlib

# Setup logging
logger = logging.getLogger(__name__)

class ReIDModel:
    """
    Re-identification model wrapper
    Uses a pre-trained feature extractor for computing embeddings
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize the Re-ID model
        
        Args:
            model_path: Path to pre-trained model (optional)
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.device = self._setup_device(device)
        self.model = self._load_model(model_path)
        self.transform = self._setup_transforms()
        self.embedding_dim = 512  # Standard embedding dimension
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _load_model(self, model_path: Optional[str]) -> nn.Module:
        """
        Load or create the Re-ID model
        For now, uses a simple ResNet-based feature extractor
        """
        try:
            if model_path and Path(model_path).exists():
                # Load custom trained model
                model = torch.load(model_path, map_location=self.device)
                logger.info(f"Loaded custom Re-ID model from {model_path}")
            else:
                # Use pre-trained ResNet as feature extractor
                from torchvision.models import resnet50
                model = resnet50(pretrained=True)
                # Remove the final classification layer
                model.fc = nn.Linear(model.fc.in_features, self.embedding_dim)
                logger.info("Using pre-trained ResNet50 as Re-ID feature extractor")
            
            model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"Failed to load Re-ID model: {e}")
            # Fallback to simple feature extractor
            return self._create_simple_model()
    
    def _create_simple_model(self) -> nn.Module:
        """Create a simple CNN-based feature extractor as fallback"""
        class SimpleReIDModel(nn.Module):
            def __init__(self, embedding_dim=512):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(256, embedding_dim),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(embedding_dim, embedding_dim)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return nn.functional.normalize(x, p=2, dim=1)
        
        model = SimpleReIDModel(self.embedding_dim)
        model.to(self.device)
        model.eval()
        logger.warning("Using simple fallback Re-ID model")
        return model
    
    def _setup_transforms(self) -> transforms.Compose:
        """Setup image preprocessing transforms"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),  # Standard Re-ID input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from an image
        
        Args:
            image: Input image as numpy array (H, W, C) in BGR format
            
        Returns:
            Feature vector as numpy array
        """
        try:
            # Convert BGR to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Preprocess image
            input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(input_tensor)
                features = nn.functional.normalize(features, p=2, dim=1)
            
            return features.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            # Return random features as fallback
            return np.random.randn(self.embedding_dim).astype(np.float32)

# Global model instance
_reid_model = None

def get_reid_model() -> ReIDModel:
    """Get or create the global Re-ID model instance"""
    global _reid_model
    if _reid_model is None:
        _reid_model = ReIDModel()
    return _reid_model

def compute_embedding(image: Union[np.ndarray, str]) -> np.ndarray:
    """
    Compute embedding for a given image
    
    Args:
        image: Input image as numpy array or path to image file
        
    Returns:
        Embedding vector as numpy array
    """
    try:
        # Load image if path is provided
        if isinstance(image, str):
            if not Path(image).exists():
                raise FileNotFoundError(f"Image file not found: {image}")
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Failed to load image: {image}")
        
        # Validate image
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be numpy array or file path")
        
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Image must be 3-channel (H, W, C)")
        
        # Get model and compute embedding
        model = get_reid_model()
        embedding = model.extract_features(image)
        
        logger.debug(f"Computed embedding with shape: {embedding.shape}")
        return embedding
        
    except Exception as e:
        logger.error(f"Failed to compute embedding: {e}")
        # Return zero vector as fallback
        return np.zeros(512, dtype=np.float32)

def match_embedding(target_emb: np.ndarray, probe_emb: np.ndarray, 
                   threshold: float = 0.7) -> Dict[str, float]:
    """
    Match two embeddings and return similarity score
    
    Args:
        target_emb: Target embedding vector
        probe_emb: Probe embedding vector to match against target
        threshold: Similarity threshold for positive match
        
    Returns:
        Dictionary with similarity score and match result
    """
    try:
        # Validate inputs
        if not isinstance(target_emb, np.ndarray) or not isinstance(probe_emb, np.ndarray):
            raise ValueError("Embeddings must be numpy arrays")
        
        if target_emb.shape != probe_emb.shape:
            raise ValueError(f"Embedding shapes don't match: {target_emb.shape} vs {probe_emb.shape}")
        
        # Normalize embeddings
        target_norm = target_emb / (np.linalg.norm(target_emb) + 1e-8)
        probe_norm = probe_emb / (np.linalg.norm(probe_emb) + 1e-8)
        
        # Compute cosine similarity
        similarity = np.dot(target_norm, probe_norm)
        
        # Ensure similarity is in valid range
        similarity = np.clip(similarity, -1.0, 1.0)
        
        # Convert to positive similarity score (0-1)
        similarity_score = (similarity + 1.0) / 2.0
        
        # Determine if it's a match
        is_match = similarity_score >= threshold
        
        result = {
            'similarity': float(similarity_score),
            'cosine_similarity': float(similarity),
            'is_match': bool(is_match),
            'threshold': threshold,
            'confidence': float(similarity_score) if is_match else 0.0
        }
        
        logger.debug(f"Embedding match result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to match embeddings: {e}")
        return {
            'similarity': 0.0,
            'cosine_similarity': 0.0,
            'is_match': False,
            'threshold': threshold,
            'confidence': 0.0,
            'error': str(e)
        }

def batch_match_embeddings(target_emb: np.ndarray, probe_embs: List[np.ndarray],
                          threshold: float = 0.7) -> List[Dict[str, float]]:
    """
    Match one target embedding against multiple probe embeddings
    
    Args:
        target_emb: Target embedding vector
        probe_embs: List of probe embedding vectors
        threshold: Similarity threshold for positive match
        
    Returns:
        List of match results for each probe embedding
    """
    results = []
    for i, probe_emb in enumerate(probe_embs):
        try:
            result = match_embedding(target_emb, probe_emb, threshold)
            result['probe_index'] = i
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to match probe {i}: {e}")
            results.append({
                'similarity': 0.0,
                'is_match': False,
                'probe_index': i,
                'error': str(e)
            })
    
    # Sort by similarity score (highest first)
    results.sort(key=lambda x: x.get('similarity', 0.0), reverse=True)
    return results

def create_embedding_id(embedding: np.ndarray) -> str:
    """
    Create a unique ID for an embedding based on its content
    
    Args:
        embedding: Embedding vector
        
    Returns:
        Unique string ID
    """
    # Create hash from embedding values
    embedding_bytes = embedding.tobytes()
    hash_obj = hashlib.sha256(embedding_bytes)
    return hash_obj.hexdigest()[:16]  # Use first 16 characters

def save_embedding(embedding: np.ndarray, metadata: Dict, 
                  save_path: str = "data/reid_embeddings.pkl") -> str:
    """
    Save an embedding with metadata to disk
    
    Args:
        embedding: Embedding vector to save
        metadata: Associated metadata (e.g., person_id, timestamp, etc.)
        save_path: Path to save the embedding database
        
    Returns:
        Unique embedding ID
    """
    try:
        # Create embedding ID
        emb_id = create_embedding_id(embedding)
        
        # Prepare embedding data
        emb_data = {
            'id': emb_id,
            'embedding': embedding,
            'metadata': metadata,
            'created_at': datetime.now().isoformat(),
            'shape': embedding.shape,
            'dtype': str(embedding.dtype)
        }
        
        # Load existing database or create new one
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_path.exists():
            with open(save_path, 'rb') as f:
                database = pickle.load(f)
        else:
            database = {}
        
        # Add new embedding
        database[emb_id] = emb_data
        
        # Save updated database
        with open(save_path, 'wb') as f:
            pickle.dump(database, f)
        
        logger.info(f"Saved embedding {emb_id} to {save_path}")
        return emb_id
        
    except Exception as e:
        logger.error(f"Failed to save embedding: {e}")
        return ""

def load_embeddings(load_path: str = "data/reid_embeddings.pkl") -> Dict:
    """
    Load embeddings database from disk
    
    Args:
        load_path: Path to load the embedding database from
        
    Returns:
        Dictionary of embeddings with their metadata
    """
    try:
        load_path = Path(load_path)
        if not load_path.exists():
            logger.warning(f"Embedding database not found: {load_path}")
            return {}
        
        with open(load_path, 'rb') as f:
            database = pickle.load(f)
        
        logger.info(f"Loaded {len(database)} embeddings from {load_path}")
        return database
        
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        return {}

# Example usage and testing functions
def test_reid_functions():
    """Test the Re-ID functions with dummy data"""
    logger.info("Testing Re-ID functions...")
    
    # Create dummy images
    dummy_image1 = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
    dummy_image2 = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
    
    # Test embedding computation
    emb1 = compute_embedding(dummy_image1)
    emb2 = compute_embedding(dummy_image2)
    
    logger.info(f"Embedding 1 shape: {emb1.shape}")
    logger.info(f"Embedding 2 shape: {emb2.shape}")
    
    # Test embedding matching
    match_result = match_embedding(emb1, emb2)
    logger.info(f"Match result: {match_result}")
    
    # Test self-matching (should be high similarity)
    self_match = match_embedding(emb1, emb1)
    logger.info(f"Self-match result: {self_match}")
    
    logger.info("Re-ID function tests completed")

if __name__ == "__main__":
    # Run tests if script is executed directly
    logging.basicConfig(level=logging.INFO)
    test_reid_functions()