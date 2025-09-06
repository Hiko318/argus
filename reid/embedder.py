"""
ReID Embedder for Person Re-Identification

This module implements person re-identification embedding generation
using various backbone models and feature extraction techniques.
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Using fallback feature extraction.")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX Runtime not available. ONNX models disabled.")


class EmbeddingModel(Enum):
    """Available embedding models"""
    RESNET50 = "resnet50"
    OSNET = "osnet"
    MOBILENET = "mobilenet"
    EFFICIENTNET = "efficientnet"
    CUSTOM = "custom"
    SIMPLE_CNN = "simple_cnn"
    ORB_FEATURES = "orb_features"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    model_type: EmbeddingModel = EmbeddingModel.RESNET50
    model_path: Optional[str] = None
    input_size: Tuple[int, int] = (256, 128)  # (height, width)
    embedding_dim: int = 512
    normalize: bool = True
    use_gpu: bool = True
    batch_size: int = 1


class ReIDEmbedder:
    """Person Re-Identification Embedder"""
    
    def __init__(self, config: EmbeddingConfig = None):
        """
        Initialize ReID embedder
        
        Args:
            config: Embedding configuration
        """
        self.config = config or EmbeddingConfig()
        self.model = None
        self.device = None
        self.transforms = None
        
        # Initialize model
        self._init_model()
        self._init_transforms()
    
    def _init_model(self):
        """Initialize the embedding model"""
        try:
            if self.config.model_type == EmbeddingModel.CUSTOM and self.config.model_path:
                self._load_custom_model()
            elif TORCH_AVAILABLE and self.config.model_type in [
                EmbeddingModel.RESNET50, EmbeddingModel.OSNET, 
                EmbeddingModel.MOBILENET, EmbeddingModel.EFFICIENTNET
            ]:
                self._load_torch_model()
            elif self.config.model_type == EmbeddingModel.SIMPLE_CNN:
                self._create_simple_cnn()
            else:
                # Fallback to traditional features
                self.config.model_type = EmbeddingModel.ORB_FEATURES
                self._create_orb_extractor()
                
        except Exception as e:
            logging.error(f"Failed to initialize model {self.config.model_type}: {e}")
            # Fallback to ORB features
            self.config.model_type = EmbeddingModel.ORB_FEATURES
            self._create_orb_extractor()
    
    def _load_custom_model(self):
        """Load custom model from file"""
        model_path = Path(self.config.model_path)
        
        if model_path.suffix == '.onnx' and ONNX_AVAILABLE:
            # Load ONNX model
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.config.use_gpu else ['CPUExecutionProvider']
            self.model = ort.InferenceSession(str(model_path), providers=providers)
            self.device = 'onnx'
        elif model_path.suffix in ['.pt', '.pth'] and TORCH_AVAILABLE:
            # Load PyTorch model
            self.device = torch.device('cuda' if torch.cuda.is_available() and self.config.use_gpu else 'cpu')
            self.model = torch.jit.load(str(model_path), map_location=self.device)
            self.model.eval()
        else:
            raise ValueError(f"Unsupported model format: {model_path.suffix}")
    
    def _load_torch_model(self):
        """Load PyTorch model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.config.use_gpu else 'cpu')
        
        if self.config.model_type == EmbeddingModel.RESNET50:
            from torchvision.models import resnet50
            self.model = resnet50(pretrained=True)
            # Modify for ReID (remove classification layer)
            self.model.fc = nn.Identity()
        elif self.config.model_type == EmbeddingModel.MOBILENET:
            from torchvision.models import mobilenet_v2
            self.model = mobilenet_v2(pretrained=True)
            self.model.classifier = nn.Identity()
        else:
            # Create simple ResNet-like model
            self.model = self._create_simple_resnet()
        
        self.model.to(self.device)
        self.model.eval()
    
    def _create_simple_resnet(self):
        """Create simple ResNet-like model for ReID"""
        class SimpleReIDNet(nn.Module):
            def __init__(self, embedding_dim=512):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, stride=2, padding=1),
                    
                    # ResBlock 1
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    
                    # ResBlock 2
                    nn.Conv2d(128, 256, 3, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    
                    # ResBlock 3
                    nn.Conv2d(256, 512, 3, stride=2, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                
                self.fc = nn.Linear(512, embedding_dim)
            
            def forward(self, x):
                x = self.backbone(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        return SimpleReIDNet(self.config.embedding_dim)
    
    def _create_simple_cnn(self):
        """Create simple CNN for feature extraction"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for Simple CNN model")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.config.use_gpu else 'cpu')
        self.model = self._create_simple_resnet()
        self.model.to(self.device)
        self.model.eval()
    
    def _create_orb_extractor(self):
        """Create ORB feature extractor as fallback"""
        self.model = cv2.ORB_create(nfeatures=1000)
        self.device = 'cpu'
    
    def _init_transforms(self):
        """Initialize image transforms"""
        if self.config.model_type == EmbeddingModel.ORB_FEATURES:
            self.transforms = None
            return
        
        if TORCH_AVAILABLE:
            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.config.input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transforms = None
    
    def extract_embedding(self, image: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Extract embedding from image
        
        Args:
            image: Input image (BGR format)
            bbox: Bounding box (x1, y1, x2, y2) to crop person region
            
        Returns:
            Normalized embedding vector
        """
        # Crop person region if bbox provided
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            person_crop = image[y1:y2, x1:x2]
        else:
            person_crop = image
        
        if person_crop.size == 0:
            return np.zeros(self.config.embedding_dim)
        
        # Extract features based on model type
        if self.config.model_type == EmbeddingModel.ORB_FEATURES:
            return self._extract_orb_features(person_crop)
        else:
            return self._extract_deep_features(person_crop)
    
    def _extract_deep_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features using deep learning model"""
        try:
            # Preprocess image
            if self.transforms:
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                input_tensor = self.transforms(rgb_image)
                
                if TORCH_AVAILABLE and hasattr(input_tensor, 'unsqueeze'):
                    input_tensor = input_tensor.unsqueeze(0).to(self.device)
            else:
                # Manual preprocessing
                resized = cv2.resize(image, (self.config.input_size[1], self.config.input_size[0]))
                normalized = resized.astype(np.float32) / 255.0
                input_tensor = np.transpose(normalized, (2, 0, 1))
                input_tensor = np.expand_dims(input_tensor, axis=0)
            
            # Forward pass
            if self.device == 'onnx':
                # ONNX inference
                input_name = self.model.get_inputs()[0].name
                outputs = self.model.run(None, {input_name: input_tensor})
                embedding = outputs[0][0]
            else:
                # PyTorch inference
                with torch.no_grad():
                    if isinstance(input_tensor, np.ndarray):
                        input_tensor = torch.from_numpy(input_tensor).to(self.device)
                    embedding = self.model(input_tensor)
                    embedding = embedding.cpu().numpy()[0]
            
            # Normalize if required
            if self.config.normalize and np.linalg.norm(embedding) > 0:
                embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logging.error(f"Deep feature extraction failed: {e}")
            return self._extract_orb_features(image)
    
    def _extract_orb_features(self, image: np.ndarray) -> np.ndarray:
        """Extract ORB features as fallback"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Resize to standard size
        resized = cv2.resize(gray, self.config.input_size[::-1])
        
        # Extract ORB features
        keypoints, descriptors = self.model.detectAndCompute(resized, None)
        
        if descriptors is not None and len(descriptors) > 0:
            # Aggregate descriptors
            feature_vector = np.mean(descriptors, axis=0)
            
            # Pad or truncate to desired dimension
            if len(feature_vector) < self.config.embedding_dim:
                feature_vector = np.pad(feature_vector, (0, self.config.embedding_dim - len(feature_vector)))
            else:
                feature_vector = feature_vector[:self.config.embedding_dim]
        else:
            # Use histogram features as last resort
            hist = cv2.calcHist([resized], [0], None, [256], [0, 256])
            feature_vector = hist.flatten()
            
            # Pad or truncate
            if len(feature_vector) < self.config.embedding_dim:
                feature_vector = np.pad(feature_vector, (0, self.config.embedding_dim - len(feature_vector)))
            else:
                feature_vector = feature_vector[:self.config.embedding_dim]
        
        # Normalize
        if self.config.normalize and np.linalg.norm(feature_vector) > 0:
            feature_vector = feature_vector / np.linalg.norm(feature_vector)
        
        return feature_vector.astype(np.float32)
    
    def extract_batch_embeddings(self, images: List[np.ndarray], 
                                bboxes: Optional[List[Tuple[int, int, int, int]]] = None) -> List[np.ndarray]:
        """
        Extract embeddings from batch of images
        
        Args:
            images: List of input images
            bboxes: Optional list of bounding boxes
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i, image in enumerate(images):
            bbox = bboxes[i] if bboxes else None
            embedding = self.extract_embedding(image, bbox)
            embeddings.append(embedding)
        
        return embeddings
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1, higher is more similar)
        """
        # Ensure embeddings are normalized
        if np.linalg.norm(embedding1) > 0:
            embedding1 = embedding1 / np.linalg.norm(embedding1)
        if np.linalg.norm(embedding2) > 0:
            embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2)
        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_type': self.config.model_type.value,
            'model_path': self.config.model_path,
            'input_size': self.config.input_size,
            'embedding_dim': self.config.embedding_dim,
            'device': str(self.device),
            'normalize': self.config.normalize,
            'torch_available': TORCH_AVAILABLE,
            'onnx_available': ONNX_AVAILABLE
        }
    
    def save_config(self, path: str):
        """Save embedder configuration"""
        import json
        
        config_dict = {
            'model_type': self.config.model_type.value,
            'model_path': self.config.model_path,
            'input_size': self.config.input_size,
            'embedding_dim': self.config.embedding_dim,
            'normalize': self.config.normalize,
            'use_gpu': self.config.use_gpu,
            'batch_size': self.config.batch_size
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_config(cls, path: str) -> 'ReIDEmbedder':
        """Load embedder from configuration file"""
        import json
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        config = EmbeddingConfig(
            model_type=EmbeddingModel(config_dict['model_type']),
            model_path=config_dict.get('model_path'),
            input_size=tuple(config_dict['input_size']),
            embedding_dim=config_dict['embedding_dim'],
            normalize=config_dict['normalize'],
            use_gpu=config_dict['use_gpu'],
            batch_size=config_dict['batch_size']
        )
        
        return cls(config)