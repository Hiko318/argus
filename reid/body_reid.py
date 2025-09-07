"""Full-body re-identification module for person tracking.

Implements full-body person re-identification using deep learning models
for cases where face recognition is not possible (occlusion, distance, pose).
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from torchvision.models import resnet50, mobilenet_v3_large
    import timm
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Body re-ID disabled.")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX Runtime not available. ONNX models disabled.")


class BodyReIDModel(Enum):
    """Available body re-identification models"""
    RESNET50 = "resnet50"
    OSNET = "osnet"
    MOBILENET = "mobilenet_v3"
    EFFICIENTNET = "efficientnet_b0"
    SWIN_TRANSFORMER = "swin_tiny"
    CUSTOM_CNN = "custom_cnn"
    ONNX_MODEL = "onnx_model"


@dataclass
class BodyReIDConfig:
    """Configuration for body re-identification"""
    model_type: BodyReIDModel = BodyReIDModel.RESNET50
    model_path: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    input_size: Tuple[int, int] = (256, 128)  # (height, width)
    embedding_dim: int = 512
    similarity_threshold: float = 0.6
    
    # Preprocessing settings
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Augmentation settings (for training)
    use_augmentation: bool = False
    horizontal_flip_prob: float = 0.5
    random_crop_prob: float = 0.3
    color_jitter_prob: float = 0.2
    
    # Feature extraction settings
    use_global_features: bool = True
    use_part_features: bool = True
    num_parts: int = 6  # Number of body parts for part-based features
    feature_fusion: str = "concat"  # "concat", "add", "attention"
    
    # Performance settings
    batch_size: int = 32
    use_mixed_precision: bool = True
    optimize_for_inference: bool = True


class BodyPartExtractor:
    """Extract features from different body parts"""
    
    def __init__(self, num_parts: int = 6):
        self.num_parts = num_parts
        self.logger = logging.getLogger(__name__)
    
    def extract_parts(self, feature_map: torch.Tensor) -> List[torch.Tensor]:
        """Extract part-based features from feature map
        
        Args:
            feature_map: Feature map tensor (B, C, H, W)
            
        Returns:
            List of part feature tensors
        """
        B, C, H, W = feature_map.shape
        part_height = H // self.num_parts
        
        parts = []
        for i in range(self.num_parts):
            start_h = i * part_height
            end_h = (i + 1) * part_height if i < self.num_parts - 1 else H
            
            part_feature = feature_map[:, :, start_h:end_h, :]
            # Global average pooling for each part
            part_feature = F.adaptive_avg_pool2d(part_feature, (1, 1))
            part_feature = part_feature.view(B, C)
            parts.append(part_feature)
        
        return parts


class CustomReIDNetwork(nn.Module):
    """Custom CNN network for person re-identification"""
    
    def __init__(self, embedding_dim: int = 512, num_classes: int = 1000):
        super().__init__()
        
        # Backbone
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Classification head (for training)
        self.classifier = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x: torch.Tensor, return_features: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Extract features
        features = self.backbone(x)
        features = self.global_pool(features)
        features = features.view(features.size(0), -1)
        
        # Project to embedding space
        embeddings = self.feature_proj(features)
        
        if return_features:
            return embeddings
        else:
            # For training with classification
            logits = self.classifier(embeddings)
            return embeddings, logits


class BodyReIDEmbedder:
    """Full-body person re-identification embedder"""
    
    def __init__(self, config: BodyReIDConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.model = None
        self.device = torch.device(config.device) if TORCH_AVAILABLE else None
        self.transform = None
        self.part_extractor = None
        
        if config.use_part_features:
            self.part_extractor = BodyPartExtractor(config.num_parts)
        
        self._initialize_model()
        self._initialize_transforms()
    
    def _initialize_model(self):
        """Initialize the re-identification model"""
        if not TORCH_AVAILABLE:
            self.logger.error("PyTorch not available. Cannot initialize body re-ID model.")
            return
        
        try:
            if self.config.model_type == BodyReIDModel.RESNET50:
                self._init_resnet50()
            elif self.config.model_type == BodyReIDModel.OSNET:
                self._init_osnet()
            elif self.config.model_type == BodyReIDModel.MOBILENET:
                self._init_mobilenet()
            elif self.config.model_type == BodyReIDModel.EFFICIENTNET:
                self._init_efficientnet()
            elif self.config.model_type == BodyReIDModel.SWIN_TRANSFORMER:
                self._init_swin_transformer()
            elif self.config.model_type == BodyReIDModel.CUSTOM_CNN:
                self._init_custom_cnn()
            elif self.config.model_type == BodyReIDModel.ONNX_MODEL:
                self._init_onnx_model()
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
            if self.model is not None:
                self.model.eval()
                if self.config.optimize_for_inference:
                    self.model = torch.jit.script(self.model)
                
                self.logger.info(f"Initialized body re-ID model: {self.config.model_type.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize body re-ID model: {e}")
            self._init_fallback()
    
    def _init_resnet50(self):
        """Initialize ResNet-50 based model"""
        if self.config.model_path and Path(self.config.model_path).exists():
            # Load custom trained model
            self.model = torch.load(self.config.model_path, map_location=self.device)
        else:
            # Use pretrained ResNet-50 and adapt for re-ID
            backbone = resnet50(pretrained=True)
            
            # Remove classification layer
            backbone = nn.Sequential(*list(backbone.children())[:-1])
            
            # Add re-ID head
            self.model = nn.Sequential(
                backbone,
                nn.Flatten(),
                nn.Linear(2048, self.config.embedding_dim),
                nn.BatchNorm1d(self.config.embedding_dim),
                nn.ReLU(inplace=True)
            )
        
        self.model = self.model.to(self.device)
    
    def _init_osnet(self):
        """Initialize OSNet model"""
        try:
            # Try to load OSNet from timm
            self.model = timm.create_model('osnet_x1_0', pretrained=True, num_classes=self.config.embedding_dim)
            self.model = self.model.to(self.device)
        except Exception as e:
            self.logger.warning(f"OSNet not available in timm: {e}. Using ResNet-50 instead.")
            self._init_resnet50()
    
    def _init_mobilenet(self):
        """Initialize MobileNet-based model"""
        backbone = mobilenet_v3_large(pretrained=True)
        
        # Remove classification layer
        backbone.classifier = nn.Identity()
        
        # Add re-ID head
        self.model = nn.Sequential(
            backbone,
            nn.Linear(960, self.config.embedding_dim),
            nn.BatchNorm1d(self.config.embedding_dim),
            nn.ReLU(inplace=True)
        )
        
        self.model = self.model.to(self.device)
    
    def _init_efficientnet(self):
        """Initialize EfficientNet-based model"""
        try:
            self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=self.config.embedding_dim)
            self.model = self.model.to(self.device)
        except Exception as e:
            self.logger.warning(f"EfficientNet not available: {e}. Using ResNet-50 instead.")
            self._init_resnet50()
    
    def _init_swin_transformer(self):
        """Initialize Swin Transformer model"""
        try:
            self.model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=self.config.embedding_dim)
            self.model = self.model.to(self.device)
        except Exception as e:
            self.logger.warning(f"Swin Transformer not available: {e}. Using ResNet-50 instead.")
            self._init_resnet50()
    
    def _init_custom_cnn(self):
        """Initialize custom CNN model"""
        self.model = CustomReIDNetwork(embedding_dim=self.config.embedding_dim)
        
        if self.config.model_path and Path(self.config.model_path).exists():
            state_dict = torch.load(self.config.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        
        self.model = self.model.to(self.device)
    
    def _init_onnx_model(self):
        """Initialize ONNX model"""
        if not ONNX_AVAILABLE:
            self.logger.error("ONNX Runtime not available")
            self._init_resnet50()
            return
        
        if not self.config.model_path or not Path(self.config.model_path).exists():
            self.logger.error(f"ONNX model path not found: {self.config.model_path}")
            self._init_resnet50()
            return
        
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.config.device == 'cuda' else ['CPUExecutionProvider']
            self.model = ort.InferenceSession(self.config.model_path, providers=providers)
            self.logger.info(f"Loaded ONNX model from {self.config.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load ONNX model: {e}")
            self._init_resnet50()
    
    def _init_fallback(self):
        """Initialize fallback feature extractor"""
        self.logger.warning("Using fallback feature extraction (ORB + color histogram)")
        self.model = None
    
    def _initialize_transforms(self):
        """Initialize image preprocessing transforms"""
        if not TORCH_AVAILABLE:
            return
        
        transform_list = [
            transforms.Resize(self.config.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.normalize_mean, std=self.config.normalize_std)
        ]
        
        if self.config.use_augmentation:
            augment_list = [
                transforms.Resize((int(self.config.input_size[0] * 1.1), int(self.config.input_size[1] * 1.1))),
                transforms.RandomCrop(self.config.input_size),
                transforms.RandomHorizontalFlip(p=self.config.horizontal_flip_prob),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1) if self.config.color_jitter_prob > 0 else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config.normalize_mean, std=self.config.normalize_std)
            ]
            self.transform = transforms.Compose(augment_list)
        else:
            self.transform = transforms.Compose(transform_list)
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed tensor
        """
        if not TORCH_AVAILABLE:
            return None
        
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            if self.transform is not None:
                # Convert to PIL Image for transforms
                from PIL import Image
                pil_image = Image.fromarray(rgb_image)
                tensor = self.transform(pil_image)
            else:
                # Manual preprocessing
                resized = cv2.resize(rgb_image, (self.config.input_size[1], self.config.input_size[0]))
                tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
                
                # Normalize
                mean = torch.tensor(self.config.normalize_mean).view(3, 1, 1)
                std = torch.tensor(self.config.normalize_std).view(3, 1, 1)
                tensor = (tensor - mean) / std
            
            return tensor.unsqueeze(0).to(self.device)
            
        except Exception as e:
            self.logger.error(f"Image preprocessing failed: {e}")
            return None
    
    def extract_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract re-identification embedding from person image
        
        Args:
            image: Person image (BGR format)
            
        Returns:
            Embedding vector or None if extraction fails
        """
        if self.model is None:
            return self._extract_fallback_features(image)
        
        if self.config.model_type == BodyReIDModel.ONNX_MODEL:
            return self._extract_embedding_onnx(image)
        else:
            return self._extract_embedding_pytorch(image)
    
    def _extract_embedding_pytorch(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding using PyTorch model"""
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(image)
            if input_tensor is None:
                return None
            
            # Extract features
            with torch.no_grad():
                if self.config.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        embedding = self.model(input_tensor)
                else:
                    embedding = self.model(input_tensor)
                
                # Normalize embedding
                embedding = F.normalize(embedding, p=2, dim=1)
            
            return embedding.cpu().numpy().flatten()
            
        except Exception as e:
            self.logger.error(f"PyTorch embedding extraction failed: {e}")
            return None
    
    def _extract_embedding_onnx(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding using ONNX model"""
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(image)
            if input_tensor is None:
                return None
            
            # Convert to numpy
            input_array = input_tensor.cpu().numpy()
            
            # Run inference
            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            
            result = self.model.run([output_name], {input_name: input_array})
            embedding = result[0].flatten()
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"ONNX embedding extraction failed: {e}")
            return None
    
    def _extract_fallback_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract fallback features using traditional CV methods"""
        try:
            # Resize image
            resized = cv2.resize(image, self.config.input_size[::-1])
            
            # Extract ORB features
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            orb = cv2.ORB_create(nfeatures=100)
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            
            # Create ORB feature vector
            orb_features = np.zeros(100 * 32)  # 100 keypoints * 32 descriptor length
            if descriptors is not None:
                flat_desc = descriptors.flatten()
                orb_features[:len(flat_desc)] = flat_desc
            
            # Extract color histogram
            hist_b = cv2.calcHist([resized], [0], None, [32], [0, 256])
            hist_g = cv2.calcHist([resized], [1], None, [32], [0, 256])
            hist_r = cv2.calcHist([resized], [2], None, [32], [0, 256])
            color_features = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
            
            # Combine features
            features = np.concatenate([orb_features, color_features])
            
            # Normalize
            features = features / (np.linalg.norm(features) + 1e-8)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Fallback feature extraction failed: {e}")
            return None
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score (0-1)
        """
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
    
    def is_same_person(self, embedding1: np.ndarray, embedding2: np.ndarray) -> bool:
        """Determine if two embeddings represent the same person
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            True if embeddings represent the same person
        """
        similarity = self.compute_similarity(embedding1, embedding2)
        return similarity >= self.config.similarity_threshold
    
    def process_batch(self, images: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """Process a batch of images for efficient embedding extraction
        
        Args:
            images: List of person images (BGR format)
            
        Returns:
            List of embedding vectors
        """
        if not TORCH_AVAILABLE or self.model is None:
            return [self.extract_embedding(img) for img in images]
        
        try:
            # Preprocess all images
            batch_tensors = []
            for image in images:
                tensor = self.preprocess_image(image)
                if tensor is not None:
                    batch_tensors.append(tensor.squeeze(0))
            
            if not batch_tensors:
                return [None] * len(images)
            
            # Create batch
            batch = torch.stack(batch_tensors).to(self.device)
            
            # Extract embeddings
            with torch.no_grad():
                if self.config.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        embeddings = self.model(batch)
                else:
                    embeddings = self.model(batch)
                
                # Normalize embeddings
                embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Convert to numpy
            embeddings_np = embeddings.cpu().numpy()
            
            # Return list of individual embeddings
            return [emb for emb in embeddings_np]
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            return [self.extract_embedding(img) for img in images]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_type': self.config.model_type.value,
            'device': str(self.device) if self.device else 'cpu',
            'input_size': self.config.input_size,
            'embedding_dim': self.config.embedding_dim,
            'similarity_threshold': self.config.similarity_threshold,
            'use_part_features': self.config.use_part_features,
            'num_parts': self.config.num_parts if self.config.use_part_features else 0,
            'model_path': self.config.model_path
        }