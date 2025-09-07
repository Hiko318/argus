"""Face embedding module for re-identification pipeline.

Implements face detection and embedding generation using ArcFace/FaceNet models
for high-accuracy face recognition in SAR operations.
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
    import torch.nn.functional as F
    from facenet_pytorch import MTCNN, InceptionResnetV1
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False
    logging.warning("FaceNet-PyTorch not available. Face embedding disabled.")

try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("InsightFace not available. ArcFace embedding disabled.")


class FaceModel(Enum):
    """Available face recognition models"""
    FACENET = "facenet"
    ARCFACE = "arcface"
    MTCNN_FACENET = "mtcnn_facenet"
    RETINAFACE_ARCFACE = "retinaface_arcface"


@dataclass
class FaceDetection:
    """Face detection result"""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    landmarks: Optional[np.ndarray] = None
    aligned_face: Optional[np.ndarray] = None
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of face bounding box"""
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)
    
    @property
    def area(self) -> int:
        """Get area of face bounding box"""
        _, _, w, h = self.bbox
        return w * h


@dataclass
class FaceEmbeddingConfig:
    """Configuration for face embedding"""
    model_type: FaceModel = FaceModel.FACENET
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    detection_threshold: float = 0.7
    embedding_threshold: float = 0.6
    min_face_size: int = 40
    max_face_size: int = 1000
    face_margin: float = 0.2  # Margin around detected face
    align_faces: bool = True
    extract_multiple: bool = False  # Extract all faces or just the largest
    
    # Model-specific settings
    mtcnn_thresholds: Tuple[float, float, float] = (0.6, 0.7, 0.7)
    arcface_model_name: str = "buffalo_l"  # InsightFace model
    facenet_pretrained: str = "vggface2"  # FaceNet pretrained weights


class FaceEmbedder:
    """Face detection and embedding generation for re-identification"""
    
    def __init__(self, config: FaceEmbeddingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize models based on configuration
        self.detector = None
        self.embedder = None
        self.device = torch.device(config.device) if torch.cuda.is_available() else torch.device("cpu")
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize face detection and embedding models"""
        try:
            if self.config.model_type == FaceModel.FACENET:
                self._init_facenet()
            elif self.config.model_type == FaceModel.ARCFACE:
                self._init_arcface()
            elif self.config.model_type == FaceModel.MTCNN_FACENET:
                self._init_mtcnn_facenet()
            elif self.config.model_type == FaceModel.RETINAFACE_ARCFACE:
                self._init_retinaface_arcface()
            else:
                raise ValueError(f"Unsupported face model: {self.config.model_type}")
                
            self.logger.info(f"Initialized face embedder with {self.config.model_type.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize face models: {e}")
            self._init_fallback()
    
    def _init_facenet(self):
        """Initialize FaceNet model"""
        if not FACENET_AVAILABLE:
            raise ImportError("FaceNet-PyTorch not available")
        
        # MTCNN for face detection
        self.detector = MTCNN(
            image_size=160,
            margin=int(160 * self.config.face_margin),
            min_face_size=self.config.min_face_size,
            thresholds=list(self.config.mtcnn_thresholds),
            factor=0.709,
            post_process=True,
            device=self.device
        )
        
        # InceptionResnetV1 for embedding
        self.embedder = InceptionResnetV1(
            pretrained=self.config.facenet_pretrained
        ).eval().to(self.device)
    
    def _init_arcface(self):
        """Initialize ArcFace model"""
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("InsightFace not available")
        
        # Initialize InsightFace app
        self.app = insightface.app.FaceAnalysis(
            name=self.config.arcface_model_name,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0 if self.config.device == "cuda" else -1, det_size=(640, 640))
    
    def _init_mtcnn_facenet(self):
        """Initialize MTCNN + FaceNet combination"""
        self._init_facenet()  # Uses MTCNN + FaceNet
    
    def _init_retinaface_arcface(self):
        """Initialize RetinaFace + ArcFace combination"""
        self._init_arcface()  # InsightFace includes RetinaFace
    
    def _init_fallback(self):
        """Initialize fallback face detection using OpenCV"""
        self.logger.warning("Using OpenCV Haar cascade fallback for face detection")
        
        # Load OpenCV face cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        self.embedder = None  # No embedding model available
    
    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces in image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of face detections
        """
        if self.config.model_type in [FaceModel.FACENET, FaceModel.MTCNN_FACENET]:
            return self._detect_faces_mtcnn(image)
        elif self.config.model_type in [FaceModel.ARCFACE, FaceModel.RETINAFACE_ARCFACE]:
            return self._detect_faces_insightface(image)
        else:
            return self._detect_faces_opencv(image)
    
    def _detect_faces_mtcnn(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces using MTCNN"""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            boxes, probs, landmarks = self.detector.detect(rgb_image, landmarks=True)
            
            detections = []
            if boxes is not None:
                for i, (box, prob, landmark) in enumerate(zip(boxes, probs, landmarks)):
                    if prob >= self.config.detection_threshold:
                        x1, y1, x2, y2 = box.astype(int)
                        w, h = x2 - x1, y2 - y1
                        
                        # Filter by size
                        if self.config.min_face_size <= min(w, h) <= self.config.max_face_size:
                            # Extract aligned face
                            aligned_face = None
                            if self.config.align_faces and landmark is not None:
                                aligned_face = self._align_face(rgb_image, landmark)
                            
                            detection = FaceDetection(
                                bbox=(x1, y1, w, h),
                                confidence=float(prob),
                                landmarks=landmark,
                                aligned_face=aligned_face
                            )
                            detections.append(detection)
            
            # Sort by confidence and return top detection if not extracting multiple
            detections.sort(key=lambda x: x.confidence, reverse=True)
            if not self.config.extract_multiple and detections:
                detections = [detections[0]]
            
            return detections
            
        except Exception as e:
            self.logger.error(f"MTCNN face detection failed: {e}")
            return []
    
    def _detect_faces_insightface(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces using InsightFace"""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = self.app.get(rgb_image)
            
            detections = []
            for face in faces:
                if face.det_score >= self.config.detection_threshold:
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox
                    w, h = x2 - x1, y2 - y1
                    
                    # Filter by size
                    if self.config.min_face_size <= min(w, h) <= self.config.max_face_size:
                        # Get landmarks
                        landmarks = face.kps if hasattr(face, 'kps') else None
                        
                        # Extract aligned face
                        aligned_face = None
                        if self.config.align_faces and landmarks is not None:
                            aligned_face = self._align_face_insightface(rgb_image, face)
                        
                        detection = FaceDetection(
                            bbox=(x1, y1, w, h),
                            confidence=float(face.det_score),
                            landmarks=landmarks,
                            aligned_face=aligned_face
                        )
                        detections.append(detection)
            
            # Sort by confidence and return top detection if not extracting multiple
            detections.sort(key=lambda x: x.confidence, reverse=True)
            if not self.config.extract_multiple and detections:
                detections = [detections[0]]
            
            return detections
            
        except Exception as e:
            self.logger.error(f"InsightFace detection failed: {e}")
            return []
    
    def _detect_faces_opencv(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces using OpenCV Haar cascade (fallback)"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.config.min_face_size, self.config.min_face_size)
            )
            
            detections = []
            for (x, y, w, h) in faces:
                if self.config.min_face_size <= min(w, h) <= self.config.max_face_size:
                    detection = FaceDetection(
                        bbox=(x, y, w, h),
                        confidence=0.8,  # Default confidence for OpenCV
                        landmarks=None,
                        aligned_face=None
                    )
                    detections.append(detection)
            
            # Sort by area (larger faces first) and return top detection if not extracting multiple
            detections.sort(key=lambda x: x.area, reverse=True)
            if not self.config.extract_multiple and detections:
                detections = [detections[0]]
            
            return detections
            
        except Exception as e:
            self.logger.error(f"OpenCV face detection failed: {e}")
            return []
    
    def _align_face(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Align face using landmarks (MTCNN format)"""
        try:
            # Simple alignment based on eye positions
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            
            # Calculate angle
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Calculate center
            center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
            
            # Rotate image
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            
            return aligned
            
        except Exception as e:
            self.logger.error(f"Face alignment failed: {e}")
            return image
    
    def _align_face_insightface(self, image: np.ndarray, face) -> np.ndarray:
        """Align face using InsightFace"""
        try:
            # Use InsightFace's built-in alignment
            aligned_face = insightface.utils.face_align.norm_crop(image, face.kps, 112)
            return aligned_face
            
        except Exception as e:
            self.logger.error(f"InsightFace alignment failed: {e}")
            return image
    
    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding from aligned face image
        
        Args:
            face_image: Aligned face image (RGB format)
            
        Returns:
            Face embedding vector or None if extraction fails
        """
        if self.config.model_type in [FaceModel.FACENET, FaceModel.MTCNN_FACENET]:
            return self._extract_embedding_facenet(face_image)
        elif self.config.model_type in [FaceModel.ARCFACE, FaceModel.RETINAFACE_ARCFACE]:
            return self._extract_embedding_arcface(face_image)
        else:
            self.logger.warning("No embedding model available")
            return None
    
    def _extract_embedding_facenet(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding using FaceNet"""
        try:
            # Preprocess image
            face_tensor = torch.from_numpy(face_image).permute(2, 0, 1).float()
            face_tensor = (face_tensor - 127.5) / 128.0  # Normalize to [-1, 1]
            face_tensor = face_tensor.unsqueeze(0).to(self.device)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.embedder(face_tensor)
                embedding = F.normalize(embedding, p=2, dim=1)
            
            return embedding.cpu().numpy().flatten()
            
        except Exception as e:
            self.logger.error(f"FaceNet embedding extraction failed: {e}")
            return None
    
    def _extract_embedding_arcface(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding using ArcFace (via InsightFace)"""
        try:
            # InsightFace expects BGR format
            bgr_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            
            # Get face analysis (includes embedding)
            faces = self.app.get(bgr_image)
            
            if faces:
                # Return embedding from the first (and likely only) face
                embedding = faces[0].embedding
                # Normalize embedding
                embedding = embedding / np.linalg.norm(embedding)
                return embedding
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"ArcFace embedding extraction failed: {e}")
            return None
    
    def process_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Process image and extract face embeddings
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of face detection results with embeddings
        """
        results = []
        
        # Detect faces
        detections = self.detect_faces(image)
        
        for detection in detections:
            result = {
                'bbox': detection.bbox,
                'confidence': detection.confidence,
                'landmarks': detection.landmarks,
                'embedding': None,
                'face_image': None
            }
            
            # Extract face region
            x, y, w, h = detection.bbox
            face_region = image[y:y+h, x:x+w]
            
            # Use aligned face if available, otherwise use cropped region
            face_for_embedding = detection.aligned_face if detection.aligned_face is not None else face_region
            
            # Convert to RGB for embedding extraction
            if face_for_embedding.shape[2] == 3:  # Ensure it's a color image
                rgb_face = cv2.cvtColor(face_for_embedding, cv2.COLOR_BGR2RGB)
                
                # Extract embedding
                embedding = self.extract_embedding(rgb_face)
                if embedding is not None:
                    result['embedding'] = embedding
                    result['face_image'] = rgb_face
            
            results.append(result)
        
        return results
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two face embeddings
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Normalize embeddings
            emb1_norm = embedding1 / np.linalg.norm(embedding1)
            emb2_norm = embedding2 / np.linalg.norm(embedding2)
            
            # Compute cosine similarity
            similarity = np.dot(emb1_norm, emb2_norm)
            
            # Convert to 0-1 range
            similarity = (similarity + 1) / 2
            
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Similarity computation failed: {e}")
            return 0.0
    
    def is_same_person(self, embedding1: np.ndarray, embedding2: np.ndarray) -> bool:
        """Determine if two embeddings represent the same person
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            True if embeddings represent the same person
        """
        similarity = self.compute_similarity(embedding1, embedding2)
        return similarity >= self.config.embedding_threshold
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded models"""
        return {
            'model_type': self.config.model_type.value,
            'device': str(self.device),
            'detection_threshold': self.config.detection_threshold,
            'embedding_threshold': self.config.embedding_threshold,
            'min_face_size': self.config.min_face_size,
            'max_face_size': self.config.max_face_size,
            'align_faces': self.config.align_faces,
            'extract_multiple': self.config.extract_multiple
        }