#!/usr/bin/env python3
"""
YOLOv8 Detection Module

Provides model loading and inference wrapper with device selection.
Supports CUDA acceleration with TensorRT fallback and CPU fallback.
"""

import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available, CPU-only mode")

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logging.warning("Ultralytics not available, detection disabled")

logger = logging.getLogger(__name__)

class DetectionResult:
    """Container for detection results"""
    def __init__(self, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, 
                 class_names: List[str], inference_time: float):
        self.boxes = boxes  # [x1, y1, x2, y2] format
        self.scores = scores
        self.classes = classes
        self.class_names = class_names
        self.inference_time = inference_time
        
    def __len__(self):
        return len(self.boxes)
        
    def filter_by_confidence(self, min_confidence: float = 0.5) -> 'DetectionResult':
        """Filter detections by minimum confidence score"""
        mask = self.scores >= min_confidence
        return DetectionResult(
            boxes=self.boxes[mask],
            scores=self.scores[mask],
            classes=self.classes[mask],
            class_names=self.class_names,
            inference_time=self.inference_time
        )
        
    def filter_by_classes(self, target_classes: List[str]) -> 'DetectionResult':
        """Filter detections by target class names"""
        target_indices = [i for i, name in enumerate(self.class_names) if name in target_classes]
        mask = np.isin(self.classes, target_indices)
        return DetectionResult(
            boxes=self.boxes[mask],
            scores=self.scores[mask],
            classes=self.classes[mask],
            class_names=self.class_names,
            inference_time=self.inference_time
        )

class DeviceManager:
    """Manages device selection and optimization"""
    
    @staticmethod
    def get_best_device() -> str:
        """Determine the best available device for inference"""
        if not TORCH_AVAILABLE:
            logger.info("PyTorch not available, using CPU")
            return "cpu"
            
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"CUDA available with {device_count} device(s)")
            return "cuda:0"
        else:
            logger.info("CUDA not available, using CPU")
            return "cpu"
    
    @staticmethod
    def check_tensorrt_support() -> bool:
        """Check if TensorRT is available"""
        try:
            import tensorrt
            return True
        except ImportError:
            return False
    
    @staticmethod
    def optimize_model_for_device(model_path: str, device: str) -> str:
        """Optimize model for specific device (TensorRT, etc.)"""
        if device.startswith("cuda") and DeviceManager.check_tensorrt_support():
            # TensorRT optimization would go here
            logger.info("TensorRT optimization available but not implemented")
        return model_path

class YOLODetector:
    """YOLOv8 object detector optimized for human detection with aerial dataset support"""
    
    def __init__(self, model_path: str = "yolov8n.pt", device: Optional[str] = None, 
                 confidence_threshold: float = 0.25, iou_threshold: float = 0.45,
                 human_only: bool = True, aerial_optimized: bool = False):
        """
        Initialize YOLOv8 detector
        
        Args:
            model_path: Path to YOLOv8 model file
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            human_only: Filter detections to only include humans/persons
            aerial_optimized: Use optimizations for aerial/SAR imagery
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics YOLO not available. Install with: pip install ultralytics")
            
        self.model_path = Path(model_path)
        self.device = device or DeviceManager.get_best_device()
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.human_only = human_only
        self.aerial_optimized = aerial_optimized
        self.model = None
        self.class_names = []
        self.is_loaded = False
        
        # Human detection class indices (COCO dataset)
        self.human_class_indices = [0]  # 'person' class in COCO
        self.human_class_names = ['person']
        
        logger.info(f"Initializing YOLODetector with device: {self.device}")
        logger.info(f"Human-only mode: {human_only}, Aerial optimized: {aerial_optimized}")
        
    def load_model(self) -> bool:
        """Load the YOLOv8 model"""
        try:
            if not self.model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                return False
                
            logger.info(f"Loading YOLOv8 model from {self.model_path}")
            
            # Optimize model path for device
            optimized_path = DeviceManager.optimize_model_for_device(str(self.model_path), self.device)
            
            # Load model
            self.model = YOLO(optimized_path)
            
            # Move to device
            if TORCH_AVAILABLE and hasattr(self.model.model, 'to'):
                self.model.model.to(self.device)
                
            # Get class names
            self.class_names = list(self.model.names.values())
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully. Classes: {len(self.class_names)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def detect(self, image: np.ndarray) -> DetectionResult:
        """Run detection on image with human-specific optimizations"""
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError("Model not loaded and failed to load")
        
        start_time = time.time()
        
        try:
            # Apply aerial optimizations if enabled
            processed_image = self._preprocess_aerial(image) if self.aerial_optimized else image
            
            # Run inference
            results = self.model(
                processed_image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )
            
            inference_time = time.time() - start_time
            
            # Extract results
            if len(results) == 0 or len(results[0].boxes) == 0:
                class_names = self.human_class_names if self.human_only else self.class_names
                return DetectionResult(
                    boxes=np.empty((0, 4)),
                    scores=np.empty(0),
                    classes=np.empty(0, dtype=int),
                    class_names=class_names,
                    inference_time=inference_time
                )
            
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            # Filter for humans only if enabled
            if self.human_only:
                human_mask = np.isin(classes, self.human_class_indices)
                boxes = boxes[human_mask]
                scores = scores[human_mask]
                classes = classes[human_mask]
                
                # Apply aerial-specific post-processing
                if self.aerial_optimized:
                    boxes, scores, classes = self._postprocess_aerial(boxes, scores, classes, image.shape)
            
            # Use human class names if filtering enabled
            class_names = self.human_class_names if self.human_only else self.class_names
            
            return DetectionResult(
                boxes=boxes,
                scores=scores,
                classes=classes,
                class_names=class_names,
                inference_time=inference_time
            )
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            inference_time = time.time() - start_time
            class_names = self.human_class_names if self.human_only else self.class_names
            return DetectionResult(
                boxes=np.empty((0, 4)),
                scores=np.empty(0),
                classes=np.empty(0, dtype=int),
                class_names=class_names,
                inference_time=inference_time
            )
    
    def detect_batch(self, images: List[np.ndarray]) -> List[DetectionResult]:
        """Run detection on batch of images"""
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError("Model not loaded and failed to load")
        
        results = []
        for image in images:
            result = self.detect(image)
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_path": str(self.model_path),
            "device": self.device,
            "is_loaded": self.is_loaded,
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold
        }
    
    def set_thresholds(self, confidence: float = None, iou: float = None):
        """Update detection thresholds"""
        if confidence is not None:
            self.confidence_threshold = confidence
        if iou is not None:
            self.iou_threshold = iou
        logger.info(f"Updated thresholds - conf: {self.confidence_threshold}, iou: {self.iou_threshold}")
    
    def _preprocess_aerial(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing optimizations for aerial/SAR imagery
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed image optimized for aerial detection
        """
        # Convert to RGB for processing
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Enhance contrast for small objects
        lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # Apply slight sharpening for small object detection
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Blend original and sharpened (70% original, 30% sharpened)
        result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
        
        # Convert back to BGR for YOLO
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    def _postprocess_aerial(self, boxes: np.ndarray, scores: np.ndarray, 
                           classes: np.ndarray, image_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply post-processing optimizations for aerial detections
        
        Args:
            boxes: Detection boxes [x1, y1, x2, y2]
            scores: Detection confidence scores
            classes: Detection class indices
            image_shape: Original image shape (H, W, C)
            
        Returns:
            Filtered boxes, scores, and classes
        """
        if len(boxes) == 0:
            return boxes, scores, classes
        
        # Calculate box areas and aspect ratios
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights
        aspect_ratios = widths / (heights + 1e-6)
        
        # Filter by minimum size (humans should be at least 10x10 pixels in aerial view)
        min_area = 100  # pixels
        area_mask = areas >= min_area
        
        # Filter by aspect ratio (humans typically 0.3 to 3.0 ratio)
        aspect_mask = (aspect_ratios >= 0.2) & (aspect_ratios <= 5.0)
        
        # Combine filters
        valid_mask = area_mask & aspect_mask
        
        # Apply confidence boost for detections in typical human size range
        image_area = image_shape[0] * image_shape[1]
        relative_areas = areas / image_area
        
        # Boost confidence for small objects (typical for aerial view)
        small_object_mask = (relative_areas < 0.01) & (relative_areas > 0.0001)
        scores[small_object_mask] = np.minimum(scores[small_object_mask] * 1.1, 1.0)
        
        return boxes[valid_mask], scores[valid_mask], classes[valid_mask]
    
    def fine_tune_aerial(self, dataset_path: str, epochs: int = 50, batch_size: int = 16) -> bool:
        """
        Fine-tune model on aerial/SAR dataset
        
        Args:
            dataset_path: Path to aerial dataset (YOLO format)
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            True if fine-tuning successful, False otherwise
        """
        if not self.is_loaded:
            logger.error("Model must be loaded before fine-tuning")
            return False
        
        try:
            logger.info(f"Starting fine-tuning on aerial dataset: {dataset_path}")
            
            # Configure training parameters for aerial imagery
            train_args = {
                'data': dataset_path,
                'epochs': epochs,
                'batch': batch_size,
                'imgsz': 640,
                'device': self.device,
                'project': 'aerial_training',
                'name': 'human_detection_aerial',
                'exist_ok': True,
                'patience': 10,
                'save_period': 5,
                'cache': True,
                'workers': 4,
                'optimizer': 'AdamW',
                'lr0': 0.001,
                'weight_decay': 0.0005,
                'mosaic': 0.5,  # Reduced mosaic for aerial imagery
                'mixup': 0.1,
                'copy_paste': 0.1,
                'degrees': 5.0,  # Reduced rotation for aerial
                'translate': 0.1,
                'scale': 0.2,
                'shear': 2.0,
                'perspective': 0.0,  # No perspective for aerial
                'flipud': 0.5,  # Allow vertical flip for aerial
                'fliplr': 0.5,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4
            }
            
            # Start training
            results = self.model.train(**train_args)
            
            # Load the best weights
            best_model_path = results.save_dir / 'weights' / 'best.pt'
            if best_model_path.exists():
                self.model = YOLO(str(best_model_path))
                logger.info(f"Fine-tuning completed. Best model loaded from {best_model_path}")
                return True
            else:
                logger.error("Fine-tuning completed but best model not found")
                return False
                
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            return False

# Legacy compatibility
class DummyDetector:
    """Legacy detector for backward compatibility"""
    def __init__(self, min_area=500):
        self.min_area = min_area
        logger.warning("DummyDetector is deprecated. Use YOLODetector instead.")

    def detect(self, image):
        # image: BGR
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, th = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area: continue
            x,y,w,h = cv2.boundingRect(c)
            dets.append({"bbox":[int(x),int(y),int(w),int(h)], "score": 0.5, "class":"person"})
        return dets

def create_detector(model_path: str = "yolov8n.pt", **kwargs) -> YOLODetector:
    """Factory function to create detector instance"""
    return YOLODetector(model_path=model_path, **kwargs)

def demo_detection():
    """Demo function for testing detection"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv8 Detection Demo")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLOv8 model")
    parser.add_argument("--device", help="Device to use (cuda/cpu)")
    parser.add_argument("--image", help="Path to test image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create detector
    detector = YOLODetector(
        model_path=args.model,
        device=args.device,
        confidence_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Load model
    if not detector.load_model():
        logger.error("Failed to load model")
        return
    
    # Print model info
    info = detector.get_model_info()
    logger.info(f"Model info: {info}")
    
    # Test with image if provided
    if args.image:
        image_path = Path(args.image)
        if image_path.exists():
            image = cv2.imread(str(image_path))
            if image is not None:
                logger.info(f"Running detection on {image_path}")
                result = detector.detect(image)
                logger.info(f"Detected {len(result)} objects in {result.inference_time:.3f}s")
                
                for i, (box, score, cls) in enumerate(zip(result.boxes, result.scores, result.classes)):
                    class_name = result.class_names[cls]
                    logger.info(f"  {i+1}: {class_name} ({score:.3f}) at {box}")
            else:
                logger.error(f"Failed to load image: {image_path}")
        else:
            logger.error(f"Image not found: {image_path}")
    else:
        # Create dummy image for testing
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        logger.info("Running detection on dummy image")
        result = detector.detect(dummy_image)
        logger.info(f"Detected {len(result)} objects in {result.inference_time:.3f}s")

if __name__ == "__main__":
    demo_detection()
