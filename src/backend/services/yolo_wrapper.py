#!/usr/bin/env python3
"""
YOLO Detection Wrapper for FPV Feeds

Provides a unified interface for YOLO object detection with support for:
- Multiple YOLO model versions (YOLOv5, YOLOv8, YOLOv10)
- Custom fine-tuned models
- Real-time detection optimization
- Confidence and NMS filtering
- Custom class filtering for SAR operations

Author: Foresight AI Team
Date: 2024
"""

import cv2
import numpy as np
import torch
import logging
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import time
import threading
from enum import Enum

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Supported YOLO model types"""
    YOLOV5 = "yolov5"
    YOLOV8 = "yolov8"
    YOLOV10 = "yolov10"
    YOLO11 = "yolo11"
    CUSTOM = "custom"

@dataclass
class Detection:
    """Single object detection result"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None

@dataclass
class YOLOConfig:
    """YOLO model configuration"""
    model_path: str
    model_type: ModelType = ModelType.YOLOV8
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    max_detections: int = 100
    input_size: Tuple[int, int] = (640, 640)
    device: str = "auto"  # auto, cpu, cuda, mps
    half_precision: bool = True
    target_classes: Optional[List[str]] = None  # Filter for specific classes
    sar_mode: bool = True  # Optimize for SAR operations

class YOLOWrapper:
    """
    Unified YOLO Detection Wrapper
    
    Provides a consistent interface for different YOLO models with optimizations
    for real-time FPV detection in search and rescue operations.
    """
    
    def __init__(self, config: YOLOConfig):
        self.config = config
        self.model = None
        self.device = None
        self.class_names = []
        self.is_loaded = False
        self.inference_lock = threading.Lock()
        
        # Performance tracking
        self.stats = {
            'total_inferences': 0,
            'avg_inference_time_ms': 0.0,
            'last_inference_time_ms': 0.0,
            'detections_per_frame': 0.0,
            'model_loaded': False
        }
        
        # SAR-specific class mapping
        self.sar_classes = {
            'person', 'human', 'people', 'man', 'woman', 'child',
            'backpack', 'handbag', 'suitcase',
            'bicycle', 'motorcycle', 'car', 'truck', 'boat',
            'tent', 'umbrella'
        }
        
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load the YOLO model"""
        try:
            logger.info(f"Loading YOLO model: {self.config.model_path}")
            
            # Determine device
            self.device = self._get_device()
            logger.info(f"Using device: {self.device}")
            
            # Load model based on type
            if self.config.model_type == ModelType.YOLOV5:
                self._load_yolov5()
            elif self.config.model_type == ModelType.YOLOV8:
                self._load_yolov8()
            elif self.config.model_type == ModelType.YOLOV10:
                self._load_yolov10()
            elif self.config.model_type == ModelType.YOLO11:
                self._load_yolo11()
            elif self.config.model_type == ModelType.CUSTOM:
                self._load_custom()
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
            # Warm up model
            self._warmup_model()
            
            self.is_loaded = True
            self.stats['model_loaded'] = True
            logger.info("YOLO model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return False
    
    def _get_device(self) -> str:
        """Determine the best available device"""
        if self.config.device != "auto":
            return self.config.device
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_yolov5(self):
        """Load YOLOv5 model"""
        try:
            import ultralytics
            # For YOLOv5, we'll use the ultralytics package
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                      path=self.config.model_path, 
                                      device=self.device)
            self.class_names = self.model.names
        except Exception as e:
            # Fallback to local YOLOv5
            logger.warning(f"Failed to load via torch.hub, trying local: {e}")
            self._load_custom()
    
    def _load_yolov8(self):
        """Load YOLOv8 model"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.config.model_path)
            self.model.to(self.device)
            
            # Get class names
            if hasattr(self.model.model, 'names'):
                self.class_names = self.model.model.names
            else:
                self.class_names = self.model.names
                
        except ImportError:
            logger.error("ultralytics package not found. Install with: pip install ultralytics")
            raise
    
    def _load_yolov10(self):
        """Load YOLOv10 model"""
        # YOLOv10 typically uses the same interface as YOLOv8
        self._load_yolov8()
    
    def _load_yolo11(self):
        """Load YOLO11 model"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.config.model_path)
            self.model.to(self.device)
            
            # Get class names
            if hasattr(self.model.model, 'names'):
                self.class_names = self.model.model.names
            else:
                self.class_names = self.model.names
                
        except ImportError:
            logger.error("ultralytics package not found. Install with: pip install ultralytics")
            raise
    
    def _load_custom(self):
        """Load custom model (PyTorch format)"""
        try:
            # Load PyTorch model directly
            self.model = torch.load(self.config.model_path, map_location=self.device)
            self.model.eval()
            
            # Try to extract class names
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
            elif hasattr(self.model, 'module') and hasattr(self.model.module, 'names'):
                self.class_names = self.model.module.names
            else:
                # Default COCO classes
                self.class_names = self._get_coco_classes()
                
        except Exception as e:
            logger.error(f"Failed to load custom model: {e}")
            raise
    
    def _get_coco_classes(self) -> List[str]:
        """Get COCO dataset class names"""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    def _warmup_model(self):
        """Warm up the model with dummy input"""
        try:
            dummy_input = np.random.randint(0, 255, 
                                          (self.config.input_size[1], self.config.input_size[0], 3), 
                                          dtype=np.uint8)
            self.detect(dummy_input)
            logger.info("Model warmed up successfully")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Perform object detection on a frame"""
        if not self.is_loaded:
            logger.warning("Model not loaded")
            return []
        
        with self.inference_lock:
            start_time = time.time()
            
            try:
                # Preprocess frame
                processed_frame = self._preprocess_frame(frame)
                
                # Run inference
                if self.config.model_type in [ModelType.YOLOV8, ModelType.YOLOV10]:
                    results = self._detect_ultralytics(processed_frame)
                else:
                    results = self._detect_torch(processed_frame)
                
                # Post-process results
                detections = self._postprocess_results(results, frame.shape)
                
                # Filter for SAR-relevant classes if enabled
                if self.config.sar_mode:
                    detections = self._filter_sar_classes(detections)
                
                # Update statistics
                inference_time = (time.time() - start_time) * 1000
                self._update_stats(inference_time, len(detections))
                
                return detections
                
            except Exception as e:
                logger.error(f"Detection failed: {e}")
                return []
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for YOLO input"""
        # Resize to model input size
        resized = cv2.resize(frame, self.config.input_size)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Add batch dimension and transpose to CHW format
        tensor = np.transpose(normalized, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        
        return tensor
    
    def _detect_ultralytics(self, frame: np.ndarray) -> List:
        """Run detection using ultralytics models"""
        # Convert numpy to tensor
        tensor = torch.from_numpy(frame).to(self.device)
        
        if self.config.half_precision and self.device != "cpu":
            tensor = tensor.half()
        
        # Run inference
        with torch.no_grad():
            results = self.model(tensor, 
                               conf=self.config.confidence_threshold,
                               iou=self.config.nms_threshold,
                               max_det=self.config.max_detections)
        
        return results
    
    def _detect_torch(self, frame: np.ndarray) -> torch.Tensor:
        """Run detection using raw PyTorch models"""
        tensor = torch.from_numpy(frame).to(self.device)
        
        if self.config.half_precision and self.device != "cpu":
            tensor = tensor.half()
        
        with torch.no_grad():
            results = self.model(tensor)
        
        return results
    
    def _postprocess_results(self, results, original_shape: Tuple[int, int, int]) -> List[Detection]:
        """Post-process detection results"""
        detections = []
        
        try:
            if self.config.model_type in [ModelType.YOLOV8, ModelType.YOLOV10]:
                # Ultralytics format
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for i in range(len(boxes)):
                            # Get box coordinates (xyxy format)
                            box = boxes.xyxy[i].cpu().numpy()
                            conf = float(boxes.conf[i].cpu().numpy())
                            cls_id = int(boxes.cls[i].cpu().numpy())
                            
                            # Scale coordinates to original image size
                            x1, y1, x2, y2 = self._scale_coordinates(box, original_shape)
                            
                            # Get class name
                            class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
                            
                            detection = Detection(
                                bbox=(int(x1), int(y1), int(x2), int(y2)),
                                confidence=conf,
                                class_id=cls_id,
                                class_name=class_name
                            )
                            detections.append(detection)
            
            else:
                # Raw PyTorch format (needs custom parsing based on model)
                # This is a simplified version - adjust based on your model's output format
                if isinstance(results, torch.Tensor):
                    results = results.cpu().numpy()
                
                # Assuming results shape: [batch, detections, 6] where 6 = [x1, y1, x2, y2, conf, class]
                if len(results.shape) == 3:
                    for detection in results[0]:  # First batch
                        if len(detection) >= 6:
                            x1, y1, x2, y2, conf, cls_id = detection[:6]
                            
                            if conf >= self.config.confidence_threshold:
                                # Scale coordinates
                                x1, y1, x2, y2 = self._scale_coordinates([x1, y1, x2, y2], original_shape)
                                
                                class_name = self.class_names[int(cls_id)] if int(cls_id) < len(self.class_names) else f"class_{int(cls_id)}"
                                
                                detection_obj = Detection(
                                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                                    confidence=float(conf),
                                    class_id=int(cls_id),
                                    class_name=class_name
                                )
                                detections.append(detection_obj)
        
        except Exception as e:
            logger.error(f"Error post-processing results: {e}")
        
        return detections
    
    def _scale_coordinates(self, box: List[float], original_shape: Tuple[int, int, int]) -> Tuple[float, float, float, float]:
        """Scale coordinates from model input size to original image size"""
        orig_h, orig_w = original_shape[:2]
        model_h, model_w = self.config.input_size
        
        # Calculate scale factors
        scale_x = orig_w / model_w
        scale_y = orig_h / model_h
        
        # Scale coordinates
        x1 = box[0] * scale_x
        y1 = box[1] * scale_y
        x2 = box[2] * scale_x
        y2 = box[3] * scale_y
        
        # Clamp to image boundaries
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))
        
        return x1, y1, x2, y2
    
    def _filter_sar_classes(self, detections: List[Detection]) -> List[Detection]:
        """Filter detections for SAR-relevant classes"""
        if not self.config.sar_mode:
            return detections
        
        filtered = []
        for detection in detections:
            class_name_lower = detection.class_name.lower()
            if any(sar_class in class_name_lower for sar_class in self.sar_classes):
                filtered.append(detection)
        
        return filtered
    
    def _update_stats(self, inference_time_ms: float, num_detections: int):
        """Update performance statistics"""
        self.stats['total_inferences'] += 1
        self.stats['last_inference_time_ms'] = inference_time_ms
        
        # Update running average
        total = self.stats['total_inferences']
        current_avg = self.stats['avg_inference_time_ms']
        self.stats['avg_inference_time_ms'] = (current_avg * (total - 1) + inference_time_ms) / total
        
        # Update detections per frame
        current_det_avg = self.stats['detections_per_frame']
        self.stats['detections_per_frame'] = (current_det_avg * (total - 1) + num_detections) / total
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        return self.stats.copy()
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection], 
                       draw_confidence: bool = True, draw_class: bool = True) -> np.ndarray:
        """Draw detection bounding boxes on frame"""
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Choose color based on class (person = red, others = green)
            color = (0, 0, 255) if 'person' in detection.class_name.lower() else (0, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label_parts = []
            if draw_class:
                label_parts.append(detection.class_name)
            if draw_confidence:
                label_parts.append(f"{detection.confidence:.2f}")
            
            if label_parts:
                label = " ".join(label_parts)
                
                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Draw label background
                cv2.rectangle(annotated_frame, 
                            (x1, y1 - text_height - baseline - 5), 
                            (x1 + text_width, y1), 
                            color, -1)
                
                # Draw label text
                cv2.putText(annotated_frame, label, (x1, y1 - baseline - 2), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_frame
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self.is_loaded
    
    def get_class_names(self) -> List[str]:
        """Get list of class names"""
        return self.class_names.copy()

def create_yolo_wrapper(model_path: str, model_type: str = "yolov8", **kwargs) -> YOLOWrapper:
    """Factory function to create YOLO wrapper"""
    config = YOLOConfig(
        model_path=model_path,
        model_type=ModelType(model_type),
        **kwargs
    )
    return YOLOWrapper(config)

if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    # Example with YOLOv8
    try:
        yolo = create_yolo_wrapper("yolo11n.pt", "yolo11")
        
        if yolo.is_model_loaded():
            print("YOLO model loaded successfully")
            print(f"Classes: {yolo.get_class_names()[:10]}...")  # First 10 classes
            
            # Test with dummy image
            dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            detections = yolo.detect(dummy_frame)
            
            print(f"Detected {len(detections)} objects")
            print(f"Stats: {yolo.get_stats()}")
        else:
            print("Failed to load YOLO model")
            
    except Exception as e:
        print(f"Error: {e}")