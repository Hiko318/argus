"""Unit tests for vision module.

Tests for object detection, image processing, and computer vision utilities.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import cv2

# Import vision modules (with error handling for missing dependencies)
try:
    from src.vision.detector import ObjectDetector, DetectionResult
    from src.vision.image_processor import ImageProcessor
    from src.vision.utils import (
        resize_image, normalize_bbox, calculate_iou,
        draw_detections, apply_nms
    )
except ImportError as e:
    pytest.skip(f"Vision modules not available: {e}", allow_module_level=True)


class TestObjectDetector:
    """Test cases for ObjectDetector class."""
    
    @pytest.fixture
    def mock_model(self):
        """Mock YOLO model for testing."""
        model = Mock()
        model.predict = Mock()
        model.names = {0: "person", 1: "car", 2: "bicycle"}
        return model
    
    @pytest.fixture
    def detector(self, mock_model):
        """Create ObjectDetector instance with mocked model."""
        with patch('src.vision.detector.YOLO', return_value=mock_model):
            detector = ObjectDetector(model_path="mock_model.pt")
            detector.model = mock_model
            return detector
    
    def test_detector_initialization(self, detector):
        """Test detector initialization."""
        assert detector.model is not None
        assert detector.confidence_threshold == 0.5
        assert detector.iou_threshold == 0.45
        assert detector.is_loaded is True
    
    def test_detector_with_custom_thresholds(self, mock_model):
        """Test detector with custom confidence and IoU thresholds."""
        with patch('src.vision.detector.YOLO', return_value=mock_model):
            detector = ObjectDetector(
                model_path="mock_model.pt",
                confidence_threshold=0.7,
                iou_threshold=0.3
            )
            assert detector.confidence_threshold == 0.7
            assert detector.iou_threshold == 0.3
    
    def test_detect_objects_success(self, detector, sample_image):
        """Test successful object detection."""
        # Mock detection results
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.xyxy = np.array([[100, 100, 200, 200, 0.85, 0]])
        mock_result.boxes.conf = np.array([0.85])
        mock_result.boxes.cls = np.array([0])
        
        detector.model.predict.return_value = [mock_result]
        
        detections = detector.detect(sample_image)
        
        assert len(detections) == 1
        assert isinstance(detections[0], DetectionResult)
        assert detections[0].bbox == [100, 100, 200, 200]
        assert detections[0].confidence == 0.85
        assert detections[0].class_id == 0
        assert detections[0].class_name == "person"
    
    def test_detect_objects_no_detections(self, detector, sample_image):
        """Test detection when no objects are found."""
        mock_result = Mock()
        mock_result.boxes = None
        
        detector.model.predict.return_value = [mock_result]
        
        detections = detector.detect(sample_image)
        
        assert len(detections) == 0
    
    def test_detect_objects_confidence_filtering(self, detector, sample_image):
        """Test that low-confidence detections are filtered out."""
        # Mock detection results with low confidence
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.xyxy = np.array([
            [100, 100, 200, 200, 0.3, 0],  # Low confidence
            [300, 300, 400, 400, 0.8, 1]   # High confidence
        ])
        mock_result.boxes.conf = np.array([0.3, 0.8])
        mock_result.boxes.cls = np.array([0, 1])
        
        detector.model.predict.return_value = [mock_result]
        
        detections = detector.detect(sample_image)
        
        # Only high-confidence detection should remain
        assert len(detections) == 1
        assert detections[0].confidence == 0.8
        assert detections[0].class_id == 1
    
    def test_detect_batch(self, detector):
        """Test batch detection on multiple images."""
        images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(3)]
        
        # Mock batch results
        mock_results = []
        for i in range(3):
            mock_result = Mock()
            mock_result.boxes = Mock()
            mock_result.boxes.xyxy = np.array([[100+i*10, 100+i*10, 200+i*10, 200+i*10, 0.8, 0]])
            mock_result.boxes.conf = np.array([0.8])
            mock_result.boxes.cls = np.array([0])
            mock_results.append(mock_result)
        
        detector.model.predict.return_value = mock_results
        
        batch_detections = detector.detect_batch(images)
        
        assert len(batch_detections) == 3
        for i, detections in enumerate(batch_detections):
            assert len(detections) == 1
            assert detections[0].bbox[0] == 100 + i * 10
    
    @pytest.mark.performance
    def test_detection_performance(self, detector, sample_image, performance_monitor):
        """Test detection performance."""
        # Mock fast detection
        mock_result = Mock()
        mock_result.boxes = None
        detector.model.predict.return_value = [mock_result]
        
        with performance_monitor() as monitor:
            for _ in range(10):
                detector.detect(sample_image)
        
        # Should complete 10 detections in reasonable time
        assert monitor.duration < 1.0  # Less than 1 second for 10 detections
    
    def test_model_loading_error(self):
        """Test handling of model loading errors."""
        with patch('src.vision.detector.YOLO', side_effect=Exception("Model not found")):
            with pytest.raises(Exception, match="Model not found"):
                ObjectDetector(model_path="nonexistent_model.pt")


class TestImageProcessor:
    """Test cases for ImageProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create ImageProcessor instance."""
        return ImageProcessor()
    
    def test_resize_image(self, processor, sample_image):
        """Test image resizing."""
        target_size = (320, 240)
        resized = processor.resize(sample_image, target_size)
        
        assert resized.shape[:2] == target_size[::-1]  # Height, Width
        assert resized.dtype == sample_image.dtype
    
    def test_normalize_image(self, processor, sample_image):
        """Test image normalization."""
        normalized = processor.normalize(sample_image)
        
        assert normalized.dtype == np.float32
        assert 0.0 <= normalized.min() <= 1.0
        assert 0.0 <= normalized.max() <= 1.0
    
    def test_apply_blur(self, processor, sample_image):
        """Test image blurring."""
        blurred = processor.apply_blur(sample_image, kernel_size=5)
        
        assert blurred.shape == sample_image.shape
        assert blurred.dtype == sample_image.dtype
        # Blurred image should be different from original
        assert not np.array_equal(blurred, sample_image)
    
    def test_enhance_contrast(self, processor, sample_image):
        """Test contrast enhancement."""
        enhanced = processor.enhance_contrast(sample_image, alpha=1.5)
        
        assert enhanced.shape == sample_image.shape
        assert enhanced.dtype == sample_image.dtype
    
    def test_crop_region(self, processor, sample_image):
        """Test region cropping."""
        bbox = [100, 100, 300, 200]  # x1, y1, x2, y2
        cropped = processor.crop_region(sample_image, bbox)
        
        expected_height = bbox[3] - bbox[1]  # y2 - y1
        expected_width = bbox[2] - bbox[0]   # x2 - x1
        
        assert cropped.shape[:2] == (expected_height, expected_width)
    
    def test_apply_privacy_filter(self, processor, sample_image):
        """Test privacy filter application."""
        # Mock face detection
        faces = [[150, 150, 250, 250]]  # Mock detected face
        
        with patch.object(processor, 'detect_faces', return_value=faces):
            filtered = processor.apply_privacy_filter(sample_image)
        
        assert filtered.shape == sample_image.shape
        # Face region should be blurred/masked
        face_region = filtered[150:250, 150:250]
        original_face = sample_image[150:250, 150:250]
        assert not np.array_equal(face_region, original_face)
    
    def test_batch_processing(self, processor):
        """Test batch image processing."""
        images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(5)]
        
        processed = processor.process_batch(images, operations=['resize', 'normalize'])
        
        assert len(processed) == 5
        for img in processed:
            assert img.dtype == np.float32
            assert 0.0 <= img.min() <= 1.0
            assert 0.0 <= img.max() <= 1.0


class TestVisionUtils:
    """Test cases for vision utility functions."""
    
    def test_resize_image_function(self, sample_image):
        """Test standalone resize function."""
        target_size = (320, 240)
        resized = resize_image(sample_image, target_size)
        
        assert resized.shape[:2] == target_size[::-1]
    
    def test_normalize_bbox(self):
        """Test bounding box normalization."""
        bbox = [100, 100, 200, 200]
        image_size = (640, 480)
        
        normalized = normalize_bbox(bbox, image_size)
        
        expected = [100/640, 100/480, 200/640, 200/480]
        np.testing.assert_array_almost_equal(normalized, expected)
    
    def test_calculate_iou(self):
        """Test IoU calculation."""
        bbox1 = [100, 100, 200, 200]
        bbox2 = [150, 150, 250, 250]
        
        iou = calculate_iou(bbox1, bbox2)
        
        # Calculate expected IoU
        intersection_area = 50 * 50  # 2500
        union_area = (100 * 100) + (100 * 100) - intersection_area  # 17500
        expected_iou = intersection_area / union_area
        
        assert abs(iou - expected_iou) < 1e-6
    
    def test_calculate_iou_no_overlap(self):
        """Test IoU calculation with no overlap."""
        bbox1 = [100, 100, 200, 200]
        bbox2 = [300, 300, 400, 400]
        
        iou = calculate_iou(bbox1, bbox2)
        
        assert iou == 0.0
    
    def test_calculate_iou_perfect_overlap(self):
        """Test IoU calculation with perfect overlap."""
        bbox1 = [100, 100, 200, 200]
        bbox2 = [100, 100, 200, 200]
        
        iou = calculate_iou(bbox1, bbox2)
        
        assert iou == 1.0
    
    def test_apply_nms(self):
        """Test Non-Maximum Suppression."""
        detections = [
            DetectionResult([100, 100, 200, 200], 0.9, 0, "person"),
            DetectionResult([110, 110, 210, 210], 0.8, 0, "person"),  # Overlapping
            DetectionResult([300, 300, 400, 400], 0.7, 1, "car")       # Different location
        ]
        
        filtered = apply_nms(detections, iou_threshold=0.5)
        
        # Should keep highest confidence detection and non-overlapping detection
        assert len(filtered) == 2
        assert filtered[0].confidence == 0.9
        assert filtered[1].confidence == 0.7
    
    def test_draw_detections(self, sample_image):
        """Test drawing detections on image."""
        detections = [
            DetectionResult([100, 100, 200, 200], 0.9, 0, "person"),
            DetectionResult([300, 300, 400, 400], 0.8, 1, "car")
        ]
        
        annotated = draw_detections(sample_image.copy(), detections)
        
        assert annotated.shape == sample_image.shape
        # Image should be modified (annotations added)
        assert not np.array_equal(annotated, sample_image)
    
    @pytest.mark.parametrize("bbox,image_size,expected", [
        ([0, 0, 100, 100], (640, 480), [0.0, 0.0, 100/640, 100/480]),
        ([320, 240, 640, 480], (640, 480), [0.5, 0.5, 1.0, 1.0]),
        ([50, 50, 150, 150], (200, 200), [0.25, 0.25, 0.75, 0.75])
    ])
    def test_normalize_bbox_parametrized(self, bbox, image_size, expected):
        """Test bbox normalization with various inputs."""
        result = normalize_bbox(bbox, image_size)
        np.testing.assert_array_almost_equal(result, expected)


class TestDetectionResult:
    """Test cases for DetectionResult class."""
    
    def test_detection_result_creation(self):
        """Test DetectionResult creation."""
        bbox = [100, 100, 200, 200]
        confidence = 0.85
        class_id = 0
        class_name = "person"
        
        detection = DetectionResult(bbox, confidence, class_id, class_name)
        
        assert detection.bbox == bbox
        assert detection.confidence == confidence
        assert detection.class_id == class_id
        assert detection.class_name == class_name
    
    def test_detection_result_properties(self):
        """Test DetectionResult computed properties."""
        detection = DetectionResult([100, 100, 200, 200], 0.85, 0, "person")
        
        assert detection.width == 100
        assert detection.height == 100
        assert detection.area == 10000
        assert detection.center == (150, 150)
    
    def test_detection_result_to_dict(self):
        """Test DetectionResult serialization."""
        detection = DetectionResult([100, 100, 200, 200], 0.85, 0, "person")
        
        result_dict = detection.to_dict()
        
        expected = {
            "bbox": [100, 100, 200, 200],
            "confidence": 0.85,
            "class_id": 0,
            "class_name": "person"
        }
        
        assert result_dict == expected
    
    def test_detection_result_from_dict(self):
        """Test DetectionResult deserialization."""
        data = {
            "bbox": [100, 100, 200, 200],
            "confidence": 0.85,
            "class_id": 0,
            "class_name": "person"
        }
        
        detection = DetectionResult.from_dict(data)
        
        assert detection.bbox == [100, 100, 200, 200]
        assert detection.confidence == 0.85
        assert detection.class_id == 0
        assert detection.class_name == "person"


@pytest.mark.integration
class TestVisionIntegration:
    """Integration tests for vision components."""
    
    def test_detector_processor_pipeline(self, sample_image):
        """Test complete detection and processing pipeline."""
        # Mock components
        with patch('src.vision.detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_model.names = {0: "person"}
            mock_yolo.return_value = mock_model
            
            # Mock detection result
            mock_result = Mock()
            mock_result.boxes = Mock()
            mock_result.boxes.xyxy = np.array([[100, 100, 200, 200, 0.85, 0]])
            mock_result.boxes.conf = np.array([0.85])
            mock_result.boxes.cls = np.array([0])
            mock_model.predict.return_value = [mock_result]
            
            # Create pipeline
            detector = ObjectDetector("mock_model.pt")
            processor = ImageProcessor()
            
            # Process image
            processed_image = processor.resize(sample_image, (640, 480))
            detections = detector.detect(processed_image)
            annotated_image = draw_detections(processed_image.copy(), detections)
            
            assert len(detections) == 1
            assert detections[0].class_name == "person"
            assert annotated_image.shape == processed_image.shape
    
    @pytest.mark.slow
    def test_batch_processing_pipeline(self):
        """Test batch processing pipeline."""
        # Create batch of test images
        images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)]
        
        with patch('src.vision.detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_model.names = {0: "person"}
            mock_yolo.return_value = mock_model
            
            # Mock batch results
            mock_results = []
            for _ in range(10):
                mock_result = Mock()
                mock_result.boxes = None  # No detections
                mock_results.append(mock_result)
            mock_model.predict.return_value = mock_results
            
            # Process batch
            detector = ObjectDetector("mock_model.pt")
            processor = ImageProcessor()
            
            processed_images = processor.process_batch(images, ['resize', 'normalize'])
            batch_detections = detector.detect_batch(processed_images)
            
            assert len(batch_detections) == 10
            for detections in batch_detections:
                assert isinstance(detections, list)


@pytest.mark.gpu
class TestGPUVision:
    """GPU-specific vision tests."""
    
    def test_gpu_detection(self, sample_image):
        """Test detection with GPU acceleration."""
        with patch('src.vision.detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_model.to = Mock(return_value=mock_model)
            mock_yolo.return_value = mock_model
            
            detector = ObjectDetector("mock_model.pt", device="cuda")
            
            # Verify model was moved to GPU
            mock_model.to.assert_called_with("cuda")
    
    def test_gpu_memory_management(self, sample_image):
        """Test GPU memory management during detection."""
        with patch('torch.cuda.empty_cache') as mock_empty_cache:
            with patch('src.vision.detector.YOLO') as mock_yolo:
                mock_model = Mock()
                mock_yolo.return_value = mock_model
                
                detector = ObjectDetector("mock_model.pt", device="cuda")
                detector.detect(sample_image)
                
                # Verify cache was cleared
                mock_empty_cache.assert_called()