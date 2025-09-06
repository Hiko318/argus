"""Unit tests for the YOLO detector module."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import the modules to test
try:
    from src.backend.detector import YOLODetector
    from vision.detector import ObjectDetector
except ImportError:
    # Handle case where modules might not be importable in test environment
    YOLODetector = None
    ObjectDetector = None


@pytest.mark.unit
class TestYOLODetector:
    """Test cases for YOLO detector functionality."""
    
    @pytest.fixture
    def mock_ultralytics_model(self):
        """Mock ultralytics YOLO model."""
        mock_model = Mock()
        mock_model.predict.return_value = [Mock(
            boxes=Mock(
                xyxy=np.array([[100, 100, 200, 200], [300, 150, 400, 250]]),
                conf=np.array([0.85, 0.92]),
                cls=np.array([0, 1])
            ),
            names={0: "person", 1: "vehicle"}
        )]
        return mock_model
    
    @pytest.fixture
    def detector_config(self):
        """Detector configuration for testing."""
        return {
            "model_path": "models/yolov8n.pt",
            "confidence_threshold": 0.5,
            "nms_threshold": 0.4,
            "device": "cpu",
            "classes": ["person", "vehicle", "bicycle"]
        }
    
    @patch('ultralytics.YOLO')
    def test_detector_initialization(self, mock_yolo, detector_config):
        """Test detector initialization with valid config."""
        if YOLODetector is None:
            pytest.skip("YOLODetector not available")
        
        mock_yolo.return_value = Mock()
        detector = YOLODetector(detector_config)
        
        assert detector.confidence_threshold == 0.5
        assert detector.nms_threshold == 0.4
        assert detector.device == "cpu"
        mock_yolo.assert_called_once_with(detector_config["model_path"])
    
    def test_detector_initialization_invalid_model(self, detector_config):
        """Test detector initialization with invalid model path."""
        if YOLODetector is None:
            pytest.skip("YOLODetector not available")
        
        detector_config["model_path"] = "nonexistent/model.pt"
        
        with patch('ultralytics.YOLO') as mock_yolo:
            mock_yolo.side_effect = FileNotFoundError("Model file not found")
            
            with pytest.raises(FileNotFoundError):
                YOLODetector(detector_config)
    
    @patch('ultralytics.YOLO')
    def test_detect_objects_success(self, mock_yolo, detector_config, sample_image, mock_ultralytics_model):
        """Test successful object detection."""
        if YOLODetector is None:
            pytest.skip("YOLODetector not available")
        
        mock_yolo.return_value = mock_ultralytics_model
        detector = YOLODetector(detector_config)
        
        result = detector.detect(sample_image)
        
        assert "boxes" in result
        assert "frame_id" in result
        assert "processing_time" in result
        assert len(result["boxes"]) == 2
        
        # Check first detection
        first_box = result["boxes"][0]
        assert first_box["confidence"] == 0.85
        assert first_box["class"] == "person"
        assert first_box["x1"] == 100
        assert first_box["y1"] == 100
        assert first_box["x2"] == 200
        assert first_box["y2"] == 200
    
    @patch('ultralytics.YOLO')
    def test_detect_empty_image(self, mock_yolo, detector_config):
        """Test detection on empty/black image."""
        if YOLODetector is None:
            pytest.skip("YOLODetector not available")
        
        # Mock model that returns no detections
        mock_model = Mock()
        mock_model.predict.return_value = [Mock(
            boxes=Mock(
                xyxy=np.array([]).reshape(0, 4),
                conf=np.array([]),
                cls=np.array([])
            ),
            names={0: "person", 1: "vehicle"}
        )]
        mock_yolo.return_value = mock_model
        
        detector = YOLODetector(detector_config)
        empty_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = detector.detect(empty_image)
        
        assert "boxes" in result
        assert len(result["boxes"]) == 0
        assert result["processing_time"] > 0
    
    @patch('ultralytics.YOLO')
    def test_detect_invalid_image(self, mock_yolo, detector_config):
        """Test detection with invalid image input."""
        if YOLODetector is None:
            pytest.skip("YOLODetector not available")
        
        mock_yolo.return_value = Mock()
        detector = YOLODetector(detector_config)
        
        # Test with None image
        with pytest.raises((ValueError, TypeError)):
            detector.detect(None)
        
        # Test with wrong image shape
        invalid_image = np.zeros((100,), dtype=np.uint8)  # 1D array
        with pytest.raises((ValueError, TypeError)):
            detector.detect(invalid_image)
    
    @patch('ultralytics.YOLO')
    def test_confidence_filtering(self, mock_yolo, detector_config, mock_ultralytics_model):
        """Test that low confidence detections are filtered out."""
        if YOLODetector is None:
            pytest.skip("YOLODetector not available")
        
        # Mock model with mixed confidence scores
        mock_model = Mock()
        mock_model.predict.return_value = [Mock(
            boxes=Mock(
                xyxy=np.array([[100, 100, 200, 200], [300, 150, 400, 250], [500, 200, 600, 300]]),
                conf=np.array([0.85, 0.3, 0.92]),  # Middle one below threshold
                cls=np.array([0, 1, 0])
            ),
            names={0: "person", 1: "vehicle"}
        )]
        mock_yolo.return_value = mock_model
        
        detector = YOLODetector(detector_config)
        sample_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = detector.detect(sample_image)
        
        # Should only have 2 detections (confidence >= 0.5)
        assert len(result["boxes"]) == 2
        confidences = [box["confidence"] for box in result["boxes"]]
        assert all(conf >= 0.5 for conf in confidences)
    
    @patch('ultralytics.YOLO')
    def test_class_filtering(self, mock_yolo, detector_config):
        """Test filtering by specific classes."""
        if YOLODetector is None:
            pytest.skip("YOLODetector not available")
        
        # Mock model with multiple classes
        mock_model = Mock()
        mock_model.predict.return_value = [Mock(
            boxes=Mock(
                xyxy=np.array([[100, 100, 200, 200], [300, 150, 400, 250], [500, 200, 600, 300]]),
                conf=np.array([0.85, 0.92, 0.78]),
                cls=np.array([0, 1, 2])  # person, vehicle, bicycle
            ),
            names={0: "person", 1: "vehicle", 2: "bicycle"}
        )]
        mock_yolo.return_value = mock_model
        
        # Filter to only detect persons
        detector_config["target_classes"] = ["person"]
        detector = YOLODetector(detector_config)
        sample_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = detector.detect(sample_image)
        
        # Should only have 1 detection (person class)
        assert len(result["boxes"]) == 1
        assert result["boxes"][0]["class"] == "person"
    
    @patch('ultralytics.YOLO')
    def test_batch_detection(self, mock_yolo, detector_config, sample_video_frame_sequence):
        """Test batch detection on multiple frames."""
        if YOLODetector is None:
            pytest.skip("YOLODetector not available")
        
        mock_model = Mock()
        # Return consistent detections for each frame
        mock_results = []
        for i in range(len(sample_video_frame_sequence)):
            mock_results.append(Mock(
                boxes=Mock(
                    xyxy=np.array([[100 + i*5, 100, 200 + i*5, 200]]),  # Moving detection
                    conf=np.array([0.85]),
                    cls=np.array([0])
                ),
                names={0: "person"}
            ))
        mock_model.predict.side_effect = mock_results
        mock_yolo.return_value = mock_model
        
        detector = YOLODetector(detector_config)
        
        results = []
        for frame in sample_video_frame_sequence:
            result = detector.detect(frame)
            results.append(result)
        
        assert len(results) == len(sample_video_frame_sequence)
        
        # Check that detections move across frames
        first_x = results[0]["boxes"][0]["x1"]
        last_x = results[-1]["boxes"][0]["x1"]
        assert last_x > first_x  # Object should have moved
    
    def test_performance_metrics(self, mock_detector):
        """Test that performance metrics are tracked."""
        sample_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = mock_detector.detect(sample_image)
        
        assert "processing_time" in result
        assert isinstance(result["processing_time"], float)
        assert result["processing_time"] > 0


@pytest.mark.unit
class TestObjectDetector:
    """Test cases for generic object detector interface."""
    
    def test_detector_interface(self, mock_detector):
        """Test that detector implements required interface."""
        # Check required methods exist
        assert hasattr(mock_detector, 'detect')
        assert hasattr(mock_detector, 'is_loaded')
        assert callable(mock_detector.detect)
        assert callable(mock_detector.is_loaded)
    
    def test_detector_state(self, mock_detector):
        """Test detector state management."""
        assert mock_detector.is_loaded() is True
        assert hasattr(mock_detector, 'model_info')
        assert isinstance(mock_detector.model_info, dict)


@pytest.mark.smoke
class TestDetectorSmoke:
    """Smoke tests for detector functionality."""
    
    def test_detector_can_process_image(self, mock_detector, sample_image):
        """Smoke test: detector can process a sample image without errors."""
        result = mock_detector.detect(sample_image)
        
        # Basic structure checks
        assert isinstance(result, dict)
        assert "boxes" in result
        assert "frame_id" in result
        assert isinstance(result["boxes"], list)
    
    def test_detector_handles_multiple_frames(self, mock_detector, sample_video_frame_sequence):
        """Smoke test: detector can handle multiple consecutive frames."""
        results = []
        
        for i, frame in enumerate(sample_video_frame_sequence[:3]):  # Test first 3 frames
            result = mock_detector.detect(frame)
            results.append(result)
        
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)
        assert all("boxes" in r for r in results)