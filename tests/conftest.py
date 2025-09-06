"""Pytest configuration and shared fixtures for the Foresight SAR test suite."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
import numpy as np
import cv2
from typing import Generator, Dict, Any

# Test data constants
TEST_IMAGE_WIDTH = 640
TEST_IMAGE_HEIGHT = 480
TEST_VIDEO_FPS = 30
TEST_MODEL_PATH = "models/yolov8n.pt"


@pytest.fixture(scope="session")
def test_data_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="foresight_test_")
    test_dir = Path(temp_dir)
    yield test_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_image() -> np.ndarray:
    """Generate a sample test image."""
    # Create a simple test image with some geometric shapes
    image = np.zeros((TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH, 3), dtype=np.uint8)
    
    # Add some colored rectangles to simulate objects
    cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue rectangle
    cv2.rectangle(image, (300, 150), (400, 250), (0, 255, 0), -1)  # Green rectangle
    cv2.circle(image, (500, 200), 50, (0, 0, 255), -1)  # Red circle
    
    return image


@pytest.fixture
def sample_video_frame_sequence(sample_image: np.ndarray) -> list:
    """Generate a sequence of video frames for testing."""
    frames = []
    base_image = sample_image.copy()
    
    # Create 10 frames with slightly moving objects
    for i in range(10):
        frame = base_image.copy()
        # Move the circle slightly in each frame
        cv2.circle(frame, (500 + i * 5, 200), 50, (0, 0, 255), -1)
        frames.append(frame)
    
    return frames


@pytest.fixture
def mock_detection_result() -> Dict[str, Any]:
    """Mock detection result for testing."""
    return {
        "boxes": [
            {"x1": 100, "y1": 100, "x2": 200, "y2": 200, "confidence": 0.85, "class": "person"},
            {"x1": 300, "y1": 150, "x2": 400, "y2": 250, "confidence": 0.92, "class": "vehicle"},
            {"x1": 450, "y1": 150, "x2": 550, "y2": 250, "confidence": 0.78, "class": "person"}
        ],
        "frame_id": 1,
        "timestamp": 1234567890.0,
        "processing_time": 0.045
    }


@pytest.fixture
def mock_gps_data() -> Dict[str, float]:
    """Mock GPS data for testing geolocation."""
    return {
        "latitude": 37.7749,
        "longitude": -122.4194,
        "altitude": 100.0,
        "heading": 45.0,
        "speed": 5.0
    }


@pytest.fixture
def mock_camera_params() -> Dict[str, Any]:
    """Mock camera calibration parameters."""
    return {
        "camera_matrix": np.array([
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0]
        ]),
        "distortion_coeffs": np.array([0.1, -0.2, 0.0, 0.0, 0.0]),
        "fov_horizontal": 60.0,
        "fov_vertical": 45.0
    }


@pytest.fixture
def mock_stream_source():
    """Mock video stream source."""
    mock_stream = Mock()
    mock_stream.is_connected.return_value = True
    mock_stream.get_frame.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    mock_stream.get_metadata.return_value = {
        "timestamp": 1234567890.0,
        "frame_id": 1,
        "source": "test_stream"
    }
    return mock_stream


@pytest.fixture
def mock_detector():
    """Mock YOLO detector."""
    mock_det = Mock()
    mock_det.detect.return_value = {
        "boxes": [[100, 100, 200, 200, 0.85, 0]],  # x1, y1, x2, y2, conf, class
        "frame_id": 1,
        "processing_time": 0.045
    }
    mock_det.is_loaded.return_value = True
    mock_det.model_info = {"name": "yolov8n", "classes": ["person", "vehicle"]}
    return mock_det


@pytest.fixture
def mock_tracker():
    """Mock object tracker."""
    mock_track = Mock()
    mock_track.update.return_value = [
        {"id": 1, "bbox": [100, 100, 200, 200], "confidence": 0.85, "class": "person"},
        {"id": 2, "bbox": [300, 150, 400, 250], "confidence": 0.92, "class": "vehicle"}
    ]
    mock_track.get_active_tracks.return_value = [1, 2]
    return mock_track


@pytest.fixture
def mock_geolocator():
    """Mock geolocation service."""
    mock_geo = Mock()
    mock_geo.project_to_ground.return_value = {
        "latitude": 37.7749,
        "longitude": -122.4194,
        "accuracy": 5.0
    }
    mock_geo.is_calibrated.return_value = True
    return mock_geo


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Test configuration settings."""
    return {
        "model": {
            "path": TEST_MODEL_PATH,
            "confidence_threshold": 0.5,
            "nms_threshold": 0.4,
            "device": "cpu"
        },
        "tracking": {
            "max_disappeared": 30,
            "max_distance": 100,
            "tracker_type": "sort"
        },
        "geolocation": {
            "enable_projection": True,
            "ground_height": 0.0,
            "camera_height": 100.0
        },
        "stream": {
            "buffer_size": 10,
            "timeout": 5.0,
            "retry_attempts": 3
        }
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment variables and paths."""
    # Set test environment variables
    monkeypatch.setenv("FORESIGHT_ENV", "test")
    monkeypatch.setenv("FORESIGHT_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("FORESIGHT_MODEL_PATH", TEST_MODEL_PATH)
    
    # Mock external dependencies that might not be available in test environment
    import sys
    from unittest.mock import MagicMock
    
    # Mock CUDA/TensorRT if not available
    if "torch" not in sys.modules:
        sys.modules["torch"] = MagicMock()
    if "ultralytics" not in sys.modules:
        sys.modules["ultralytics"] = MagicMock()


# Pytest markers for test categorization
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
]


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "smoke: mark test as a smoke test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )