"""Pytest configuration and shared fixtures for Foresight SAR System.

This module provides shared fixtures, configuration, and utilities
for all tests in the Foresight SAR test suite.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "tests" / "fixtures" / "data"
TEST_ASSETS_DIR = Path(__file__).parent / "tests" / "fixtures" / "assets"


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Set test environment variables
    os.environ["FORESIGHT_ENV"] = "test"
    os.environ["FORESIGHT_DEBUG"] = "1"
    os.environ["FORESIGHT_LOG_LEVEL"] = "DEBUG"
    
    # Disable GPU for tests by default (can be overridden)
    if "FORESIGHT_ENABLE_GPU" not in os.environ:
        os.environ["FORESIGHT_ENABLE_GPU"] = "false"
    
    # Create test data directories
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TEST_ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and skip conditions."""
    # Add markers based on test file location
    for item in items:
        # Add markers based on file path
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "acceptance" in str(item.fspath):
            item.add_marker(pytest.mark.acceptance)
        
        # Skip GPU tests if GPU not available
        if item.get_closest_marker("gpu"):
            if not _gpu_available():
                item.add_marker(pytest.mark.skip(reason="GPU not available"))
        
        # Skip network tests if offline
        if item.get_closest_marker("network"):
            if not _network_available():
                item.add_marker(pytest.mark.skip(reason="Network not available"))
        
        # Skip platform-specific tests
        if item.get_closest_marker("windows") and sys.platform != "win32":
            item.add_marker(pytest.mark.skip(reason="Windows-only test"))
        
        if item.get_closest_marker("linux") and sys.platform == "win32":
            item.add_marker(pytest.mark.skip(reason="Linux-only test"))


def _gpu_available() -> bool:
    """Check if GPU is available for testing."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _network_available() -> bool:
    """Check if network is available for testing."""
    import socket
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False


# ============================================================================
# Core Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Provide path to test data directory."""
    return TEST_DATA_DIR


@pytest.fixture(scope="session")
def test_assets_dir() -> Path:
    """Provide path to test assets directory."""
    return TEST_ASSETS_DIR


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration for testing."""
    return {
        "app": {
            "name": "Foresight SAR Test",
            "version": "1.0.0-test",
            "debug": True
        },
        "server": {
            "host": "127.0.0.1",
            "port": 8000
        },
        "database": {
            "type": "sqlite",
            "path": ":memory:"
        },
        "logging": {
            "level": "DEBUG"
        },
        "gpu": {
            "enabled": False
        },
        "models": {
            "path": str(TEST_ASSETS_DIR / "models")
        },
        "evidence": {
            "path": str(TEST_DATA_DIR / "evidence")
        }
    }


# ============================================================================
# Video and Image Fixtures
# ============================================================================

@pytest.fixture
def sample_image() -> np.ndarray:
    """Provide a sample test image."""
    # Create a simple test image (640x480, RGB)
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return image


@pytest.fixture
def sample_video_frame() -> np.ndarray:
    """Provide a sample video frame."""
    # Create a test video frame with some patterns
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some patterns for testing
    frame[100:200, 100:200] = [255, 0, 0]  # Red square
    frame[300:400, 300:400] = [0, 255, 0]  # Green square
    frame[200:300, 400:500] = [0, 0, 255]  # Blue square
    
    return frame


@pytest.fixture
def sample_detection() -> Dict[str, Any]:
    """Provide a sample object detection result."""
    return {
        "bbox": [100, 100, 200, 200],  # x1, y1, x2, y2
        "confidence": 0.85,
        "class_id": 0,
        "class_name": "person",
        "timestamp": "2024-01-15T10:30:00Z"
    }


@pytest.fixture
def sample_tracking_data() -> Dict[str, Any]:
    """Provide sample tracking data."""
    return {
        "track_id": 1,
        "detections": [
            {
                "frame_id": 0,
                "bbox": [100, 100, 200, 200],
                "confidence": 0.85
            },
            {
                "frame_id": 1,
                "bbox": [105, 105, 205, 205],
                "confidence": 0.87
            }
        ],
        "trajectory": [[150, 150], [155, 155]],
        "status": "active"
    }


# ============================================================================
# Geolocation Fixtures
# ============================================================================

@pytest.fixture
def sample_camera_params() -> Dict[str, Any]:
    """Provide sample camera calibration parameters."""
    return {
        "intrinsic_matrix": [
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0]
        ],
        "distortion_coeffs": [0.1, -0.2, 0.0, 0.0, 0.0],
        "image_size": [640, 480]
    }


@pytest.fixture
def sample_gps_data() -> Dict[str, Any]:
    """Provide sample GPS data."""
    return {
        "latitude": 37.7749,
        "longitude": -122.4194,
        "altitude": 100.0,
        "accuracy": 5.0,
        "timestamp": "2024-01-15T10:30:00Z"
    }


@pytest.fixture
def sample_imu_data() -> Dict[str, Any]:
    """Provide sample IMU data."""
    return {
        "accelerometer": [0.1, 0.2, 9.8],
        "gyroscope": [0.01, 0.02, 0.03],
        "magnetometer": [25.0, 30.0, 45.0],
        "timestamp": "2024-01-15T10:30:00Z"
    }


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture
def mock_database():
    """Provide a mock database for testing."""
    db = Mock()
    db.connect = Mock()
    db.disconnect = Mock()
    db.execute = Mock()
    db.fetch = Mock(return_value=[])
    db.insert = Mock(return_value=1)
    db.update = Mock(return_value=True)
    db.delete = Mock(return_value=True)
    return db


# ============================================================================
# Network and API Fixtures
# ============================================================================

@pytest.fixture
def mock_http_client():
    """Provide a mock HTTP client for testing."""
    client = Mock()
    client.get = Mock()
    client.post = Mock()
    client.put = Mock()
    client.delete = Mock()
    return client


@pytest.fixture
def mock_websocket():
    """Provide a mock WebSocket for testing."""
    ws = Mock()
    ws.send = Mock()
    ws.receive = Mock()
    ws.close = Mock()
    return ws


# ============================================================================
# Hardware Fixtures
# ============================================================================

@pytest.fixture
def mock_camera():
    """Provide a mock camera for testing."""
    camera = Mock()
    camera.connect = Mock(return_value=True)
    camera.disconnect = Mock()
    camera.capture_frame = Mock()
    camera.start_stream = Mock()
    camera.stop_stream = Mock()
    camera.is_connected = Mock(return_value=True)
    return camera


@pytest.fixture
def mock_gpu():
    """Provide a mock GPU for testing."""
    gpu = Mock()
    gpu.is_available = Mock(return_value=True)
    gpu.memory_usage = Mock(return_value=0.5)
    gpu.temperature = Mock(return_value=65.0)
    return gpu


# ============================================================================
# Model Fixtures
# ============================================================================

@pytest.fixture
def mock_detection_model():
    """Provide a mock object detection model."""
    model = Mock()
    model.load = Mock()
    model.predict = Mock(return_value=[])
    model.is_loaded = Mock(return_value=True)
    return model


@pytest.fixture
def mock_reid_model():
    """Provide a mock re-identification model."""
    model = Mock()
    model.load = Mock()
    model.extract_features = Mock(return_value=np.random.rand(512))
    model.compare_features = Mock(return_value=0.8)
    model.is_loaded = Mock(return_value=True)
    return model


# ============================================================================
# Security Fixtures
# ============================================================================

@pytest.fixture
def mock_vault_client():
    """Provide a mock HashiCorp Vault client."""
    vault = Mock()
    vault.is_authenticated = Mock(return_value=True)
    vault.encrypt = Mock(return_value=b"encrypted_data")
    vault.decrypt = Mock(return_value=b"decrypted_data")
    vault.sign = Mock(return_value=b"signature")
    vault.verify = Mock(return_value=True)
    return vault


@pytest.fixture
def sample_evidence_package() -> Dict[str, Any]:
    """Provide a sample evidence package."""
    return {
        "id": "evidence_001",
        "timestamp": "2024-01-15T10:30:00Z",
        "location": {
            "latitude": 37.7749,
            "longitude": -122.4194,
            "altitude": 100.0
        },
        "files": [
            {
                "name": "video.mp4",
                "hash": "sha256:abc123...",
                "size": 1024000
            }
        ],
        "metadata": {
            "operator": "test_operator",
            "mission": "test_mission",
            "equipment": "test_drone"
        },
        "signatures": {
            "digital_signature": "signature_data",
            "timestamp_proof": "ots_proof"
        }
    }


# ============================================================================
# Performance Fixtures
# ============================================================================

@pytest.fixture
def performance_monitor():
    """Provide a performance monitoring context manager."""
    import time
    import psutil
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.start_memory = None
            self.end_memory = None
            self.process = psutil.Process()
        
        def __enter__(self):
            self.start_time = time.time()
            self.start_memory = self.process.memory_info().rss
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.end_time = time.time()
            self.end_memory = self.process.memory_info().rss
        
        @property
        def duration(self) -> float:
            return self.end_time - self.start_time
        
        @property
        def memory_delta(self) -> int:
            return self.end_memory - self.start_memory
    
    return PerformanceMonitor


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Automatically cleanup test files after each test."""
    yield
    
    # Cleanup temporary test files
    temp_patterns = [
        "test_*.tmp",
        "*.test",
        "temp_*"
    ]
    
    for pattern in temp_patterns:
        for file_path in Path(".").glob(pattern):
            try:
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
            except (OSError, PermissionError):
                pass  # Ignore cleanup errors


# ============================================================================
# Parametrized Fixtures
# ============================================================================

@pytest.fixture(params=["cpu", "gpu"])
def device_type(request):
    """Parametrized fixture for testing on different device types."""
    if request.param == "gpu" and not _gpu_available():
        pytest.skip("GPU not available")
    return request.param


@pytest.fixture(params=["sqlite", "postgresql", "mysql"])
def database_type(request):
    """Parametrized fixture for testing different database types."""
    return request.param


@pytest.fixture(params=["yolov8n", "yolov8s", "yolov8m"])
def model_variant(request):
    """Parametrized fixture for testing different model variants."""
    return request.param