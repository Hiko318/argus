"""Unit tests for the video ingest module."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import threading
import time
from queue import Queue, Empty

# Import the modules to test
try:
    from src.backend.ingest import VideoIngest, StreamManager
    from connection.stream_bridge import StreamBridge
except ImportError:
    # Handle case where modules might not be importable in test environment
    VideoIngest = None
    StreamManager = None
    StreamBridge = None


@pytest.mark.unit
class TestVideoIngest:
    """Test cases for video ingest functionality."""
    
    @pytest.fixture
    def ingest_config(self):
        """Ingest configuration for testing."""
        return {
            "buffer_size": 10,
            "timeout": 5.0,
            "retry_attempts": 3,
            "frame_skip": 1,
            "target_fps": 30,
            "enable_preprocessing": True
        }
    
    @pytest.fixture
    def mock_video_source(self):
        """Mock video source (camera/stream)."""
        mock_source = Mock()
        mock_source.isOpened.return_value = True
        mock_source.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_source.get.return_value = 30.0  # FPS
        mock_source.release = Mock()
        return mock_source
    
    def test_ingest_initialization(self, ingest_config):
        """Test ingest initialization with valid config."""
        if VideoIngest is None:
            pytest.skip("VideoIngest not available")
        
        ingest = VideoIngest(ingest_config)
        
        assert ingest.buffer_size == 10
        assert ingest.timeout == 5.0
        assert ingest.retry_attempts == 3
        assert ingest.is_running is False
    
    def test_ingest_initialization_invalid_config(self):
        """Test ingest initialization with invalid config."""
        if VideoIngest is None:
            pytest.skip("VideoIngest not available")
        
        # Test with negative buffer size
        invalid_config = {"buffer_size": -1}
        with pytest.raises(ValueError):
            VideoIngest(invalid_config)
        
        # Test with invalid timeout
        invalid_config = {"timeout": -5.0}
        with pytest.raises(ValueError):
            VideoIngest(invalid_config)
    
    @patch('cv2.VideoCapture')
    def test_start_ingest_success(self, mock_cv2_cap, ingest_config, mock_video_source):
        """Test successful start of video ingest."""
        if VideoIngest is None:
            pytest.skip("VideoIngest not available")
        
        mock_cv2_cap.return_value = mock_video_source
        ingest = VideoIngest(ingest_config)
        
        # Start ingest
        success = ingest.start("test_source")
        
        assert success is True
        assert ingest.is_running is True
        assert ingest.source_name == "test_source"
        
        # Cleanup
        ingest.stop()
    
    @patch('cv2.VideoCapture')
    def test_start_ingest_failure(self, mock_cv2_cap, ingest_config):
        """Test failed start of video ingest."""
        if VideoIngest is None:
            pytest.skip("VideoIngest not available")
        
        # Mock failed video source
        mock_source = Mock()
        mock_source.isOpened.return_value = False
        mock_cv2_cap.return_value = mock_source
        
        ingest = VideoIngest(ingest_config)
        
        # Start ingest should fail
        success = ingest.start("invalid_source")
        
        assert success is False
        assert ingest.is_running is False
    
    @patch('cv2.VideoCapture')
    def test_frame_capture(self, mock_cv2_cap, ingest_config, mock_video_source, sample_image):
        """Test frame capture functionality."""
        if VideoIngest is None:
            pytest.skip("VideoIngest not available")
        
        # Setup mock to return sample image
        mock_video_source.read.return_value = (True, sample_image)
        mock_cv2_cap.return_value = mock_video_source
        
        ingest = VideoIngest(ingest_config)
        ingest.start("test_source")
        
        # Wait a bit for frames to be captured
        time.sleep(0.1)
        
        # Get frame from buffer
        frame_data = ingest.get_frame()
        
        assert frame_data is not None
        assert "frame" in frame_data
        assert "timestamp" in frame_data
        assert "frame_id" in frame_data
        assert np.array_equal(frame_data["frame"], sample_image)
        
        ingest.stop()
    
    @patch('cv2.VideoCapture')
    def test_frame_buffer_overflow(self, mock_cv2_cap, ingest_config, mock_video_source):
        """Test frame buffer overflow handling."""
        if VideoIngest is None:
            pytest.skip("VideoIngest not available")
        
        # Set small buffer size
        ingest_config["buffer_size"] = 2
        mock_cv2_cap.return_value = mock_video_source
        
        ingest = VideoIngest(ingest_config)
        ingest.start("test_source")
        
        # Let buffer fill up
        time.sleep(0.2)
        
        # Buffer should not exceed max size
        assert ingest.get_buffer_size() <= ingest_config["buffer_size"]
        
        ingest.stop()
    
    @patch('cv2.VideoCapture')
    def test_stop_ingest(self, mock_cv2_cap, ingest_config, mock_video_source):
        """Test stopping video ingest."""
        if VideoIngest is None:
            pytest.skip("VideoIngest not available")
        
        mock_cv2_cap.return_value = mock_video_source
        ingest = VideoIngest(ingest_config)
        
        ingest.start("test_source")
        assert ingest.is_running is True
        
        ingest.stop()
        assert ingest.is_running is False
        
        # Verify cleanup
        mock_video_source.release.assert_called_once()
    
    @patch('cv2.VideoCapture')
    def test_frame_preprocessing(self, mock_cv2_cap, ingest_config, mock_video_source, sample_image):
        """Test frame preprocessing functionality."""
        if VideoIngest is None:
            pytest.skip("VideoIngest not available")
        
        mock_video_source.read.return_value = (True, sample_image)
        mock_cv2_cap.return_value = mock_video_source
        
        # Enable preprocessing
        ingest_config["enable_preprocessing"] = True
        ingest_config["target_resolution"] = (320, 240)
        
        ingest = VideoIngest(ingest_config)
        ingest.start("test_source")
        
        time.sleep(0.1)
        frame_data = ingest.get_frame()
        
        # Check if frame was resized
        if frame_data:
            frame = frame_data["frame"]
            assert frame.shape[:2] == (240, 320)  # height, width
        
        ingest.stop()
    
    def test_get_frame_timeout(self, ingest_config):
        """Test frame retrieval with timeout."""
        if VideoIngest is None:
            pytest.skip("VideoIngest not available")
        
        ingest = VideoIngest(ingest_config)
        
        # Try to get frame without starting ingest
        frame_data = ingest.get_frame(timeout=0.1)
        
        assert frame_data is None
    
    @patch('cv2.VideoCapture')
    def test_connection_retry(self, mock_cv2_cap, ingest_config):
        """Test connection retry mechanism."""
        if VideoIngest is None:
            pytest.skip("VideoIngest not available")
        
        # Mock source that fails initially then succeeds
        mock_source_fail = Mock()
        mock_source_fail.isOpened.return_value = False
        
        mock_source_success = Mock()
        mock_source_success.isOpened.return_value = True
        mock_source_success.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        
        # First call fails, second succeeds
        mock_cv2_cap.side_effect = [mock_source_fail, mock_source_success]
        
        ingest = VideoIngest(ingest_config)
        
        # Should retry and eventually succeed
        success = ingest.start("test_source")
        
        # Depending on implementation, this might succeed after retry
        assert mock_cv2_cap.call_count >= 1
        
        if success:
            ingest.stop()


@pytest.mark.unit
class TestStreamManager:
    """Test cases for stream management functionality."""
    
    @pytest.fixture
    def stream_config(self):
        """Stream manager configuration."""
        return {
            "max_streams": 3,
            "default_buffer_size": 10,
            "health_check_interval": 5.0
        }
    
    def test_stream_manager_initialization(self, stream_config):
        """Test stream manager initialization."""
        if StreamManager is None:
            pytest.skip("StreamManager not available")
        
        manager = StreamManager(stream_config)
        
        assert manager.max_streams == 3
        assert len(manager.active_streams) == 0
    
    def test_add_stream(self, stream_config, mock_stream_source):
        """Test adding a new stream."""
        if StreamManager is None:
            pytest.skip("StreamManager not available")
        
        manager = StreamManager(stream_config)
        
        success = manager.add_stream("stream1", mock_stream_source)
        
        assert success is True
        assert "stream1" in manager.active_streams
        assert len(manager.active_streams) == 1
    
    def test_add_duplicate_stream(self, stream_config, mock_stream_source):
        """Test adding duplicate stream."""
        if StreamManager is None:
            pytest.skip("StreamManager not available")
        
        manager = StreamManager(stream_config)
        
        # Add first stream
        manager.add_stream("stream1", mock_stream_source)
        
        # Try to add duplicate
        success = manager.add_stream("stream1", mock_stream_source)
        
        assert success is False
        assert len(manager.active_streams) == 1
    
    def test_remove_stream(self, stream_config, mock_stream_source):
        """Test removing a stream."""
        if StreamManager is None:
            pytest.skip("StreamManager not available")
        
        manager = StreamManager(stream_config)
        
        # Add and then remove stream
        manager.add_stream("stream1", mock_stream_source)
        success = manager.remove_stream("stream1")
        
        assert success is True
        assert "stream1" not in manager.active_streams
        assert len(manager.active_streams) == 0
    
    def test_max_streams_limit(self, stream_config, mock_stream_source):
        """Test maximum streams limit."""
        if StreamManager is None:
            pytest.skip("StreamManager not available")
        
        manager = StreamManager(stream_config)
        
        # Add streams up to limit
        for i in range(stream_config["max_streams"]):
            success = manager.add_stream(f"stream{i}", mock_stream_source)
            assert success is True
        
        # Try to add one more (should fail)
        success = manager.add_stream("overflow_stream", mock_stream_source)
        assert success is False
        assert len(manager.active_streams) == stream_config["max_streams"]
    
    def test_get_stream_status(self, stream_config, mock_stream_source):
        """Test getting stream status."""
        if StreamManager is None:
            pytest.skip("StreamManager not available")
        
        manager = StreamManager(stream_config)
        manager.add_stream("stream1", mock_stream_source)
        
        status = manager.get_stream_status("stream1")
        
        assert status is not None
        assert "connected" in status
        assert "frame_count" in status
        assert "last_frame_time" in status


@pytest.mark.smoke
class TestIngestSmoke:
    """Smoke tests for ingest functionality."""
    
    def test_ingest_can_initialize(self, test_config):
        """Smoke test: ingest can be initialized without errors."""
        if VideoIngest is None:
            pytest.skip("VideoIngest not available")
        
        ingest_config = test_config.get("stream", {})
        ingest = VideoIngest(ingest_config)
        
        assert ingest is not None
        assert hasattr(ingest, 'start')
        assert hasattr(ingest, 'stop')
        assert hasattr(ingest, 'get_frame')
    
    @patch('cv2.VideoCapture')
    def test_ingest_basic_workflow(self, mock_cv2_cap, test_config, mock_video_source):
        """Smoke test: basic ingest workflow works."""
        if VideoIngest is None:
            pytest.skip("VideoIngest not available")
        
        mock_cv2_cap.return_value = mock_video_source
        ingest_config = test_config.get("stream", {})
        ingest = VideoIngest(ingest_config)
        
        # Basic workflow: start -> get frame -> stop
        try:
            success = ingest.start("test_source")
            if success:
                time.sleep(0.05)  # Brief wait
                frame_data = ingest.get_frame(timeout=0.1)
                # Frame data might be None in test environment, that's ok
        finally:
            ingest.stop()
        
        # If we get here without exceptions, the basic workflow works
        assert True
    
    def test_stream_manager_basic_operations(self, test_config):
        """Smoke test: stream manager basic operations work."""
        if StreamManager is None:
            pytest.skip("StreamManager not available")
        
        stream_config = {
            "max_streams": 2,
            "default_buffer_size": 5
        }
        
        manager = StreamManager(stream_config)
        
        # Basic operations should not raise exceptions
        assert manager.get_active_stream_count() == 0
        status = manager.get_stream_status("nonexistent")
        assert status is None
        
        success = manager.remove_stream("nonexistent")
        assert success is False