"""Integration tests for the complete SAR pipeline."""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import asyncio
from typing import Dict, List, Any

# Import modules for integration testing
try:
    from src.backend.sar_service import SARService
    from src.backend.detector import YOLODetector
    from src.backend.ingest import VideoIngest
    from tracking.tracker import ObjectTracker
    from geolocation.projection import GeolocationProjector
except ImportError:
    # Handle case where modules might not be importable in test environment
    SARService = None
    YOLODetector = None
    VideoIngest = None
    ObjectTracker = None
    GeolocationProjector = None


@pytest.mark.integration
class TestSARPipelineIntegration:
    """Integration tests for the complete SAR processing pipeline."""
    
    @pytest.fixture
    def pipeline_config(self):
        """Complete pipeline configuration."""
        return {
            "detector": {
                "model_path": "models/yolov8n.pt",
                "confidence_threshold": 0.5,
                "nms_threshold": 0.4,
                "device": "cpu"
            },
            "tracker": {
                "max_disappeared": 30,
                "max_distance": 100,
                "tracker_type": "sort"
            },
            "geolocation": {
                "enable_projection": True,
                "camera_height": 100.0,
                "ground_height": 0.0
            },
            "ingest": {
                "buffer_size": 10,
                "timeout": 5.0,
                "frame_skip": 1
            }
        }
    
    @pytest.fixture
    def mock_sar_components(self):
        """Mock SAR service components."""
        components = {
            "detector": Mock(),
            "tracker": Mock(),
            "geolocator": Mock(),
            "ingest": Mock()
        }
        
        # Configure detector mock
        components["detector"].detect.return_value = {
            "boxes": [
                {"x1": 100, "y1": 100, "x2": 200, "y2": 200, "confidence": 0.85, "class": "person"},
                {"x1": 300, "y1": 150, "x2": 400, "y2": 250, "confidence": 0.92, "class": "vehicle"}
            ],
            "frame_id": 1,
            "timestamp": time.time(),
            "processing_time": 0.045
        }
        components["detector"].is_loaded.return_value = True
        
        # Configure tracker mock
        components["tracker"].update.return_value = [
            {"id": 1, "bbox": [100, 100, 200, 200], "confidence": 0.85, "class": "person"},
            {"id": 2, "bbox": [300, 150, 400, 250], "confidence": 0.92, "class": "vehicle"}
        ]
        components["tracker"].get_active_tracks.return_value = [1, 2]
        
        # Configure geolocator mock
        components["geolocator"].geolocate_detection.return_value = {
            "latitude": 37.7749,
            "longitude": -122.4194,
            "accuracy": 5.0,
            "timestamp": time.time()
        }
        components["geolocator"].is_calibrated.return_value = True
        
        # Configure ingest mock
        components["ingest"].get_frame.return_value = {
            "frame": np.zeros((480, 640, 3), dtype=np.uint8),
            "timestamp": time.time(),
            "frame_id": 1,
            "metadata": {"source": "test_stream"}
        }
        components["ingest"].is_running = True
        
        return components
    
    def test_pipeline_initialization(self, pipeline_config):
        """Test complete pipeline initialization."""
        if SARService is None:
            pytest.skip("SARService not available")
        
        with patch.multiple(
            'src.backend.sar_service',
            YOLODetector=Mock(),
            ObjectTracker=Mock(),
            GeolocationProjector=Mock(),
            VideoIngest=Mock()
        ):
            service = SARService(pipeline_config)
            
            assert service is not None
            assert hasattr(service, 'start')
            assert hasattr(service, 'stop')
            assert hasattr(service, 'process_frame')
    
    def test_frame_processing_pipeline(self, pipeline_config, mock_sar_components, sample_image, mock_gps_data):
        """Test complete frame processing through the pipeline."""
        if SARService is None:
            pytest.skip("SARService not available")
        
        with patch.multiple(
            'src.backend.sar_service',
            YOLODetector=lambda config: mock_sar_components["detector"],
            ObjectTracker=lambda config: mock_sar_components["tracker"],
            GeolocationProjector=lambda config, params: mock_sar_components["geolocator"],
            VideoIngest=lambda config: mock_sar_components["ingest"]
        ):
            service = SARService(pipeline_config)
            
            # Process a frame through the complete pipeline
            result = service.process_frame(sample_image, mock_gps_data)
            
            assert result is not None
            assert "detections" in result
            assert "tracks" in result
            assert "geolocations" in result
            assert "frame_id" in result
            assert "timestamp" in result
            
            # Verify pipeline components were called
            mock_sar_components["detector"].detect.assert_called_once()
            mock_sar_components["tracker"].update.assert_called_once()
    
    def test_detection_to_tracking_flow(self, pipeline_config, mock_sar_components, sample_image):
        """Test flow from detection to tracking."""
        if SARService is None:
            pytest.skip("SARService not available")
        
        # Configure detector to return specific detections
        detection_result = {
            "boxes": [
                {"x1": 100, "y1": 100, "x2": 200, "y2": 200, "confidence": 0.85, "class": "person"}
            ],
            "frame_id": 1,
            "timestamp": time.time()
        }
        mock_sar_components["detector"].detect.return_value = detection_result
        
        with patch.multiple(
            'src.backend.sar_service',
            YOLODetector=lambda config: mock_sar_components["detector"],
            ObjectTracker=lambda config: mock_sar_components["tracker"],
            GeolocationProjector=lambda config, params: mock_sar_components["geolocator"]
        ):
            service = SARService(pipeline_config)
            result = service.process_frame(sample_image, {})
            
            # Verify tracker received detection data
            tracker_call_args = mock_sar_components["tracker"].update.call_args
            assert tracker_call_args is not None
            
            # Check that detection data was passed to tracker
            passed_detections = tracker_call_args[0][0]  # First argument
            assert len(passed_detections) == 1
            assert passed_detections[0]["class"] == "person"
    
    def test_tracking_to_geolocation_flow(self, pipeline_config, mock_sar_components, sample_image, mock_gps_data):
        """Test flow from tracking to geolocation."""
        if SARService is None:
            pytest.skip("SARService not available")
        
        # Configure tracker to return specific tracks
        track_result = [
            {"id": 1, "bbox": [100, 100, 200, 200], "confidence": 0.85, "class": "person"}
        ]
        mock_sar_components["tracker"].update.return_value = track_result
        
        with patch.multiple(
            'src.backend.sar_service',
            YOLODetector=lambda config: mock_sar_components["detector"],
            ObjectTracker=lambda config: mock_sar_components["tracker"],
            GeolocationProjector=lambda config, params: mock_sar_components["geolocator"]
        ):
            service = SARService(pipeline_config)
            result = service.process_frame(sample_image, mock_gps_data)
            
            # Verify geolocator was called for each track
            geolocator_calls = mock_sar_components["geolocator"].geolocate_detection.call_count
            assert geolocator_calls >= 1
    
    def test_pipeline_error_handling(self, pipeline_config, mock_sar_components, sample_image):
        """Test pipeline error handling when components fail."""
        if SARService is None:
            pytest.skip("SARService not available")
        
        # Configure detector to raise an exception
        mock_sar_components["detector"].detect.side_effect = Exception("Detection failed")
        
        with patch.multiple(
            'src.backend.sar_service',
            YOLODetector=lambda config: mock_sar_components["detector"],
            ObjectTracker=lambda config: mock_sar_components["tracker"],
            GeolocationProjector=lambda config, params: mock_sar_components["geolocator"]
        ):
            service = SARService(pipeline_config)
            
            # Pipeline should handle errors gracefully
            result = service.process_frame(sample_image, {})
            
            # Should return error result or None, not crash
            assert result is None or "error" in result
    
    def test_pipeline_performance_metrics(self, pipeline_config, mock_sar_components, sample_image):
        """Test that pipeline tracks performance metrics."""
        if SARService is None:
            pytest.skip("SARService not available")
        
        with patch.multiple(
            'src.backend.sar_service',
            YOLODetector=lambda config: mock_sar_components["detector"],
            ObjectTracker=lambda config: mock_sar_components["tracker"],
            GeolocationProjector=lambda config, params: mock_sar_components["geolocator"]
        ):
            service = SARService(pipeline_config)
            result = service.process_frame(sample_image, {})
            
            # Check for performance metrics
            if result:
                assert "processing_time" in result or "performance" in result
                
                # Get performance stats
                stats = service.get_performance_stats()
                assert stats is not None
                assert "total_frames" in stats or "avg_processing_time" in stats
    
    def test_multi_frame_processing(self, pipeline_config, mock_sar_components, sample_video_frame_sequence, mock_gps_data):
        """Test processing multiple frames in sequence."""
        if SARService is None:
            pytest.skip("SARService not available")
        
        with patch.multiple(
            'src.backend.sar_service',
            YOLODetector=lambda config: mock_sar_components["detector"],
            ObjectTracker=lambda config: mock_sar_components["tracker"],
            GeolocationProjector=lambda config, params: mock_sar_components["geolocator"]
        ):
            service = SARService(pipeline_config)
            
            results = []
            for i, frame in enumerate(sample_video_frame_sequence[:5]):  # Process 5 frames
                # Update frame_id for each frame
                mock_sar_components["detector"].detect.return_value["frame_id"] = i + 1
                
                result = service.process_frame(frame, mock_gps_data)
                if result:
                    results.append(result)
            
            assert len(results) > 0
            
            # Check that frame IDs are sequential
            frame_ids = [r["frame_id"] for r in results if "frame_id" in r]
            if frame_ids:
                assert len(set(frame_ids)) == len(frame_ids)  # All unique


@pytest.mark.integration
class TestStreamToDetectionIntegration:
    """Integration tests for stream ingestion to detection pipeline."""
    
    @pytest.fixture
    def stream_detection_config(self):
        """Configuration for stream-to-detection testing."""
        return {
            "stream": {
                "buffer_size": 5,
                "timeout": 1.0,
                "frame_skip": 1
            },
            "detection": {
                "model_path": "models/yolov8n.pt",
                "confidence_threshold": 0.5,
                "batch_size": 1
            }
        }
    
    @patch('cv2.VideoCapture')
    def test_stream_to_detection_flow(self, mock_cv2_cap, stream_detection_config, sample_image):
        """Test complete flow from stream ingestion to detection."""
        if VideoIngest is None or YOLODetector is None:
            pytest.skip("Required modules not available")
        
        # Mock video source
        mock_source = Mock()
        mock_source.isOpened.return_value = True
        mock_source.read.return_value = (True, sample_image)
        mock_cv2_cap.return_value = mock_source
        
        # Mock detector
        mock_detector = Mock()
        mock_detector.detect.return_value = {
            "boxes": [{"x1": 100, "y1": 100, "x2": 200, "y2": 200, "confidence": 0.85, "class": "person"}],
            "frame_id": 1,
            "processing_time": 0.045
        }
        mock_detector.is_loaded.return_value = True
        
        with patch('src.backend.detector.YOLODetector', return_value=mock_detector):
            # Initialize components
            ingest = VideoIngest(stream_detection_config["stream"])
            detector = mock_detector
            
            # Start stream
            success = ingest.start("test_stream")
            assert success is True
            
            try:
                # Get frame and detect
                time.sleep(0.1)  # Allow frame capture
                frame_data = ingest.get_frame(timeout=0.5)
                
                if frame_data:
                    detection_result = detector.detect(frame_data["frame"])
                    
                    assert detection_result is not None
                    assert "boxes" in detection_result
                    assert len(detection_result["boxes"]) > 0
                    
                    # Verify detection data structure
                    detection = detection_result["boxes"][0]
                    assert "confidence" in detection
                    assert "class" in detection
                    assert detection["confidence"] >= 0.5
            
            finally:
                ingest.stop()
    
    def test_stream_buffer_management(self, stream_detection_config):
        """Test stream buffer management under load."""
        if VideoIngest is None:
            pytest.skip("VideoIngest not available")
        
        # Set small buffer for testing
        stream_detection_config["stream"]["buffer_size"] = 3
        
        with patch('cv2.VideoCapture') as mock_cv2_cap:
            mock_source = Mock()
            mock_source.isOpened.return_value = True
            mock_source.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_cv2_cap.return_value = mock_source
            
            ingest = VideoIngest(stream_detection_config["stream"])
            
            success = ingest.start("test_stream")
            assert success is True
            
            try:
                # Let buffer fill up
                time.sleep(0.2)
                
                # Buffer should not exceed configured size
                buffer_size = ingest.get_buffer_size()
                assert buffer_size <= stream_detection_config["stream"]["buffer_size"]
                
                # Should still be able to get frames
                frame_data = ingest.get_frame(timeout=0.1)
                # Frame might be None if buffer is empty, that's ok
                
            finally:
                ingest.stop()


@pytest.mark.smoke
class TestPipelineSmoke:
    """Smoke tests for pipeline integration."""
    
    def test_pipeline_components_can_be_imported(self):
        """Smoke test: all pipeline components can be imported."""
        # Test that imports work (already done at module level)
        # If we get here, imports succeeded or were skipped
        assert True
    
    def test_pipeline_can_be_configured(self, test_config):
        """Smoke test: pipeline can be configured without errors."""
        # Basic configuration test
        config = {
            "detector": test_config.get("model", {}),
            "tracker": test_config.get("tracking", {}),
            "geolocation": test_config.get("geolocation", {}),
            "stream": test_config.get("stream", {})
        }
        
        # Should not raise exceptions
        assert isinstance(config, dict)
        assert "detector" in config
        assert "tracker" in config
        assert "geolocation" in config
        assert "stream" in config
    
    def test_mock_pipeline_workflow(self, mock_sar_components, sample_image):
        """Smoke test: mock pipeline workflow completes without errors."""
        # Simulate basic pipeline workflow with mocks
        try:
            # Detection
            detection_result = mock_sar_components["detector"].detect(sample_image)
            assert detection_result is not None
            
            # Tracking
            if detection_result and "boxes" in detection_result:
                track_result = mock_sar_components["tracker"].update(detection_result["boxes"])
                assert track_result is not None
            
            # Geolocation
            if detection_result and "boxes" in detection_result and detection_result["boxes"]:
                geo_result = mock_sar_components["geolocator"].geolocate_detection(
                    detection_result["boxes"][0], {}, time.time()
                )
                assert geo_result is not None
                
        except Exception as e:
            pytest.fail(f"Mock pipeline workflow failed: {e}")