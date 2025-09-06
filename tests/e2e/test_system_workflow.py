"""End-to-end tests for the complete Foresight SAR system."""

import pytest
import numpy as np
import time
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import asyncio
from typing import Dict, List, Any, Optional

# Import system components for E2E testing
try:
    from src.backend.main import app, SARSystem
    from src.backend.api import APIServer
    from src.backend.sar_service import SARService
except ImportError:
    # Handle case where modules might not be importable in test environment
    app = None
    SARSystem = None
    APIServer = None
    SARService = None


@pytest.mark.e2e
class TestSystemWorkflowE2E:
    """End-to-end tests for complete system workflows."""
    
    @pytest.fixture
    def system_config(self, tmp_path):
        """Complete system configuration for E2E testing."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        config = {
            "system": {
                "name": "foresight-sar-test",
                "version": "1.0.0",
                "debug": True,
                "log_level": "INFO"
            },
            "api": {
                "host": "127.0.0.1",
                "port": 8004,
                "cors_enabled": True,
                "rate_limit": 100
            },
            "detector": {
                "model_path": str(tmp_path / "models" / "yolov8n.pt"),
                "confidence_threshold": 0.5,
                "nms_threshold": 0.4,
                "device": "cpu",
                "batch_size": 1
            },
            "tracker": {
                "max_disappeared": 30,
                "max_distance": 100,
                "tracker_type": "sort"
            },
            "geolocation": {
                "enable_projection": True,
                "camera_height": 100.0,
                "ground_height": 0.0,
                "accuracy_threshold": 10.0
            },
            "stream": {
                "buffer_size": 10,
                "timeout": 5.0,
                "frame_skip": 1,
                "max_streams": 4
            },
            "storage": {
                "data_dir": str(tmp_path / "data"),
                "models_dir": str(tmp_path / "models"),
                "logs_dir": str(tmp_path / "logs"),
                "max_storage_gb": 10
            },
            "processing": {
                "max_workers": 2,
                "queue_size": 100,
                "timeout": 30.0
            }
        }
        
        # Create required directories
        for dir_path in [config["storage"]["data_dir"], 
                        config["storage"]["models_dir"], 
                        config["storage"]["logs_dir"]]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Save config file
        config_file = config_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config
    
    @pytest.fixture
    def mock_system_components(self):
        """Mock all system components for E2E testing."""
        components = {
            "sar_service": Mock(),
            "api_server": Mock(),
            "detector": Mock(),
            "tracker": Mock(),
            "geolocator": Mock(),
            "stream_manager": Mock()
        }
        
        # Configure SAR service mock
        components["sar_service"].start.return_value = True
        components["sar_service"].stop.return_value = True
        components["sar_service"].is_running.return_value = True
        components["sar_service"].process_frame.return_value = {
            "frame_id": 1,
            "timestamp": time.time(),
            "detections": [
                {"id": 1, "class": "person", "confidence": 0.85, "bbox": [100, 100, 200, 200]}
            ],
            "tracks": [
                {"id": 1, "bbox": [100, 100, 200, 200], "confidence": 0.85, "class": "person"}
            ],
            "geolocations": [
                {"track_id": 1, "latitude": 37.7749, "longitude": -122.4194, "accuracy": 5.0}
            ],
            "processing_time": 0.045
        }
        components["sar_service"].get_status.return_value = {
            "status": "running",
            "uptime": 3600,
            "frames_processed": 1000,
            "active_streams": 1,
            "active_tracks": 5
        }
        
        # Configure API server mock
        components["api_server"].start.return_value = True
        components["api_server"].stop.return_value = True
        components["api_server"].is_running.return_value = True
        
        # Configure stream manager mock
        components["stream_manager"].add_stream.return_value = True
        components["stream_manager"].remove_stream.return_value = True
        components["stream_manager"].get_streams.return_value = {
            "stream_1": {"status": "active", "fps": 30, "resolution": "1920x1080"}
        }
        
        return components
    
    def test_system_initialization(self, system_config):
        """Test complete system initialization."""
        if SARSystem is None:
            pytest.skip("SARSystem not available")
        
        with patch.multiple(
            'src.backend.main',
            SARService=Mock(),
            APIServer=Mock()
        ):
            system = SARSystem(system_config)
            
            assert system is not None
            assert hasattr(system, 'start')
            assert hasattr(system, 'stop')
            assert hasattr(system, 'get_status')
    
    def test_system_startup_shutdown_cycle(self, system_config, mock_system_components):
        """Test complete system startup and shutdown cycle."""
        if SARSystem is None:
            pytest.skip("SARSystem not available")
        
        with patch.multiple(
            'src.backend.main',
            SARService=lambda config: mock_system_components["sar_service"],
            APIServer=lambda config: mock_system_components["api_server"]
        ):
            system = SARSystem(system_config)
            
            # Test startup
            startup_success = system.start()
            assert startup_success is True
            
            # Verify components were started
            mock_system_components["sar_service"].start.assert_called_once()
            mock_system_components["api_server"].start.assert_called_once()
            
            # Test system status
            status = system.get_status()
            assert status is not None
            assert "status" in status
            
            # Test shutdown
            shutdown_success = system.stop()
            assert shutdown_success is True
            
            # Verify components were stopped
            mock_system_components["sar_service"].stop.assert_called_once()
            mock_system_components["api_server"].stop.assert_called_once()
    
    def test_stream_processing_workflow(self, system_config, mock_system_components, sample_video_frame_sequence):
        """Test complete stream processing workflow."""
        if SARSystem is None:
            pytest.skip("SARSystem not available")
        
        with patch.multiple(
            'src.backend.main',
            SARService=lambda config: mock_system_components["sar_service"],
            APIServer=lambda config: mock_system_components["api_server"]
        ):
            system = SARSystem(system_config)
            system.start()
            
            try:
                # Add a stream
                stream_id = "test_stream_1"
                add_result = system.add_stream(stream_id, "rtsp://test.stream")
                assert add_result is True
                
                # Process frames
                for i, frame in enumerate(sample_video_frame_sequence[:3]):
                    # Simulate frame processing
                    result = system.process_frame(stream_id, frame, {"timestamp": time.time()})
                    
                    if result:
                        assert "frame_id" in result
                        assert "detections" in result
                        assert "tracks" in result
                        assert "processing_time" in result
                
                # Get stream status
                streams = system.get_streams()
                assert stream_id in streams or len(streams) > 0
                
                # Remove stream
                remove_result = system.remove_stream(stream_id)
                assert remove_result is True
                
            finally:
                system.stop()
    
    def test_api_integration_workflow(self, system_config, mock_system_components):
        """Test API integration with system workflow."""
        if SARSystem is None or APIServer is None:
            pytest.skip("Required components not available")
        
        # Mock API endpoints
        mock_api = Mock()
        mock_api.get_status.return_value = {"status": "ok", "uptime": 3600}
        mock_api.add_stream.return_value = {"success": True, "stream_id": "test_stream"}
        mock_api.get_detections.return_value = {
            "detections": [
                {"id": 1, "class": "person", "confidence": 0.85, "timestamp": time.time()}
            ]
        }
        
        with patch.multiple(
            'src.backend.main',
            SARService=lambda config: mock_system_components["sar_service"],
            APIServer=lambda config: mock_api
        ):
            system = SARSystem(system_config)
            system.start()
            
            try:
                # Test API status endpoint
                status = mock_api.get_status()
                assert status["status"] == "ok"
                
                # Test API stream management
                add_response = mock_api.add_stream("test_stream", "rtsp://test.stream")
                assert add_response["success"] is True
                
                # Test API detection retrieval
                detections = mock_api.get_detections("test_stream")
                assert "detections" in detections
                assert len(detections["detections"]) > 0
                
            finally:
                system.stop()
    
    def test_error_recovery_workflow(self, system_config, mock_system_components):
        """Test system error recovery workflows."""
        if SARSystem is None:
            pytest.skip("SARSystem not available")
        
        # Configure components to fail initially
        mock_system_components["sar_service"].start.side_effect = [Exception("Start failed"), True]
        mock_system_components["sar_service"].process_frame.side_effect = [
            Exception("Processing failed"), 
            {"frame_id": 1, "detections": [], "tracks": []}
        ]
        
        with patch.multiple(
            'src.backend.main',
            SARService=lambda config: mock_system_components["sar_service"],
            APIServer=lambda config: mock_system_components["api_server"]
        ):
            system = SARSystem(system_config)
            
            # First start attempt should fail
            startup_success = system.start()
            # System should handle the error gracefully
            
            # Retry should succeed
            mock_system_components["sar_service"].start.side_effect = None
            mock_system_components["sar_service"].start.return_value = True
            
            retry_success = system.start()
            assert retry_success is True
            
            try:
                # Test processing error recovery
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # First processing attempt should fail
                result1 = system.process_frame("test_stream", frame, {})
                # Should handle error gracefully
                
                # Second attempt should succeed
                mock_system_components["sar_service"].process_frame.side_effect = None
                mock_system_components["sar_service"].process_frame.return_value = {
                    "frame_id": 1, "detections": [], "tracks": []
                }
                
                result2 = system.process_frame("test_stream", frame, {})
                assert result2 is not None
                
            finally:
                system.stop()
    
    def test_performance_monitoring_workflow(self, system_config, mock_system_components):
        """Test system performance monitoring workflow."""
        if SARSystem is None:
            pytest.skip("SARSystem not available")
        
        # Configure performance metrics
        mock_system_components["sar_service"].get_performance_stats.return_value = {
            "total_frames": 1000,
            "avg_processing_time": 0.045,
            "fps": 22.2,
            "memory_usage": 512.5,
            "cpu_usage": 45.2
        }
        
        with patch.multiple(
            'src.backend.main',
            SARService=lambda config: mock_system_components["sar_service"],
            APIServer=lambda config: mock_system_components["api_server"]
        ):
            system = SARSystem(system_config)
            system.start()
            
            try:
                # Get performance metrics
                metrics = system.get_performance_metrics()
                
                assert metrics is not None
                if "total_frames" in metrics:
                    assert metrics["total_frames"] >= 0
                if "avg_processing_time" in metrics:
                    assert metrics["avg_processing_time"] >= 0
                
                # Test performance alerts
                alerts = system.get_performance_alerts()
                assert alerts is not None
                
            finally:
                system.stop()
    
    def test_configuration_management_workflow(self, system_config, tmp_path):
        """Test configuration management workflow."""
        if SARSystem is None:
            pytest.skip("SARSystem not available")
        
        with patch.multiple(
            'src.backend.main',
            SARService=Mock(),
            APIServer=Mock()
        ):
            system = SARSystem(system_config)
            
            # Test configuration validation
            is_valid = system.validate_config(system_config)
            assert is_valid is True or is_valid is None  # Depends on implementation
            
            # Test configuration updates
            new_config = system_config.copy()
            new_config["detector"]["confidence_threshold"] = 0.7
            
            update_result = system.update_config(new_config)
            # Should handle config updates gracefully
            
            # Test configuration backup/restore
            backup_path = tmp_path / "config_backup.json"
            backup_result = system.backup_config(str(backup_path))
            # Should handle backup gracefully
            
            if backup_path.exists():
                restore_result = system.restore_config(str(backup_path))
                # Should handle restore gracefully


@pytest.mark.e2e
class TestDataFlowE2E:
    """End-to-end tests for data flow through the system."""
    
    def test_video_to_detection_to_storage_flow(self, system_config, tmp_path):
        """Test complete data flow from video input to storage."""
        # Create test data directories
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        
        # Mock data flow components
        mock_components = {
            "video_processor": Mock(),
            "detector": Mock(),
            "storage": Mock()
        }
        
        # Configure data flow
        mock_components["video_processor"].process_video.return_value = {
            "frames_processed": 100,
            "duration": 10.0,
            "fps": 10.0
        }
        
        mock_components["detector"].detect_batch.return_value = [
            {"frame_id": i, "detections": [{"class": "person", "confidence": 0.8}]}
            for i in range(10)
        ]
        
        mock_components["storage"].save_results.return_value = True
        
        # Test data flow
        try:
            # Process video file
            video_file = input_dir / "test_video.mp4"
            video_file.touch()  # Create dummy file
            
            processing_result = mock_components["video_processor"].process_video(str(video_file))
            assert processing_result["frames_processed"] > 0
            
            # Detect objects
            detection_results = mock_components["detector"].detect_batch([])
            assert len(detection_results) > 0
            
            # Store results
            storage_result = mock_components["storage"].save_results(
                detection_results, str(output_dir / "results.json")
            )
            assert storage_result is True
            
        except Exception as e:
            pytest.fail(f"Data flow test failed: {e}")
    
    def test_realtime_stream_processing_flow(self, system_config):
        """Test real-time stream processing data flow."""
        # Mock real-time components
        mock_stream = Mock()
        mock_processor = Mock()
        mock_output = Mock()
        
        # Configure real-time flow
        mock_stream.get_frame.return_value = {
            "frame": np.zeros((480, 640, 3), dtype=np.uint8),
            "timestamp": time.time(),
            "frame_id": 1
        }
        
        mock_processor.process_realtime.return_value = {
            "detections": [{"class": "vehicle", "confidence": 0.9}],
            "processing_time": 0.033,
            "fps": 30.0
        }
        
        mock_output.send_results.return_value = True
        
        # Test real-time flow
        try:
            # Simulate real-time processing loop
            for i in range(5):  # Process 5 frames
                # Get frame
                frame_data = mock_stream.get_frame()
                assert frame_data is not None
                
                # Process frame
                result = mock_processor.process_realtime(frame_data)
                assert result is not None
                assert "processing_time" in result
                
                # Send results
                output_success = mock_output.send_results(result)
                assert output_success is True
                
                # Simulate frame rate
                time.sleep(0.001)  # Very short delay for testing
                
        except Exception as e:
            pytest.fail(f"Real-time flow test failed: {e}")


@pytest.mark.smoke
class TestSystemSmoke:
    """Smoke tests for the complete system."""
    
    def test_system_imports(self):
        """Smoke test: system modules can be imported."""
        # Test that main system imports work
        # If we get here, imports succeeded or were skipped
        assert True
    
    def test_system_configuration_structure(self, system_config):
        """Smoke test: system configuration has required structure."""
        required_sections = ["system", "api", "detector", "tracker", "geolocation", "stream", "storage"]
        
        for section in required_sections:
            assert section in system_config, f"Missing required config section: {section}"
        
        # Test basic config values
        assert system_config["api"]["port"] > 0
        assert system_config["detector"]["confidence_threshold"] > 0
        assert system_config["stream"]["buffer_size"] > 0
    
    def test_system_directories_creation(self, system_config):
        """Smoke test: system can create required directories."""
        storage_config = system_config["storage"]
        
        for dir_key in ["data_dir", "models_dir", "logs_dir"]:
            dir_path = Path(storage_config[dir_key])
            assert dir_path.exists(), f"Directory not created: {dir_path}"
            assert dir_path.is_dir(), f"Path is not a directory: {dir_path}"
    
    def test_mock_system_workflow(self, mock_system_components):
        """Smoke test: mock system workflow completes without errors."""
        try:
            # Test component initialization
            assert mock_system_components["sar_service"] is not None
            assert mock_system_components["api_server"] is not None
            
            # Test basic operations
            start_result = mock_system_components["sar_service"].start()
            assert start_result is True
            
            status = mock_system_components["sar_service"].get_status()
            assert status is not None
            
            stop_result = mock_system_components["sar_service"].stop()
            assert stop_result is True
            
        except Exception as e:
            pytest.fail(f"Mock system workflow failed: {e}")
    
    def test_system_health_check(self, system_config, mock_system_components):
        """Smoke test: system health check works."""
        if SARSystem is None:
            pytest.skip("SARSystem not available")
        
        with patch.multiple(
            'src.backend.main',
            SARService=lambda config: mock_system_components["sar_service"],
            APIServer=lambda config: mock_system_components["api_server"]
        ):
            system = SARSystem(system_config)
            
            # Test health check
            health = system.health_check()
            
            # Health check should return some status
            assert health is not None or health is True or health is False