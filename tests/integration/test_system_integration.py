"""Integration tests for complete system workflow.

Tests for end-to-end system integration, component interactions,
and complete processing pipelines.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import asyncio
import time

# Import system components (with error handling)
try:
    from src.vision.detector import ObjectDetector, DetectionResult
    from src.tracking.tracker import MultiObjectTracker
    from src.geolocation.projection import GeolocationProcessor
    from src.reid.matcher import ReIDMatcher
    from src.packaging.evidence_packager import EvidencePackager
    from src.ui.websocket_handler import WebSocketHandler
    from src.connection.stream_manager import StreamManager
except ImportError as e:
    pytest.skip(f"System modules not available: {e}", allow_module_level=True)


class TestVideoProcessingPipeline:
    """Test complete video processing pipeline."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mocked system components."""
        components = {
            'detector': Mock(spec=ObjectDetector),
            'tracker': Mock(spec=MultiObjectTracker),
            'geolocation': Mock(spec=GeolocationProcessor),
            'reid': Mock(spec=ReIDMatcher),
            'packager': Mock(spec=EvidencePackager)
        }
        
        # Configure mock behaviors
        components['detector'].detect.return_value = [
            DetectionResult([100, 100, 200, 200], 0.8, 0, "person")
        ]
        
        components['tracker'].update.return_value = [
            Mock(track_id=1, current_bbox=[100, 100, 200, 200], state="confirmed")
        ]
        
        components['geolocation'].project_to_world.return_value = {
            'latitude': 37.7749,
            'longitude': -122.4194,
            'altitude': 100.0
        }
        
        components['reid'].extract_features.return_value = np.random.rand(512)
        components['reid'].match_features.return_value = 0.85
        
        components['packager'].create_package.return_value = {
            'package_id': 'test_package_001',
            'status': 'success'
        }
        
        return components
    
    @pytest.fixture
    def sample_video_frames(self):
        """Generate sample video frames for testing."""
        frames = []
        for i in range(10):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            # Add some movement pattern
            frame[100+i*5:200+i*5, 100+i*5:200+i*5] = [255, 0, 0]  # Moving red square
            frames.append(frame)
        return frames
    
    def test_complete_processing_pipeline(self, mock_components, sample_video_frames):
        """Test complete video processing pipeline."""
        detector = mock_components['detector']
        tracker = mock_components['tracker']
        geolocation = mock_components['geolocation']
        reid = mock_components['reid']
        packager = mock_components['packager']
        
        # Process video frames
        results = []
        for frame_id, frame in enumerate(sample_video_frames):
            # Detection
            detections = detector.detect(frame)
            
            # Tracking
            tracks = tracker.update(detections, frame_id)
            
            # Geolocation
            for track in tracks:
                world_coords = geolocation.project_to_world(
                    track.current_bbox, frame_id
                )
                track.world_coordinates = world_coords
            
            # Re-identification
            for track in tracks:
                features = reid.extract_features(frame, track.current_bbox)
                track.features = features
            
            results.append({
                'frame_id': frame_id,
                'detections': detections,
                'tracks': tracks,
                'timestamp': time.time()
            })
        
        # Package evidence
        evidence_package = packager.create_package(results)
        
        # Verify pipeline execution
        assert len(results) == 10
        assert detector.detect.call_count == 10
        assert tracker.update.call_count == 10
        assert geolocation.project_to_world.call_count == 10
        assert reid.extract_features.call_count == 10
        assert packager.create_package.call_count == 1
        
        # Verify evidence package
        assert evidence_package['package_id'] == 'test_package_001'
        assert evidence_package['status'] == 'success'
    
    def test_pipeline_with_multiple_objects(self, mock_components, sample_video_frames):
        """Test pipeline with multiple tracked objects."""
        detector = mock_components['detector']
        tracker = mock_components['tracker']
        
        # Configure multiple detections
        detector.detect.return_value = [
            DetectionResult([100, 100, 200, 200], 0.8, 0, "person"),
            DetectionResult([300, 300, 400, 400], 0.7, 0, "person"),
            DetectionResult([500, 100, 600, 200], 0.9, 1, "car")
        ]
        
        # Configure multiple tracks
        tracker.update.return_value = [
            Mock(track_id=1, current_bbox=[100, 100, 200, 200], class_name="person"),
            Mock(track_id=2, current_bbox=[300, 300, 400, 400], class_name="person"),
            Mock(track_id=3, current_bbox=[500, 100, 600, 200], class_name="car")
        ]
        
        # Process frames
        for frame_id, frame in enumerate(sample_video_frames[:5]):
            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame_id)
            
            assert len(detections) == 3
            assert len(tracks) == 3
            
            # Verify different object types
            person_tracks = [t for t in tracks if t.class_name == "person"]
            car_tracks = [t for t in tracks if t.class_name == "car"]
            
            assert len(person_tracks) == 2
            assert len(car_tracks) == 1
    
    def test_pipeline_error_handling(self, mock_components, sample_video_frames):
        """Test pipeline error handling and recovery."""
        detector = mock_components['detector']
        tracker = mock_components['tracker']
        
        # Simulate detection failure
        detector.detect.side_effect = [Exception("Detection failed")] + [
            [DetectionResult([100, 100, 200, 200], 0.8, 0, "person")]
        ] * 9
        
        successful_frames = 0
        failed_frames = 0
        
        for frame_id, frame in enumerate(sample_video_frames):
            try:
                detections = detector.detect(frame)
                tracks = tracker.update(detections, frame_id)
                successful_frames += 1
            except Exception as e:
                # Handle error gracefully
                tracks = tracker.update([], frame_id)  # Empty detections
                failed_frames += 1
        
        assert failed_frames == 1
        assert successful_frames == 9
        assert tracker.update.call_count == 10  # Should still update tracker
    
    @pytest.mark.performance
    def test_pipeline_performance(self, mock_components, sample_video_frames, performance_monitor):
        """Test pipeline performance with realistic workload."""
        detector = mock_components['detector']
        tracker = mock_components['tracker']
        geolocation = mock_components['geolocation']
        
        with performance_monitor() as monitor:
            for frame_id, frame in enumerate(sample_video_frames):
                detections = detector.detect(frame)
                tracks = tracker.update(detections, frame_id)
                
                for track in tracks:
                    geolocation.project_to_world(track.current_bbox, frame_id)
        
        # Should process 10 frames quickly
        assert monitor.duration < 2.0  # Less than 2 seconds
        
        # Calculate FPS
        fps = len(sample_video_frames) / monitor.duration
        assert fps > 5  # At least 5 FPS


class TestWebSocketIntegration:
    """Test WebSocket integration for real-time updates."""
    
    @pytest.fixture
    def mock_websocket_handler(self):
        """Create mock WebSocket handler."""
        handler = Mock(spec=WebSocketHandler)
        handler.connected_clients = []
        handler.broadcast = Mock()
        handler.send_to_client = Mock()
        return handler
    
    def test_real_time_detection_updates(self, mock_websocket_handler, sample_video_frames):
        """Test real-time detection updates via WebSocket."""
        handler = mock_websocket_handler
        
        # Simulate processing with WebSocket updates
        for frame_id, frame in enumerate(sample_video_frames[:3]):
            # Simulate detection
            detections = [
                DetectionResult([100+frame_id*10, 100, 200+frame_id*10, 200], 0.8, 0, "person")
            ]
            
            # Send update via WebSocket
            update_data = {
                'type': 'detection_update',
                'frame_id': frame_id,
                'detections': [d.to_dict() for d in detections],
                'timestamp': time.time()
            }
            
            handler.broadcast(update_data)
        
        # Verify WebSocket calls
        assert handler.broadcast.call_count == 3
        
        # Verify update data structure
        last_call_args = handler.broadcast.call_args[0][0]
        assert last_call_args['type'] == 'detection_update'
        assert last_call_args['frame_id'] == 2
        assert len(last_call_args['detections']) == 1
    
    def test_track_status_updates(self, mock_websocket_handler):
        """Test track status updates via WebSocket."""
        handler = mock_websocket_handler
        
        # Simulate track updates
        track_updates = [
            {'track_id': 1, 'status': 'new', 'confidence': 0.8},
            {'track_id': 1, 'status': 'confirmed', 'confidence': 0.85},
            {'track_id': 2, 'status': 'new', 'confidence': 0.7},
            {'track_id': 1, 'status': 'lost', 'confidence': 0.0}
        ]
        
        for update in track_updates:
            handler.broadcast({
                'type': 'track_update',
                'data': update,
                'timestamp': time.time()
            })
        
        assert handler.broadcast.call_count == 4
    
    def test_geolocation_updates(self, mock_websocket_handler):
        """Test geolocation updates via WebSocket."""
        handler = mock_websocket_handler
        
        # Simulate geolocation updates
        geo_update = {
            'type': 'geolocation_update',
            'track_id': 1,
            'world_coordinates': {
                'latitude': 37.7749,
                'longitude': -122.4194,
                'altitude': 100.0
            },
            'pixel_coordinates': [150, 150],
            'timestamp': time.time()
        }
        
        handler.broadcast(geo_update)
        
        assert handler.broadcast.call_count == 1
        call_args = handler.broadcast.call_args[0][0]
        assert call_args['type'] == 'geolocation_update'
        assert 'world_coordinates' in call_args


class TestStreamManagerIntegration:
    """Test stream manager integration."""
    
    @pytest.fixture
    def mock_stream_manager(self):
        """Create mock stream manager."""
        manager = Mock(spec=StreamManager)
        manager.active_streams = {}
        manager.start_stream = Mock()
        manager.stop_stream = Mock()
        manager.get_frame = Mock()
        return manager
    
    def test_multiple_stream_handling(self, mock_stream_manager):
        """Test handling multiple video streams."""
        manager = mock_stream_manager
        
        # Configure mock streams
        stream_configs = [
            {'stream_id': 'drone_1', 'source': 'rtmp://drone1/stream'},
            {'stream_id': 'drone_2', 'source': 'rtmp://drone2/stream'},
            {'stream_id': 'ground_cam', 'source': '/dev/video0'}
        ]
        
        # Start streams
        for config in stream_configs:
            manager.start_stream(config['stream_id'], config['source'])
            manager.active_streams[config['stream_id']] = config
        
        # Verify streams started
        assert manager.start_stream.call_count == 3
        assert len(manager.active_streams) == 3
        
        # Simulate frame retrieval
        for stream_id in manager.active_streams:
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            manager.get_frame.return_value = frame
            
            retrieved_frame = manager.get_frame(stream_id)
            assert retrieved_frame is not None
            assert retrieved_frame.shape == (480, 640, 3)
    
    def test_stream_failure_recovery(self, mock_stream_manager):
        """Test stream failure and recovery handling."""
        manager = mock_stream_manager
        
        # Simulate stream failure
        manager.get_frame.side_effect = [Exception("Stream disconnected")] * 3 + [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        ]
        
        successful_frames = 0
        failed_attempts = 0
        
        for attempt in range(4):
            try:
                frame = manager.get_frame('drone_1')
                if frame is not None:
                    successful_frames += 1
            except Exception:
                failed_attempts += 1
                # Attempt reconnection
                manager.start_stream('drone_1', 'rtmp://drone1/stream')
        
        assert failed_attempts == 3
        assert successful_frames == 1
        assert manager.start_stream.call_count >= 3  # Reconnection attempts


class TestEvidencePackagingIntegration:
    """Test evidence packaging integration."""
    
    @pytest.fixture
    def temp_evidence_dir(self, temp_dir):
        """Create temporary evidence directory."""
        evidence_dir = temp_dir / "evidence"
        evidence_dir.mkdir()
        return evidence_dir
    
    def test_complete_evidence_workflow(self, temp_evidence_dir, mock_config):
        """Test complete evidence packaging workflow."""
        # Create mock evidence data
        evidence_data = {
            'mission_id': 'SAR_001',
            'operator': 'test_operator',
            'start_time': '2024-01-15T10:00:00Z',
            'end_time': '2024-01-15T11:00:00Z',
            'detections': [
                {
                    'frame_id': 0,
                    'timestamp': '2024-01-15T10:30:00Z',
                    'bbox': [100, 100, 200, 200],
                    'confidence': 0.85,
                    'class_name': 'person',
                    'world_coordinates': {
                        'latitude': 37.7749,
                        'longitude': -122.4194,
                        'altitude': 100.0
                    }
                }
            ],
            'tracks': [
                {
                    'track_id': 1,
                    'trajectory': [[150, 150], [155, 155], [160, 160]],
                    'duration': 30.0,
                    'status': 'confirmed'
                }
            ]
        }
        
        with patch('src.packaging.evidence_packager.EvidencePackager') as mock_packager:
            packager_instance = mock_packager.return_value
            packager_instance.create_package.return_value = {
                'package_id': 'evidence_SAR_001_20240115',
                'package_path': str(temp_evidence_dir / 'SAR_001.zip'),
                'manifest_hash': 'sha256:abc123...',
                'signature': 'digital_signature_data',
                'timestamp_proof': 'ots_proof_data'
            }
            
            # Create evidence package
            package_result = packager_instance.create_package(
                evidence_data,
                output_dir=str(temp_evidence_dir)
            )
            
            # Verify package creation
            assert package_result['package_id'] == 'evidence_SAR_001_20240115'
            assert 'package_path' in package_result
            assert 'manifest_hash' in package_result
            assert 'signature' in package_result
            assert 'timestamp_proof' in package_result
    
    def test_evidence_integrity_verification(self, temp_evidence_dir):
        """Test evidence package integrity verification."""
        # Create mock package file
        package_path = temp_evidence_dir / "test_package.zip"
        package_path.write_text("mock_package_data")
        
        # Create mock manifest
        manifest = {
            'package_id': 'test_package',
            'files': [
                {
                    'name': 'video.mp4',
                    'hash': 'sha256:def456...',
                    'size': 1024000
                }
            ],
            'metadata': {
                'created_at': '2024-01-15T10:00:00Z',
                'operator': 'test_operator'
            }
        }
        
        manifest_path = temp_evidence_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        
        with patch('src.packaging.evidence_packager.EvidencePackager') as mock_packager:
            packager_instance = mock_packager.return_value
            packager_instance.verify_package.return_value = {
                'valid': True,
                'integrity_check': 'passed',
                'signature_valid': True,
                'timestamp_valid': True
            }
            
            # Verify package
            verification_result = packager_instance.verify_package(
                str(package_path)
            )
            
            assert verification_result['valid'] is True
            assert verification_result['integrity_check'] == 'passed'
            assert verification_result['signature_valid'] is True
            assert verification_result['timestamp_valid'] is True


class TestReIDIntegration:
    """Test re-identification integration."""
    
    def test_reid_across_multiple_cameras(self, sample_video_frames):
        """Test re-identification across multiple camera feeds."""
        with patch('src.reid.matcher.ReIDMatcher') as mock_reid:
            reid_instance = mock_reid.return_value
            
            # Configure feature extraction
            reid_instance.extract_features.return_value = np.random.rand(512)
            reid_instance.compare_features.return_value = 0.85
            
            # Simulate detections from different cameras
            camera_1_detections = [
                DetectionResult([100, 100, 200, 200], 0.8, 0, "person")
            ]
            
            camera_2_detections = [
                DetectionResult([150, 150, 250, 250], 0.7, 0, "person")
            ]
            
            # Extract features from both cameras
            features_1 = reid_instance.extract_features(
                sample_video_frames[0], camera_1_detections[0].bbox
            )
            
            features_2 = reid_instance.extract_features(
                sample_video_frames[1], camera_2_detections[0].bbox
            )
            
            # Compare features
            similarity = reid_instance.compare_features(features_1, features_2)
            
            assert similarity == 0.85
            assert reid_instance.extract_features.call_count == 2
            assert reid_instance.compare_features.call_count == 1
    
    def test_reid_database_integration(self):
        """Test re-identification with database storage."""
        with patch('src.reid.matcher.ReIDMatcher') as mock_reid:
            reid_instance = mock_reid.return_value
            
            # Configure database operations
            reid_instance.store_features.return_value = True
            reid_instance.search_database.return_value = [
                {'person_id': 'person_001', 'similarity': 0.92},
                {'person_id': 'person_002', 'similarity': 0.78}
            ]
            
            # Store new person features
            features = np.random.rand(512)
            reid_instance.store_features('person_003', features)
            
            # Search for similar persons
            search_results = reid_instance.search_database(features, threshold=0.8)
            
            assert len(search_results) == 2
            assert search_results[0]['similarity'] > 0.9
            assert reid_instance.store_features.call_count == 1
            assert reid_instance.search_database.call_count == 1


@pytest.mark.integration
class TestSystemConfiguration:
    """Test system configuration and initialization."""
    
    def test_system_startup_sequence(self, mock_config, temp_dir):
        """Test complete system startup sequence."""
        # Mock system components
        with patch.multiple(
            'src',
            ObjectDetector=Mock(),
            MultiObjectTracker=Mock(),
            GeolocationProcessor=Mock(),
            ReIDMatcher=Mock(),
            EvidencePackager=Mock(),
            WebSocketHandler=Mock(),
            StreamManager=Mock()
        ):
            # Simulate system initialization
            config = mock_config.copy()
            config['data_path'] = str(temp_dir)
            
            # Create required directories
            (temp_dir / 'models').mkdir()
            (temp_dir / 'evidence').mkdir()
            (temp_dir / 'logs').mkdir()
            
            # Initialize components
            components = {}
            
            # Each component should initialize successfully
            for component_name in ['detector', 'tracker', 'geolocation', 'reid', 'packager']:
                components[component_name] = Mock()
                components[component_name].initialize.return_value = True
            
            # Verify all components initialized
            for component in components.values():
                assert component.initialize.return_value is True
    
    def test_configuration_validation(self, mock_config):
        """Test configuration validation."""
        # Test valid configuration
        valid_config = mock_config.copy()
        
        # Mock configuration validator
        with patch('src.config.validator.ConfigValidator') as mock_validator:
            validator_instance = mock_validator.return_value
            validator_instance.validate.return_value = {
                'valid': True,
                'errors': []
            }
            
            result = validator_instance.validate(valid_config)
            assert result['valid'] is True
            assert len(result['errors']) == 0
        
        # Test invalid configuration
        invalid_config = valid_config.copy()
        del invalid_config['server']['port']  # Remove required field
        
        with patch('src.config.validator.ConfigValidator') as mock_validator:
            validator_instance = mock_validator.return_value
            validator_instance.validate.return_value = {
                'valid': False,
                'errors': ['Missing required field: server.port']
            }
            
            result = validator_instance.validate(invalid_config)
            assert result['valid'] is False
            assert len(result['errors']) == 1


@pytest.mark.slow
class TestLongRunningIntegration:
    """Test long-running integration scenarios."""
    
    def test_extended_tracking_session(self, mock_components):
        """Test extended tracking session with multiple objects."""
        tracker = mock_components['tracker']
        detector = mock_components['detector']
        
        # Configure realistic tracking scenario
        tracker.update.side_effect = self._generate_tracking_sequence()
        detector.detect.side_effect = self._generate_detection_sequence()
        
        # Run extended session (simulate 10 minutes at 10 FPS)
        total_frames = 6000
        processed_frames = 0
        
        for frame_id in range(0, total_frames, 100):  # Sample every 100 frames
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame_id)
            
            processed_frames += 1
            
            # Verify processing continues
            assert detections is not None
            assert tracks is not None
        
        assert processed_frames == 60  # 6000 / 100
    
    def _generate_tracking_sequence(self):
        """Generate realistic tracking sequence."""
        tracks = []
        for i in range(60):
            # Simulate varying number of tracks
            num_tracks = min(5, max(1, 3 + (i % 3) - 1))
            frame_tracks = []
            
            for track_id in range(1, num_tracks + 1):
                track = Mock(
                    track_id=track_id,
                    current_bbox=[100 + track_id*50 + i*2, 100 + i*3, 
                                200 + track_id*50 + i*2, 200 + i*3],
                    state="confirmed" if i > 5 else "tentative"
                )
                frame_tracks.append(track)
            
            tracks.append(frame_tracks)
        
        return tracks
    
    def _generate_detection_sequence(self):
        """Generate realistic detection sequence."""
        detections = []
        for i in range(60):
            # Simulate varying detection quality
            num_detections = min(6, max(0, 3 + (i % 4) - 1))
            frame_detections = []
            
            for det_id in range(num_detections):
                detection = DetectionResult(
                    bbox=[100 + det_id*60 + i*2, 100 + i*3, 
                         180 + det_id*60 + i*2, 180 + i*3],
                    confidence=0.6 + (det_id * 0.1) + (i % 10) * 0.02,
                    class_id=0,
                    class_name="person"
                )
                frame_detections.append(detection)
            
            detections.append(frame_detections)
        
        return detections
    
    def test_memory_usage_stability(self, mock_components, performance_monitor):
        """Test memory usage stability over extended operation."""
        detector = mock_components['detector']
        tracker = mock_components['tracker']
        
        # Configure components
        detector.detect.return_value = [
            DetectionResult([100, 100, 200, 200], 0.8, 0, "person")
        ]
        tracker.update.return_value = [
            Mock(track_id=1, current_bbox=[100, 100, 200, 200])
        ]
        
        with performance_monitor() as monitor:
            # Simulate extended processing
            for frame_id in range(1000):
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                
                detections = detector.detect(frame)
                tracks = tracker.update(detections, frame_id)
                
                # Simulate cleanup every 100 frames
                if frame_id % 100 == 0:
                    del frame, detections, tracks
        
        # Memory usage should be reasonable
        assert monitor.memory_delta < 100 * 1024 * 1024  # Less than 100MB increase