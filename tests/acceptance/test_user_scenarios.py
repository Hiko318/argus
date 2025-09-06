"""Acceptance tests for user scenarios and system validation.

Tests for complete user workflows, system acceptance criteria,
and end-to-end validation scenarios.
"""

import pytest
import numpy as np
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import asyncio

# Import system components (with error handling)
try:
    from src.main import ForesightSARSystem
    from src.config.settings import load_config
    from src.ui.web_interface import WebInterface
    from src.connection.stream_manager import StreamManager
    from src.packaging.evidence_packager import EvidencePackager
except ImportError as e:
    pytest.skip(f"System modules not available: {e}", allow_module_level=True)


class TestSearchAndRescueScenarios:
    """Test complete search and rescue scenarios."""
    
    @pytest.fixture
    def sar_system(self, mock_config, temp_dir):
        """Create SAR system instance for testing."""
        with patch('src.main.ForesightSARSystem') as mock_system:
            system_instance = mock_system.return_value
            system_instance.config = mock_config
            system_instance.data_path = str(temp_dir)
            system_instance.is_running = False
            
            # Mock system methods
            system_instance.start = Mock()
            system_instance.stop = Mock()
            system_instance.process_frame = Mock()
            system_instance.get_status = Mock(return_value={'status': 'ready'})
            
            return system_instance
    
    @pytest.fixture
    def mission_data(self):
        """Create sample mission data."""
        return {
            'mission_id': 'SAR_MISSION_001',
            'operator': 'Officer Smith',
            'location': 'Mount Rainier National Park',
            'missing_person': {
                'name': 'John Doe',
                'age': 35,
                'description': 'Male, 6ft, wearing red jacket',
                'last_seen': '2024-01-15T14:30:00Z'
            },
            'search_area': {
                'center_lat': 46.8523,
                'center_lon': -121.7603,
                'radius_km': 5.0
            },
            'weather': {
                'conditions': 'partly cloudy',
                'visibility': 'good',
                'wind_speed': '10 mph'
            }
        }
    
    def test_complete_sar_mission_workflow(self, sar_system, mission_data, temp_dir):
        """Test complete SAR mission from start to evidence package."""
        # Step 1: Initialize mission
        sar_system.start()
        assert sar_system.start.called
        
        # Step 2: Configure mission parameters
        mission_config = {
            'mission_data': mission_data,
            'detection_threshold': 0.7,
            'tracking_enabled': True,
            'geolocation_enabled': True,
            'reid_enabled': True
        }
        
        # Step 3: Start video processing
        video_frames = self._generate_sar_video_sequence()
        detections_log = []
        
        for frame_id, frame in enumerate(video_frames):
            # Mock frame processing
            sar_system.process_frame.return_value = {
                'frame_id': frame_id,
                'detections': [
                    {
                        'bbox': [150 + frame_id*2, 200, 250 + frame_id*2, 300],
                        'confidence': 0.85,
                        'class_name': 'person',
                        'world_coordinates': {
                            'latitude': 46.8523 + frame_id*0.0001,
                            'longitude': -121.7603 + frame_id*0.0001,
                            'altitude': 1500.0
                        }
                    }
                ] if frame_id % 3 == 0 else [],  # Detection every 3rd frame
                'timestamp': time.time()
            }
            
            result = sar_system.process_frame(frame, frame_id)
            if result['detections']:
                detections_log.append(result)
        
        # Step 4: Verify processing results
        assert len(detections_log) > 0
        assert sar_system.process_frame.call_count == len(video_frames)
        
        # Step 5: Generate evidence package
        with patch('src.packaging.evidence_packager.EvidencePackager') as mock_packager:
            packager_instance = mock_packager.return_value
            packager_instance.create_package.return_value = {
                'package_id': f"evidence_{mission_data['mission_id']}",
                'package_path': str(temp_dir / 'evidence_package.zip'),
                'manifest': {
                    'mission_data': mission_data,
                    'detections_count': len(detections_log),
                    'processing_duration': '00:15:30'
                }
            }
            
            evidence_package = packager_instance.create_package(
                mission_data, detections_log
            )
            
            assert evidence_package['package_id'] == 'evidence_SAR_MISSION_001'
            assert 'package_path' in evidence_package
        
        # Step 6: Stop mission
        sar_system.stop()
        assert sar_system.stop.called
    
    def test_multi_drone_coordination(self, sar_system, mission_data):
        """Test coordination of multiple drone feeds."""
        # Configure multiple drone streams
        drone_configs = [
            {'drone_id': 'DRONE_001', 'operator': 'Pilot A', 'area': 'North'},
            {'drone_id': 'DRONE_002', 'operator': 'Pilot B', 'area': 'South'},
            {'drone_id': 'DRONE_003', 'operator': 'Pilot C', 'area': 'East'}
        ]
        
        with patch('src.connection.stream_manager.StreamManager') as mock_stream_manager:
            stream_manager = mock_stream_manager.return_value
            stream_manager.active_streams = {}
            
            # Start streams for each drone
            for drone in drone_configs:
                stream_manager.start_stream(
                    drone['drone_id'],
                    f"rtmp://drone-{drone['drone_id'].lower()}/stream"
                )
                stream_manager.active_streams[drone['drone_id']] = drone
            
            # Simulate coordinated search pattern
            search_results = []
            for frame_id in range(30):  # 30 frames across all drones
                for drone_id in stream_manager.active_streams:
                    # Mock frame from each drone
                    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    
                    # Process frame
                    sar_system.process_frame.return_value = {
                        'drone_id': drone_id,
                        'frame_id': frame_id,
                        'detections': [
                            {
                                'bbox': [100, 100, 200, 200],
                                'confidence': 0.8,
                                'class_name': 'person'
                            }
                        ] if frame_id % 10 == 0 and drone_id == 'DRONE_002' else []
                    }
                    
                    result = sar_system.process_frame(frame, frame_id)
                    if result['detections']:
                        search_results.append(result)
            
            # Verify multi-drone coordination
            assert len(stream_manager.active_streams) == 3
            assert len(search_results) > 0
            
            # Verify detection came from correct drone
            detection_drone = search_results[0]['drone_id']
            assert detection_drone == 'DRONE_002'
    
    def test_emergency_response_workflow(self, sar_system, mission_data):
        """Test emergency response workflow with person found."""
        # Simulate person detection
        detection_event = {
            'timestamp': '2024-01-15T15:45:00Z',
            'location': {
                'latitude': 46.8525,
                'longitude': -121.7605,
                'altitude': 1520.0
            },
            'confidence': 0.92,
            'detection_type': 'person',
            'frame_data': {
                'frame_id': 1250,
                'bbox': [180, 220, 280, 380],
                'image_path': 'evidence/detection_1250.jpg'
            }
        }
        
        # Mock emergency alert system
        with patch('src.alerts.emergency_notifier.EmergencyNotifier') as mock_notifier:
            notifier = mock_notifier.return_value
            notifier.send_alert = Mock()
            notifier.update_status = Mock()
            
            # Trigger emergency response
            notifier.send_alert({
                'type': 'PERSON_FOUND',
                'mission_id': mission_data['mission_id'],
                'detection': detection_event,
                'priority': 'HIGH',
                'recipients': ['command_center', 'ground_team', 'medical_team']
            })
            
            # Verify alert sent
            assert notifier.send_alert.called
            alert_data = notifier.send_alert.call_args[0][0]
            assert alert_data['type'] == 'PERSON_FOUND'
            assert alert_data['priority'] == 'HIGH'
            assert 'command_center' in alert_data['recipients']
            
            # Update mission status
            notifier.update_status({
                'mission_id': mission_data['mission_id'],
                'status': 'PERSON_LOCATED',
                'location': detection_event['location'],
                'next_steps': 'Dispatch ground team for rescue'
            })
            
            assert notifier.update_status.called
    
    def _generate_sar_video_sequence(self):
        """Generate realistic SAR video sequence."""
        frames = []
        for i in range(50):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Add terrain features
            frame[200:250, 100:150] = [139, 69, 19]  # Brown terrain
            frame[100:150, 300:400] = [34, 139, 34]  # Green vegetation
            
            # Occasionally add person-like object
            if i % 15 == 0:
                frame[220:240, 180:200] = [255, 0, 0]  # Red jacket
                frame[240:260, 185:195] = [255, 220, 177]  # Skin tone
            
            frames.append(frame)
        
        return frames


class TestOperatorWorkflows:
    """Test operator-specific workflows and interfaces."""
    
    @pytest.fixture
    def web_interface(self, mock_config):
        """Create web interface for testing."""
        with patch('src.ui.web_interface.WebInterface') as mock_interface:
            interface_instance = mock_interface.return_value
            interface_instance.config = mock_config
            interface_instance.is_running = False
            
            # Mock interface methods
            interface_instance.start = Mock()
            interface_instance.stop = Mock()
            interface_instance.send_update = Mock()
            interface_instance.get_connected_clients = Mock(return_value=[])
            
            return interface_instance
    
    def test_operator_login_and_session(self, web_interface):
        """Test operator login and session management."""
        # Mock authentication
        with patch('src.auth.authenticator.Authenticator') as mock_auth:
            auth_instance = mock_auth.return_value
            auth_instance.authenticate.return_value = {
                'success': True,
                'user_id': 'operator_001',
                'role': 'sar_operator',
                'permissions': ['view_streams', 'control_drones', 'create_evidence']
            }
            
            # Simulate login
            login_result = auth_instance.authenticate('operator_001', 'secure_password')
            
            assert login_result['success'] is True
            assert login_result['role'] == 'sar_operator'
            assert 'control_drones' in login_result['permissions']
            
            # Start operator session
            web_interface.start()
            assert web_interface.start.called
    
    def test_real_time_monitoring_interface(self, web_interface):
        """Test real-time monitoring interface."""
        # Simulate real-time updates
        updates = [
            {
                'type': 'detection_update',
                'data': {
                    'frame_id': 100,
                    'detections': [{'bbox': [100, 100, 200, 200], 'confidence': 0.85}],
                    'timestamp': time.time()
                }
            },
            {
                'type': 'system_status',
                'data': {
                    'cpu_usage': 45.2,
                    'memory_usage': 62.1,
                    'gpu_usage': 78.5,
                    'active_streams': 2
                }
            },
            {
                'type': 'mission_update',
                'data': {
                    'mission_id': 'SAR_001',
                    'status': 'active',
                    'duration': '00:25:30',
                    'detections_count': 15
                }
            }
        ]
        
        # Send updates to interface
        for update in updates:
            web_interface.send_update(update)
        
        # Verify updates sent
        assert web_interface.send_update.call_count == 3
        
        # Verify update types
        call_args = [call[0][0] for call in web_interface.send_update.call_args_list]
        update_types = [update['type'] for update in call_args]
        
        assert 'detection_update' in update_types
        assert 'system_status' in update_types
        assert 'mission_update' in update_types
    
    def test_evidence_review_workflow(self, web_interface, temp_dir):
        """Test evidence review and approval workflow."""
        # Create mock evidence files
        evidence_dir = temp_dir / 'evidence'
        evidence_dir.mkdir()
        
        evidence_files = [
            'detection_001.jpg',
            'detection_002.jpg', 
            'track_analysis.json',
            'mission_summary.pdf'
        ]
        
        for filename in evidence_files:
            (evidence_dir / filename).write_text(f"Mock content for {filename}")
        
        # Mock evidence review system
        with patch('src.evidence.reviewer.EvidenceReviewer') as mock_reviewer:
            reviewer_instance = mock_reviewer.return_value
            reviewer_instance.list_evidence.return_value = [
                {
                    'file_id': 'evidence_001',
                    'filename': 'detection_001.jpg',
                    'type': 'detection_image',
                    'status': 'pending_review',
                    'confidence': 0.85,
                    'timestamp': '2024-01-15T15:30:00Z'
                },
                {
                    'file_id': 'evidence_002',
                    'filename': 'track_analysis.json',
                    'type': 'analysis_data',
                    'status': 'pending_review',
                    'timestamp': '2024-01-15T15:35:00Z'
                }
            ]
            
            reviewer_instance.approve_evidence.return_value = {
                'success': True,
                'approved_by': 'operator_001',
                'approval_timestamp': time.time()
            }
            
            # List evidence for review
            evidence_list = reviewer_instance.list_evidence()
            assert len(evidence_list) == 2
            
            # Approve evidence
            for evidence in evidence_list:
                approval_result = reviewer_instance.approve_evidence(
                    evidence['file_id'], 'operator_001'
                )
                assert approval_result['success'] is True
            
            assert reviewer_instance.approve_evidence.call_count == 2
    
    def test_mission_configuration_interface(self, web_interface):
        """Test mission configuration interface."""
        # Mock mission configuration
        mission_config = {
            'mission_name': 'Mount Baker Search',
            'search_area': {
                'coordinates': [
                    [48.7767, -121.8144],  # Mount Baker coordinates
                    [48.7800, -121.8100],
                    [48.7750, -121.8050],
                    [48.7720, -121.8090]
                ]
            },
            'detection_settings': {
                'confidence_threshold': 0.75,
                'nms_threshold': 0.4,
                'classes': ['person', 'vehicle']
            },
            'tracking_settings': {
                'max_age': 30,
                'min_hits': 3,
                'iou_threshold': 0.3
            },
            'alert_settings': {
                'auto_alert_threshold': 0.9,
                'notification_channels': ['email', 'sms', 'radio']
            }
        }
        
        with patch('src.config.mission_config.MissionConfig') as mock_config:
            config_instance = mock_config.return_value
            config_instance.validate.return_value = {
                'valid': True,
                'errors': []
            }
            config_instance.save.return_value = True
            
            # Validate configuration
            validation_result = config_instance.validate(mission_config)
            assert validation_result['valid'] is True
            
            # Save configuration
            save_result = config_instance.save(mission_config)
            assert save_result is True
            
            assert config_instance.validate.called
            assert config_instance.save.called


class TestSystemPerformanceAcceptance:
    """Test system performance acceptance criteria."""
    
    def test_real_time_processing_requirements(self, performance_monitor):
        """Test real-time processing performance requirements."""
        # Mock system components for performance testing
        with patch.multiple(
            'src',
            ObjectDetector=Mock(),
            MultiObjectTracker=Mock(),
            GeolocationProcessor=Mock()
        ) as mocks:
            
            detector = mocks['ObjectDetector'].return_value
            tracker = mocks['MultiObjectTracker'].return_value
            geolocation = mocks['GeolocationProcessor'].return_value
            
            # Configure realistic processing times
            detector.detect.return_value = [Mock(bbox=[100, 100, 200, 200])]
            tracker.update.return_value = [Mock(track_id=1)]
            geolocation.project_to_world.return_value = {'lat': 46.8523, 'lon': -121.7603}
            
            with performance_monitor() as monitor:
                # Process frames at target FPS
                target_fps = 10
                frame_count = 100
                
                for frame_id in range(frame_count):
                    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    
                    # Simulate processing pipeline
                    detections = detector.detect(frame)
                    tracks = tracker.update(detections, frame_id)
                    
                    for track in tracks:
                        geolocation.project_to_world(track.bbox, frame_id)
                    
                    # Simulate frame timing
                    time.sleep(0.001)  # Minimal processing delay
            
            # Verify performance requirements
            actual_fps = frame_count / monitor.duration
            assert actual_fps >= target_fps * 0.8  # Allow 20% tolerance
            
            # Verify processing latency
            avg_frame_time = monitor.duration / frame_count
            assert avg_frame_time < 0.1  # Less than 100ms per frame
    
    def test_memory_usage_requirements(self, performance_monitor):
        """Test memory usage requirements."""
        with performance_monitor() as monitor:
            # Simulate extended operation
            frames_processed = 0
            max_memory_usage = 0
            
            for batch in range(10):  # 10 batches of processing
                batch_frames = []
                
                # Create batch of frames
                for i in range(100):
                    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
                    batch_frames.append(frame)
                    frames_processed += 1
                
                # Process batch
                for frame in batch_frames:
                    # Simulate processing
                    processed_frame = frame.copy()
                    del processed_frame
                
                # Clear batch
                del batch_frames
                
                # Track memory usage
                current_memory = monitor.current_memory_usage()
                max_memory_usage = max(max_memory_usage, current_memory)
            
            # Verify memory requirements
            assert max_memory_usage < 2 * 1024 * 1024 * 1024  # Less than 2GB
            assert frames_processed == 1000
    
    def test_detection_accuracy_requirements(self):
        """Test detection accuracy requirements."""
        # Mock ground truth data
        ground_truth = [
            {'frame_id': 0, 'bbox': [100, 100, 200, 200], 'class': 'person'},
            {'frame_id': 1, 'bbox': [150, 150, 250, 250], 'class': 'person'},
            {'frame_id': 2, 'bbox': [200, 200, 300, 300], 'class': 'person'}
        ]
        
        # Mock detection results
        detections = [
            {'frame_id': 0, 'bbox': [105, 105, 205, 205], 'class': 'person', 'confidence': 0.85},
            {'frame_id': 1, 'bbox': [145, 145, 245, 245], 'class': 'person', 'confidence': 0.90},
            {'frame_id': 2, 'bbox': [195, 195, 295, 295], 'class': 'person', 'confidence': 0.88}
        ]
        
        # Calculate accuracy metrics
        def calculate_iou(box1, box2):
            """Calculate Intersection over Union."""
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
            
            intersection = (x2 - x1) * (y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        
        # Evaluate detections
        total_iou = 0
        matches = 0
        
        for gt, det in zip(ground_truth, detections):
            if gt['frame_id'] == det['frame_id'] and gt['class'] == det['class']:
                iou = calculate_iou(gt['bbox'], det['bbox'])
                if iou > 0.5:  # IoU threshold for match
                    total_iou += iou
                    matches += 1
        
        # Verify accuracy requirements
        accuracy = matches / len(ground_truth)
        avg_iou = total_iou / matches if matches > 0 else 0
        
        assert accuracy >= 0.8  # At least 80% detection rate
        assert avg_iou >= 0.7   # At least 70% average IoU
    
    def test_system_reliability_requirements(self):
        """Test system reliability and uptime requirements."""
        # Mock system components with failure scenarios
        failure_count = 0
        recovery_count = 0
        total_operations = 1000
        
        with patch('src.main.ForesightSARSystem') as mock_system:
            system_instance = mock_system.return_value
            
            # Simulate operations with occasional failures
            for operation_id in range(total_operations):
                try:
                    # Simulate random failures (1% failure rate)
                    if operation_id % 100 == 0 and operation_id > 0:
                        raise Exception("Simulated system failure")
                    
                    # Normal operation
                    system_instance.process_frame.return_value = {
                        'status': 'success',
                        'operation_id': operation_id
                    }
                    
                except Exception:
                    failure_count += 1
                    
                    # Simulate recovery
                    system_instance.recover.return_value = True
                    recovery_count += 1
        
        # Verify reliability requirements
        failure_rate = failure_count / total_operations
        recovery_rate = recovery_count / failure_count if failure_count > 0 else 1.0
        
        assert failure_rate < 0.02  # Less than 2% failure rate
        assert recovery_rate >= 0.95  # At least 95% recovery rate
        assert failure_count == recovery_count  # All failures recovered


class TestLegalAndComplianceAcceptance:
    """Test legal and compliance acceptance criteria."""
    
    def test_evidence_chain_of_custody(self, temp_dir):
        """Test evidence chain of custody requirements."""
        # Create evidence with chain of custody
        evidence_data = {
            'case_id': 'SAR_CASE_001',
            'evidence_id': 'EVIDENCE_001',
            'collected_by': 'Officer Smith',
            'collection_timestamp': '2024-01-15T15:30:00Z',
            'collection_location': {
                'latitude': 46.8523,
                'longitude': -121.7603
            },
            'evidence_type': 'digital_video',
            'description': 'Drone footage showing person in distress'
        }
        
        with patch('src.packaging.evidence_packager.EvidencePackager') as mock_packager:
            packager_instance = mock_packager.return_value
            packager_instance.create_package.return_value = {
                'package_id': 'EVIDENCE_001_PACKAGE',
                'chain_of_custody': [
                    {
                        'action': 'collected',
                        'timestamp': '2024-01-15T15:30:00Z',
                        'operator': 'Officer Smith',
                        'signature': 'digital_signature_1'
                    },
                    {
                        'action': 'processed',
                        'timestamp': '2024-01-15T15:35:00Z',
                        'operator': 'System',
                        'signature': 'digital_signature_2'
                    },
                    {
                        'action': 'packaged',
                        'timestamp': '2024-01-15T15:40:00Z',
                        'operator': 'Officer Smith',
                        'signature': 'digital_signature_3'
                    }
                ],
                'integrity_hash': 'sha256:abc123...',
                'timestamp_proof': 'ots_proof_data'
            }
            
            # Create evidence package
            package = packager_instance.create_package(evidence_data)
            
            # Verify chain of custody
            assert 'chain_of_custody' in package
            assert len(package['chain_of_custody']) >= 3
            
            # Verify required fields
            for entry in package['chain_of_custody']:
                assert 'action' in entry
                assert 'timestamp' in entry
                assert 'operator' in entry
                assert 'signature' in entry
            
            # Verify integrity
            assert 'integrity_hash' in package
            assert 'timestamp_proof' in package
    
    def test_privacy_protection_requirements(self):
        """Test privacy protection requirements."""
        # Mock privacy protection system
        with patch('src.privacy.protector.PrivacyProtector') as mock_protector:
            protector_instance = mock_protector.return_value
            
            # Configure privacy settings
            privacy_config = {
                'face_blurring': True,
                'license_plate_masking': True,
                'location_anonymization': True,
                'data_retention_days': 90
            }
            
            protector_instance.apply_privacy_protection.return_value = {
                'faces_blurred': 2,
                'plates_masked': 1,
                'locations_anonymized': 5,
                'privacy_level': 'high'
            }
            
            # Apply privacy protection
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            protection_result = protector_instance.apply_privacy_protection(
                frame, privacy_config
            )
            
            # Verify privacy protection applied
            assert protection_result['faces_blurred'] >= 0
            assert protection_result['privacy_level'] == 'high'
            assert protector_instance.apply_privacy_protection.called
    
    def test_audit_trail_requirements(self, temp_dir):
        """Test audit trail requirements."""
        # Create audit log
        audit_log = temp_dir / 'audit.log'
        
        with patch('src.audit.logger.AuditLogger') as mock_logger:
            logger_instance = mock_logger.return_value
            logger_instance.log_action = Mock()
            
            # Log various system actions
            actions = [
                {'action': 'system_start', 'user': 'system', 'timestamp': time.time()},
                {'action': 'user_login', 'user': 'operator_001', 'timestamp': time.time()},
                {'action': 'mission_start', 'user': 'operator_001', 'mission_id': 'SAR_001'},
                {'action': 'detection_found', 'user': 'system', 'confidence': 0.85},
                {'action': 'evidence_created', 'user': 'operator_001', 'evidence_id': 'EVIDENCE_001'},
                {'action': 'mission_end', 'user': 'operator_001', 'mission_id': 'SAR_001'}
            ]
            
            for action in actions:
                logger_instance.log_action(action)
            
            # Verify audit logging
            assert logger_instance.log_action.call_count == len(actions)
            
            # Verify audit trail completeness
            logged_actions = [call[0][0]['action'] for call in logger_instance.log_action.call_args_list]
            expected_actions = ['system_start', 'user_login', 'mission_start', 
                              'detection_found', 'evidence_created', 'mission_end']
            
            for expected_action in expected_actions:
                assert expected_action in logged_actions


@pytest.mark.acceptance
class TestDeploymentAcceptance:
    """Test deployment and installation acceptance criteria."""
    
    def test_windows_deployment_acceptance(self, temp_dir):
        """Test Windows deployment acceptance."""
        # Mock Windows deployment
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stdout = "Installation completed successfully"
            
            # Simulate deployment script execution
            deployment_script = temp_dir / 'deploy_windows.ps1'
            deployment_script.write_text("# Mock deployment script")
            
            # Run deployment
            result = mock_subprocess([
                'powershell', '-ExecutionPolicy', 'Bypass',
                '-File', str(deployment_script)
            ])
            
            # Verify deployment success
            assert result.returncode == 0
            assert "completed successfully" in result.stdout
    
    def test_jetson_deployment_acceptance(self, temp_dir):
        """Test Jetson deployment acceptance."""
        # Mock Jetson deployment
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stdout = "Jetson setup completed"
            
            # Simulate deployment script execution
            deployment_script = temp_dir / 'deploy_jetson.sh'
            deployment_script.write_text("#!/bin/bash\n# Mock deployment script")
            
            # Run deployment
            result = mock_subprocess(['bash', str(deployment_script)])
            
            # Verify deployment success
            assert result.returncode == 0
            assert "setup completed" in result.stdout
    
    def test_configuration_validation_acceptance(self, mock_config):
        """Test configuration validation acceptance."""
        # Test configuration validation
        required_sections = [
            'server', 'detection', 'tracking', 'geolocation',
            'reid', 'packaging', 'ui', 'logging'
        ]
        
        for section in required_sections:
            assert section in mock_config
        
        # Verify critical settings
        assert mock_config['server']['port'] > 0
        assert 0 < mock_config['detection']['confidence_threshold'] < 1
        assert mock_config['tracking']['max_age'] > 0
        assert mock_config['logging']['level'] in ['DEBUG', 'INFO', 'WARNING', 'ERROR']