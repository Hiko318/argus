"""Unit tests for tracking module.

Tests for object tracking, trajectory analysis, and tracking utilities.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import time

# Import tracking modules (with error handling for missing dependencies)
try:
    from src.tracking.tracker import MultiObjectTracker, Track, TrackState
    from src.tracking.kalman_filter import KalmanFilter
    from src.tracking.association import HungarianAssociation
    from src.tracking.utils import (
        calculate_track_velocity, smooth_trajectory,
        predict_future_position, calculate_track_metrics
    )
    from src.vision.detector import DetectionResult
except ImportError as e:
    pytest.skip(f"Tracking modules not available: {e}", allow_module_level=True)


class TestTrack:
    """Test cases for Track class."""
    
    def test_track_creation(self, sample_detection):
        """Test track creation from detection."""
        detection = DetectionResult(
            bbox=sample_detection["bbox"],
            confidence=sample_detection["confidence"],
            class_id=sample_detection["class_id"],
            class_name=sample_detection["class_name"]
        )
        
        track = Track(track_id=1, initial_detection=detection, frame_id=0)
        
        assert track.track_id == 1
        assert track.state == TrackState.TENTATIVE
        assert len(track.detections) == 1
        assert track.age == 1
        assert track.time_since_update == 0
    
    def test_track_update(self, sample_detection):
        """Test track update with new detection."""
        detection1 = DetectionResult([100, 100, 200, 200], 0.8, 0, "person")
        detection2 = DetectionResult([105, 105, 205, 205], 0.85, 0, "person")
        
        track = Track(track_id=1, initial_detection=detection1, frame_id=0)
        track.update(detection2, frame_id=1)
        
        assert len(track.detections) == 2
        assert track.age == 2
        assert track.time_since_update == 0
        assert track.current_bbox == [105, 105, 205, 205]
    
    def test_track_prediction(self, sample_detection):
        """Test track state prediction."""
        detection = DetectionResult([100, 100, 200, 200], 0.8, 0, "person")
        track = Track(track_id=1, initial_detection=detection, frame_id=0)
        
        # Add some movement
        for i in range(1, 5):
            new_detection = DetectionResult(
                [100 + i*5, 100 + i*5, 200 + i*5, 200 + i*5],
                0.8, 0, "person"
            )
            track.update(new_detection, frame_id=i)
        
        # Predict next position
        predicted_bbox = track.predict()
        
        # Should predict continued movement
        assert predicted_bbox[0] > 120  # x should continue increasing
        assert predicted_bbox[1] > 120  # y should continue increasing
    
    def test_track_state_transitions(self, sample_detection):
        """Test track state transitions."""
        detection = DetectionResult([100, 100, 200, 200], 0.8, 0, "person")
        track = Track(track_id=1, initial_detection=detection, frame_id=0)
        
        # Initially tentative
        assert track.state == TrackState.TENTATIVE
        
        # Confirm after multiple updates
        for i in range(1, 4):
            new_detection = DetectionResult(
                [100 + i, 100 + i, 200 + i, 200 + i],
                0.8, 0, "person"
            )
            track.update(new_detection, frame_id=i)
        
        assert track.state == TrackState.CONFIRMED
        
        # Mark as deleted after no updates
        for _ in range(10):
            track.mark_missed()
        
        assert track.state == TrackState.DELETED
    
    def test_track_velocity_calculation(self, sample_detection):
        """Test track velocity calculation."""
        detection = DetectionResult([100, 100, 200, 200], 0.8, 0, "person")
        track = Track(track_id=1, initial_detection=detection, frame_id=0)
        
        # Add movement over time
        for i in range(1, 5):
            new_detection = DetectionResult(
                [100 + i*10, 100 + i*5, 200 + i*10, 200 + i*5],
                0.8, 0, "person"
            )
            track.update(new_detection, frame_id=i)
        
        velocity = track.get_velocity()
        
        # Should have positive velocity in both directions
        assert velocity[0] > 0  # x velocity
        assert velocity[1] > 0  # y velocity
    
    def test_track_trajectory(self, sample_detection):
        """Test track trajectory extraction."""
        detection = DetectionResult([100, 100, 200, 200], 0.8, 0, "person")
        track = Track(track_id=1, initial_detection=detection, frame_id=0)
        
        # Add several detections
        positions = [(100, 100), (110, 105), (120, 110), (130, 115)]
        for i, (x, y) in enumerate(positions[1:], 1):
            new_detection = DetectionResult([x, y, x+100, y+100], 0.8, 0, "person")
            track.update(new_detection, frame_id=i)
        
        trajectory = track.get_trajectory()
        
        assert len(trajectory) == 4
        for i, (x, y) in enumerate(positions):
            assert abs(trajectory[i][0] - (x + 50)) < 1  # Center x
            assert abs(trajectory[i][1] - (y + 50)) < 1  # Center y


class TestKalmanFilter:
    """Test cases for KalmanFilter class."""
    
    def test_kalman_filter_initialization(self):
        """Test Kalman filter initialization."""
        kf = KalmanFilter()
        
        assert kf.state_dim == 8  # [x, y, w, h, vx, vy, vw, vh]
        assert kf.measurement_dim == 4  # [x, y, w, h]
        assert kf.state is not None
        assert kf.covariance is not None
    
    def test_kalman_filter_prediction(self):
        """Test Kalman filter prediction step."""
        kf = KalmanFilter()
        
        # Initialize with a detection
        initial_bbox = [100, 100, 200, 200]
        kf.initiate(initial_bbox)
        
        initial_state = kf.state.copy()
        
        # Predict next state
        kf.predict()
        
        # State should change (position updated based on velocity)
        assert not np.array_equal(kf.state, initial_state)
    
    def test_kalman_filter_update(self):
        """Test Kalman filter update step."""
        kf = KalmanFilter()
        
        # Initialize and predict
        initial_bbox = [100, 100, 200, 200]
        kf.initiate(initial_bbox)
        kf.predict()
        
        predicted_state = kf.state.copy()
        
        # Update with measurement
        measurement = [105, 105, 205, 205]
        kf.update(measurement)
        
        # State should be corrected based on measurement
        assert not np.array_equal(kf.state, predicted_state)
        
        # Position should be closer to measurement
        assert abs(kf.state[0] - 155) < abs(predicted_state[0] - 155)  # x center
        assert abs(kf.state[1] - 155) < abs(predicted_state[1] - 155)  # y center
    
    def test_kalman_filter_multiple_updates(self):
        """Test Kalman filter with multiple prediction-update cycles."""
        kf = KalmanFilter()
        
        # Initialize
        kf.initiate([100, 100, 200, 200])
        
        # Simulate tracking with consistent movement
        measurements = [
            [105, 105, 205, 205],
            [110, 110, 210, 210],
            [115, 115, 215, 215],
            [120, 120, 220, 220]
        ]
        
        states = []
        for measurement in measurements:
            kf.predict()
            kf.update(measurement)
            states.append(kf.state.copy())
        
        # Velocity should be learned
        final_velocity = kf.state[4:6]  # vx, vy
        assert final_velocity[0] > 0  # Moving right
        assert final_velocity[1] > 0  # Moving down
    
    def test_kalman_filter_covariance_evolution(self):
        """Test covariance matrix evolution."""
        kf = KalmanFilter()
        kf.initiate([100, 100, 200, 200])
        
        initial_covariance = kf.covariance.copy()
        
        # Prediction should increase uncertainty
        kf.predict()
        predicted_covariance = kf.covariance.copy()
        
        # Update should decrease uncertainty
        kf.update([105, 105, 205, 205])
        updated_covariance = kf.covariance.copy()
        
        # Prediction increases uncertainty
        assert np.trace(predicted_covariance) > np.trace(initial_covariance)
        
        # Update decreases uncertainty
        assert np.trace(updated_covariance) < np.trace(predicted_covariance)


class TestHungarianAssociation:
    """Test cases for HungarianAssociation class."""
    
    def test_association_initialization(self):
        """Test association algorithm initialization."""
        associator = HungarianAssociation(max_distance=50.0)
        
        assert associator.max_distance == 50.0
    
    def test_distance_calculation(self):
        """Test distance calculation between tracks and detections."""
        associator = HungarianAssociation()
        
        # Create mock tracks and detections
        tracks = [
            Mock(current_bbox=[100, 100, 200, 200]),
            Mock(current_bbox=[300, 300, 400, 400])
        ]
        
        detections = [
            DetectionResult([105, 105, 205, 205], 0.8, 0, "person"),
            DetectionResult([295, 295, 395, 395], 0.8, 0, "person"),
            DetectionResult([500, 500, 600, 600], 0.8, 0, "person")
        ]
        
        distance_matrix = associator.calculate_distance_matrix(tracks, detections)
        
        assert distance_matrix.shape == (2, 3)
        
        # First track should be closest to first detection
        assert distance_matrix[0, 0] < distance_matrix[0, 1]
        assert distance_matrix[0, 0] < distance_matrix[0, 2]
        
        # Second track should be closest to second detection
        assert distance_matrix[1, 1] < distance_matrix[1, 0]
        assert distance_matrix[1, 1] < distance_matrix[1, 2]
    
    def test_association_matching(self):
        """Test track-detection association."""
        associator = HungarianAssociation(max_distance=50.0)
        
        # Create tracks and detections
        tracks = [
            Mock(track_id=1, current_bbox=[100, 100, 200, 200]),
            Mock(track_id=2, current_bbox=[300, 300, 400, 400])
        ]
        
        detections = [
            DetectionResult([105, 105, 205, 205], 0.8, 0, "person"),
            DetectionResult([295, 295, 395, 395], 0.8, 0, "person")
        ]
        
        matches, unmatched_tracks, unmatched_detections = associator.associate(
            tracks, detections
        )
        
        assert len(matches) == 2
        assert len(unmatched_tracks) == 0
        assert len(unmatched_detections) == 0
        
        # Check correct matching
        track_indices = [match[0] for match in matches]
        detection_indices = [match[1] for match in matches]
        
        assert 0 in track_indices
        assert 1 in track_indices
        assert 0 in detection_indices
        assert 1 in detection_indices
    
    def test_association_with_unmatched(self):
        """Test association with unmatched tracks and detections."""
        associator = HungarianAssociation(max_distance=30.0)
        
        tracks = [
            Mock(track_id=1, current_bbox=[100, 100, 200, 200])
        ]
        
        detections = [
            DetectionResult([105, 105, 205, 205], 0.8, 0, "person"),  # Close
            DetectionResult([500, 500, 600, 600], 0.8, 0, "person")   # Far
        ]
        
        matches, unmatched_tracks, unmatched_detections = associator.associate(
            tracks, detections
        )
        
        assert len(matches) == 1
        assert len(unmatched_tracks) == 0
        assert len(unmatched_detections) == 1
        assert unmatched_detections[0] == 1  # Far detection


class TestMultiObjectTracker:
    """Test cases for MultiObjectTracker class."""
    
    @pytest.fixture
    def tracker(self):
        """Create MultiObjectTracker instance."""
        return MultiObjectTracker(
            max_disappeared=10,
            max_distance=50.0,
            min_hits=3
        )
    
    def test_tracker_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker.max_disappeared == 10
        assert tracker.max_distance == 50.0
        assert tracker.min_hits == 3
        assert len(tracker.tracks) == 0
        assert tracker.next_track_id == 1
    
    def test_tracker_first_frame(self, tracker):
        """Test tracker with first frame detections."""
        detections = [
            DetectionResult([100, 100, 200, 200], 0.8, 0, "person"),
            DetectionResult([300, 300, 400, 400], 0.8, 0, "person")
        ]
        
        tracks = tracker.update(detections, frame_id=0)
        
        assert len(tracks) == 2
        assert tracker.next_track_id == 3
        
        # All tracks should be tentative initially
        for track in tracks:
            assert track.state == TrackState.TENTATIVE
    
    def test_tracker_track_confirmation(self, tracker):
        """Test track confirmation after multiple detections."""
        # First frame
        detections1 = [
            DetectionResult([100, 100, 200, 200], 0.8, 0, "person")
        ]
        tracker.update(detections1, frame_id=0)
        
        # Subsequent frames with consistent detections
        for i in range(1, 5):
            detections = [
                DetectionResult(
                    [100 + i*5, 100 + i*5, 200 + i*5, 200 + i*5],
                    0.8, 0, "person"
                )
            ]
            tracks = tracker.update(detections, frame_id=i)
        
        # Track should be confirmed
        assert len(tracks) == 1
        assert tracks[0].state == TrackState.CONFIRMED
    
    def test_tracker_track_deletion(self, tracker):
        """Test track deletion after missed detections."""
        # Create and confirm a track
        for i in range(5):
            detections = [
                DetectionResult(
                    [100 + i*5, 100 + i*5, 200 + i*5, 200 + i*5],
                    0.8, 0, "person"
                )
            ]
            tracker.update(detections, frame_id=i)
        
        # No detections for many frames
        for i in range(5, 20):
            tracks = tracker.update([], frame_id=i)
        
        # Track should be deleted
        active_tracks = [t for t in tracker.tracks if t.state != TrackState.DELETED]
        assert len(active_tracks) == 0
    
    def test_tracker_multiple_objects(self, tracker):
        """Test tracking multiple objects simultaneously."""
        # Create multiple objects moving in different directions
        for frame_id in range(10):
            detections = [
                # Object 1: moving right
                DetectionResult(
                    [100 + frame_id*10, 100, 200 + frame_id*10, 200],
                    0.8, 0, "person"
                ),
                # Object 2: moving down
                DetectionResult(
                    [300, 100 + frame_id*10, 400, 200 + frame_id*10],
                    0.8, 0, "person"
                ),
                # Object 3: stationary
                DetectionResult(
                    [500, 500, 600, 600],
                    0.8, 0, "person"
                )
            ]
            tracks = tracker.update(detections, frame_id)
        
        # Should have 3 confirmed tracks
        confirmed_tracks = [t for t in tracks if t.state == TrackState.CONFIRMED]
        assert len(confirmed_tracks) == 3
        
        # Check trajectories
        for track in confirmed_tracks:
            trajectory = track.get_trajectory()
            assert len(trajectory) >= 5  # Should have substantial trajectory
    
    def test_tracker_occlusion_handling(self, tracker):
        """Test tracker behavior during occlusions."""
        # Create a track
        for i in range(5):
            detections = [
                DetectionResult(
                    [100 + i*5, 100, 200 + i*5, 200],
                    0.8, 0, "person"
                )
            ]
            tracker.update(detections, frame_id=i)
        
        # Simulate occlusion (no detections)
        for i in range(5, 8):
            tracker.update([], frame_id=i)
        
        # Object reappears
        for i in range(8, 12):
            detections = [
                DetectionResult(
                    [100 + i*5, 100, 200 + i*5, 200],
                    0.8, 0, "person"
                )
            ]
            tracks = tracker.update(detections, frame_id=i)
        
        # Should maintain the same track
        confirmed_tracks = [t for t in tracks if t.state == TrackState.CONFIRMED]
        assert len(confirmed_tracks) == 1
        assert confirmed_tracks[0].track_id == 1
    
    @pytest.mark.performance
    def test_tracker_performance(self, tracker, performance_monitor):
        """Test tracker performance with many objects."""
        # Create many detections
        detections = [
            DetectionResult(
                [i*50, j*50, i*50+40, j*50+40],
                0.8, 0, "person"
            )
            for i in range(10) for j in range(10)
        ]
        
        with performance_monitor() as monitor:
            for frame_id in range(10):
                tracker.update(detections, frame_id)
        
        # Should complete in reasonable time
        assert monitor.duration < 5.0  # Less than 5 seconds


class TestTrackingUtils:
    """Test cases for tracking utility functions."""
    
    def test_calculate_track_velocity(self):
        """Test track velocity calculation."""
        trajectory = [
            (100, 100),
            (110, 105),
            (120, 110),
            (130, 115)
        ]
        timestamps = [0, 1, 2, 3]
        
        velocity = calculate_track_velocity(trajectory, timestamps)
        
        # Should have positive velocity
        assert velocity[0] > 0  # x velocity
        assert velocity[1] > 0  # y velocity
        
        # Check approximate values
        assert abs(velocity[0] - 10.0) < 1.0  # ~10 pixels/frame in x
        assert abs(velocity[1] - 5.0) < 1.0   # ~5 pixels/frame in y
    
    def test_smooth_trajectory(self):
        """Test trajectory smoothing."""
        # Create noisy trajectory
        trajectory = [
            (100, 100),
            (112, 103),  # Noisy
            (118, 108),  # Noisy
            (135, 112),  # Noisy
            (140, 120)
        ]
        
        smoothed = smooth_trajectory(trajectory, window_size=3)
        
        assert len(smoothed) == len(trajectory)
        
        # Smoothed trajectory should be less noisy
        # Check that middle points are averaged
        assert smoothed[2][0] != trajectory[2][0]  # Should be different
    
    def test_predict_future_position(self):
        """Test future position prediction."""
        trajectory = [
            (100, 100),
            (110, 105),
            (120, 110),
            (130, 115)
        ]
        
        future_pos = predict_future_position(trajectory, steps_ahead=2)
        
        # Should predict continued movement
        assert future_pos[0] > 130  # x should continue increasing
        assert future_pos[1] > 115  # y should continue increasing
        
        # Check approximate prediction
        expected_x = 130 + 2 * 10  # Continue velocity of 10 px/frame
        expected_y = 115 + 2 * 5   # Continue velocity of 5 px/frame
        
        assert abs(future_pos[0] - expected_x) < 5
        assert abs(future_pos[1] - expected_y) < 5
    
    def test_calculate_track_metrics(self):
        """Test track quality metrics calculation."""
        # Create a track with good consistency
        detections = [
            DetectionResult([100+i*10, 100+i*5, 200+i*10, 200+i*5], 0.8+i*0.01, 0, "person")
            for i in range(10)
        ]
        
        metrics = calculate_track_metrics(detections)
        
        assert "length" in metrics
        assert "avg_confidence" in metrics
        assert "confidence_std" in metrics
        assert "velocity_consistency" in metrics
        assert "bbox_stability" in metrics
        
        assert metrics["length"] == 10
        assert 0.8 <= metrics["avg_confidence"] <= 0.9
        assert metrics["confidence_std"] >= 0
    
    @pytest.mark.parametrize("trajectory,expected_velocity", [
        ([(0, 0), (10, 0), (20, 0)], (10.0, 0.0)),
        ([(0, 0), (0, 10), (0, 20)], (0.0, 10.0)),
        ([(0, 0), (10, 10), (20, 20)], (10.0, 10.0)),
        ([(0, 0), (5, 5), (10, 10)], (5.0, 5.0))
    ])
    def test_velocity_calculation_parametrized(self, trajectory, expected_velocity):
        """Test velocity calculation with various trajectories."""
        timestamps = list(range(len(trajectory)))
        velocity = calculate_track_velocity(trajectory, timestamps)
        
        assert abs(velocity[0] - expected_velocity[0]) < 0.1
        assert abs(velocity[1] - expected_velocity[1]) < 0.1


@pytest.mark.integration
class TestTrackingIntegration:
    """Integration tests for tracking components."""
    
    def test_complete_tracking_pipeline(self):
        """Test complete tracking pipeline with detection integration."""
        tracker = MultiObjectTracker()
        
        # Simulate detection results over multiple frames
        detection_sequences = [
            # Frame 0
            [DetectionResult([100, 100, 200, 200], 0.8, 0, "person")],
            # Frame 1
            [DetectionResult([110, 105, 210, 205], 0.85, 0, "person")],
            # Frame 2
            [DetectionResult([120, 110, 220, 210], 0.9, 0, "person")],
            # Frame 3 - new object appears
            [
                DetectionResult([125, 115, 225, 215], 0.8, 0, "person"),
                DetectionResult([300, 300, 400, 400], 0.7, 0, "person")
            ],
            # Frame 4
            [
                DetectionResult([130, 120, 230, 220], 0.85, 0, "person"),
                DetectionResult([305, 305, 405, 405], 0.75, 0, "person")
            ]
        ]
        
        all_tracks = []
        for frame_id, detections in enumerate(detection_sequences):
            tracks = tracker.update(detections, frame_id)
            all_tracks.append(tracks)
        
        # Should have 2 tracks by the end
        final_tracks = all_tracks[-1]
        active_tracks = [t for t in final_tracks if t.state != TrackState.DELETED]
        assert len(active_tracks) == 2
        
        # First track should have longer trajectory
        track1 = min(active_tracks, key=lambda t: t.track_id)
        assert len(track1.detections) == 5
        
        # Second track should have shorter trajectory
        track2 = max(active_tracks, key=lambda t: t.track_id)
        assert len(track2.detections) == 2
    
    def test_tracking_with_kalman_prediction(self):
        """Test tracking with Kalman filter prediction."""
        tracker = MultiObjectTracker()
        
        # Create consistent movement pattern
        for frame_id in range(10):
            if frame_id < 5:
                # Object present
                detections = [
                    DetectionResult(
                        [100 + frame_id*20, 100 + frame_id*10, 
                         200 + frame_id*20, 200 + frame_id*10],
                        0.8, 0, "person"
                    )
                ]
            else:
                # Object temporarily missing (occlusion)
                detections = []
            
            tracks = tracker.update(detections, frame_id)
        
        # Track should still exist and have predicted positions
        active_tracks = [t for t in tracks if t.state != TrackState.DELETED]
        assert len(active_tracks) == 1
        
        track = active_tracks[0]
        # Should have predictions during occlusion
        assert track.age > 5
    
    @pytest.mark.slow
    def test_long_sequence_tracking(self):
        """Test tracking over a long sequence."""
        tracker = MultiObjectTracker(max_disappeared=30)
        
        # Simulate 100 frames with 3 objects
        for frame_id in range(100):
            detections = []
            
            # Object 1: consistent movement
            if frame_id % 10 != 0:  # Occasionally missing
                detections.append(
                    DetectionResult(
                        [50 + frame_id*2, 50, 150 + frame_id*2, 150],
                        0.8, 0, "person"
                    )
                )
            
            # Object 2: circular movement
            import math
            x = 300 + 50 * math.cos(frame_id * 0.1)
            y = 300 + 50 * math.sin(frame_id * 0.1)
            detections.append(
                DetectionResult(
                    [x, y, x+100, y+100],
                    0.7, 0, "person"
                )
            )
            
            # Object 3: appears later
            if frame_id > 50:
                detections.append(
                    DetectionResult(
                        [500, 100 + (frame_id-50)*3, 600, 200 + (frame_id-50)*3],
                        0.9, 0, "person"
                    )
                )
            
            tracks = tracker.update(detections, frame_id)
        
        # Should have 3 confirmed tracks
        confirmed_tracks = [t for t in tracks if t.state == TrackState.CONFIRMED]
        assert len(confirmed_tracks) >= 2  # At least 2 should be confirmed
        
        # Check track lengths
        for track in confirmed_tracks:
            assert len(track.detections) > 10  # Should have substantial history


@pytest.mark.performance
class TestTrackingPerformance:
    """Performance tests for tracking components."""
    
    def test_tracker_scalability(self, performance_monitor):
        """Test tracker performance with increasing number of objects."""
        tracker = MultiObjectTracker()
        
        # Test with increasing number of objects
        object_counts = [10, 50, 100]
        
        for count in object_counts:
            detections = [
                DetectionResult(
                    [i*30, j*30, i*30+25, j*30+25],
                    0.8, 0, "person"
                )
                for i in range(int(count**0.5))
                for j in range(int(count**0.5))
            ][:count]
            
            with performance_monitor() as monitor:
                for frame_id in range(5):
                    tracker.update(detections, frame_id)
            
            # Performance should scale reasonably
            time_per_object = monitor.duration / (count * 5)
            assert time_per_object < 0.01  # Less than 10ms per object per frame
    
    def test_kalman_filter_performance(self, performance_monitor):
        """Test Kalman filter performance."""
        kf = KalmanFilter()
        kf.initiate([100, 100, 200, 200])
        
        with performance_monitor() as monitor:
            for _ in range(1000):
                kf.predict()
                kf.update([105, 105, 205, 205])
        
        # Should complete 1000 cycles quickly
        assert monitor.duration < 1.0  # Less than 1 second
    
    def test_association_performance(self, performance_monitor):
        """Test association algorithm performance."""
        associator = HungarianAssociation()
        
        # Create many tracks and detections
        tracks = [Mock(current_bbox=[i*50, i*50, i*50+40, i*50+40]) for i in range(50)]
        detections = [
            DetectionResult([i*50+5, i*50+5, i*50+45, i*50+45], 0.8, 0, "person")
            for i in range(50)
        ]
        
        with performance_monitor() as monitor:
            for _ in range(10):
                associator.associate(tracks, detections)
        
        # Should complete 10 associations quickly
        assert monitor.duration < 2.0  # Less than 2 seconds