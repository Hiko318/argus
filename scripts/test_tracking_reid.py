#!/usr/bin/env python3
"""
Tracking + Re-ID Integration Test for FORESIGHT System

Tests the integration between object tracking and re-identification
to verify that person IDs are maintained correctly across frames.

Usage:
    python scripts/test_tracking_reid.py
    python scripts/test_tracking_reid.py --verbose
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json

try:
    import numpy as np
    import cv2
except ImportError:
    print("Error: OpenCV and NumPy required. Install with: pip install opencv-python numpy")
    sys.exit(1)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from tracking.sort_tracker import SORTTracker, Detection, Track
except ImportError:
    print("Error: Could not import tracking modules")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrackingReIDTester:
    """Test tracking and re-identification integration."""
    
    def __init__(self, verbose: bool = False):
        """Initialize the tester.
        
        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Initialize tracker
        self.tracker = SORTTracker(
            max_disappeared=10,
            min_hits=3,
            iou_threshold=0.3
        )
        
        # Test statistics
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'total_tracks': 0,
            'id_switches': 0,
            'track_lifetimes': [],
            'tracking_accuracy': 0.0
        }
        
        # Track history for analysis
        self.track_history = {}
        self.ground_truth = {}
    
    def create_synthetic_detections(self, frame_id: int) -> List[Detection]:
        """Create synthetic detections for testing.
        
        Args:
            frame_id: Current frame number
            
        Returns:
            List of synthetic detections
        """
        detections = []
        
        # Person 1: Moving right
        if frame_id < 50:  # Present for first 50 frames
            x1 = 100 + frame_id * 5
            y1 = 100
            x2 = x1 + 80
            y2 = y1 + 160
            
            # Add some noise
            noise = np.random.normal(0, 2, 4)
            bbox = (x1 + noise[0], y1 + noise[1], x2 + noise[2], y2 + noise[3])
            
            detection = Detection(
                bbox=bbox,
                confidence=0.8 + np.random.normal(0, 0.1),
                class_id=0,
                features=np.random.rand(512)  # Simulated ReID features
            )
            detections.append(detection)
            
            # Ground truth: this should be track ID 1
            self.ground_truth[frame_id] = self.ground_truth.get(frame_id, {})
            self.ground_truth[frame_id]['person_1'] = bbox
        
        # Person 2: Moving down
        if frame_id >= 10 and frame_id < 60:  # Appears at frame 10
            x1 = 300
            y1 = 50 + (frame_id - 10) * 4
            x2 = x1 + 80
            y2 = y1 + 160
            
            # Add some noise
            noise = np.random.normal(0, 2, 4)
            bbox = (x1 + noise[0], y1 + noise[1], x2 + noise[2], y2 + noise[3])
            
            detection = Detection(
                bbox=bbox,
                confidence=0.75 + np.random.normal(0, 0.1),
                class_id=0,
                features=np.random.rand(512)  # Simulated ReID features
            )
            detections.append(detection)
            
            # Ground truth: this should be track ID 2
            self.ground_truth[frame_id] = self.ground_truth.get(frame_id, {})
            self.ground_truth[frame_id]['person_2'] = bbox
        
        # Person 3: Circular movement
        if frame_id >= 20:  # Appears at frame 20
            center_x, center_y = 500, 300
            radius = 80
            angle = (frame_id - 20) * 0.1
            
            x1 = center_x + radius * np.cos(angle) - 40
            y1 = center_y + radius * np.sin(angle) - 80
            x2 = x1 + 80
            y2 = y1 + 160
            
            # Add some noise
            noise = np.random.normal(0, 2, 4)
            bbox = (x1 + noise[0], y1 + noise[1], x2 + noise[2], y2 + noise[3])
            
            detection = Detection(
                bbox=bbox,
                confidence=0.85 + np.random.normal(0, 0.1),
                class_id=0,
                features=np.random.rand(512)  # Simulated ReID features
            )
            detections.append(detection)
            
            # Ground truth: this should be track ID 3
            self.ground_truth[frame_id] = self.ground_truth.get(frame_id, {})
            self.ground_truth[frame_id]['person_3'] = bbox
        
        # Occasionally miss detections (occlusion simulation)
        if np.random.random() < 0.05:  # 5% chance of missing all detections
            detections = []
        
        return detections
    
    def calculate_iou(self, box1: Tuple[float, float, float, float], 
                     box2: Tuple[float, float, float, float]) -> float:
        """Calculate IoU between two bounding boxes.
        
        Args:
            box1: First bounding box (x1, y1, x2, y2)
            box2: Second bounding box (x1, y1, x2, y2)
            
        Returns:
            IoU value between 0 and 1
        """
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def analyze_tracking_performance(self) -> Dict[str, Any]:
        """Analyze tracking performance against ground truth.
        
        Returns:
            Dictionary with performance metrics
        """
        total_matches = 0
        total_ground_truth = 0
        id_switches = 0
        
        # Track ID mapping (ground truth person -> assigned track ID)
        id_mapping = {}
        
        for frame_id in range(self.stats['total_frames']):
            if frame_id not in self.ground_truth:
                continue
            
            gt_persons = self.ground_truth[frame_id]
            frame_tracks = self.track_history.get(frame_id, [])
            
            total_ground_truth += len(gt_persons)
            
            # Match tracks to ground truth persons
            for person_name, gt_bbox in gt_persons.items():
                best_match = None
                best_iou = 0.0
                
                for track in frame_tracks:
                    iou = self.calculate_iou(gt_bbox, track.bbox)
                    if iou > best_iou and iou > 0.3:  # Minimum IoU threshold
                        best_iou = iou
                        best_match = track
                
                if best_match:
                    total_matches += 1
                    
                    # Check for ID switches
                    if person_name in id_mapping:
                        if id_mapping[person_name] != best_match.id:
                            id_switches += 1
                            logger.warning(f"ID switch detected for {person_name}: "
                                         f"{id_mapping[person_name]} -> {best_match.id} at frame {frame_id}")
                    
                    id_mapping[person_name] = best_match.id
        
        # Calculate metrics
        tracking_accuracy = total_matches / total_ground_truth if total_ground_truth > 0 else 0.0
        
        return {
            'tracking_accuracy': tracking_accuracy,
            'total_matches': total_matches,
            'total_ground_truth': total_ground_truth,
            'id_switches': id_switches,
            'unique_tracks': len(set(id_mapping.values())),
            'ground_truth_persons': len(id_mapping)
        }
    
    def run_test(self, num_frames: int = 70) -> Dict[str, Any]:
        """Run the tracking + ReID integration test.
        
        Args:
            num_frames: Number of frames to simulate
            
        Returns:
            Test results and statistics
        """
        logger.info(f"Starting tracking + ReID integration test ({num_frames} frames)")
        
        start_time = time.time()
        
        for frame_id in range(num_frames):
            # Create synthetic detections
            detections = self.create_synthetic_detections(frame_id)
            
            # Update tracker
            tracks = self.tracker.update(detections)
            
            # Store track history
            self.track_history[frame_id] = tracks.copy()
            
            # Update statistics
            self.stats['total_frames'] += 1
            self.stats['total_detections'] += len(detections)
            
            if self.verbose and frame_id % 10 == 0:
                logger.debug(f"Frame {frame_id}: {len(detections)} detections, {len(tracks)} tracks")
        
        end_time = time.time()
        
        # Analyze performance
        performance = self.analyze_tracking_performance()
        
        # Calculate final statistics
        active_tracks = [t for t in self.tracker.trackers if t.time_since_update < 5]
        self.stats['total_tracks'] = len(self.tracker.trackers)
        self.stats['active_tracks'] = len(active_tracks)
        self.stats['processing_time'] = end_time - start_time
        self.stats['fps'] = num_frames / (end_time - start_time)
        
        # Merge performance metrics
        self.stats.update(performance)
        
        return self.stats
    
    def print_results(self, results: Dict[str, Any]):
        """Print test results in a formatted way.
        
        Args:
            results: Test results dictionary
        """
        print("\n" + "="*60)
        print("TRACKING + RE-ID INTEGRATION TEST RESULTS")
        print("="*60)
        
        print(f"Test Configuration:")
        print(f"  Total Frames: {results['total_frames']}")
        print(f"  Total Detections: {results['total_detections']}")
        print(f"  Processing Time: {results['processing_time']:.2f}s")
        print(f"  Processing FPS: {results['fps']:.1f}")
        
        print(f"\nTracking Performance:")
        print(f"  Total Tracks Created: {results['total_tracks']}")
        print(f"  Active Tracks: {results['active_tracks']}")
        print(f"  Tracking Accuracy: {results['tracking_accuracy']:.1%}")
        print(f"  Total Matches: {results['total_matches']}/{results['total_ground_truth']}")
        
        print(f"\nRe-ID Performance:")
        print(f"  ID Switches: {results['id_switches']}")
        print(f"  Unique Tracks: {results['unique_tracks']}")
        print(f"  Ground Truth Persons: {results['ground_truth_persons']}")
        
        # Determine test status
        success_criteria = [
            results['tracking_accuracy'] >= 0.8,  # 80% tracking accuracy
            results['id_switches'] <= 2,  # Max 2 ID switches
            results['unique_tracks'] <= results['ground_truth_persons'] + 1  # Reasonable track count
        ]
        
        if all(success_criteria):
            print(f"\n✅ TEST PASSED - Tracking + ReID integration working correctly")
            status = "PASSED"
        else:
            print(f"\n❌ TEST FAILED - Issues detected in tracking/ReID integration")
            status = "FAILED"
            
            if results['tracking_accuracy'] < 0.8:
                print(f"   - Low tracking accuracy: {results['tracking_accuracy']:.1%} < 80%")
            if results['id_switches'] > 2:
                print(f"   - Too many ID switches: {results['id_switches']} > 2")
            if results['unique_tracks'] > results['ground_truth_persons'] + 1:
                print(f"   - Too many tracks created: {results['unique_tracks']} > {results['ground_truth_persons'] + 1}")
        
        results['test_status'] = status
        return status

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Test tracking + ReID integration",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--frames', '-f', type=int, default=70,
                       help='Number of frames to simulate (default: 70)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--output', '-o',
                       help='Output file for results (JSON format)')
    
    args = parser.parse_args()
    
    try:
        # Run test
        tester = TrackingReIDTester(verbose=args.verbose)
        results = tester.run_test(args.frames)
        
        # Print results
        status = tester.print_results(results)
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")
        
        # Exit with appropriate code
        sys.exit(0 if status == "PASSED" else 1)
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()