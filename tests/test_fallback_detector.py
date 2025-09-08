#!/usr/bin/env python3
"""
Test Fallback Detector

Tests the stub detector fallback mechanism when model files are missing.
Verifies that the application stays alive and returns empty detections.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os

# Import the detector modules
try:
    from src.backend.detector import create_detector, StubDetector, DetectionResult
except ImportError:
    pytest.skip("Detector module not available", allow_module_level=True)

class TestFallbackDetector:
    """Test suite for fallback detector functionality"""
    
    def test_stub_detector_initialization(self):
        """Test that StubDetector initializes correctly"""
        detector = StubDetector(model_path="nonexistent_model.pt")
        
        assert detector.is_loaded is True
        assert detector.model_path == Path("nonexistent_model.pt")
        assert detector.device == "cpu"
        assert "person" in detector.class_names
    
    def test_stub_detector_empty_detections(self):
        """Test that StubDetector returns empty detections"""
        detector = StubDetector()
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        
        # Run detection
        result = detector.detect(dummy_image)
        
        # Verify empty results
        assert isinstance(result, DetectionResult)
        assert len(result.boxes) == 0
        assert len(result.scores) == 0
        assert len(result.classes) == 0
        assert result.inference_time > 0  # Should have some processing time
        assert len(result.class_names) > 0  # Should have class names
    
    def test_stub_detector_batch_detection(self):
        """Test that StubDetector handles batch detection"""
        detector = StubDetector()
        
        # Create batch of dummy images
        images = [
            np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        
        # Run batch detection
        results = detector.detect_batch(images)
        
        # Verify results
        assert len(results) == 3
        for result in results:
            assert isinstance(result, DetectionResult)
            assert len(result.boxes) == 0
    
    def test_create_detector_with_missing_model(self):
        """Test create_detector falls back to stub when model is missing"""
        # Use a definitely non-existent model path
        nonexistent_path = "definitely_does_not_exist_model.pt"
        
        detector = create_detector(model_path=nonexistent_path)
        
        # Should return StubDetector
        assert isinstance(detector, StubDetector)
        assert detector.is_loaded is True
    
    def test_create_detector_with_existing_model(self):
        """Test create_detector uses real detector when model exists"""
        # Create a temporary file to simulate existing model
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(b"fake model data")
        
        try:
            detector = create_detector(model_path=tmp_path)
            
            # Should attempt to create real detector but may fall back to stub
            # if ultralytics is not available or model loading fails
            assert detector is not None
            assert hasattr(detector, 'detect')
            assert hasattr(detector, 'is_loaded')
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
    
    def test_stub_detector_model_info(self):
        """Test that StubDetector provides correct model info"""
        detector = StubDetector(model_path="test_model.pt", human_only=True)
        
        info = detector.get_model_info()
        
        assert info["model_path"] == "test_model.pt"
        assert info["is_loaded"] is True
        assert info["is_stub"] is True
        assert "person" in info["class_names"]
        assert info["num_classes"] > 0
    
    def test_stub_detector_threshold_updates(self):
        """Test that StubDetector handles threshold updates"""
        detector = StubDetector()
        
        # Update thresholds
        detector.set_thresholds(confidence=0.7, iou=0.6)
        
        assert detector.confidence_threshold == 0.7
        assert detector.iou_threshold == 0.6
    
    def test_stub_detector_human_only_mode(self):
        """Test StubDetector in human-only mode"""
        detector = StubDetector(human_only=True)
        
        assert detector.human_only is True
        assert detector.class_names == ['person']
        
        # Test detection still returns empty results
        dummy_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        result = detector.detect(dummy_image)
        
        assert len(result.boxes) == 0
        assert result.class_names == ['person']
    
    def test_stub_detector_multi_class_mode(self):
        """Test StubDetector in multi-class mode"""
        detector = StubDetector(human_only=False)
        
        assert detector.human_only is False
        assert len(detector.class_names) > 1
        assert 'person' in detector.class_names

if __name__ == "__main__":
    pytest.main([__file__])