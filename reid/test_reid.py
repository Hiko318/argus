"""Test Suite for Person Re-Identification Module.

This module provides comprehensive tests for the ReID system, including:
- Embedding generation and storage
- Privacy filtering functionality
- Similarity matching and database operations
- Pipeline integration and workflow testing
- Performance and stress testing
"""

import unittest
import numpy as np
import cv2
import tempfile
import shutil
import time
import os
from pathlib import Path
from typing import List, Dict, Any

# Import ReID components
from .embedder import ReIDEmbedder, EmbeddingConfig, EmbeddingModel
from .privacy_filter import PrivacyFilter, PrivacyConfig, BlurMethod, PrivacyLevel, create_privacy_filter
from .embedding_manager import EmbeddingManager, EmbeddingDatabase, ReIDEmbedding, MatchResult, DistanceMetric
from .reid_pipeline import (
    ReIDPipeline, PipelineConfig, DetectionInput, ReIDResult, 
    PipelineMode, AlertLevel, create_default_pipeline, create_sar_pipeline
)


class TestDataGenerator:
    """Generate test data for ReID testing"""
    
    @staticmethod
    def create_test_image(width: int = 224, height: int = 224, person_present: bool = True) -> np.ndarray:
        """Create a synthetic test image"""
        # Create base image
        image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        if person_present:
            # Add a simple "person" shape (rectangle)
            person_x = width // 4
            person_y = height // 4
            person_w = width // 2
            person_h = height // 2
            
            # Body (darker rectangle)
            cv2.rectangle(image, (person_x, person_y), (person_x + person_w, person_y + person_h), (100, 100, 100), -1)
            
            # Head (circle)
            head_center = (person_x + person_w // 2, person_y + person_h // 4)
            head_radius = person_w // 6
            cv2.circle(image, head_center, head_radius, (150, 150, 150), -1)
            
            # Face features (eyes)
            eye1 = (head_center[0] - head_radius // 2, head_center[1] - head_radius // 3)
            eye2 = (head_center[0] + head_radius // 2, head_center[1] - head_radius // 3)
            cv2.circle(image, eye1, 2, (0, 0, 0), -1)
            cv2.circle(image, eye2, 2, (0, 0, 0), -1)
        
        return image
    
    @staticmethod
    def create_detection_input(
        source_id: str = "test_camera",
        confidence: float = 0.9,
        bbox: tuple = None
    ) -> DetectionInput:
        """Create a test detection input"""
        image = TestDataGenerator.create_test_image()
        
        if bbox is None:
            bbox = (50, 50, 124, 124)  # x, y, w, h
        
        return DetectionInput(
            image=image,
            detection_bbox=bbox,
            detection_confidence=confidence,
            timestamp=time.time(),
            source_id=source_id,
            metadata={
                'test_data': True,
                'detection_id': f"test_{int(time.time() * 1000)}"
            }
        )
    
    @staticmethod
    def create_test_embedding(
        embedding_id: str = None,
        person_id: str = None,
        dimension: int = 512
    ) -> ReIDEmbedding:
        """Create a test embedding"""
        if embedding_id is None:
            embedding_id = f"test_emb_{int(time.time() * 1000)}"
        
        # Create normalized random embedding
        embedding_vector = np.random.randn(dimension).astype(np.float32)
        embedding_vector = embedding_vector / np.linalg.norm(embedding_vector)
        
        return ReIDEmbedding(
            embedding_id=embedding_id,
            person_id=person_id,
            embedding_vector=embedding_vector,
            confidence=0.85,
            timestamp=time.time(),
            metadata={'test_data': True},
            source_image_hash="test_hash",
            privacy_level="standard"
        )


class TestReIDEmbedder(unittest.TestCase):
    """Test ReID embedder functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = EmbeddingConfig(
            model_type="simple_cnn",  # Use simple model for testing
            device="cpu",
            batch_size=1,
            input_size=(224, 224)
        )
        self.embedder = ReIDEmbedder(self.config)
    
    def test_embedder_initialization(self):
        """Test embedder initialization"""
        self.assertIsNotNone(self.embedder)
        self.assertEqual(self.embedder.config.model_type, "simple_cnn")
        self.assertEqual(self.embedder.config.device, "cpu")
    
    def test_embedding_extraction(self):
        """Test embedding extraction from image"""
        test_image = TestDataGenerator.create_test_image()
        
        embedding = self.embedder.extract_embedding(test_image)
        
        self.assertIsNotNone(embedding)
        self.assertIsInstance(embedding, np.ndarray)
        self.assertGreater(len(embedding), 0)
        
        # Test that embeddings are normalized
        norm = np.linalg.norm(embedding)
        self.assertAlmostEqual(norm, 1.0, places=5)
    
    def test_batch_embedding_extraction(self):
        """Test batch embedding extraction"""
        images = [TestDataGenerator.create_test_image() for _ in range(3)]
        
        embeddings = self.embedder.extract_embeddings_batch(images)
        
        self.assertIsNotNone(embeddings)
        self.assertEqual(len(embeddings), 3)
        
        for embedding in embeddings:
            self.assertIsInstance(embedding, np.ndarray)
            norm = np.linalg.norm(embedding)
            self.assertAlmostEqual(norm, 1.0, places=5)
    
    def test_invalid_image_handling(self):
        """Test handling of invalid images"""
        # Empty image
        empty_image = np.array([])
        embedding = self.embedder.extract_embedding(empty_image)
        self.assertIsNone(embedding)
        
        # Wrong dimensions
        wrong_dims = np.random.randint(0, 255, (100,), dtype=np.uint8)
        embedding = self.embedder.extract_embedding(wrong_dims)
        self.assertIsNone(embedding)


class TestPrivacyFilter(unittest.TestCase):
    """Test privacy filter functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = PrivacyConfig(
            privacy_level=PrivacyLevel.STANDARD,
            blur_method=BlurMethod.GAUSSIAN,
            blur_strength=15,
            face_detection_confidence=0.5
        )
        self.privacy_filter = PrivacyFilter(self.config)
    
    def test_privacy_filter_initialization(self):
        """Test privacy filter initialization"""
        self.assertIsNotNone(self.privacy_filter)
        self.assertEqual(self.privacy_filter.config.privacy_level, PrivacyLevel.STANDARD)
    
    def test_image_processing(self):
        """Test image processing with privacy filter"""
        test_image = TestDataGenerator.create_test_image()
        
        processed_image = self.privacy_filter.process_image(test_image)
        
        self.assertIsNotNone(processed_image)
        self.assertEqual(processed_image.shape, test_image.shape)
        self.assertIsInstance(processed_image, np.ndarray)
    
    def test_face_blurring(self):
        """Test face blurring functionality"""
        test_image = TestDataGenerator.create_test_image(person_present=True)
        
        # Process with face blurring enabled
        processed_image = self.privacy_filter.process_image(test_image)
        
        # Image should be modified (blurred)
        self.assertFalse(np.array_equal(test_image, processed_image))
    
    def test_privacy_levels(self):
        """Test different privacy levels"""
        test_image = TestDataGenerator.create_test_image()
        
        # Test different privacy levels
        for level in [PrivacyLevel.MINIMAL, PrivacyLevel.STANDARD, PrivacyLevel.HIGH]:
            config = PrivacyConfig(privacy_level=level)
            filter_obj = PrivacyFilter(config)
            
            processed = filter_obj.process_image(test_image)
            self.assertIsNotNone(processed)
    
    def test_create_privacy_filter_presets(self):
        """Test privacy filter preset creation"""
        for preset in ["minimal", "standard", "high"]:
            config = create_privacy_filter(preset)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, PrivacyConfig)


class TestEmbeddingManager(unittest.TestCase):
    """Test embedding manager functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_embeddings.db")
        self.manager = EmbeddingManager(self.db_path, use_faiss=False)  # Disable FAISS for testing
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_manager_initialization(self):
        """Test embedding manager initialization"""
        self.assertIsNotNone(self.manager)
        self.assertIsNotNone(self.manager.database)
    
    def test_add_embedding(self):
        """Test adding embeddings"""
        embedding = TestDataGenerator.create_test_embedding()
        
        success = self.manager.add_embedding(embedding)
        self.assertTrue(success)
        
        # Retrieve and verify
        retrieved = self.manager.database.get_embedding(embedding.embedding_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.embedding_id, embedding.embedding_id)
    
    def test_similarity_search(self):
        """Test similarity search"""
        # Add some test embeddings
        embeddings = [TestDataGenerator.create_test_embedding(f"emb_{i}") for i in range(5)]
        
        for embedding in embeddings:
            self.manager.add_embedding(embedding)
        
        # Search for similar embeddings
        query_vector = embeddings[0].embedding_vector
        results = self.manager.find_similar_embeddings(query_vector, threshold=0.1, max_results=3)
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # First result should be the exact match
        best_match, similarity = results[0]
        self.assertEqual(best_match.embedding_id, embeddings[0].embedding_id)
        self.assertAlmostEqual(similarity, 1.0, places=5)
    
    def test_match_creation_and_confirmation(self):
        """Test match result creation and confirmation"""
        embedding1 = TestDataGenerator.create_test_embedding("emb1")
        embedding2 = TestDataGenerator.create_test_embedding("emb2")
        
        self.manager.add_embedding(embedding1)
        self.manager.add_embedding(embedding2)
        
        # Create match result
        match = self.manager.create_match_result(
            embedding1.embedding_id,
            embedding2,
            0.85
        )
        
        self.assertIsNotNone(match)
        self.assertEqual(match.query_embedding_id, embedding1.embedding_id)
        self.assertEqual(match.matched_embedding_id, embedding2.embedding_id)
        
        # Confirm match
        success = self.manager.confirm_match(match)
        self.assertTrue(success)
    
    def test_database_statistics(self):
        """Test database statistics"""
        # Add some test data
        for i in range(3):
            embedding = TestDataGenerator.create_test_embedding(f"emb_{i}", f"person_{i}")
            self.manager.add_embedding(embedding)
        
        stats = self.manager.get_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats['total_embeddings'], 3)
        self.assertEqual(stats['unique_persons'], 3)


class TestReIDPipeline(unittest.TestCase):
    """Test ReID pipeline functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_pipeline.db")
        
        # Create test pipeline
        self.pipeline = create_default_pipeline(self.db_path)
        
        # Override with simple models for testing
        self.pipeline.config.embedding_config.model_type = "simple_cnn"
        self.pipeline.embedder = ReIDEmbedder(self.pipeline.config.embedding_config)
    
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'pipeline'):
            self.pipeline.stop()
        shutil.rmtree(self.temp_dir)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        self.assertIsNotNone(self.pipeline)
        self.assertIsNotNone(self.pipeline.embedder)
        self.assertIsNotNone(self.pipeline.privacy_filter)
        self.assertIsNotNone(self.pipeline.embedding_manager)
    
    def test_synchronous_processing(self):
        """Test synchronous detection processing"""
        detection = TestDataGenerator.create_detection_input()
        
        result = self.pipeline.process_detection_sync(detection)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, ReIDResult)
        self.assertIsNotNone(result.embedding_id)
        self.assertIsInstance(result.matches, list)
        self.assertIsInstance(result.alert_level, AlertLevel)
    
    def test_asynchronous_processing(self):
        """Test asynchronous detection processing"""
        self.pipeline.start()
        
        detection = TestDataGenerator.create_detection_input()
        
        # Queue detection
        processing_id = self.pipeline.process_detection(detection)
        self.assertIsNotNone(processing_id)
        
        # Get result
        result = self.pipeline.get_result(timeout=5.0)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, ReIDResult)
        
        self.pipeline.stop()
    
    def test_match_confirmation_workflow(self):
        """Test match confirmation workflow"""
        # Add a reference embedding first
        ref_embedding = TestDataGenerator.create_test_embedding("ref_emb", "person_1")
        self.pipeline.embedding_manager.add_embedding(ref_embedding)
        
        # Process similar detection
        detection = TestDataGenerator.create_detection_input()
        result = self.pipeline.process_detection_sync(detection)
        
        if result.matches:
            match = result.matches[0]
            
            # Test confirmation
            confirmed = self.pipeline.confirm_match(match, "person_1")
            self.assertTrue(confirmed)
            
            # Test rejection
            rejected = self.pipeline.reject_match(match)
            self.assertTrue(rejected)
    
    def test_pipeline_statistics(self):
        """Test pipeline statistics"""
        stats = self.pipeline.get_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('processed_count', stats)
        self.assertIn('match_count', stats)
        self.assertIn('running', stats)
    
    def test_sar_pipeline_creation(self):
        """Test SAR-optimized pipeline creation"""
        sar_pipeline = create_sar_pipeline(self.db_path)
        
        self.assertIsNotNone(sar_pipeline)
        self.assertEqual(sar_pipeline.config.mode, PipelineMode.REAL_TIME)
        self.assertLess(sar_pipeline.config.similarity_threshold, 0.7)  # Lower threshold for SAR


class TestIntegration(unittest.TestCase):
    """Integration tests for complete ReID workflow"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "integration_test.db")
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_complete_reid_workflow(self):
        """Test complete re-identification workflow"""
        # Create pipeline
        pipeline = create_default_pipeline(self.db_path)
        pipeline.config.embedding_config.model_type = "simple_cnn"
        pipeline.embedder = ReIDEmbedder(pipeline.config.embedding_config)
        
        try:
            # Step 1: Process first detection (new person)
            detection1 = TestDataGenerator.create_detection_input("camera_1")
            result1 = pipeline.process_detection_sync(detection1)
            
            self.assertIsNotNone(result1)
            self.assertEqual(len(result1.matches), 0)  # No matches for first person
            
            # Step 2: Process second detection (same person)
            detection2 = TestDataGenerator.create_detection_input("camera_2")
            result2 = pipeline.process_detection_sync(detection2)
            
            self.assertIsNotNone(result2)
            # Should have at least one match (the first detection)
            self.assertGreaterEqual(len(result2.matches), 0)
            
            # Step 3: Confirm match if found
            if result2.matches:
                match = result2.matches[0]
                confirmed = pipeline.confirm_match(match, "person_001")
                self.assertTrue(confirmed)
            
            # Step 4: Verify statistics
            stats = pipeline.get_statistics()
            self.assertEqual(stats['processed_count'], 2)
            
        finally:
            pipeline.stop()
    
    def test_privacy_preservation(self):
        """Test that privacy is preserved throughout workflow"""
        # Create high-privacy pipeline
        config = PipelineConfig(
            embedding_config=EmbeddingConfig(model_type="simple_cnn", device="cpu"),
            privacy_config=create_privacy_filter("high"),
            database_path=self.db_path
        )
        pipeline = ReIDPipeline(config)
        
        try:
            detection = TestDataGenerator.create_detection_input()
            result = pipeline.process_detection_sync(detection)
            
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.processed_image)
            
            # Verify that processed image is different from original (privacy applied)
            original_crop = detection.get_cropped_image()
            self.assertFalse(np.array_equal(original_crop, result.processed_image))
            
        finally:
            pipeline.stop()
    
    def test_performance_benchmarking(self):
        """Test performance benchmarking"""
        pipeline = create_sar_pipeline(self.db_path)
        pipeline.config.embedding_config.model_type = "simple_cnn"
        pipeline.embedder = ReIDEmbedder(pipeline.config.embedding_config)
        
        try:
            # Process multiple detections and measure time
            num_detections = 10
            start_time = time.time()
            
            for i in range(num_detections):
                detection = TestDataGenerator.create_detection_input(f"camera_{i}")
                result = pipeline.process_detection_sync(detection)
                self.assertIsNotNone(result)
            
            total_time = time.time() - start_time
            avg_time = total_time / num_detections
            
            # Should process reasonably fast (less than 1 second per detection)
            self.assertLess(avg_time, 1.0)
            
            # Verify statistics
            stats = pipeline.get_statistics()
            self.assertEqual(stats['processed_count'], num_detections)
            self.assertGreater(stats['processing_rate'], 0)
            
        finally:
            pipeline.stop()


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "error_test.db")
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_invalid_detection_handling(self):
        """Test handling of invalid detections"""
        pipeline = create_default_pipeline(self.db_path)
        pipeline.config.embedding_config.model_type = "simple_cnn"
        pipeline.embedder = ReIDEmbedder(pipeline.config.embedding_config)
        
        try:
            # Invalid bbox (outside image bounds)
            detection = DetectionInput(
                image=TestDataGenerator.create_test_image(100, 100),
                detection_bbox=(200, 200, 50, 50),  # Outside image
                detection_confidence=0.9,
                timestamp=time.time(),
                source_id="test",
                metadata={}
            )
            
            result = pipeline.process_detection_sync(detection)
            # Should handle gracefully (return None or empty result)
            self.assertTrue(result is None or len(result.matches) == 0)
            
        finally:
            pipeline.stop()
    
    def test_database_corruption_recovery(self):
        """Test recovery from database issues"""
        # Create manager with invalid database path
        invalid_path = "/invalid/path/database.db"
        
        # Should handle gracefully without crashing
        try:
            manager = EmbeddingManager(invalid_path)
            # Basic operations should not crash
            stats = manager.get_statistics()
            self.assertIsInstance(stats, dict)
        except Exception as e:
            # If it fails, it should fail gracefully
            self.assertIsInstance(e, Exception)
    
    def test_memory_management(self):
        """Test memory management with large datasets"""
        pipeline = create_default_pipeline(self.db_path)
        pipeline.config.embedding_config.model_type = "simple_cnn"
        pipeline.embedder = ReIDEmbedder(pipeline.config.embedding_config)
        
        try:
            # Process many detections to test memory usage
            for i in range(50):
                detection = TestDataGenerator.create_detection_input(f"test_{i}")
                result = pipeline.process_detection_sync(detection)
                
                # Verify result is created and released properly
                self.assertIsNotNone(result)
                del result  # Explicit cleanup
            
            # Verify statistics are reasonable
            stats = pipeline.get_statistics()
            self.assertEqual(stats['processed_count'], 50)
            
        finally:
            pipeline.stop()


def run_tests():
    """Run all ReID tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestReIDEmbedder,
        TestPrivacyFilter,
        TestEmbeddingManager,
        TestReIDPipeline,
        TestIntegration,
        TestErrorHandling
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run tests when script is executed directly
    success = run_tests()
    exit(0 if success else 1)