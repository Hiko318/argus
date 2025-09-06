# Person Re-Identification (ReID) Module

The ReID module provides privacy-first person re-identification capabilities for the Foresight SAR system. It enables tracking and identification of individuals across multiple camera feeds while maintaining strict privacy controls and operator oversight.

## Features

### Core Capabilities
- **Deep Learning Embeddings**: Generate robust person embeddings using state-of-the-art models
- **Privacy-First Design**: Automatic face blurring and configurable privacy levels
- **Real-time Processing**: Optimized for real-time SAR operations
- **Operator Oversight**: Human-in-the-loop confirmation for critical matches
- **Multi-Modal Support**: Works with various input sources and formats

### Privacy Features
- **Automatic Face Blurring**: Configurable face detection and blurring
- **Local Processing**: No cloud dependencies, all processing on-device
- **Privacy Levels**: Minimal, Standard, and High privacy configurations
- **Secure Storage**: Encrypted embedding storage with automatic cleanup
- **Audit Trail**: Complete logging of all matches and confirmations

### Technical Features
- **Multiple Models**: Support for ResNet50, MobileNet, and custom models
- **ONNX/TensorRT**: Optimized inference for production deployment
- **FAISS Integration**: Fast similarity search for large databases
- **Batch Processing**: Efficient batch embedding generation
- **Configurable Thresholds**: Adjustable similarity and confidence thresholds

## Quick Start

### Basic Usage

```python
from foresight.reid import create_sar_pipeline, DetectionInput
import cv2
import time

# Create SAR-optimized pipeline
pipeline = create_sar_pipeline("sar_embeddings.db")
pipeline.start()

# Process a detection
image = cv2.imread("person_detection.jpg")
detection = DetectionInput(
    image=image,
    detection_bbox=(100, 100, 200, 300),  # x, y, w, h
    detection_confidence=0.9,
    timestamp=time.time(),
    source_id="camera_01",
    metadata={"location": "search_area_alpha"}
)

# Process synchronously
result = pipeline.process_detection_sync(detection)

if result.matches:
    print(f"Found {len(result.matches)} potential matches")
    for match in result.matches:
        print(f"  Similarity: {match.similarity_score:.3f}")
        
        if result.requires_confirmation:
            # Operator confirmation required
            confirmed = input("Confirm match? (y/n): ").lower() == 'y'
            if confirmed:
                pipeline.confirm_match(match, person_id="person_001")
            else:
                pipeline.reject_match(match)

pipeline.stop()
```

### Asynchronous Processing

```python
from foresight.reid import create_sar_pipeline

# Create and start pipeline
pipeline = create_sar_pipeline()
pipeline.start()

# Set up callback for matches
def handle_match(result):
    if result.alert_level.value in ['high', 'critical']:
        print(f"ALERT: High confidence match detected!")
        # Trigger operator notification
        notify_operator(result)

pipeline.set_match_callback(handle_match)

# Queue detections for processing
for detection in detection_stream:
    processing_id = pipeline.process_detection(detection)
    if processing_id:
        print(f"Queued detection {processing_id}")

# Get results
while True:
    result = pipeline.get_result(timeout=1.0)
    if result:
        process_result(result)
```

## Configuration

### Pipeline Configuration

```python
from foresight.reid import (
    ReIDPipeline, PipelineConfig, EmbeddingConfig, 
    PrivacyConfig, PipelineMode, create_privacy_filter
)

# Custom embedding configuration
embedding_config = EmbeddingConfig(
    model_type="resnet50",
    device="cuda",  # or "cpu"
    batch_size=4,
    input_size=(224, 224),
    normalize=True,
    use_onnx=True  # Use ONNX for faster inference
)

# Privacy configuration
privacy_config = create_privacy_filter("high")  # minimal, standard, high

# Pipeline configuration
config = PipelineConfig(
    embedding_config=embedding_config,
    privacy_config=privacy_config,
    mode=PipelineMode.INTERACTIVE,
    similarity_threshold=0.7,
    high_confidence_threshold=0.85,
    critical_threshold=0.95,
    require_confirmation_above=0.8,
    max_matches=5,
    database_path="custom_embeddings.db"
)

pipeline = ReIDPipeline(config)
```

### Privacy Levels

```python
from foresight.reid import create_privacy_filter, PrivacyLevel, BlurMethod

# Preset configurations
minimal_privacy = create_privacy_filter("minimal")
standard_privacy = create_privacy_filter("standard")
high_privacy = create_privacy_filter("high")

# Custom privacy configuration
custom_privacy = PrivacyConfig(
    privacy_level=PrivacyLevel.HIGH,
    blur_method=BlurMethod.GAUSSIAN,
    blur_strength=25,
    face_detection_confidence=0.3,
    blur_entire_head=True,
    preserve_body=True
)
```

## Model Support

### Available Models

| Model | Speed | Accuracy | Memory | Use Case |
|-------|-------|----------|--------|-----------|
| `simple_cnn` | Fast | Basic | Low | Testing/Development |
| `mobilenet_v3` | Fast | Good | Low | Real-time SAR |
| `resnet50` | Medium | High | Medium | Balanced performance |
| `custom` | Varies | Varies | Varies | Custom trained models |

### Model Selection Guidelines

- **Real-time SAR Operations**: Use `mobilenet_v3` for best speed/accuracy balance
- **High Accuracy Requirements**: Use `resnet50` for maximum accuracy
- **Resource Constrained**: Use `simple_cnn` for minimal resource usage
- **Custom Scenarios**: Train and use `custom` models for specific requirements

### Custom Model Integration

```python
from foresight.reid import EmbeddingConfig, ReIDEmbedder

# Load custom ONNX model
config = EmbeddingConfig(
    model_type="custom",
    model_path="path/to/custom_model.onnx",
    device="cuda",
    input_size=(256, 128),  # Custom input size
    normalize=True
)

embedder = ReIDEmbedder(config)
```

## Database Management

### Embedding Storage

```python
from foresight.reid import EmbeddingManager, ReIDEmbedding
import numpy as np

# Create embedding manager
manager = EmbeddingManager("embeddings.db", use_faiss=True)

# Create and store embedding
embedding = ReIDEmbedding(
    embedding_id="person_001_frame_123",
    person_id="person_001",
    embedding_vector=np.random.randn(512),
    confidence=0.95,
    timestamp=time.time(),
    metadata={"camera": "cam_01", "location": "area_alpha"},
    privacy_level="standard"
)

manager.add_embedding(embedding)

# Search for similar embeddings
query_vector = np.random.randn(512)
similar = manager.find_similar_embeddings(
    query_vector, 
    threshold=0.7, 
    max_results=5
)

for embedding, similarity in similar:
    print(f"Match: {embedding.person_id}, Similarity: {similarity:.3f}")
```

### Database Maintenance

```python
# Get database statistics
stats = manager.get_statistics()
print(f"Total embeddings: {stats['total_embeddings']}")
print(f"Unique persons: {stats['unique_persons']}")
print(f"Match counts: {stats['match_counts']}")

# Cleanup old embeddings (older than 30 days)
deleted_count = manager.database.cleanup_old_embeddings(max_age_days=30)
print(f"Cleaned up {deleted_count} old embeddings")
```

## Performance Optimization

### Hardware Acceleration

```python
# GPU acceleration
config = EmbeddingConfig(
    model_type="resnet50",
    device="cuda",  # Use GPU
    batch_size=8,   # Larger batch for GPU
    use_onnx=True   # ONNX for optimization
)

# CPU optimization
config = EmbeddingConfig(
    model_type="mobilenet_v3",
    device="cpu",
    batch_size=1,   # Smaller batch for CPU
    use_onnx=True
)
```

### FAISS Integration

```python
# Enable FAISS for fast similarity search
manager = EmbeddingManager(
    "embeddings.db", 
    use_faiss=True  # Requires faiss-cpu or faiss-gpu
)

# FAISS provides significant speedup for large databases (>1000 embeddings)
```

### Batch Processing

```python
# Process multiple images in batch
images = [cv2.imread(f"person_{i}.jpg") for i in range(10)]
embeddings = embedder.extract_embeddings_batch(images)

# More efficient than processing individually
for i, embedding in enumerate(embeddings):
    if embedding is not None:
        print(f"Processed image {i}: embedding shape {embedding.shape}")
```

## Testing

### Running Tests

```bash
# Run all tests
python -m foresight.reid.test_reid

# Run specific test class
python -m unittest foresight.reid.test_reid.TestReIDPipeline

# Run with verbose output
python -m unittest foresight.reid.test_reid -v
```

### Test Coverage

The test suite covers:
- Embedding generation and extraction
- Privacy filtering functionality
- Database operations and similarity search
- Pipeline integration and workflows
- Error handling and edge cases
- Performance benchmarking
- Memory management

### Custom Test Data

```python
from foresight.reid.test_reid import TestDataGenerator

# Generate test images
test_image = TestDataGenerator.create_test_image(224, 224, person_present=True)

# Create test detection
detection = TestDataGenerator.create_detection_input(
    source_id="test_camera",
    confidence=0.9
)

# Create test embedding
embedding = TestDataGenerator.create_test_embedding(
    embedding_id="test_001",
    person_id="test_person"
)
```

## Integration with Foresight

### Main Application Integration

```python
# In main.py or detection pipeline
from foresight.reid import create_sar_pipeline

class ForesightSystem:
    def __init__(self):
        self.reid_pipeline = create_sar_pipeline("foresight_reid.db")
        self.reid_pipeline.start()
        
        # Set up callbacks
        self.reid_pipeline.set_match_callback(self.handle_reid_match)
    
    def handle_detection(self, detection_result):
        """Handle person detection from YOLO"""
        if detection_result.class_name == "person":
            # Create ReID detection input
            reid_detection = DetectionInput(
                image=detection_result.image,
                detection_bbox=detection_result.bbox,
                detection_confidence=detection_result.confidence,
                timestamp=detection_result.timestamp,
                source_id=detection_result.source_id,
                metadata=detection_result.metadata
            )
            
            # Process with ReID pipeline
            self.reid_pipeline.process_detection(reid_detection)
    
    def handle_reid_match(self, reid_result):
        """Handle ReID match results"""
        if reid_result.alert_level.value in ['high', 'critical']:
            # High confidence match - alert operator
            self.alert_operator(reid_result)
        
        # Log all matches
        self.log_reid_result(reid_result)
```

### UI Integration

```python
# For UI confirmation workflows
def show_match_confirmation_dialog(match_result):
    """Show UI dialog for match confirmation"""
    # Display match information
    print(f"Potential match found:")
    print(f"Similarity: {match_result.similarity_score:.3f}")
    print(f"Confidence: {match_result.confidence:.3f}")
    
    # Show images (with privacy filtering applied)
    # display_match_images(match_result)
    
    # Get operator decision
    decision = get_operator_decision()  # UI implementation
    
    return decision == "confirm"

# Set confirmation callback
pipeline.set_confirmation_callback(show_match_confirmation_dialog)
```

## Troubleshooting

### Common Issues

#### Model Loading Errors
```python
# Check model availability
from foresight.reid import EmbeddingModel
print("Available models:", [model.value for model in EmbeddingModel])

# Use fallback model
config = EmbeddingConfig(
    model_type="simple_cnn",  # Always available fallback
    device="cpu"
)
```

#### Memory Issues
```python
# Reduce batch size
config.batch_size = 1

# Use smaller model
config.model_type = "mobilenet_v3"

# Enable cleanup
config.max_embedding_age_days = 7  # Shorter retention
```

#### Performance Issues
```python
# Enable GPU if available
config.device = "cuda" if torch.cuda.is_available() else "cpu"

# Use ONNX optimization
config.use_onnx = True

# Enable FAISS for large databases
manager = EmbeddingManager(db_path, use_faiss=True)
```

#### Database Issues
```python
# Check database statistics
stats = manager.get_statistics()
if stats['total_embeddings'] == 0:
    print("Database is empty or corrupted")

# Recreate database if needed
import os
if os.path.exists("corrupted.db"):
    os.remove("corrupted.db")
manager = EmbeddingManager("new_database.db")
```

### Logging and Debugging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Pipeline-specific logging
config.log_level = "DEBUG"
config.log_matches = True
config.log_rejections = True

# Monitor pipeline statistics
stats = pipeline.get_statistics()
print(f"Processing rate: {stats['processing_rate']:.2f} detections/sec")
print(f"Match rate: {stats['match_rate']:.2%}")
print(f"Error count: {stats['error_count']}")
```

## Security Considerations

### Privacy Protection
- All face detection and blurring happens locally
- No biometric data is transmitted to external services
- Embeddings are stored encrypted in local database
- Automatic cleanup of old embeddings
- Configurable privacy levels for different scenarios

### Data Handling
- Original images are never stored, only processed embeddings
- Face regions are automatically blurred in processed images
- Metadata is sanitized to remove sensitive information
- Audit trail maintains record of all operator decisions

### Access Control
- Database access requires appropriate file permissions
- Operator confirmation required for high-confidence matches
- All match confirmations and rejections are logged
- Configurable thresholds for automatic vs. manual processing

## Contributing

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest foresight/reid/test_reid.py -v

# Run linting
flake8 foresight/reid/
black foresight/reid/
```

### Adding New Models

1. Implement model loading in `embedder.py`
2. Add model type to `EmbeddingModel` enum
3. Update model selection logic
4. Add tests for new model
5. Update documentation

### Performance Optimization

1. Profile code with `cProfile`
2. Optimize bottlenecks identified
3. Add performance tests
4. Update benchmarks

## License

This module is part of the Foresight SAR system and follows the same licensing terms.

## Support

For issues and questions:
1. Check this documentation
2. Review test cases for usage examples
3. Check logs for error details
4. File issues with detailed reproduction steps