# Geolocation Module for Foresight SAR System

The geolocation module provides comprehensive geographic positioning capabilities for the Foresight Search and Rescue (SAR) system. It enables accurate conversion between image pixel coordinates and real-world geographic coordinates using camera calibration, pinhole projection models, coordinate transformations, and digital elevation model (DEM) correction.

## Features

### Core Capabilities
- **Camera Calibration**: Support for pinhole camera models with various distortion corrections
- **Pinhole Projection**: Convert between pixel coordinates and 3D rays in world space
- **Coordinate Transformations**: Convert between geodetic (WGS84), UTM, and local ENU coordinate systems
- **DEM Correction**: Terrain-aware geolocation using digital elevation models
- **Ray-Ground Intersection**: Calculate ground intersection points with terrain correction
- **Batch Processing**: Efficient processing of multiple pixel coordinates

### Supported Formats
- **Camera Models**: Pinhole camera with radial-tangential distortion
- **DEM Formats**: GeoTIFF, SRTM, ASTER, ALOS, and custom synthetic DEMs
- **Coordinate Systems**: WGS84, UTM, ENU (East-North-Up), NED (North-East-Down)
- **Interpolation Methods**: Nearest neighbor, bilinear, bicubic interpolation

### Privacy & Security
- **Local Processing**: All geolocation calculations performed locally
- **No External Dependencies**: Works offline without internet connectivity
- **Configurable Precision**: Adjustable accuracy vs. performance trade-offs
- **Error Handling**: Robust error handling with confidence metrics

## Quick Start

### Basic Usage

```python
from geolocation import (
    GeolocationService, CalibrationData, GeodeticCoordinate, CameraPose
)
import numpy as np

# 1. Create camera calibration data
camera_matrix = np.array([
    [800.0, 0.0, 320.0],
    [0.0, 800.0, 240.0],
    [0.0, 0.0, 1.0]
])

calibration = CalibrationData(
    camera_matrix=camera_matrix,
    distortion_coeffs=np.array([0.1, -0.2, 0.001, 0.002, 0.05]),
    image_width=640,
    image_height=480
)

# 2. Set reference point (e.g., takeoff location)
reference_point = GeodeticCoordinate(
    latitude=51.5074,   # London
    longitude=-0.1278,
    altitude=0.0
)

# 3. Create geolocation service
service = GeolocationService(
    calibration_data=calibration,
    reference_point=reference_point
)

# 4. Define camera pose (drone position and orientation)
camera_pose = CameraPose(
    position=np.array([100.0, 200.0, 50.0]),  # ENU coordinates
    euler_angles=np.array([0.0, np.radians(15), 0.0])  # roll, pitch, yaw
)

# 5. Geolocate a pixel
pixel_coords = (320, 240)  # Center of image
result = service.geolocate_pixel(
    pixel_coords=pixel_coords,
    camera_pose=camera_pose
)

if result:
    print(f"Latitude: {result.geodetic_coordinate.latitude:.6f}")
    print(f"Longitude: {result.geodetic_coordinate.longitude:.6f}")
    print(f"Confidence: {result.confidence:.2f}")
```

### With DEM Correction

```python
# Load DEM for terrain correction
service.dem_corrector.load_dem_file(
    file_path="path/to/dem.tif",
    dem_format=DEMFormat.GEOTIFF
)

# Or create synthetic DEM for testing
service.dem_corrector.create_synthetic_dem(
    bounds=(-1.0, 51.0, 1.0, 53.0),  # min_lon, min_lat, max_lon, max_lat
    resolution=30.0,  # meters
    base_elevation=50.0,
    terrain_variation=100.0
)

# Geolocate with DEM correction
result = service.geolocate_pixel(
    pixel_coords=pixel_coords,
    camera_pose=camera_pose,
    use_dem_correction=True
)
```

### Batch Processing

```python
# Process multiple pixels efficiently
pixel_list = [(100, 100), (320, 240), (500, 400)]

results = service.geolocate_pixels_batch(
    pixel_coords_list=pixel_list,
    camera_pose=camera_pose,
    use_dem_correction=True
)

for i, result in enumerate(results):
    if result:
        print(f"Pixel {pixel_list[i]}: {result.geodetic_coordinate.latitude:.6f}, {result.geodetic_coordinate.longitude:.6f}")
```

## Configuration

### Camera Calibration

```python
from geolocation import CameraCalibrator, CameraModel, DistortionModel

# Create calibrator for camera calibration
calibrator = CameraCalibrator(
    camera_model=CameraModel.PINHOLE,
    distortion_model=DistortionModel.RADIAL_TANGENTIAL
)

# Calibrate from chessboard images (if available)
# calibration_data = calibrator.calibrate_from_images(image_paths, chessboard_size)

# Or create from known parameters
calibration_data = CalibrationData(
    camera_matrix=camera_matrix,
    distortion_coeffs=distortion_coeffs,
    image_width=1920,
    image_height=1080,
    camera_model=CameraModel.PINHOLE,
    distortion_model=DistortionModel.RADIAL_TANGENTIAL
)
```

### Geolocation Configuration

```python
from geolocation import GeolocationConfig, GeolocationMethod, ConfidenceLevel

config = GeolocationConfig(
    method=GeolocationMethod.DEM_CORRECTED,
    confidence_threshold=ConfidenceLevel.MEDIUM,
    max_iterations=10,
    convergence_tolerance=1.0,  # meters
    use_cache=True,
    cache_size_limit=10000
)

service = GeolocationService(
    calibration_data=calibration_data,
    reference_point=reference_point,
    config=config
)
```

### DEM Configuration

```python
from geolocation import DEMCorrector, InterpolationMethod

# Configure DEM corrector
dem_corrector = DEMCorrector()
dem_corrector.set_default_elevation(0.0)  # Sea level default

# Load multiple DEM tiles
dem_files = ["dem_tile_1.tif", "dem_tile_2.tif"]
for dem_file in dem_files:
    dem_corrector.load_dem_file(dem_file)

# Configure interpolation
elevation = dem_corrector.get_elevation(
    lon=0.0, lat=52.0,
    method=InterpolationMethod.BILINEAR
)
```

## Coordinate Systems

### Supported Systems

1. **Geodetic (WGS84)**: Latitude, longitude, altitude
2. **UTM**: Universal Transverse Mercator projection
3. **ENU**: East-North-Up local coordinate system
4. **NED**: North-East-Down coordinate system
5. **ECEF**: Earth-Centered Earth-Fixed coordinates

### Coordinate Conversion

```python
from geolocation import CoordinateTransformer, GeodeticCoordinate

# Create transformer with reference point
transformer = CoordinateTransformer(reference_point)

# Convert geodetic to ENU
geodetic_coord = GeodeticCoordinate(51.5074, -0.1278, 100.0)
enu_coord = transformer.geodetic_to_enu(geodetic_coord)

# Convert ENU back to geodetic
geodetic_back = transformer.enu_to_geodetic(enu_coord)

# Convert to UTM
utm_coord = transformer.geodetic_to_utm(geodetic_coord)
```

## Camera Pose Management

### Creating Camera Poses

```python
from geolocation import CameraPose, CoordinateSystem
import numpy as np

# From position and Euler angles
pose = CameraPose(
    position=np.array([100.0, 200.0, 50.0]),
    euler_angles=np.array([0.0, np.radians(15), np.radians(30)]),
    coordinate_system=CoordinateSystem.ENU
)

# From position and rotation matrix
rotation_matrix = np.array([
    [0.866, -0.5, 0.0],
    [0.5, 0.866, 0.0],
    [0.0, 0.0, 1.0]
])

pose = CameraPose(
    position=np.array([100.0, 200.0, 50.0]),
    rotation_matrix=rotation_matrix
)

# From quaternion
quaternion = np.array([0.966, 0.0, 0.0, 0.259])  # w, x, y, z
pose = CameraPose(
    position=np.array([100.0, 200.0, 50.0]),
    quaternion=quaternion
)
```

### Pose Transformations

```python
# Get transformation matrix
T = pose.get_transformation_matrix()  # 4x4 matrix

# Get inverse pose
inverse_pose = pose.inverse()

# Validate rotation matrix
is_valid = pose._is_valid_rotation_matrix(pose.rotation_matrix)
```

## DEM Management

### Loading DEM Data

```python
from geolocation import DEMCorrector, DEMFormat

dem_corrector = DEMCorrector()

# Load from GeoTIFF file
dem_tile = dem_corrector.load_dem_file(
    file_path="srtm_data.tif",
    dem_format=DEMFormat.GEOTIFF
)

# Create synthetic DEM for testing
synthetic_dem = dem_corrector.create_synthetic_dem(
    bounds=(-1.0, 51.0, 1.0, 53.0),
    resolution=30.0,
    base_elevation=50.0,
    terrain_variation=100.0
)
```

### DEM Operations

```python
# Get elevation at point
elevation = dem_corrector.get_elevation(lon=0.0, lat=52.0)

# Get terrain slope
slope_x, slope_y = dem_corrector.get_terrain_slope(lon=0.0, lat=52.0)

# Get terrain normal vector
normal = dem_corrector.get_terrain_normal(lon=0.0, lat=52.0)

# Get DEM coverage area
coverage = dem_corrector.get_coverage_area()

# Get DEM information
info = dem_corrector.get_dem_info()
print(f"DEM Status: {info['status']}")
print(f"Coverage: {info['coverage_area']}")
```

## Performance Optimization

### Caching

```python
# Enable elevation caching
dem_corrector.cache_size_limit = 10000

# Clear cache when needed
dem_corrector.clear_cache()
```

### Batch Processing

```python
# Process multiple pixels efficiently
pixel_batch = [(i*50, j*50) for i in range(10) for j in range(10)]

results = service.geolocate_pixels_batch(
    pixel_coords_list=pixel_batch,
    camera_pose=camera_pose,
    use_dem_correction=True
)
```

### Hardware Acceleration

```python
# Use optimized NumPy operations
import numpy as np
np.seterr(all='ignore')  # Suppress warnings for performance

# Consider using numba for critical loops (if available)
try:
    from numba import jit
    # Apply @jit decorator to performance-critical functions
except ImportError:
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest geolocation/test_geolocation.py -v

# Run specific test class
python -m pytest geolocation/test_geolocation.py::TestProjection -v

# Run with coverage
python -m pytest geolocation/test_geolocation.py --cov=geolocation --cov-report=html
```

### Test Categories

1. **Unit Tests**: Individual component testing
   - Camera calibration
   - Projection mathematics
   - Coordinate transformations
   - DEM operations

2. **Integration Tests**: End-to-end pipeline testing
   - Complete geolocation workflow
   - Multi-component interactions
   - Performance benchmarks

3. **Accuracy Tests**: Validation with known coordinates
   - Round-trip conversion accuracy
   - Known point validation
   - Error analysis

### Custom Test Data

```python
# Create test scenario
from geolocation.test_geolocation import TestIntegration

test = TestIntegration()
test.setUp()

# Run specific test
test.test_complete_pipeline()
test.test_accuracy_assessment()
```

## Integration with Foresight

### Main Application Integration

```python
# In main.py or detection pipeline
from geolocation import GeolocationService

class SARSystem:
    def __init__(self):
        self.geolocation_service = GeolocationService(
            calibration_data=self.load_camera_calibration(),
            reference_point=self.get_reference_point()
        )
    
    def process_detection(self, detection_bbox, camera_pose):
        # Get center of detection
        center_x = (detection_bbox[0] + detection_bbox[2]) / 2
        center_y = (detection_bbox[1] + detection_bbox[3]) / 2
        
        # Geolocate detection
        result = self.geolocation_service.geolocate_pixel(
            pixel_coords=(center_x, center_y),
            camera_pose=camera_pose,
            use_dem_correction=True
        )
        
        return result
```

### UI Integration

```javascript
// In sar_interface.js
class GeolocationDisplay {
    updateDetectionLocation(detection, geolocation_result) {
        if (geolocation_result && geolocation_result.confidence > 0.5) {
            const lat = geolocation_result.geodetic_coordinate.latitude;
            const lon = geolocation_result.geodetic_coordinate.longitude;
            
            // Update map marker
            this.addMapMarker(lat, lon, detection.class_name);
            
            // Update coordinate display
            this.updateCoordinateDisplay(lat, lon, geolocation_result.confidence);
        }
    }
}
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```python
   # Ensure all dependencies are installed
   pip install numpy opencv-python rasterio pyproj
   ```

2. **DEM Loading Issues**
   ```python
   # Check if rasterio is available
   try:
       import rasterio
       print("Rasterio available")
   except ImportError:
       print("Install rasterio: pip install rasterio")
   ```

3. **Coordinate Conversion Errors**
   ```python
   # Validate coordinate ranges
   assert -90 <= latitude <= 90
   assert -180 <= longitude <= 180
   ```

4. **Low Geolocation Confidence**
   - Check camera calibration accuracy
   - Verify camera pose estimation
   - Ensure DEM coverage for area of interest
   - Validate pixel coordinates are within image bounds

### Debugging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Check geolocation service status
info = service.get_service_info()
print(f"Service status: {info}")

# Validate camera calibration
fov_h, fov_v = service.projector.get_field_of_view()
print(f"Field of view: {fov_h:.1f}° x {fov_v:.1f}°")

# Check DEM coverage
dem_info = service.dem_corrector.get_dem_info()
print(f"DEM info: {dem_info}")
```

### Performance Issues

1. **Slow DEM Operations**
   - Reduce DEM resolution
   - Enable caching
   - Use appropriate interpolation method

2. **Memory Usage**
   - Limit cache size
   - Process in batches
   - Clear cache periodically

3. **Accuracy vs Speed**
   - Adjust convergence tolerance
   - Reduce max iterations
   - Use simpler interpolation methods

## API Reference

### Core Classes

- `GeolocationService`: Main service class
- `CalibrationData`: Camera calibration parameters
- `CameraPose`: Camera position and orientation
- `GeolocationResult`: Geolocation output with confidence
- `PinholeProjector`: Pinhole camera projection
- `CoordinateTransformer`: Coordinate system conversions
- `DEMCorrector`: Digital elevation model operations

### Data Classes

- `GeodeticCoordinate`: Latitude, longitude, altitude
- `UTMCoordinate`: UTM projection coordinates
- `Ray3D`: 3D ray representation
- `DEMTile`: Digital elevation model tile
- `DEMMetadata`: DEM metadata information

### Enums

- `CameraModel`: Supported camera models
- `DistortionModel`: Distortion correction models
- `CoordinateSystem`: Coordinate system types
- `GeolocationMethod`: Geolocation algorithms
- `ConfidenceLevel`: Confidence thresholds
- `InterpolationMethod`: DEM interpolation methods
- `DEMFormat`: Supported DEM formats

## Contributing

### Development Setup

```bash
# Clone repository
git clone <repository_url>
cd foresight

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Install in development mode
pip install -e .
```

### Code Style

```bash
# Format code
black geolocation/

# Check style
flake8 geolocation/

# Type checking (if using mypy)
mypy geolocation/
```

### Testing Guidelines

1. Write tests for all new functionality
2. Maintain >90% code coverage
3. Include integration tests for complex workflows
4. Test with realistic data scenarios
5. Validate accuracy with known coordinates

### Submitting Changes

1. Create feature branch
2. Write tests for new functionality
3. Ensure all tests pass
4. Update documentation
5. Submit pull request

## License

This module is part of the Foresight SAR system and is subject to the project's license terms.

## Support

For technical support or questions:

1. Check the troubleshooting section
2. Review test cases for usage examples
3. Consult the API documentation
4. Submit issues through the project repository

---

*Last updated: 2024*
*Version: 1.0.0*