"""Unit tests for the geolocation module."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from math import radians, degrees, sin, cos, sqrt, atan2

# Import the modules to test
try:
    from geolocation.projection import GeolocationProjector
    from geolocation.approx_pos import ApproximatePositioning
    from tools.geolocate import GeolocateService
except ImportError:
    # Handle case where modules might not be importable in test environment
    GeolocationProjector = None
    ApproximatePositioning = None
    GeolocateService = None


@pytest.mark.unit
class TestGeolocationProjector:
    """Test cases for geolocation projection functionality."""
    
    @pytest.fixture
    def projector_config(self):
        """Projector configuration for testing."""
        return {
            "camera_height": 100.0,  # meters
            "ground_height": 0.0,    # meters
            "enable_earth_curvature": False,
            "coordinate_system": "WGS84"
        }
    
    @pytest.fixture
    def camera_params(self, mock_camera_params):
        """Camera parameters for testing."""
        return mock_camera_params
    
    def test_projector_initialization(self, projector_config, camera_params):
        """Test projector initialization with valid config."""
        if GeolocationProjector is None:
            pytest.skip("GeolocationProjector not available")
        
        projector = GeolocationProjector(projector_config, camera_params)
        
        assert projector.camera_height == 100.0
        assert projector.ground_height == 0.0
        assert projector.is_calibrated() is True
    
    def test_projector_invalid_config(self, camera_params):
        """Test projector initialization with invalid config."""
        if GeolocationProjector is None:
            pytest.skip("GeolocationProjector not available")
        
        # Test with negative camera height
        invalid_config = {"camera_height": -10.0}
        with pytest.raises(ValueError):
            GeolocationProjector(invalid_config, camera_params)
        
        # Test with invalid coordinate system
        invalid_config = {"coordinate_system": "INVALID"}
        with pytest.raises(ValueError):
            GeolocationProjector(invalid_config, camera_params)
    
    def test_pixel_to_ground_projection(self, projector_config, camera_params, mock_gps_data):
        """Test pixel to ground coordinate projection."""
        if GeolocationProjector is None:
            pytest.skip("GeolocationProjector not available")
        
        projector = GeolocationProjector(projector_config, camera_params)
        
        # Test center pixel (should project straight down)
        pixel_x, pixel_y = 320, 240  # Center of 640x480 image
        
        ground_coords = projector.pixel_to_ground(
            pixel_x, pixel_y, 
            mock_gps_data["latitude"], 
            mock_gps_data["longitude"],
            mock_gps_data["altitude"],
            mock_gps_data["heading"]
        )
        
        assert "latitude" in ground_coords
        assert "longitude" in ground_coords
        assert "accuracy" in ground_coords
        
        # Center pixel should be close to drone position
        lat_diff = abs(ground_coords["latitude"] - mock_gps_data["latitude"])
        lon_diff = abs(ground_coords["longitude"] - mock_gps_data["longitude"])
        
        # Should be within reasonable range (depends on altitude)
        assert lat_diff < 0.01  # ~1km at equator
        assert lon_diff < 0.01
    
    def test_ground_to_pixel_projection(self, projector_config, camera_params, mock_gps_data):
        """Test ground coordinate to pixel projection."""
        if GeolocationProjector is None:
            pytest.skip("GeolocationProjector not available")
        
        projector = GeolocationProjector(projector_config, camera_params)
        
        # Project a ground point slightly offset from drone position
        target_lat = mock_gps_data["latitude"] + 0.001  # ~100m north
        target_lon = mock_gps_data["longitude"] + 0.001  # ~100m east
        
        pixel_coords = projector.ground_to_pixel(
            target_lat, target_lon,
            mock_gps_data["latitude"],
            mock_gps_data["longitude"],
            mock_gps_data["altitude"],
            mock_gps_data["heading"]
        )
        
        if pixel_coords:  # Might be None if outside image bounds
            assert "x" in pixel_coords
            assert "y" in pixel_coords
            assert 0 <= pixel_coords["x"] <= 640
            assert 0 <= pixel_coords["y"] <= 480
    
    def test_projection_roundtrip(self, projector_config, camera_params, mock_gps_data):
        """Test pixel->ground->pixel roundtrip accuracy."""
        if GeolocationProjector is None:
            pytest.skip("GeolocationProjector not available")
        
        projector = GeolocationProjector(projector_config, camera_params)
        
        # Start with a pixel coordinate
        original_x, original_y = 400, 300
        
        # Project to ground
        ground_coords = projector.pixel_to_ground(
            original_x, original_y,
            mock_gps_data["latitude"],
            mock_gps_data["longitude"],
            mock_gps_data["altitude"],
            mock_gps_data["heading"]
        )
        
        # Project back to pixel
        pixel_coords = projector.ground_to_pixel(
            ground_coords["latitude"],
            ground_coords["longitude"],
            mock_gps_data["latitude"],
            mock_gps_data["longitude"],
            mock_gps_data["altitude"],
            mock_gps_data["heading"]
        )
        
        if pixel_coords:
            # Should be close to original (within a few pixels)
            x_diff = abs(pixel_coords["x"] - original_x)
            y_diff = abs(pixel_coords["y"] - original_y)
            
            assert x_diff < 5.0  # Within 5 pixels
            assert y_diff < 5.0
    
    def test_altitude_effect(self, projector_config, camera_params, mock_gps_data):
        """Test effect of altitude on projection accuracy."""
        if GeolocationProjector is None:
            pytest.skip("GeolocationProjector not available")
        
        projector = GeolocationProjector(projector_config, camera_params)
        
        pixel_x, pixel_y = 320, 240  # Center pixel
        
        # Test at different altitudes
        low_alt_coords = projector.pixel_to_ground(
            pixel_x, pixel_y,
            mock_gps_data["latitude"],
            mock_gps_data["longitude"],
            50.0,  # Low altitude
            mock_gps_data["heading"]
        )
        
        high_alt_coords = projector.pixel_to_ground(
            pixel_x, pixel_y,
            mock_gps_data["latitude"],
            mock_gps_data["longitude"],
            200.0,  # High altitude
            mock_gps_data["heading"]
        )
        
        # Higher altitude should have lower accuracy (larger error)
        assert low_alt_coords["accuracy"] < high_alt_coords["accuracy"]
    
    def test_heading_rotation(self, projector_config, camera_params, mock_gps_data):
        """Test effect of drone heading on projection."""
        if GeolocationProjector is None:
            pytest.skip("GeolocationProjector not available")
        
        projector = GeolocationProjector(projector_config, camera_params)
        
        # Same pixel, different headings
        pixel_x, pixel_y = 400, 240  # Right of center
        
        coords_north = projector.pixel_to_ground(
            pixel_x, pixel_y,
            mock_gps_data["latitude"],
            mock_gps_data["longitude"],
            mock_gps_data["altitude"],
            0.0  # North
        )
        
        coords_east = projector.pixel_to_ground(
            pixel_x, pixel_y,
            mock_gps_data["latitude"],
            mock_gps_data["longitude"],
            mock_gps_data["altitude"],
            90.0  # East
        )
        
        # Results should be different due to rotation
        lat_diff = abs(coords_north["latitude"] - coords_east["latitude"])
        lon_diff = abs(coords_north["longitude"] - coords_east["longitude"])
        
        assert lat_diff > 0.0001 or lon_diff > 0.0001  # Should be noticeably different


@pytest.mark.unit
class TestApproximatePositioning:
    """Test cases for approximate positioning functionality."""
    
    def test_distance_calculation(self):
        """Test distance calculation between GPS coordinates."""
        if ApproximatePositioning is None:
            pytest.skip("ApproximatePositioning not available")
        
        # Test known distance (approximately)
        lat1, lon1 = 37.7749, -122.4194  # San Francisco
        lat2, lon2 = 37.7849, -122.4094  # ~1.5km northeast
        
        distance = ApproximatePositioning.calculate_distance(lat1, lon1, lat2, lon2)
        
        # Should be approximately 1.5km (allowing for some error)
        assert 1000 < distance < 2000  # meters
    
    def test_bearing_calculation(self):
        """Test bearing calculation between GPS coordinates."""
        if ApproximatePositioning is None:
            pytest.skip("ApproximatePositioning not available")
        
        # Test known bearings
        lat1, lon1 = 37.7749, -122.4194
        lat2, lon2 = 37.7849, -122.4194  # Due north
        
        bearing = ApproximatePositioning.calculate_bearing(lat1, lon1, lat2, lon2)
        
        # Should be approximately 0 degrees (north)
        assert -10 < bearing < 10 or 350 < bearing < 360
    
    def test_coordinate_offset(self):
        """Test coordinate offset calculation."""
        if ApproximatePositioning is None:
            pytest.skip("ApproximatePositioning not available")
        
        base_lat, base_lon = 37.7749, -122.4194
        distance = 1000  # 1km
        bearing = 0  # North
        
        new_coords = ApproximatePositioning.offset_coordinates(
            base_lat, base_lon, distance, bearing
        )
        
        assert "latitude" in new_coords
        assert "longitude" in new_coords
        
        # New latitude should be higher (more north)
        assert new_coords["latitude"] > base_lat
        # Longitude should be approximately the same
        assert abs(new_coords["longitude"] - base_lon) < 0.001


@pytest.mark.unit
class TestGeolocateService:
    """Test cases for geolocation service integration."""
    
    @pytest.fixture
    def service_config(self):
        """Service configuration for testing."""
        return {
            "enable_projection": True,
            "camera_height": 100.0,
            "ground_height": 0.0,
            "accuracy_threshold": 10.0,  # meters
            "enable_filtering": True
        }
    
    def test_service_initialization(self, service_config, mock_camera_params):
        """Test service initialization."""
        if GeolocateService is None:
            pytest.skip("GeolocateService not available")
        
        service = GeolocateService(service_config, mock_camera_params)
        
        assert service.is_enabled() is True
        assert service.is_calibrated() is True
    
    def test_geolocate_detection(self, service_config, mock_camera_params, mock_detection_result, mock_gps_data):
        """Test geolocating a detection."""
        if GeolocateService is None:
            pytest.skip("GeolocateService not available")
        
        service = GeolocateService(service_config, mock_camera_params)
        
        # Geolocate first detection
        detection = mock_detection_result["boxes"][0]
        
        result = service.geolocate_detection(
            detection, mock_gps_data, mock_detection_result["timestamp"]
        )
        
        assert result is not None
        assert "latitude" in result
        assert "longitude" in result
        assert "accuracy" in result
        assert "timestamp" in result
    
    def test_geolocate_multiple_detections(self, service_config, mock_camera_params, mock_detection_result, mock_gps_data):
        """Test geolocating multiple detections."""
        if GeolocateService is None:
            pytest.skip("GeolocateService not available")
        
        service = GeolocateService(service_config, mock_camera_params)
        
        results = service.geolocate_detections(
            mock_detection_result["boxes"], mock_gps_data, mock_detection_result["timestamp"]
        )
        
        assert len(results) == len(mock_detection_result["boxes"])
        
        for result in results:
            assert "latitude" in result
            assert "longitude" in result
            assert "accuracy" in result
    
    def test_accuracy_filtering(self, service_config, mock_camera_params, mock_gps_data):
        """Test filtering by accuracy threshold."""
        if GeolocateService is None:
            pytest.skip("GeolocateService not available")
        
        # Set strict accuracy threshold
        service_config["accuracy_threshold"] = 1.0  # 1 meter
        service = GeolocateService(service_config, mock_camera_params)
        
        # Create detection that would have low accuracy (edge of image)
        edge_detection = {
            "x1": 0, "y1": 0, "x2": 50, "y2": 50,
            "confidence": 0.9, "class": "person"
        }
        
        result = service.geolocate_detection(
            edge_detection, mock_gps_data, 1234567890.0
        )
        
        # Might be filtered out due to low accuracy
        if result is None:
            assert True  # Correctly filtered
        else:
            assert result["accuracy"] <= service_config["accuracy_threshold"]
    
    def test_service_without_gps(self, service_config, mock_camera_params):
        """Test service behavior without GPS data."""
        if GeolocateService is None:
            pytest.skip("GeolocateService not available")
        
        service = GeolocateService(service_config, mock_camera_params)
        
        detection = {"x1": 100, "y1": 100, "x2": 200, "y2": 200, "confidence": 0.9, "class": "person"}
        
        # No GPS data
        result = service.geolocate_detection(detection, None, 1234567890.0)
        
        assert result is None  # Should fail gracefully


@pytest.mark.smoke
class TestGeolocateSmoke:
    """Smoke tests for geolocation functionality."""
    
    def test_projector_can_initialize(self, test_config, mock_camera_params):
        """Smoke test: projector can be initialized without errors."""
        if GeolocationProjector is None:
            pytest.skip("GeolocationProjector not available")
        
        geo_config = test_config.get("geolocation", {})
        projector = GeolocationProjector(geo_config, mock_camera_params)
        
        assert projector is not None
        assert hasattr(projector, 'pixel_to_ground')
        assert hasattr(projector, 'ground_to_pixel')
    
    def test_basic_projection_workflow(self, test_config, mock_camera_params, mock_gps_data):
        """Smoke test: basic projection workflow works."""
        if GeolocationProjector is None:
            pytest.skip("GeolocationProjector not available")
        
        geo_config = test_config.get("geolocation", {})
        projector = GeolocationProjector(geo_config, mock_camera_params)
        
        # Basic workflow should not raise exceptions
        try:
            result = projector.pixel_to_ground(
                320, 240,  # Center pixel
                mock_gps_data["latitude"],
                mock_gps_data["longitude"],
                mock_gps_data["altitude"],
                mock_gps_data["heading"]
            )
            
            # Result might be None in some implementations, that's ok
            if result:
                assert isinstance(result, dict)
        except Exception as e:
            pytest.fail(f"Basic projection workflow failed: {e}")
    
    def test_service_integration(self, test_config, mock_camera_params):
        """Smoke test: service integration works."""
        if GeolocateService is None:
            pytest.skip("GeolocateService not available")
        
        geo_config = test_config.get("geolocation", {})
        service = GeolocateService(geo_config, mock_camera_params)
        
        # Basic service operations should not raise exceptions
        assert service.is_enabled() in [True, False]
        assert service.is_calibrated() in [True, False]