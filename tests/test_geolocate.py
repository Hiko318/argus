#!/usr/bin/env python3
"""
Unit tests for the Geolocation Pipeline

Tests pixel-to-geographic coordinate transformation with synthetic data
and validates coordinate system transformations.

Author: Foresight SAR Team
Date: 2024-01-15
"""

import unittest
import math
import numpy as np
from pathlib import Path
import sys
import tempfile
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "backend"))

from geolocate import (
    GeolocationPipeline,
    Point2D,
    Point3D,
    GeographicCoordinate,
    TelemetryData,
    CameraIntrinsics,
    CoordinateTransforms,
    DEMProvider
)

class MockDEMProvider(DEMProvider):
    """Mock DEM provider for testing"""
    
    def __init__(self):
        # Don't call parent __init__ to avoid creating cache directory
        pass
    
    def get_elevation(self, latitude: float, longitude: float) -> float:
        """Return synthetic elevation based on coordinates"""
        # Simple synthetic terrain: elevation increases with distance from origin
        distance = math.sqrt(latitude**2 + longitude**2)
        return max(0, 50 * math.sin(distance * 0.1))  # Sinusoidal terrain

class TestCoordinateTransforms(unittest.TestCase):
    """Test coordinate system transformations"""
    
    def test_rotation_matrix_identity(self):
        """Test rotation matrix with zero angles"""
        R = CoordinateTransforms.rotation_matrix_from_euler(0, 0, 0)
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(R, expected, decimal=10)
    
    def test_rotation_matrix_90_degrees(self):
        """Test rotation matrix with 90-degree rotations"""
        # 90-degree yaw rotation
        R_yaw = CoordinateTransforms.rotation_matrix_from_euler(0, 0, 90)
        test_vector = np.array([1, 0, 0])  # Point along x-axis
        result = R_yaw @ test_vector
        expected = np.array([0, 1, 0])  # Should point along y-axis
        np.testing.assert_array_almost_equal(result, expected, decimal=10)
    
    def test_wgs84_enu_roundtrip(self):
        """Test WGS84 to ENU and back conversion"""
        # Reference point (San Francisco)
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
        
        # Test point (nearby)
        test_lat, test_lon, test_alt = 37.7750, -122.4193, 105.0
        
        # Convert to ENU
        enu_point = CoordinateTransforms.wgs84_to_enu(
            test_lat, test_lon, test_alt,
            ref_lat, ref_lon, ref_alt
        )
        
        # Convert back to WGS84
        result = CoordinateTransforms.enu_to_wgs84(
            enu_point.x, enu_point.y, enu_point.z,
            ref_lat, ref_lon, ref_alt
        )
        
        # Check roundtrip accuracy
        self.assertAlmostEqual(result.latitude, test_lat, places=6)
        self.assertAlmostEqual(result.longitude, test_lon, places=6)
        self.assertAlmostEqual(result.altitude, test_alt, places=1)
    
    def test_enu_coordinate_system(self):
        """Test ENU coordinate system orientation"""
        ref_lat, ref_lon, ref_alt = 0.0, 0.0, 0.0
        
        # Point to the east
        east_point = CoordinateTransforms.wgs84_to_enu(
            0.0, 0.001, 0.0,  # Slightly east
            ref_lat, ref_lon, ref_alt
        )
        self.assertGreater(east_point.x, 0)  # East should be positive x
        self.assertAlmostEqual(east_point.y, 0, places=1)  # North should be ~0
        
        # Point to the north
        north_point = CoordinateTransforms.wgs84_to_enu(
            0.001, 0.0, 0.0,  # Slightly north
            ref_lat, ref_lon, ref_alt
        )
        self.assertAlmostEqual(north_point.x, 0, places=1)  # East should be ~0
        self.assertGreater(north_point.y, 0)  # North should be positive y
        
        # Point above
        up_point = CoordinateTransforms.wgs84_to_enu(
            0.0, 0.0, 100.0,  # 100m above
            ref_lat, ref_lon, ref_alt
        )
        self.assertAlmostEqual(up_point.x, 0, places=1)  # East should be ~0
        self.assertAlmostEqual(up_point.y, 0, places=1)  # North should be ~0
        self.assertAlmostEqual(up_point.z, 100, places=1)  # Up should be 100m

class TestCameraIntrinsics(unittest.TestCase):
    """Test camera intrinsics and distortion correction"""
    
    def setUp(self):
        """Set up test camera intrinsics"""
        self.intrinsics = CameraIntrinsics(
            fx=1000.0, fy=1000.0,
            cx=500.0, cy=400.0,
            k1=-0.1, k2=0.05, k3=0.0,
            p1=0.001, p2=-0.001,
            image_width=1000, image_height=800
        )
    
    def test_principal_point_undistortion(self):
        """Test that principal point remains unchanged after undistortion"""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                "camera_profiles": {
                    "test_camera": {
                        "camera_matrix": {
                            "fx": self.intrinsics.fx,
                            "fy": self.intrinsics.fy,
                            "cx": self.intrinsics.cx,
                            "cy": self.intrinsics.cy
                        },
                        "distortion": {
                            "k1": self.intrinsics.k1,
                            "k2": self.intrinsics.k2,
                            "k3": self.intrinsics.k3,
                            "p1": self.intrinsics.p1,
                            "p2": self.intrinsics.p2
                        },
                        "image_width_px": self.intrinsics.image_width,
                        "image_height_px": self.intrinsics.image_height
                    }
                }
            }
            json.dump(config, f)
            config_path = f.name
        
        try:
            pipeline = GeolocationPipeline(config_path)
            
            # Principal point should not be affected by distortion correction
            principal_point = Point2D(self.intrinsics.cx, self.intrinsics.cy)
            undistorted = pipeline.undistort_pixel(principal_point, self.intrinsics)
            
            self.assertAlmostEqual(undistorted.u, principal_point.u, places=3)
            self.assertAlmostEqual(undistorted.v, principal_point.v, places=3)
        finally:
            Path(config_path).unlink()
    
    def test_pixel_to_ray_center(self):
        """Test pixel to ray conversion for image center"""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                "camera_profiles": {
                    "test_camera": {
                        "camera_matrix": {
                            "fx": self.intrinsics.fx,
                            "fy": self.intrinsics.fy,
                            "cx": self.intrinsics.cx,
                            "cy": self.intrinsics.cy
                        },
                        "distortion": {
                            "k1": self.intrinsics.k1,
                            "k2": self.intrinsics.k2,
                            "k3": self.intrinsics.k3,
                            "p1": self.intrinsics.p1,
                            "p2": self.intrinsics.p2
                        },
                        "image_width_px": self.intrinsics.image_width,
                        "image_height_px": self.intrinsics.image_height
                    }
                }
            }
            json.dump(config, f)
            config_path = f.name
        
        try:
            pipeline = GeolocationPipeline(config_path)
            
            # Center pixel should produce ray pointing straight ahead
            center_pixel = Point2D(self.intrinsics.cx, self.intrinsics.cy)
            ray = pipeline.pixel_to_ray(center_pixel, self.intrinsics)
            
            # Ray should point forward (positive z) with minimal x,y components
            self.assertAlmostEqual(ray.x, 0.0, places=3)
            self.assertAlmostEqual(ray.y, 0.0, places=3)
            self.assertAlmostEqual(ray.z, 1.0, places=3)
        finally:
            Path(config_path).unlink()

class TestGeolocationPipeline(unittest.TestCase):
    """Test the main geolocation pipeline"""
    
    def setUp(self):
        """Set up test pipeline with mock DEM"""
        # Create temporary config file
        self.config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        config = {
            "camera_profiles": {
                "test_camera": {
                    "camera_matrix": {
                        "fx": 1000.0,
                        "fy": 1000.0,
                        "cx": 500.0,
                        "cy": 400.0
                    },
                    "distortion": {
                        "k1": 0.0,
                        "k2": 0.0,
                        "k3": 0.0,
                        "p1": 0.0,
                        "p2": 0.0
                    },
                    "image_width_px": 1000,
                    "image_height_px": 800
                }
            }
        }
        json.dump(config, self.config_file)
        self.config_file.close()
        
        self.pipeline = GeolocationPipeline(self.config_file.name)
        self.pipeline.dem_provider = MockDEMProvider()
    
    def tearDown(self):
        """Clean up temporary files"""
        Path(self.config_file.name).unlink()
    
    def test_nadir_projection(self):
        """Test projection with camera pointing straight down"""
        # Aircraft at 100m altitude, looking straight down
        telemetry = TelemetryData(
            latitude=37.7749,
            longitude=-122.4194,
            altitude=100.0,
            roll=0.0,
            pitch=-90.0,  # Looking straight down
            yaw=0.0,
            gimbal_pitch=0.0,
            gimbal_yaw=0.0,
            gimbal_roll=0.0,
            timestamp=1642262400.0
        )
        
        # Center pixel should project to point directly below aircraft
        center_pixel = Point2D(500.0, 400.0)
        result = self.pipeline.project_pixel_to_geo(
            center_pixel, telemetry, "test_camera"
        )
        
        self.assertIsNotNone(result)
        # Should be very close to aircraft position (within 0.001 degrees)
        self.assertAlmostEqual(result.latitude, telemetry.latitude, places=3)
        self.assertAlmostEqual(result.longitude, telemetry.longitude, places=3)
        # Altitude should be terrain elevation (from mock DEM)
        self.assertLess(result.altitude, telemetry.altitude)
    
    def test_oblique_projection(self):
        """Test projection with camera at oblique angle"""
        # Aircraft looking forward and down
        telemetry = TelemetryData(
            latitude=37.7749,
            longitude=-122.4194,
            altitude=100.0,
            roll=0.0,
            pitch=-45.0,  # Looking 45 degrees down
            yaw=0.0,      # Facing north
            gimbal_pitch=0.0,
            gimbal_yaw=0.0,
            gimbal_roll=0.0,
            timestamp=1642262400.0
        )
        
        # Center pixel should project to point north of aircraft
        center_pixel = Point2D(500.0, 400.0)
        result = self.pipeline.project_pixel_to_geo(
            center_pixel, telemetry, "test_camera"
        )
        
        self.assertIsNotNone(result)
        # Should be north of aircraft position
        self.assertGreater(result.latitude, telemetry.latitude)
        # Longitude should be similar
        self.assertAlmostEqual(result.longitude, telemetry.longitude, places=2)
    
    def test_out_of_bounds_pixel(self):
        """Test handling of out-of-bounds pixel coordinates"""
        telemetry = TelemetryData(
            latitude=37.7749, longitude=-122.4194, altitude=100.0,
            roll=0.0, pitch=-90.0, yaw=0.0,
            gimbal_pitch=0.0, gimbal_yaw=0.0, gimbal_roll=0.0,
            timestamp=1642262400.0
        )
        
        # Pixel outside image bounds
        out_of_bounds_pixel = Point2D(1500.0, 1200.0)
        result = self.pipeline.project_pixel_to_geo(
            out_of_bounds_pixel, telemetry, "test_camera"
        )
        
        self.assertIsNone(result)
    
    def test_batch_projection(self):
        """Test batch pixel projection"""
        telemetry = TelemetryData(
            latitude=37.7749, longitude=-122.4194, altitude=100.0,
            roll=0.0, pitch=-90.0, yaw=0.0,
            gimbal_pitch=0.0, gimbal_yaw=0.0, gimbal_roll=0.0,
            timestamp=1642262400.0
        )
        
        # Multiple test pixels
        pixels = [
            Point2D(500.0, 400.0),  # Center
            Point2D(250.0, 200.0),  # Top-left quadrant
            Point2D(750.0, 600.0),  # Bottom-right quadrant
            Point2D(1500.0, 1200.0)  # Out of bounds
        ]
        
        results = self.pipeline.batch_project_pixels(
            pixels, telemetry, "test_camera"
        )
        
        self.assertEqual(len(results), len(pixels))
        # First three should succeed, last should fail
        self.assertIsNotNone(results[0])
        self.assertIsNotNone(results[1])
        self.assertIsNotNone(results[2])
        self.assertIsNone(results[3])
    
    def test_camera_footprint(self):
        """Test camera footprint calculation"""
        telemetry = TelemetryData(
            latitude=37.7749, longitude=-122.4194, altitude=100.0,
            roll=0.0, pitch=-90.0, yaw=0.0,
            gimbal_pitch=0.0, gimbal_yaw=0.0, gimbal_roll=0.0,
            timestamp=1642262400.0
        )
        
        footprint = self.pipeline.get_camera_footprint(
            telemetry, "test_camera", grid_size=3
        )
        
        # Should have 9 points (3x3 grid)
        self.assertEqual(len(footprint), 9)
        
        # All points should be valid geographic coordinates
        for point in footprint:
            self.assertIsInstance(point, GeographicCoordinate)
            self.assertTrue(-90 <= point.latitude <= 90)
            self.assertTrue(-180 <= point.longitude <= 180)
    
    def test_gimbal_rotation_effect(self):
        """Test effect of gimbal rotation on projection"""
        base_telemetry = TelemetryData(
            latitude=37.7749, longitude=-122.4194, altitude=100.0,
            roll=0.0, pitch=-90.0, yaw=0.0,
            gimbal_pitch=0.0, gimbal_yaw=0.0, gimbal_roll=0.0,
            timestamp=1642262400.0
        )
        
        # Telemetry with gimbal yaw rotation
        rotated_telemetry = TelemetryData(
            latitude=37.7749, longitude=-122.4194, altitude=100.0,
            roll=0.0, pitch=-90.0, yaw=0.0,
            gimbal_pitch=0.0, gimbal_yaw=45.0, gimbal_roll=0.0,  # 45-degree yaw
            timestamp=1642262400.0
        )
        
        center_pixel = Point2D(500.0, 400.0)
        
        # Project with both telemetry sets
        result_base = self.pipeline.project_pixel_to_geo(
            center_pixel, base_telemetry, "test_camera"
        )
        result_rotated = self.pipeline.project_pixel_to_geo(
            center_pixel, rotated_telemetry, "test_camera"
        )
        
        self.assertIsNotNone(result_base)
        self.assertIsNotNone(result_rotated)
        
        # Results should be different due to gimbal rotation
        self.assertNotAlmostEqual(
            result_base.longitude, result_rotated.longitude, places=3
        )

class TestSyntheticScenarios(unittest.TestCase):
    """Test with synthetic scenarios to validate expected behavior"""
    
    def setUp(self):
        """Set up test pipeline"""
        # Create temporary config file
        self.config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        config = {
            "camera_profiles": {
                "test_camera": {
                    "camera_matrix": {
                        "fx": 1000.0,
                        "fy": 1000.0,
                        "cx": 500.0,
                        "cy": 400.0
                    },
                    "distortion": {
                        "k1": 0.0, "k2": 0.0, "k3": 0.0,
                        "p1": 0.0, "p2": 0.0
                    },
                    "image_width_px": 1000,
                    "image_height_px": 800
                }
            }
        }
        json.dump(config, self.config_file)
        self.config_file.close()
        
        self.pipeline = GeolocationPipeline(self.config_file.name)
        self.pipeline.dem_provider = MockDEMProvider()
    
    def tearDown(self):
        """Clean up"""
        Path(self.config_file.name).unlink()
    
    def test_altitude_effect_on_footprint_size(self):
        """Test that higher altitude results in larger footprint"""
        # Low altitude
        low_alt_telemetry = TelemetryData(
            latitude=37.7749, longitude=-122.4194, altitude=50.0,
            roll=0.0, pitch=-90.0, yaw=0.0,
            gimbal_pitch=0.0, gimbal_yaw=0.0, gimbal_roll=0.0,
            timestamp=1642262400.0
        )
        
        # High altitude
        high_alt_telemetry = TelemetryData(
            latitude=37.7749, longitude=-122.4194, altitude=200.0,
            roll=0.0, pitch=-90.0, yaw=0.0,
            gimbal_pitch=0.0, gimbal_yaw=0.0, gimbal_roll=0.0,
            timestamp=1642262400.0
        )
        
        # Get footprints
        low_footprint = self.pipeline.get_camera_footprint(
            low_alt_telemetry, "test_camera", grid_size=3
        )
        high_footprint = self.pipeline.get_camera_footprint(
            high_alt_telemetry, "test_camera", grid_size=3
        )
        
        # Calculate footprint spans
        def calculate_span(footprint):
            if not footprint:
                return 0, 0
            lats = [p.latitude for p in footprint]
            lons = [p.longitude for p in footprint]
            return max(lats) - min(lats), max(lons) - min(lons)
        
        low_lat_span, low_lon_span = calculate_span(low_footprint)
        high_lat_span, high_lon_span = calculate_span(high_footprint)
        
        # Higher altitude should result in larger footprint
        self.assertGreater(high_lat_span, low_lat_span)
        self.assertGreater(high_lon_span, low_lon_span)
    
    def test_consistency_across_image_regions(self):
        """Test that projections are consistent across different image regions"""
        telemetry = TelemetryData(
            latitude=37.7749, longitude=-122.4194, altitude=100.0,
            roll=0.0, pitch=-90.0, yaw=0.0,
            gimbal_pitch=0.0, gimbal_yaw=0.0, gimbal_roll=0.0,
            timestamp=1642262400.0
        )
        
        # Test pixels in different image regions
        test_pixels = [
            Point2D(100.0, 100.0),   # Top-left
            Point2D(900.0, 100.0),   # Top-right
            Point2D(100.0, 700.0),   # Bottom-left
            Point2D(900.0, 700.0),   # Bottom-right
            Point2D(500.0, 400.0),   # Center
        ]
        
        results = []
        for pixel in test_pixels:
            result = self.pipeline.project_pixel_to_geo(
                pixel, telemetry, "test_camera"
            )
            results.append(result)
        
        # All projections should succeed
        for i, result in enumerate(results):
            self.assertIsNotNone(result, f"Projection failed for pixel {test_pixels[i]}")
        
        # Results should form a reasonable spatial pattern
        # (this is a basic sanity check)
        center_result = results[4]
        for result in results[:4]:
            # Corner points should be farther from aircraft than center
            center_dist = abs(center_result.latitude - telemetry.latitude) + \
                         abs(center_result.longitude - telemetry.longitude)
            corner_dist = abs(result.latitude - telemetry.latitude) + \
                         abs(result.longitude - telemetry.longitude)
            self.assertGreaterEqual(corner_dist, center_dist * 0.5)

def run_all_tests():
    """Run all geolocation tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestCoordinateTransforms,
        TestCameraIntrinsics,
        TestGeolocationPipeline,
        TestSyntheticScenarios
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)