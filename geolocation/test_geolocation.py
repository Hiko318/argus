#!/usr/bin/env python3
"""
Unit Tests for Foresight Geolocation Module

Comprehensive test suite for camera calibration, projection, coordinate
transformation, DEM correction, and geolocation services.
"""

import unittest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
from .camera_calibration import (
    CameraCalibrator, CalibrationData, CameraModel, DistortionModel
)
from .projection import (
    PinholeProjector, CameraPose, Ray3D, CoordinateSystem
)
from .coordinate_transform import (
    CoordinateTransformer, GeodeticCoordinate, UTMCoordinate, WGS84Constants
)
from .dem_correction import (
    DEMCorrector, DEMTile, DEMMetadata, InterpolationMethod, DEMFormat
)
from .geolocation_service import (
    GeolocationService, GeolocationResult, GeolocationConfig, 
    GeolocationMethod, ConfidenceLevel
)


class TestCameraCalibration(unittest.TestCase):
    """Test camera calibration functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Known camera parameters for testing
        self.camera_matrix = np.array([
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0]
        ])
        
        self.distortion_coeffs = np.array([0.1, -0.2, 0.001, 0.002, 0.05])
        
        self.calibration_data = CalibrationData(
            camera_matrix=self.camera_matrix,
            distortion_coeffs=self.distortion_coeffs,
            image_width=640,
            image_height=480,
            camera_model=CameraModel.PINHOLE,
            distortion_model=DistortionModel.RADIAL_TANGENTIAL
        )
    
    def test_calibration_data_creation(self):
        """Test CalibrationData creation and validation"""
        self.assertEqual(self.calibration_data.image_width, 640)
        self.assertEqual(self.calibration_data.image_height, 480)
        self.assertEqual(self.calibration_data.camera_model, CameraModel.PINHOLE)
        
        # Test field of view calculation
        fov_h, fov_v = self.calibration_data.get_field_of_view()
        self.assertGreater(fov_h, 0)
        self.assertGreater(fov_v, 0)
        self.assertLess(fov_h, 180)
        self.assertLess(fov_v, 180)
    
    def test_undistort_points(self):
        """Test point undistortion"""
        # Test with center point (should be unchanged)
        center_point = np.array([[320.0, 240.0]])
        undistorted = self.calibration_data.undistort_points(center_point)
        
        # Center point should be close to original (minimal distortion)
        np.testing.assert_allclose(undistorted, center_point, atol=1.0)
        
        # Test with corner point
        corner_point = np.array([[0.0, 0.0]])
        undistorted_corner = self.calibration_data.undistort_points(corner_point)
        
        # Should be different from original due to distortion
        self.assertFalse(np.allclose(undistorted_corner, corner_point))
    
    def test_calibration_validation(self):
        """Test calibration data validation"""
        # Test invalid camera matrix
        with self.assertRaises(ValueError):
            CalibrationData(
                camera_matrix=np.array([[1, 2], [3, 4]]),  # Wrong shape
                distortion_coeffs=self.distortion_coeffs,
                image_width=640,
                image_height=480
            )
        
        # Test invalid image dimensions
        with self.assertRaises(ValueError):
            CalibrationData(
                camera_matrix=self.camera_matrix,
                distortion_coeffs=self.distortion_coeffs,
                image_width=0,  # Invalid
                image_height=480
            )


class TestProjection(unittest.TestCase):
    """Test pinhole projection functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create calibration data
        camera_matrix = np.array([
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0]
        ])
        
        self.calibration_data = CalibrationData(
            camera_matrix=camera_matrix,
            distortion_coeffs=np.zeros(5),  # No distortion for simplicity
            image_width=640,
            image_height=480
        )
        
        self.projector = PinholeProjector(self.calibration_data)
        
        # Create test camera pose
        self.camera_pose = CameraPose(
            position=np.array([0.0, 0.0, 10.0]),  # 10m above origin
            rotation_matrix=np.eye(3)  # No rotation
        )
    
    def test_camera_pose_creation(self):
        """Test CameraPose creation and validation"""
        # Test with rotation matrix
        pose = CameraPose(
            position=np.array([1.0, 2.0, 3.0]),
            rotation_matrix=np.eye(3)
        )
        
        self.assertTrue(pose._is_valid_rotation_matrix(pose.rotation_matrix))
        
        # Test with Euler angles
        pose_euler = CameraPose(
            position=np.array([1.0, 2.0, 3.0]),
            euler_angles=np.array([0.1, 0.2, 0.3])
        )
        
        self.assertTrue(pose_euler._is_valid_rotation_matrix(pose_euler.rotation_matrix))
        
        # Test transformation matrix
        T = pose.get_transformation_matrix()
        self.assertEqual(T.shape, (4, 4))
        np.testing.assert_array_equal(T[:3, 3], pose.position)
    
    def test_pixel_to_ray(self):
        """Test pixel to ray conversion"""
        # Test center pixel
        center_pixel = (320.0, 240.0)
        ray = self.projector.pixel_to_ray(center_pixel)
        
        # Ray should point forward (positive Z)
        self.assertGreater(ray.direction[2], 0)
        
        # Test with camera pose
        ray_world = self.projector.pixel_to_ray(center_pixel, self.camera_pose)
        
        # Ray origin should be at camera position
        np.testing.assert_array_almost_equal(ray_world.origin, self.camera_pose.position)
    
    def test_ray_ground_intersection(self):
        """Test ray-ground intersection"""
        # Create ray pointing down
        ray = Ray3D(
            origin=np.array([0.0, 0.0, 10.0]),
            direction=np.array([0.0, 0.0, -1.0])
        )
        
        # Intersect with ground plane at z=0
        intersection = ray.intersect_ground_plane(0.0)
        
        self.assertIsNotNone(intersection)
        np.testing.assert_array_almost_equal(intersection, [0.0, 0.0, 0.0])
        
        # Test with angled ray
        angled_ray = Ray3D(
            origin=np.array([0.0, 0.0, 10.0]),
            direction=np.array([1.0, 0.0, -1.0])
        )
        
        intersection_angled = angled_ray.intersect_ground_plane(0.0)
        self.assertIsNotNone(intersection_angled)
        self.assertAlmostEqual(intersection_angled[2], 0.0)  # Should be on ground
    
    def test_project_point(self):
        """Test 3D point projection"""
        # Project point in front of camera
        point_3d = np.array([0.0, 0.0, 5.0])  # 5m in front
        pixel = self.projector.project_point(point_3d)
        
        self.assertIsNotNone(pixel)
        # Should project to center of image
        np.testing.assert_array_almost_equal(pixel, [320.0, 240.0], decimal=1)
        
        # Test point behind camera
        point_behind = np.array([0.0, 0.0, -5.0])
        pixel_behind = self.projector.project_point(point_behind)
        
        self.assertIsNone(pixel_behind)
    
    def test_field_of_view(self):
        """Test field of view calculation"""
        fov_h, fov_v = self.projector.get_field_of_view()
        
        # Should be reasonable values
        self.assertGreater(fov_h, 30)  # At least 30 degrees
        self.assertLess(fov_h, 120)    # Less than 120 degrees
        self.assertGreater(fov_v, 20)
        self.assertLess(fov_v, 90)


class TestCoordinateTransform(unittest.TestCase):
    """Test coordinate transformation functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Reference point in London
        self.reference_point = GeodeticCoordinate(
            latitude=51.5074,
            longitude=-0.1278,
            altitude=0.0
        )
        
        self.transformer = CoordinateTransformer(self.reference_point)
    
    def test_geodetic_coordinate(self):
        """Test GeodeticCoordinate creation and validation"""
        coord = GeodeticCoordinate(51.5074, -0.1278, 100.0)
        
        self.assertEqual(coord.latitude, 51.5074)
        self.assertEqual(coord.longitude, -0.1278)
        self.assertEqual(coord.altitude, 100.0)
        
        # Test validation
        with self.assertRaises(ValueError):
            GeodeticCoordinate(91.0, 0.0, 0.0)  # Invalid latitude
        
        with self.assertRaises(ValueError):
            GeodeticCoordinate(0.0, 181.0, 0.0)  # Invalid longitude
    
    def test_enu_conversion(self):
        """Test ENU coordinate conversion"""
        # Test reference point (should be origin)
        enu_origin = self.transformer.geodetic_to_enu(self.reference_point)
        np.testing.assert_array_almost_equal(enu_origin, [0.0, 0.0, 0.0], decimal=3)
        
        # Test point 1km north
        north_point = GeodeticCoordinate(
            latitude=self.reference_point.latitude + 0.009,  # ~1km north
            longitude=self.reference_point.longitude,
            altitude=0.0
        )
        
        enu_north = self.transformer.geodetic_to_enu(north_point)
        
        # Should be approximately [0, 1000, 0]
        self.assertLess(abs(enu_north[0]), 10)  # East component small
        self.assertGreater(enu_north[1], 900)   # North component ~1000m
        self.assertLess(abs(enu_north[2]), 10)  # Up component small
        
        # Test round-trip conversion
        geodetic_back = self.transformer.enu_to_geodetic(enu_north)
        
        self.assertAlmostEqual(geodetic_back.latitude, north_point.latitude, places=5)
        self.assertAlmostEqual(geodetic_back.longitude, north_point.longitude, places=5)
        self.assertAlmostEqual(geodetic_back.altitude, north_point.altitude, places=1)
    
    def test_utm_conversion(self):
        """Test UTM coordinate conversion"""
        utm_coord = self.transformer.geodetic_to_utm(self.reference_point)
        
        self.assertIsInstance(utm_coord, UTMCoordinate)
        self.assertGreater(utm_coord.easting, 0)
        self.assertGreater(utm_coord.northing, 0)
        self.assertIsInstance(utm_coord.zone_number, int)
        self.assertIn(utm_coord.zone_letter, 'CDEFGHJKLMNPQRSTUVWX')
        
        # Test round-trip conversion
        geodetic_back = self.transformer.utm_to_geodetic(utm_coord)
        
        self.assertAlmostEqual(geodetic_back.latitude, self.reference_point.latitude, places=5)
        self.assertAlmostEqual(geodetic_back.longitude, self.reference_point.longitude, places=5)


class TestDEMCorrection(unittest.TestCase):
    """Test DEM correction functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.dem_corrector = DEMCorrector()
        
        # Create synthetic DEM for testing
        self.test_bounds = (-1.0, 51.0, 1.0, 53.0)  # Around London
        self.dem_tile = self.dem_corrector.create_synthetic_dem(
            bounds=self.test_bounds,
            resolution=30.0,
            base_elevation=50.0,
            terrain_variation=100.0
        )
    
    def test_dem_tile_creation(self):
        """Test DEM tile creation"""
        self.assertIsInstance(self.dem_tile, DEMTile)
        self.assertGreater(self.dem_tile.data.size, 0)
        self.assertEqual(self.dem_tile.metadata.format, DEMFormat.CUSTOM)
        self.assertEqual(self.dem_tile.metadata.bounds, self.test_bounds)
    
    def test_elevation_query(self):
        """Test elevation queries"""
        # Test point within bounds
        lon, lat = 0.0, 52.0  # Center of test area
        elevation = self.dem_corrector.get_elevation(lon, lat)
        
        self.assertIsInstance(elevation, float)
        self.assertGreater(elevation, -200)  # Reasonable elevation
        self.assertLess(elevation, 500)
        
        # Test point outside bounds
        elevation_outside = self.dem_corrector.get_elevation(10.0, 10.0)
        self.assertEqual(elevation_outside, self.dem_corrector.default_elevation)
    
    def test_interpolation_methods(self):
        """Test different interpolation methods"""
        lon, lat = 0.0, 52.0
        
        # Test all interpolation methods
        for method in InterpolationMethod:
            elevation = self.dem_corrector.get_elevation(lon, lat, method)
            self.assertIsInstance(elevation, float)
    
    def test_terrain_slope(self):
        """Test terrain slope calculation"""
        lon, lat = 0.0, 52.0
        slope_x, slope_y = self.dem_corrector.get_terrain_slope(lon, lat)
        
        self.assertIsInstance(slope_x, float)
        self.assertIsInstance(slope_y, float)
        
        # Slopes should be reasonable (in degrees)
        self.assertGreater(slope_x, -90)
        self.assertLess(slope_x, 90)
        self.assertGreater(slope_y, -90)
        self.assertLess(slope_y, 90)
    
    def test_terrain_normal(self):
        """Test terrain normal calculation"""
        lon, lat = 0.0, 52.0
        normal = self.dem_corrector.get_terrain_normal(lon, lat)
        
        self.assertEqual(normal.shape, (3,))
        
        # Normal should be unit vector
        self.assertAlmostEqual(np.linalg.norm(normal), 1.0, places=5)
        
        # Z component should be positive (pointing up)
        self.assertGreater(normal[2], 0)
    
    def test_ray_correction(self):
        """Test ray-ground intersection correction"""
        # Create test ray
        ray = Ray3D(
            origin=np.array([0.0, 0.0, 1000.0]),
            direction=np.array([0.0, 0.0, -1.0])
        )
        
        # Initial intersection at sea level
        initial_intersection = np.array([0.0, 0.0, 0.0])
        
        # Correct using DEM
        corrected = self.dem_corrector.correct_ray_ground_intersection(
            ray, initial_intersection
        )
        
        self.assertIsNotNone(corrected)
        # Z coordinate should be close to DEM elevation
        dem_elevation = self.dem_corrector.get_elevation(0.0, 0.0)
        self.assertAlmostEqual(corrected[2], dem_elevation, delta=5.0)


class TestGeolocationService(unittest.TestCase):
    """Test geolocation service integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create calibration data
        camera_matrix = np.array([
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0]
        ])
        
        calibration_data = CalibrationData(
            camera_matrix=camera_matrix,
            distortion_coeffs=np.zeros(5),
            image_width=640,
            image_height=480
        )
        
        # Create reference point
        reference_point = GeodeticCoordinate(51.5074, -0.1278, 0.0)
        
        # Create geolocation service
        self.geolocation_service = GeolocationService(
            calibration_data=calibration_data,
            reference_point=reference_point
        )
        
        # Create test camera pose
        self.camera_pose = CameraPose(
            position=np.array([0.0, 0.0, 100.0]),  # 100m above reference
            rotation_matrix=np.eye(3)
        )
    
    def test_service_initialization(self):
        """Test geolocation service initialization"""
        self.assertIsNotNone(self.geolocation_service.projector)
        self.assertIsNotNone(self.geolocation_service.coordinate_transformer)
        self.assertIsNotNone(self.geolocation_service.dem_corrector)
    
    def test_pixel_geolocation(self):
        """Test pixel geolocation"""
        # Test center pixel
        pixel_coords = (320.0, 240.0)
        
        result = self.geolocation_service.geolocate_pixel(
            pixel_coords=pixel_coords,
            camera_pose=self.camera_pose
        )
        
        self.assertIsInstance(result, GeolocationResult)
        self.assertIsNotNone(result.geodetic_coordinate)
        self.assertGreater(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        
        # Result should be close to reference point
        self.assertAlmostEqual(
            result.geodetic_coordinate.latitude, 
            self.geolocation_service.coordinate_transformer.reference_point.latitude,
            delta=0.01
        )
    
    def test_batch_geolocation(self):
        """Test batch geolocation"""
        # Test multiple pixels
        pixel_list = [(320, 240), (100, 100), (500, 400)]
        
        results = self.geolocation_service.geolocate_pixels_batch(
            pixel_coords_list=pixel_list,
            camera_pose=self.camera_pose
        )
        
        self.assertEqual(len(results), len(pixel_list))
        
        for result in results:
            if result is not None:
                self.assertIsInstance(result, GeolocationResult)
                self.assertIsNotNone(result.geodetic_coordinate)
    
    def test_error_handling(self):
        """Test error handling"""
        # Test with invalid pixel coordinates
        invalid_pixel = (-100, -100)
        
        result = self.geolocation_service.geolocate_pixel(
            pixel_coords=invalid_pixel,
            camera_pose=self.camera_pose
        )
        
        # Should handle gracefully
        self.assertIsNotNone(result)
        self.assertLess(result.confidence, 0.5)  # Low confidence for invalid input


class TestIntegration(unittest.TestCase):
    """Integration tests for complete geolocation pipeline"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        # Create realistic test scenario
        self.camera_matrix = np.array([
            [1000.0, 0.0, 640.0],
            [0.0, 1000.0, 360.0],
            [0.0, 0.0, 1.0]
        ])
        
        self.calibration_data = CalibrationData(
            camera_matrix=self.camera_matrix,
            distortion_coeffs=np.array([0.05, -0.1, 0.001, 0.002, 0.02]),
            image_width=1280,
            image_height=720
        )
        
        # Reference point (London)
        self.reference_point = GeodeticCoordinate(51.5074, -0.1278, 0.0)
        
        # Create geolocation service with DEM
        self.service = GeolocationService(
            calibration_data=self.calibration_data,
            reference_point=self.reference_point
        )
        
        # Add synthetic DEM
        self.service.dem_corrector.create_synthetic_dem(
            bounds=(-0.5, 51.0, 0.5, 52.0),
            resolution=30.0,
            base_elevation=50.0,
            terrain_variation=50.0
        )
        
        # Drone camera pose (100m altitude, looking down)
        self.drone_pose = CameraPose(
            position=np.array([100.0, 200.0, 100.0]),  # ENU coordinates
            euler_angles=np.array([0.0, np.radians(15), 0.0])  # Slight tilt
        )
    
    def test_complete_pipeline(self):
        """Test complete geolocation pipeline"""
        # Test multiple pixels across the image
        test_pixels = [
            (640, 360),   # Center
            (320, 180),   # Upper left quadrant
            (960, 540),   # Lower right quadrant
            (100, 100),   # Corner
        ]
        
        results = []
        for pixel in test_pixels:
            result = self.service.geolocate_pixel(
                pixel_coords=pixel,
                camera_pose=self.drone_pose,
                use_dem_correction=True
            )
            results.append(result)
        
        # Verify all results
        for i, result in enumerate(results):
            with self.subTest(pixel=test_pixels[i]):
                self.assertIsNotNone(result)
                self.assertIsInstance(result, GeolocationResult)
                
                # Check coordinate validity
                coord = result.geodetic_coordinate
                self.assertGreater(coord.latitude, 50.0)
                self.assertLess(coord.latitude, 53.0)
                self.assertGreater(coord.longitude, -2.0)
                self.assertLess(coord.longitude, 2.0)
                
                # Check confidence
                self.assertGreater(result.confidence, 0.0)
                self.assertLessEqual(result.confidence, 1.0)
    
    def test_accuracy_assessment(self):
        """Test geolocation accuracy with known points"""
        # Create a known 3D point in ENU coordinates
        known_enu_point = np.array([500.0, 1000.0, 0.0])  # 500m east, 1km north
        
        # Convert to geodetic
        known_geodetic = self.service.coordinate_transformer.enu_to_geodetic(known_enu_point)
        
        # Project to image
        projected_pixel = self.service.projector.project_point(
            known_enu_point, self.drone_pose
        )
        
        if projected_pixel is not None:
            # Geolocate the projected pixel
            result = self.service.geolocate_pixel(
                pixel_coords=projected_pixel,
                camera_pose=self.drone_pose,
                use_dem_correction=False  # Use flat ground for accuracy test
            )
            
            if result is not None:
                # Calculate error
                lat_error = abs(result.geodetic_coordinate.latitude - known_geodetic.latitude)
                lon_error = abs(result.geodetic_coordinate.longitude - known_geodetic.longitude)
                
                # Errors should be small (within reasonable tolerance)
                self.assertLess(lat_error, 0.001)  # ~100m at this latitude
                self.assertLess(lon_error, 0.001)
    
    def test_performance_benchmark(self):
        """Test performance of geolocation operations"""
        import time
        
        # Benchmark single pixel geolocation
        start_time = time.time()
        
        for _ in range(100):
            result = self.service.geolocate_pixel(
                pixel_coords=(640, 360),
                camera_pose=self.drone_pose
            )
        
        single_time = time.time() - start_time
        
        # Should be reasonably fast (< 1 second for 100 operations)
        self.assertLess(single_time, 1.0)
        
        # Benchmark batch geolocation
        pixel_batch = [(i*100, j*100) for i in range(5) for j in range(5)]
        
        start_time = time.time()
        results = self.service.geolocate_pixels_batch(
            pixel_coords_list=pixel_batch,
            camera_pose=self.drone_pose
        )
        batch_time = time.time() - start_time
        
        # Batch should be faster than individual calls
        expected_individual_time = len(pixel_batch) * (single_time / 100)
        self.assertLess(batch_time, expected_individual_time * 0.8)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestCameraCalibration,
        TestProjection,
        TestCoordinateTransform,
        TestDEMCorrection,
        TestGeolocationService,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    print(f"{'='*50}")