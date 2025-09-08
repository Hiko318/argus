#!/usr/bin/env python3
"""
Unit tests for geolocation projection and coordinate transformations.

This module tests the coordinate transformation functions in the geolocation
module to ensure accurate conversions between different coordinate systems.
"""

import unittest
import numpy as np
import math
import sys
from pathlib import Path

# Add geolocation module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "geolocation"))

try:
    from coordinate_transform import (
        CoordinateTransformer,
        GeodeticCoordinate,
        UTMCoordinate,
        WGS84Constants,
        CoordinateSystem
    )
    GEOLOCATION_AVAILABLE = True
except ImportError:
    GEOLOCATION_AVAILABLE = False


class TestCoordinateTransformations(unittest.TestCase):
    """Test coordinate transformation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not GEOLOCATION_AVAILABLE:
            self.skipTest("Geolocation module not available")
        
        # Reference point in London
        self.reference_point = GeodeticCoordinate(
            latitude=51.5074,
            longitude=-0.1278,
            altitude=0.0
        )
        
        self.transformer = CoordinateTransformer(self.reference_point)
    
    def test_geodetic_coordinate_validation(self):
        """Test geodetic coordinate validation."""
        # Valid coordinates
        valid_coord = GeodeticCoordinate(51.5074, -0.1278, 100.0)
        self.assertEqual(valid_coord.latitude, 51.5074)
        self.assertEqual(valid_coord.longitude, -0.1278)
        self.assertEqual(valid_coord.altitude, 100.0)
        
        # Invalid latitude
        with self.assertRaises(ValueError):
            GeodeticCoordinate(91.0, 0.0, 0.0)  # > 90
        
        with self.assertRaises(ValueError):
            GeodeticCoordinate(-91.0, 0.0, 0.0)  # < -90
        
        # Invalid longitude
        with self.assertRaises(ValueError):
            GeodeticCoordinate(0.0, 181.0, 0.0)  # > 180
        
        with self.assertRaises(ValueError):
            GeodeticCoordinate(0.0, -181.0, 0.0)  # < -180
    
    def test_geodetic_to_ecef_known_values(self):
        """Test geodetic to ECEF conversion with known values."""
        # Test point at equator, prime meridian
        equator_prime = GeodeticCoordinate(0.0, 0.0, 0.0)
        ecef = self.transformer.geodetic_to_ecef(equator_prime)
        
        # Should be approximately [6378137, 0, 0] (WGS84 semi-major axis)
        expected = np.array([WGS84Constants.A, 0.0, 0.0])
        np.testing.assert_array_almost_equal(ecef, expected, decimal=0)
        
        # Test point at north pole
        north_pole = GeodeticCoordinate(90.0, 0.0, 0.0)
        ecef_pole = self.transformer.geodetic_to_ecef(north_pole)
        
        # Should be approximately [0, 0, 6356752] (WGS84 semi-minor axis)
        expected_pole = np.array([0.0, 0.0, WGS84Constants.B])
        np.testing.assert_array_almost_equal(ecef_pole, expected_pole, decimal=0)
    
    def test_ecef_to_geodetic_roundtrip(self):
        """Test ECEF to geodetic round-trip conversion."""
        test_points = [
            GeodeticCoordinate(51.5074, -0.1278, 100.0),  # London
            GeodeticCoordinate(40.7128, -74.0060, 50.0),   # New York
            GeodeticCoordinate(-33.8688, 151.2093, 200.0), # Sydney
            GeodeticCoordinate(0.0, 0.0, 0.0),             # Equator/Prime
        ]
        
        for original in test_points:
            with self.subTest(point=original):
                # Convert to ECEF and back
                ecef = self.transformer.geodetic_to_ecef(original)
                recovered = self.transformer.ecef_to_geodetic(ecef)
                
                # Check accuracy (should be within 1e-6 degrees, 1e-3 meters)
                self.assertAlmostEqual(recovered.latitude, original.latitude, places=6)
                self.assertAlmostEqual(recovered.longitude, original.longitude, places=6)
                self.assertAlmostEqual(recovered.altitude, original.altitude, places=3)
    
    def test_enu_conversion_known_values(self):
        """Test ENU coordinate conversion with known values."""
        # Test reference point (should be origin)
        enu_origin = self.transformer.geodetic_to_enu(self.reference_point)
        np.testing.assert_array_almost_equal(enu_origin, [0.0, 0.0, 0.0], decimal=3)
        
        # Test point 1km north (approximately)
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
        
        # Test point 1km east (approximately)
        east_point = GeodeticCoordinate(
            latitude=self.reference_point.latitude,
            longitude=self.reference_point.longitude + 0.014,  # ~1km east at London latitude
            altitude=0.0
        )
        
        enu_east = self.transformer.geodetic_to_enu(east_point)
        
        # Should be approximately [1000, 0, 0]
        self.assertGreater(enu_east[0], 900)    # East component ~1000m
        self.assertLess(abs(enu_east[1]), 10)   # North component small
        self.assertLess(abs(enu_east[2]), 10)   # Up component small
    
    def test_enu_roundtrip_conversion(self):
        """Test ENU round-trip conversion."""
        test_points = [
            GeodeticCoordinate(51.5074, -0.1278, 100.0),
            GeodeticCoordinate(51.5084, -0.1268, 50.0),   # ~1km NE
            GeodeticCoordinate(51.5064, -0.1288, 150.0),  # ~1km SW
        ]
        
        for original in test_points:
            with self.subTest(point=original):
                # Convert to ENU and back
                enu = self.transformer.geodetic_to_enu(original)
                recovered = self.transformer.enu_to_geodetic(enu)
                
                # Check accuracy
                self.assertAlmostEqual(recovered.latitude, original.latitude, places=6)
                self.assertAlmostEqual(recovered.longitude, original.longitude, places=6)
                self.assertAlmostEqual(recovered.altitude, original.altitude, places=3)
    
    def test_utm_conversion(self):
        """Test UTM coordinate conversion."""
        utm_coord = self.transformer.geodetic_to_utm(self.reference_point)
        
        self.assertIsInstance(utm_coord, UTMCoordinate)
        self.assertGreater(utm_coord.easting, 0)
        self.assertGreater(utm_coord.northing, 0)
        self.assertIsInstance(utm_coord.zone, int)
        self.assertIn(utm_coord.hemisphere, ['N', 'S'])
        
        # London should be in UTM zone 30N
        self.assertEqual(utm_coord.zone, 30)
        self.assertEqual(utm_coord.hemisphere, 'N')
        
        # Test round-trip conversion
        geodetic_back = self.transformer.utm_to_geodetic(utm_coord)
        
        self.assertAlmostEqual(geodetic_back.latitude, self.reference_point.latitude, places=5)
        self.assertAlmostEqual(geodetic_back.longitude, self.reference_point.longitude, places=5)
    
    def test_enu_ned_conversion(self):
        """Test ENU to NED conversion."""
        enu_coords = np.array([100.0, 200.0, 50.0])  # 100m E, 200m N, 50m U
        ned_coords = self.transformer.enu_to_ned(enu_coords)
        
        # NED should be [200, 100, -50] (N, E, D)
        expected_ned = np.array([200.0, 100.0, -50.0])
        np.testing.assert_array_equal(ned_coords, expected_ned)
        
        # Test round-trip
        enu_back = self.transformer.ned_to_enu(ned_coords)
        np.testing.assert_array_equal(enu_back, enu_coords)
    
    def test_distance_calculation(self):
        """Test great circle distance calculation."""
        # Test known distance: London to Paris (~344 km)
        london = GeodeticCoordinate(51.5074, -0.1278, 0.0)
        paris = GeodeticCoordinate(48.8566, 2.3522, 0.0)
        
        distance = self.transformer.calculate_distance(london, paris)
        
        # Should be approximately 344,000 meters (±5%)
        expected_distance = 344000  # meters
        self.assertAlmostEqual(distance, expected_distance, delta=expected_distance * 0.05)
        
        # Test zero distance (same point)
        zero_distance = self.transformer.calculate_distance(london, london)
        self.assertAlmostEqual(zero_distance, 0.0, places=3)
    
    def test_bearing_calculation(self):
        """Test bearing calculation."""
        # Test bearing from London to Paris (should be roughly southeast, ~150°)
        london = GeodeticCoordinate(51.5074, -0.1278, 0.0)
        paris = GeodeticCoordinate(48.8566, 2.3522, 0.0)
        
        bearing = self.transformer.calculate_bearing(london, paris)
        
        # Should be approximately 150° (±10°)
        self.assertGreater(bearing, 140)
        self.assertLess(bearing, 160)
        
        # Test bearing range (0-360)
        self.assertGreaterEqual(bearing, 0)
        self.assertLess(bearing, 360)
    
    def test_coordinate_system_transform(self):
        """Test general coordinate system transformation."""
        # Test geodetic to ENU
        geodetic_coords = np.array([51.5074, -0.1278, 100.0])
        enu_coords = self.transformer.transform_coordinates(
            geodetic_coords, CoordinateSystem.GEODETIC, CoordinateSystem.ENU
        )
        
        # Reference point should transform to origin
        np.testing.assert_array_almost_equal(enu_coords, [0.0, 0.0, 100.0], decimal=3)
        
        # Test ENU to NED
        enu_test = np.array([100.0, 200.0, 50.0])
        ned_coords = self.transformer.transform_coordinates(
            enu_test, CoordinateSystem.ENU, CoordinateSystem.NED
        )
        
        expected_ned = np.array([200.0, 100.0, -50.0])
        np.testing.assert_array_equal(ned_coords, expected_ned)
    
    def test_wgs84_constants(self):
        """Test WGS84 constants are correct."""
        # Check semi-major axis
        self.assertAlmostEqual(WGS84Constants.A, 6378137.0, places=1)
        
        # Check flattening
        self.assertAlmostEqual(WGS84Constants.F, 1.0/298.257223563, places=10)
        
        # Check semi-minor axis calculation
        expected_b = WGS84Constants.A * (1.0 - WGS84Constants.F)
        self.assertAlmostEqual(WGS84Constants.B, expected_b, places=1)
        
        # Check eccentricity squared
        expected_e2 = WGS84Constants.F * (2.0 - WGS84Constants.F)
        self.assertAlmostEqual(WGS84Constants.E2, expected_e2, places=10)
    
    def test_transformation_info(self):
        """Test transformation information retrieval."""
        info = self.transformer.get_transformation_info()
        
        self.assertIn('supported_systems', info)
        self.assertIn('reference_point', info)
        
        # Check supported systems
        expected_systems = ['geodetic', 'ecef', 'enu', 'ned', 'utm']
        for system in expected_systems:
            self.assertIn(system, info['supported_systems'])
        
        # Check reference point info
        ref_info = info['reference_point']
        self.assertEqual(ref_info['latitude'], self.reference_point.latitude)
        self.assertEqual(ref_info['longitude'], self.reference_point.longitude)
        self.assertEqual(ref_info['altitude'], self.reference_point.altitude)


if __name__ == '__main__':
    unittest.main()