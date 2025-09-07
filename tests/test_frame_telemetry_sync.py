#!/usr/bin/env python3
"""
Test Suite for Frame-Telemetry Synchronization

Comprehensive tests for RTCP SR/PTS mapping and frame-telemetry sync.
"""

import asyncio
import unittest
import time
import struct
import socket
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'backend'))
sys.path.append(os.path.dirname(__file__))

# Import mock implementations first
from mock_connection_manager import ConnectionManager, FrameData, TelemetryPacket

# Patch the imports before importing the real modules
sys.modules['connection_manager'] = sys.modules['mock_connection_manager']

try:
    from frame_telemetry_sync import (
        RTCPParser, PTSMapper, FrameTelemetrySync, SyncQuality,
        create_frame_telemetry_sync
    )
    from enhanced_connection_manager import (
        EnhancedConnectionManager, EnhancedFrameData, RTCPReceiver,
        create_enhanced_connection_manager
    )
    from telemetry_service import TelemetryService, TelemetryData
except ImportError as e:
    print(f"Import error: {e}")
    print("Running with limited functionality...")
    
    # Define minimal classes for testing
    class SyncQuality:
        EXCELLENT = "excellent"
        GOOD = "good"
        FAIR = "fair"
        POOR = "poor"
    
    class RTCPParser:
        def parse_rtcp_packet(self, packet):
            return None
    
    class PTSMapper:
        def __init__(self, max_history=5000):
            self.max_history = max_history
            self.mappings = []
        
        def add_mapping(self, pts, wall_time):
            self.mappings.append({'pts': pts, 'wall_time': wall_time})
        
        def pts_to_wall_time(self, pts):
            return time.time()
    
    class FrameTelemetrySync:
        def __init__(self, telemetry_service):
            self.telemetry_service = telemetry_service
            self.pts_mapper = PTSMapper()
        
        def process_rtcp_packet(self, packet):
            return True
        
        def get_synchronized_telemetry(self, pts):
            return None
        
        def add_sync_callback(self, callback):
            pass
    
    def create_frame_telemetry_sync(telemetry_service, config):
        return FrameTelemetrySync(telemetry_service)
    
    class TelemetryService:
        def get_interpolated_telemetry(self, timestamp):
            return None
    
    class TelemetryData:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    # Skip enhanced connection manager tests
    SKIP_ENHANCED_TESTS = True
else:
    SKIP_ENHANCED_TESTS = False

class TestRTCPParser(unittest.TestCase):
    """Test RTCP packet parsing."""
    
    def setUp(self):
        self.parser = RTCPParser()
    
    def test_parse_valid_sr_packet(self):
        """Test parsing valid RTCP SR packet."""
        # Create mock RTCP SR packet
        # Format: V(2) P(1) RC(5) PT(8) length(16) SSRC(32) NTP_MSW(32) NTP_LSW(32) RTP_timestamp(32)
        packet = struct.pack('!BBHIIIII',
            0x80,  # V=2, P=0, RC=0
            200,   # PT=200 (SR)
            6,     # length (6 32-bit words)
            0x12345678,  # SSRC
            0x12345678,  # NTP MSW
            0x87654321,  # NTP LSW
            0x9ABCDEF0   # RTP timestamp
        )
        
        result = self.parser.parse_rtcp_packet(packet)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['packet_type'], 'SR')
        self.assertEqual(result['ssrc'], 0x12345678)
        self.assertEqual(result['rtp_timestamp'], 0x9ABCDEF0)
        self.assertIn('ntp_timestamp', result)
    
    def test_parse_invalid_packet(self):
        """Test parsing invalid packet."""
        # Too short packet
        packet = b'\x80\xc8'
        result = self.parser.parse_rtcp_packet(packet)
        self.assertIsNone(result)
        
        # Wrong version
        packet = struct.pack('!BBHIIIII',
            0x40,  # V=1 (wrong version)
            200,   # PT=200
            6,     # length
            0x12345678, 0x12345678, 0x87654321, 0x9ABCDEF0
        )
        result = self.parser.parse_rtcp_packet(packet)
        self.assertIsNone(result)
    
    def test_ntp_to_unix_conversion(self):
        """Test NTP to Unix timestamp conversion."""
        # Known NTP timestamp (Jan 1, 2020 00:00:00 UTC)
        ntp_msw = 0xe1f5c800  # NTP seconds since 1900
        ntp_lsw = 0x00000000  # Fractional seconds
        
        unix_time = self.parser._ntp_to_unix(ntp_msw, ntp_lsw)
        
        # Should be approximately Jan 1, 2020
        expected = 1577836800.0  # Unix timestamp for Jan 1, 2020
        self.assertAlmostEqual(unix_time, expected, delta=1.0)

class TestPTSMapper(unittest.TestCase):
    """Test PTS mapping functionality."""
    
    def setUp(self):
        self.mapper = PTSMapper()
    
    def test_add_mapping(self):
        """Test adding PTS mapping."""
        pts = 90000  # 1 second at 90kHz
        wall_time = time.time()
        
        self.mapper.add_mapping(pts, wall_time)
        
        # Check mapping was added
        self.assertEqual(len(self.mapper.mappings), 1)
        self.assertEqual(self.mapper.mappings[0]['pts'], pts)
        self.assertEqual(self.mapper.mappings[0]['wall_time'], wall_time)
    
    def test_pts_to_wall_time(self):
        """Test PTS to wall time conversion."""
        base_time = time.time()
        base_pts = 90000
        
        # Add base mapping
        self.mapper.add_mapping(base_pts, base_time)
        
        # Test conversion
        test_pts = base_pts + 90000  # +1 second
        wall_time = self.mapper.pts_to_wall_time(test_pts)
        
        self.assertIsNotNone(wall_time)
        self.assertAlmostEqual(wall_time, base_time + 1.0, delta=0.01)
    
    def test_drift_compensation(self):
        """Test clock drift compensation."""
        base_time = time.time()
        
        # Add multiple mappings with simulated drift
        mappings = [
            (90000, base_time),
            (180000, base_time + 1.001),  # Slight drift
            (270000, base_time + 2.002),  # More drift
        ]
        
        for pts, wall_time in mappings:
            self.mapper.add_mapping(pts, wall_time)
        
        # Test conversion with drift compensation
        test_pts = 360000  # +4 seconds from base
        wall_time = self.mapper.pts_to_wall_time(test_pts)
        
        self.assertIsNotNone(wall_time)
        # Should compensate for drift
        expected = base_time + 4.0
        self.assertAlmostEqual(wall_time, expected, delta=0.1)
    
    def test_mapping_cleanup(self):
        """Test old mapping cleanup."""
        base_time = time.time()
        
        # Add many mappings
        for i in range(6000):  # More than max_history (5000)
            self.mapper.add_mapping(i * 90, base_time + i * 0.001)
        
        # Should keep only max_history mappings
        self.assertEqual(len(self.mapper.mappings), 5000)
        
        # Should keep the most recent ones
        self.assertEqual(self.mapper.mappings[0]['pts'], 1000 * 90)  # First kept
        self.assertEqual(self.mapper.mappings[-1]['pts'], 5999 * 90)  # Last added

class TestFrameTelemetrySync(unittest.TestCase):
    """Test frame-telemetry synchronization."""
    
    def setUp(self):
        # Mock telemetry service
        self.mock_telemetry = Mock(spec=TelemetryService)
        self.sync = FrameTelemetrySync(self.mock_telemetry)
    
    def test_process_rtcp_packet(self):
        """Test RTCP packet processing."""
        # Create valid RTCP SR packet
        packet = struct.pack('!BBHIIIII',
            0x80, 200, 6, 0x12345678,
            0x12345678, 0x87654321, 0x9ABCDEF0, 0
        )
        
        result = self.sync.process_rtcp_packet(packet)
        self.assertTrue(result)
        
        # Check that mapping was added
        self.assertEqual(len(self.sync.pts_mapper.mappings), 1)
    
    def test_get_synchronized_telemetry(self):
        """Test getting synchronized telemetry."""
        # Setup mock telemetry data
        mock_telemetry_data = TelemetryData(
            timestamp=time.time(),
            latitude=37.7749,
            longitude=-122.4194,
            altitude_msl=100.0,
            altitude_agl=50.0,
            roll=0.0, pitch=0.0, yaw=90.0,
            gimbal_roll=0.0, gimbal_pitch=-30.0, gimbal_yaw=0.0,
            frame_id="test_frame",
            quality=0.95,
            source="test"
        )
        
        self.mock_telemetry.get_interpolated_telemetry.return_value = mock_telemetry_data
        
        # Add PTS mapping
        wall_time = time.time()
        pts = 90000
        self.sync.pts_mapper.add_mapping(pts, wall_time)
        
        # Get synchronized telemetry
        result = self.sync.get_synchronized_telemetry(pts)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['pts'], pts)
        self.assertEqual(result['telemetry'], mock_telemetry_data)
        self.assertIn('sync_quality', result)
        self.assertIn('time_diff_ms', result)
    
    def test_sync_quality_assessment(self):
        """Test synchronization quality assessment."""
        # Test excellent quality (small time difference)
        quality = self.sync._assess_sync_quality(5.0)  # 5ms
        self.assertEqual(quality, SyncQuality.EXCELLENT)
        
        # Test good quality
        quality = self.sync._assess_sync_quality(25.0)  # 25ms
        self.assertEqual(quality, SyncQuality.GOOD)
        
        # Test fair quality
        quality = self.sync._assess_sync_quality(75.0)  # 75ms
        self.assertEqual(quality, SyncQuality.FAIR)
        
        # Test poor quality
        quality = self.sync._assess_sync_quality(150.0)  # 150ms
        self.assertEqual(quality, SyncQuality.POOR)
    
    def test_sync_callbacks(self):
        """Test synchronization callbacks."""
        callback_data = []
        
        def test_callback(sync_data):
            callback_data.append(sync_data)
        
        self.sync.add_sync_callback(test_callback)
        
        # Setup and trigger sync
        mock_telemetry_data = TelemetryData(
            timestamp=time.time(),
            latitude=37.7749, longitude=-122.4194,
            altitude_msl=100.0, altitude_agl=50.0,
            roll=0.0, pitch=0.0, yaw=90.0,
            gimbal_roll=0.0, gimbal_pitch=-30.0, gimbal_yaw=0.0,
            frame_id="test_frame", quality=0.95, source="test"
        )
        
        self.mock_telemetry.get_interpolated_telemetry.return_value = mock_telemetry_data
        
        wall_time = time.time()
        pts = 90000
        self.sync.pts_mapper.add_mapping(pts, wall_time)
        
        # Get synchronized telemetry (should trigger callback)
        self.sync.get_synchronized_telemetry(pts)
        
        # Check callback was called
        self.assertEqual(len(callback_data), 1)
        self.assertEqual(callback_data[0]['pts'], pts)

@unittest.skipIf(SKIP_ENHANCED_TESTS, "Enhanced connection manager not available")
class TestEnhancedConnectionManager(unittest.TestCase):
    """Test enhanced connection manager."""
    
    def setUp(self):
        self.config = {
            "enable_sync": True,
            "rtcp_port": 5006,  # Use different port for testing
            "sync_config": {"max_history": 1000}
        }
    
    @patch('enhanced_connection_manager.get_telemetry_service')
    @patch('enhanced_connection_manager.ConnectionManager')
    def test_initialization(self, mock_base_manager, mock_telemetry_service):
        """Test enhanced connection manager initialization."""
        mock_telemetry = Mock(spec=TelemetryService)
        mock_telemetry_service.return_value = mock_telemetry
        
        manager = EnhancedConnectionManager(self.config)
        
        self.assertIsNotNone(manager.sync_manager)
        self.assertIsNotNone(manager.rtcp_receiver)
        self.assertTrue(manager.sync_enabled)
    
    @patch('enhanced_connection_manager.get_telemetry_service')
    @patch('enhanced_connection_manager.ConnectionManager')
    def test_pts_generation(self, mock_base_manager, mock_telemetry_service):
        """Test PTS generation for frames."""
        mock_telemetry = Mock(spec=TelemetryService)
        mock_telemetry_service.return_value = mock_telemetry
        
        manager = EnhancedConnectionManager(self.config)
        
        # Test PTS generation
        timestamp = time.time()
        pts1 = manager._generate_pts(timestamp)
        pts2 = manager._generate_pts(timestamp + 1.0)
        
        self.assertIsInstance(pts1, int)
        self.assertIsInstance(pts2, int)
        self.assertEqual(pts2 - pts1, 90000)  # 1 second at 90kHz
    
    def test_sync_callback_management(self):
        """Test sync callback management."""
        with patch('enhanced_connection_manager.get_telemetry_service'), \
             patch('enhanced_connection_manager.ConnectionManager'):
            
            manager = EnhancedConnectionManager(self.config)
            
            callback_calls = []
            
            def test_callback(frame_data):
                callback_calls.append(frame_data)
            
            # Add callback
            manager.add_sync_callback(test_callback)
            self.assertEqual(len(manager.sync_callbacks), 1)
            
            # Test callback execution (mock frame data)
            mock_frame = Mock(spec=EnhancedFrameData)
            for callback in manager.sync_callbacks:
                callback(mock_frame)
            
            self.assertEqual(len(callback_calls), 1)
            self.assertEqual(callback_calls[0], mock_frame)

@unittest.skipIf(SKIP_ENHANCED_TESTS, "RTCP receiver not available")
class TestRTCPReceiver(unittest.TestCase):
    """Test RTCP receiver functionality."""
    
    def setUp(self):
        self.mock_sync_manager = Mock()
        self.receiver = RTCPReceiver(self.mock_sync_manager, port=5007)
    
    @patch('socket.socket')
    async def test_start_stop(self, mock_socket):
        """Test RTCP receiver start/stop."""
        mock_sock = Mock()
        mock_socket.return_value = mock_sock
        
        # Test start
        result = await self.receiver.start()
        self.assertTrue(result)
        self.assertTrue(self.receiver.is_running)
        
        # Test stop
        await self.receiver.stop()
        self.assertFalse(self.receiver.is_running)
        mock_sock.close.assert_called_once()

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete sync system."""
    
    def test_create_frame_telemetry_sync(self):
        """Test factory function for frame-telemetry sync."""
        mock_telemetry = Mock(spec=TelemetryService)
        config = {"max_history": 1000}
        
        sync = create_frame_telemetry_sync(mock_telemetry, config)
        
        self.assertIsInstance(sync, FrameTelemetrySync)
        self.assertEqual(sync.pts_mapper.max_history, 1000)
    
    def test_create_enhanced_connection_manager(self):
        """Test factory function for enhanced connection manager."""
        config = {
            "enable_sync": True,
            "rtcp_port": 5008
        }
        
        with patch('enhanced_connection_manager.get_telemetry_service'), \
             patch('enhanced_connection_manager.ConnectionManager'):
            
            manager = create_enhanced_connection_manager(config)
            
            self.assertIsInstance(manager, EnhancedConnectionManager)
            self.assertTrue(manager.sync_enabled)
            self.assertEqual(manager.rtcp_receiver.port, 5008)

class TestPerformance(unittest.TestCase):
    """Performance tests for sync components."""
    
    def test_pts_mapper_performance(self):
        """Test PTS mapper performance with many mappings."""
        mapper = PTSMapper(max_history=10000)
        
        # Add many mappings
        start_time = time.time()
        base_time = time.time()
        
        for i in range(1000):
            mapper.add_mapping(i * 90, base_time + i * 0.001)
        
        add_time = time.time() - start_time
        
        # Test lookup performance
        start_time = time.time()
        
        for i in range(100):
            result = mapper.pts_to_wall_time(i * 90)
            self.assertIsNotNone(result)
        
        lookup_time = time.time() - start_time
        
        # Performance assertions (should be fast)
        self.assertLess(add_time, 1.0)  # Adding 1000 mappings < 1s
        self.assertLess(lookup_time, 0.1)  # 100 lookups < 0.1s
    
    def test_rtcp_parsing_performance(self):
        """Test RTCP parsing performance."""
        parser = RTCPParser()
        
        # Create test packet
        packet = struct.pack('!BBHIIIII',
            0x80, 200, 6, 0x12345678,
            0x12345678, 0x87654321, 0x9ABCDEF0, 0
        )
        
        # Test parsing performance
        start_time = time.time()
        
        for _ in range(1000):
            result = parser.parse_rtcp_packet(packet)
            self.assertIsNotNone(result)
        
        parse_time = time.time() - start_time
        
        # Should parse 1000 packets quickly
        self.assertLess(parse_time, 0.5)

if __name__ == '__main__':
    import logging
    
    # Configure logging for tests
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    unittest.main(verbosity=2)