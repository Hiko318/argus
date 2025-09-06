"""Test suite for DJI O4 connection implementations.

Provides unit tests and integration tests using recorded data files
to validate both Mobile SDK and Payload SDK implementations.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np
import pytest

from .dji_o4 import (
    DJIO4Connection,
    DJIO4MobileSDK,
    DJIO4PayloadSDK,
    TelemetryData,
    FrameData
)


class TestDataManager:
    """Manages test data files for DJI O4 testing."""
    
    def __init__(self, test_data_dir: str = "data/test_recordings"):
        self.test_data_dir = Path(test_data_dir)
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        
    def create_test_video(self, filename: str, duration: int = 30, fps: int = 30) -> str:
        """Create a test video file with synthetic drone footage."""
        video_path = self.test_data_dir / filename
        
        # Video codec settings
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (1920, 1080))
        
        try:
            for frame_num in range(duration * fps):
                # Create synthetic aerial view
                frame = self._generate_aerial_frame(frame_num, fps)
                writer.write(frame)
                
        finally:
            writer.release()
            
        return str(video_path)
    
    def create_test_telemetry(self, filename: str, duration: int = 30, rate: int = 10) -> str:
        """Create test telemetry data file."""
        telemetry_path = self.test_data_dir / filename
        
        telemetry_data = []
        for i in range(duration * rate):
            t = i / rate
            data = {
                'timestamp': time.time() + t,
                'latitude': 37.7749 + 0.001 * np.sin(t * 0.1),
                'longitude': -122.4194 + 0.001 * np.cos(t * 0.1),
                'altitude_msl': 100.0 + 10 * np.sin(t * 0.05),
                'altitude_agl': 50.0 + 10 * np.sin(t * 0.05),
                'heading': 180 + 30 * np.sin(t * 0.02),
                'pitch': 5 * np.sin(t * 0.03),
                'roll': 3 * np.cos(t * 0.04),
                'yaw': 0,
                'velocity_x': 5.0,
                'velocity_y': 0.0,
                'velocity_z': 0.0,
                'gimbal_pitch': -45,
                'gimbal_roll': 0,
                'gimbal_yaw': 0,
                'battery_percentage': 85.0,
                'signal_strength': 4,
                'gps_satellite_count': 12,
                'flight_mode': 'AUTO'
            }
            telemetry_data.append(data)
        
        with open(telemetry_path, 'w') as f:
            json.dump(telemetry_data, f, indent=2)
            
        return str(telemetry_path)
    
    def _generate_aerial_frame(self, frame_num: int, fps: int) -> np.ndarray:
        """Generate synthetic aerial view frame."""
        # Create base aerial landscape
        frame = np.random.randint(50, 150, (1080, 1920, 3), dtype=np.uint8)
        
        # Add ground features
        # Roads
        cv2.line(frame, (0, 400), (1920, 450), (80, 80, 80), 20)
        cv2.line(frame, (500, 0), (550, 1080), (80, 80, 80), 15)
        
        # Buildings
        for i in range(10):
            x = np.random.randint(100, 1800)
            y = np.random.randint(100, 900)
            w = np.random.randint(50, 150)
            h = np.random.randint(50, 150)
            color = (np.random.randint(100, 200), np.random.randint(100, 200), np.random.randint(100, 200))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
        
        # Add timestamp and frame info
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"Test Recording - {timestamp}", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_num}", 
                   (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Simulate camera movement
        t = frame_num / fps
        offset_x = int(10 * np.sin(t * 0.1))
        offset_y = int(5 * np.cos(t * 0.15))
        
        # Apply slight movement
        M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        frame = cv2.warpAffine(frame, M, (1920, 1080))
        
        return frame


class RecordedDataConnection(DJIO4ConnectionBase):
    """DJI O4 connection that plays back recorded data for testing."""
    
    def __init__(self, video_path: str, telemetry_path: str, config: Dict[str, Any] = None):
        super().__init__(config)
        self.video_path = video_path
        self.telemetry_path = telemetry_path
        self.cap = None
        self.telemetry_data = []
        self.fps = 30
        
    async def connect(self) -> bool:
        """Load recorded data files."""
        try:
            # Load video
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video file: {self.video_path}")
            
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # Load telemetry
            with open(self.telemetry_path, 'r') as f:
                telemetry_json = json.load(f)
                self.telemetry_data = [TelemetryData(**data) for data in telemetry_json]
            
            self._connected = True
            self.logger.info(f"Loaded recorded data: {len(self.telemetry_data)} telemetry points")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load recorded data: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Release resources."""
        if self.cap:
            self.cap.release()
        self._connected = False
    
    async def stream_frames(self) -> AsyncGenerator[Tuple[FrameData, TelemetryData], None]:
        """Stream frames from recorded data."""
        if not self._connected:
            raise RuntimeError("Not connected")
        
        frame_interval = 1.0 / self.fps
        telemetry_interval = len(self.telemetry_data) / (self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        
        while self._connected:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Get corresponding telemetry
            telemetry_idx = min(int(self._frame_count * telemetry_interval), len(self.telemetry_data) - 1)
            telemetry = self.telemetry_data[telemetry_idx]
            
            frame_data = FrameData(
                frame=frame,
                timestamp=time.time(),
                pts=self._frame_count,
                frame_number=self._frame_count,
                width=frame.shape[1],
                height=frame.shape[0]
            )
            
            self._frame_count += 1
            self._last_telemetry = telemetry
            
            yield frame_data, telemetry
            await asyncio.sleep(frame_interval)


@pytest.fixture
def test_data_manager():
    """Provide test data manager."""
    return TestDataManager()


@pytest.fixture
def recorded_connection(test_data_manager):
    """Provide recorded data connection for testing."""
    video_path = test_data_manager.create_test_video("test_flight.mp4", duration=10)
    telemetry_path = test_data_manager.create_test_telemetry("test_telemetry.json", duration=10)
    return RecordedDataConnection(video_path, telemetry_path)


class TestDJIO4Connection:
    """Test cases for DJI O4 connection implementations."""
    
    def test_factory_creation(self):
        """Test connection factory."""
        # Test mobile SDK creation
        mobile_conn = DJIO4Connection.create('mobile')
        assert isinstance(mobile_conn, DJIO4MobileSDK)
        
        # Test payload SDK creation
        payload_conn = DJIO4Connection.create('payload')
        assert isinstance(payload_conn, DJIO4PayloadSDK)
    
    def test_config_handling(self):
        """Test configuration handling."""
        config = {
            'video_fps': 60,
            'telemetry_rate': 20,
            'app_key': 'test_key'
        }
        
        mobile_conn = DJIO4MobileSDK(config)
        assert mobile_conn.video_fps == 60
        assert mobile_conn.telemetry_rate == 20
        assert mobile_conn.app_key == 'test_key'
    
    @pytest.mark.asyncio
    async def test_recorded_data_playback(self, recorded_connection):
        """Test playback of recorded data."""
        # Connect
        assert await recorded_connection.connect()
        assert recorded_connection.is_connected
        
        # Stream frames
        frame_count = 0
        async for frame_data, telemetry in recorded_connection.stream_frames():
            assert isinstance(frame_data, FrameData)
            assert isinstance(telemetry, TelemetryData)
            assert frame_data.frame.shape == (1080, 1920, 3)
            
            frame_count += 1
            if frame_count >= 30:  # Test first 30 frames
                break
        
        # Disconnect
        await recorded_connection.disconnect()
        assert not recorded_connection.is_connected
    
    @pytest.mark.asyncio
    async def test_mobile_sdk_simulation(self):
        """Test Mobile SDK simulation mode."""
        config = {'video_fps': 30, 'telemetry_rate': 10}
        connection = DJIO4MobileSDK(config)
        
        # Test connection
        assert await connection.connect()
        
        # Test streaming
        frame_count = 0
        async for frame_data, telemetry in connection.stream_frames():
            assert frame_data.width == 1920
            assert frame_data.height == 1080
            assert 37.7 < telemetry.latitude < 37.8
            assert -122.5 < telemetry.longitude < -122.4
            
            frame_count += 1
            if frame_count >= 10:
                break
        
        await connection.disconnect()
    
    @pytest.mark.asyncio
    async def test_payload_sdk_simulation(self):
        """Test Payload SDK simulation mode."""
        config = {'device_path': '/dev/video0', 'baudrate': 921600}
        connection = DJIO4PayloadSDK(config)
        
        # Test connection
        assert await connection.connect()
        
        # Test streaming
        frame_count = 0
        async for frame_data, telemetry in connection.stream_frames():
            assert frame_data.width == 1920
            assert frame_data.height == 1080
            assert telemetry.flight_mode == "MANUAL"
            
            frame_count += 1
            if frame_count >= 10:
                break
        
        await connection.disconnect()
    
    def test_telemetry_serialization(self):
        """Test telemetry data serialization."""
        telemetry = TelemetryData(
            timestamp=time.time(),
            latitude=37.7749,
            longitude=-122.4194,
            altitude_msl=100.0,
            altitude_agl=50.0,
            heading=180.0,
            pitch=5.0,
            roll=3.0,
            yaw=0.0,
            velocity_x=5.0,
            velocity_y=0.0,
            velocity_z=0.0,
            gimbal_pitch=-45.0,
            gimbal_roll=0.0,
            gimbal_yaw=0.0,
            battery_percentage=85.0,
            signal_strength=4,
            gps_satellite_count=12,
            flight_mode="AUTO"
        )
        
        # Test serialization
        data_dict = telemetry.to_dict()
        assert 'position' in data_dict
        assert 'attitude' in data_dict
        assert 'velocity' in data_dict
        assert 'gimbal' in data_dict
        assert 'status' in data_dict
        
        assert data_dict['position']['latitude'] == 37.7749
        assert data_dict['attitude']['heading'] == 180.0
        assert data_dict['status']['flight_mode'] == "AUTO"


if __name__ == "__main__":
    # Run basic test
    async def main():
        # Create test data
        manager = TestDataManager()
        video_path = manager.create_test_video("demo_flight.mp4", duration=5)
        telemetry_path = manager.create_test_telemetry("demo_telemetry.json", duration=5)
        
        print(f"Created test video: {video_path}")
        print(f"Created test telemetry: {telemetry_path}")
        
        # Test recorded data connection
        connection = RecordedDataConnection(video_path, telemetry_path)
        
        if await connection.connect():
            print("Testing recorded data playback...")
            
            frame_count = 0
            async for frame_data, telemetry in connection.stream_frames():
                print(f"Frame {frame_count}: {frame_data.shape}, "
                      f"Alt: {telemetry.altitude_agl:.1f}m, "
                      f"Heading: {telemetry.heading:.1f}°")
                
                frame_count += 1
                if frame_count >= 30:
                    break
            
            await connection.disconnect()
            print("Test completed successfully!")
        else:
            print("Failed to connect to recorded data")
    
    asyncio.run(main())