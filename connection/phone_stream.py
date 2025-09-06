"""Phone stream connection for existing phone mirroring functionality.

Integrates with existing scrcpy and phone streaming infrastructure.
"""

import asyncio
import logging
import time
from typing import AsyncGenerator, Optional, Tuple, Dict, Any

import cv2
import numpy as np
from .dji_o4 import FrameData, TelemetryData, DJIO4ConnectionBase


class PhoneStreamConnection(DJIO4ConnectionBase):
    """Connection for phone-based video streaming.
    
    Integrates with existing phone mirroring infrastructure (scrcpy, etc.)
    and provides mock telemetry for testing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.stream_url = self.config.get('stream_url', 'rtsp://127.0.0.1:8554/scrcpy')
        self.video_source = self.config.get('video_source', 0)  # Camera index or URL
        self.mock_gps = self.config.get('mock_gps', True)
        self.base_lat = self.config.get('base_latitude', 37.7749)
        self.base_lon = self.config.get('base_longitude', -122.4194)
        self.cap = None
        
    async def connect(self) -> bool:
        """Connect to phone stream."""
        try:
            self.logger.info(f"Connecting to phone stream: {self.stream_url}")
            
            # Try RTSP stream first, fallback to camera
            self.cap = cv2.VideoCapture(self.stream_url)
            
            if not self.cap.isOpened():
                self.logger.warning(f"RTSP stream failed, trying camera {self.video_source}")
                self.cap = cv2.VideoCapture(self.video_source)
            
            if not self.cap.isOpened():
                self.logger.error("Failed to open video source")
                return False
            
            # Configure capture
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self._connected = True
            self.logger.info("Successfully connected to phone stream")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to phone stream: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from phone stream."""
        if self._connected and self.cap:
            self.logger.info("Disconnecting from phone stream...")
            self.cap.release()
            self.cap = None
            self._connected = False
            self.logger.info("Disconnected from phone stream")
    
    async def stream_frames(self) -> AsyncGenerator[Tuple[FrameData, TelemetryData], None]:
        """Stream frames from phone."""
        if not self._connected or not self.cap:
            raise RuntimeError("Not connected to phone stream")
        
        self.logger.info("Starting phone video stream...")
        
        try:
            while self._connected:
                ret, frame = self.cap.read()
                
                if not ret:
                    self.logger.warning("Failed to read frame, attempting reconnection...")
                    await asyncio.sleep(1)
                    continue
                
                # Generate mock telemetry
                telemetry = self._generate_mock_telemetry()
                
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
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.001)
                
        except Exception as e:
            self.logger.error(f"Error in phone stream: {e}")
            raise
    
    def _generate_mock_telemetry(self) -> TelemetryData:
        """Generate mock telemetry for phone-based testing."""
        if not self.mock_gps:
            # Return minimal telemetry
            return TelemetryData(
                timestamp=time.time(),
                latitude=0.0,
                longitude=0.0,
                altitude_msl=0.0,
                altitude_agl=0.0,
                heading=0.0,
                pitch=0.0,
                roll=0.0,
                yaw=0.0,
                velocity_x=0.0,
                velocity_y=0.0,
                velocity_z=0.0,
                gimbal_pitch=0.0,
                gimbal_roll=0.0,
                gimbal_yaw=0.0,
                battery_percentage=100.0,
                signal_strength=5,
                gps_satellite_count=0,
                flight_mode="PHONE"
            )
        
        # Generate realistic mock GPS data for testing
        t = time.time()
        return TelemetryData(
            timestamp=t,
            latitude=self.base_lat + 0.0005 * np.sin(t * 0.05),
            longitude=self.base_lon + 0.0005 * np.cos(t * 0.05),
            altitude_msl=50.0 + 5 * np.sin(t * 0.02),
            altitude_agl=30.0 + 5 * np.sin(t * 0.02),
            heading=180 + 20 * np.sin(t * 0.01),
            pitch=2 * np.sin(t * 0.03),
            roll=1 * np.cos(t * 0.04),
            yaw=0,
            velocity_x=2.0,
            velocity_y=0.5,
            velocity_z=0.0,
            gimbal_pitch=-30,
            gimbal_roll=0,
            gimbal_yaw=0,
            battery_percentage=100.0,  # Phone doesn't have flight battery
            signal_strength=4,
            gps_satellite_count=8,
            flight_mode="PHONE"
        )