"""Phone stream connection for existing phone mirroring functionality.

Integrates with existing scrcpy and phone streaming infrastructure.
"""

import asyncio
import logging
import time
from typing import AsyncGenerator, Optional, Tuple, Dict, Any, Union

import cv2
import numpy as np
from .dji_o4 import FrameData, TelemetryData, DJIO4ConnectionBase
from .rtmp_stream import RTMPConnection


class PhoneStreamConnection(DJIO4ConnectionBase):
    """Connection for phone-based streaming (RTSP, RTMP, or camera).
    
    Supports RTSP streaming, RTMP streaming from DJI Fly app, and direct camera access.
    """
    
    def __init__(self, config: Optional[Union[Dict[str, Any], str]] = None):
        # Handle string URL input
        if isinstance(config, str):
            url = config
            if url.startswith('rtmp://'):
                config = {'video_source': 'rtmp', 'rtmp_url': url}
            elif url.startswith('rtsp://'):
                config = {'video_source': 'rtsp', 'rtsp_url': url}
            else:
                config = {'video_source': 'camera', 'camera_index': int(url) if url.isdigit() else 0}
        
        super().__init__(config)
        self.video_source = self.config.get('video_source', 'rtsp')  # 'rtsp', 'rtmp', or 'camera'
        self.rtsp_url = self.config.get('rtsp_url', 'rtsp://192.168.1.100:8554/stream')
        self.rtmp_url = self.config.get('rtmp_url', 'rtmp://localhost:1935/live/dji_stream')
        self.camera_index = self.config.get('camera_index', 0)
        self.cap = None
        self.mock_gps = self.config.get('mock_gps', True)
        self.base_lat = self.config.get('base_latitude', 37.7749)
        self.base_lon = self.config.get('base_longitude', -122.4194)
        
        # RTMP connection for DJI Fly app integration
        self.rtmp_connection = None
        if self.video_source == 'rtmp':
            self.rtmp_connection = RTMPConnection(self.config)
        
    async def connect(self) -> bool:
        """Connect to phone stream."""
        try:
            if self.video_source == 'rtmp':
                # Use RTMP connection for DJI Fly app
                self.logger.info("Connecting to RTMP stream from DJI Fly app")
                if self.rtmp_connection:
                    success = await self.rtmp_connection.connect()
                    if success:
                        self._connected = True
                        return True
                    else:
                        self.logger.error("Failed to start RTMP server")
                        return False
                else:
                    self.logger.error("RTMP connection not initialized")
                    return False
            
            elif self.video_source == 'rtsp':
                # RTSP streaming
                self.logger.info(f"Connecting to RTSP stream: {self.rtsp_url}")
                self.cap = cv2.VideoCapture(self.rtsp_url)
            
            elif self.video_source == 'camera':
                # Direct camera access
                self.logger.info(f"Connecting to camera: {self.camera_index}")
                self.cap = cv2.VideoCapture(self.camera_index)
            
            else:
                self.logger.error(f"Unknown video source: {self.video_source}")
                return False
            
            # For RTSP and camera sources, test the connection
            if self.video_source in ['rtsp', 'camera']:
                if not self.cap.isOpened():
                    self.logger.error("Failed to open video source")
                    return False
                
                # Configure capture
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Test frame capture
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.error("Failed to read from video source")
                    return False
                
                self.logger.info(f"Connected to {self.video_source} stream. Frame size: {frame.shape}")
            
            self._connected = True
            self.logger.info("Successfully connected to phone stream")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to phone stream: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from phone stream."""
        if self._connected:
            self.logger.info("Disconnecting from phone stream...")
            
            if self.video_source == 'rtmp' and self.rtmp_connection:
                await self.rtmp_connection.disconnect()
            
            if self.cap:
                self.cap.release()
                self.cap = None
            
            self._connected = False
            self.logger.info("Disconnected from phone stream")
    
    async def stream_frames(self) -> AsyncGenerator[Tuple[FrameData, TelemetryData], None]:
        """Stream frames from phone."""
        if not self._connected:
            raise RuntimeError("Not connected to phone stream")
        
        if self.video_source == 'rtmp' and self.rtmp_connection:
            # Use RTMP connection's stream_frames method
            self.logger.info("Starting RTMP video stream from DJI Fly app...")
            async for frame_data, telemetry_data in self.rtmp_connection.stream_frames():
                yield frame_data, telemetry_data
        else:
            # Use traditional RTSP/camera streaming
            if not self.cap:
                raise RuntimeError("Not connected to phone stream")
            
            self.logger.info(f"Starting {self.video_source} video stream...")
            
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
            flight_mode="PHONE_STREAM" if self.video_source != 'rtmp' else "RTMP_STREAM"
        )
    
    def get_stream_info(self) -> Dict[str, Any]:
        """Get stream configuration and setup information."""
        if self.video_source == 'rtmp' and self.rtmp_connection:
            return self.rtmp_connection.get_stream_info()
        else:
            return {
                'video_source': self.video_source,
                'rtsp_url': self.rtsp_url if self.video_source == 'rtsp' else None,
                'camera_index': self.camera_index if self.video_source == 'camera' else None,
                'mock_gps': self.mock_gps,
                'instructions': {
                    'rtsp_setup': [
                        '1. Set up RTSP server on your phone or network',
                        f'2. Configure RTSP URL: {self.rtsp_url}',
                        '3. Ensure network connectivity',
                        '4. Start streaming'
                    ] if self.video_source == 'rtsp' else [],
                    'camera_setup': [
                        f'1. Connect camera to index {self.camera_index}',
                        '2. Ensure camera permissions are granted',
                        '3. Start streaming'
                    ] if self.video_source == 'camera' else []
                }
            }