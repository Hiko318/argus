"""RTMP stream connection for DJI Fly app integration.

Enables phone-as-bridge functionality where DJI Fly app streams
RTMP video to PC for processing.
"""

import asyncio
import logging
import time
import subprocess
import threading
from typing import AsyncGenerator, Optional, Tuple, Dict, Any

import cv2
import numpy as np
from .dji_o4 import FrameData, TelemetryData, DJIO4ConnectionBase


class RTMPConnection(DJIO4ConnectionBase):
    """Connection for RTMP streams from DJI Fly app.
    
    Receives RTMP streams from DJI Fly app running on smartphone
    and provides real-time video processing capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.rtmp_port = self.config.get('rtmp_port', 1935)
        self.rtmp_app = self.config.get('rtmp_app', 'live')
        self.rtmp_stream_key = self.config.get('rtmp_stream_key', 'dji_stream')
        self.rtmp_url = f"rtmp://localhost:{self.rtmp_port}/{self.rtmp_app}/{self.rtmp_stream_key}"
        self.output_url = f"rtmp://localhost:{self.rtmp_port}/{self.rtmp_app}/{self.rtmp_stream_key}_out"
        
        # RTMP server process
        self.rtmp_server_process = None
        self.ffmpeg_process = None
        self.cap = None
        
        # Stream settings
        self.target_fps = self.config.get('target_fps', 30)
        self.target_width = self.config.get('target_width', 1920)
        self.target_height = self.config.get('target_height', 1080)
        
    async def connect(self) -> bool:
        """Start RTMP server and prepare for incoming streams."""
        try:
            self.logger.info(f"Starting RTMP server on port {self.rtmp_port}")
            
            # Start simple RTMP server using FFmpeg
            await self._start_rtmp_server()
            
            # Wait a moment for server to start
            await asyncio.sleep(2)
            
            self.logger.info(f"RTMP server ready. Stream URL: {self.rtmp_url}")
            self.logger.info("Configure DJI Fly app to stream to this URL")
            
            self._connected = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start RTMP server: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Stop RTMP server and cleanup."""
        if self._connected:
            self.logger.info("Stopping RTMP server...")
            
            if self.cap:
                self.cap.release()
                self.cap = None
                
            if self.ffmpeg_process:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process = None
                
            if self.rtmp_server_process:
                self.rtmp_server_process.terminate()
                self.rtmp_server_process = None
                
            self._connected = False
            self.logger.info("RTMP server stopped")
    
    async def _start_rtmp_server(self) -> None:
        """Start RTMP server using FFmpeg."""
        try:
            # Create a simple RTMP relay using FFmpeg
            # This will accept RTMP input and make it available for OpenCV
            cmd = [
                'ffmpeg',
                '-f', 'flv',
                '-listen', '1',
                '-i', f'rtmp://localhost:{self.rtmp_port}/{self.rtmp_app}/{self.rtmp_stream_key}',
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-f', 'rtsp',
                f'rtsp://localhost:8554/{self.rtmp_stream_key}'
            ]
            
            self.logger.info(f"Starting FFmpeg RTMP relay: {' '.join(cmd)}")
            
            # Start FFmpeg in background thread
            def run_ffmpeg():
                try:
                    self.ffmpeg_process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    
                    # Log FFmpeg output
                    for line in self.ffmpeg_process.stderr:
                        if line.strip():
                            self.logger.debug(f"FFmpeg: {line.strip()}")
                            
                except Exception as e:
                    self.logger.error(f"FFmpeg process error: {e}")
            
            ffmpeg_thread = threading.Thread(target=run_ffmpeg, daemon=True)
            ffmpeg_thread.start()
            
        except Exception as e:
            self.logger.error(f"Failed to start FFmpeg RTMP relay: {e}")
            raise
    
    async def wait_for_stream(self, timeout: int = 30) -> bool:
        """Wait for RTMP stream to become available."""
        self.logger.info("Waiting for RTMP stream from DJI Fly app...")
        
        rtsp_url = f"rtsp://localhost:8554/{self.rtmp_stream_key}"
        
        for i in range(timeout):
            try:
                self.cap = cv2.VideoCapture(rtsp_url)
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self.logger.info("RTMP stream detected and ready!")
                        return True
                
                if self.cap:
                    self.cap.release()
                    self.cap = None
                    
            except Exception as e:
                self.logger.debug(f"Stream check attempt {i+1}: {e}")
            
            await asyncio.sleep(1)
        
        self.logger.warning(f"No RTMP stream received within {timeout} seconds")
        return False
    
    async def stream_frames(self) -> AsyncGenerator[Tuple[FrameData, TelemetryData], None]:
        """Stream frames from RTMP input."""
        if not self._connected:
            raise RuntimeError("Not connected to RTMP server")
        
        # Wait for stream to become available
        if not await self.wait_for_stream():
            raise RuntimeError("No RTMP stream available")
        
        self.logger.info("Starting RTMP video stream processing...")
        
        try:
            while self._connected and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    self.logger.warning("Lost RTMP stream, attempting reconnection...")
                    await asyncio.sleep(1)
                    
                    # Try to reconnect
                    if await self.wait_for_stream(timeout=5):
                        continue
                    else:
                        break
                
                # Generate telemetry (could be enhanced with actual DJI data)
                telemetry = self._generate_telemetry()
                
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
                
                # Control frame rate
                await asyncio.sleep(1.0 / self.target_fps)
                
        except Exception as e:
            self.logger.error(f"Error in RTMP stream: {e}")
            raise
    
    def _generate_telemetry(self) -> TelemetryData:
        """Generate telemetry data.
        
        Note: This generates mock telemetry. In a real implementation,
        you might extract telemetry from DJI SDK or other sources.
        """
        t = time.time()
        
        # Generate realistic flight telemetry
        return TelemetryData(
            timestamp=t,
            latitude=37.7749 + 0.001 * np.sin(t * 0.01),  # Mock GPS movement
            longitude=-122.4194 + 0.001 * np.cos(t * 0.01),
            altitude_msl=100.0 + 20 * np.sin(t * 0.005),
            altitude_agl=80.0 + 20 * np.sin(t * 0.005),
            heading=180 + 30 * np.sin(t * 0.02),
            pitch=5 * np.sin(t * 0.03),
            roll=3 * np.cos(t * 0.04),
            yaw=0,
            velocity_x=5.0 + 2 * np.sin(t * 0.01),
            velocity_y=1.0,
            velocity_z=0.5 * np.sin(t * 0.02),
            gimbal_pitch=-45 + 10 * np.sin(t * 0.01),
            gimbal_roll=0,
            gimbal_yaw=0,
            battery_percentage=85.0,
            signal_strength=4,
            gps_satellite_count=12,
            flight_mode="RTMP_STREAM"
        )
    
    def get_stream_info(self) -> Dict[str, Any]:
        """Get RTMP stream configuration info."""
        return {
            'rtmp_url': self.rtmp_url,
            'rtmp_port': self.rtmp_port,
            'rtmp_app': self.rtmp_app,
            'stream_key': self.rtmp_stream_key,
            'instructions': {
                'dji_fly_setup': [
                    '1. Open DJI Fly app on your smartphone',
                    '2. Connect to your drone/goggles',
                    '3. Go to Settings > Live Streaming',
                    '4. Select "Custom RTMP"',
                    f'5. Enter RTMP URL: {self.rtmp_url}',
                    '6. Start streaming',
                    '7. Video will appear in Foresight application'
                ]
            }
        }