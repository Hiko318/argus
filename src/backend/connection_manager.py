"""DJI O4 Connection Manager for stream and telemetry fusion."""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, AsyncGenerator
from datetime import datetime
import cv2
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class TelemetryPacket:
    """Telemetry data packet from DJI O4."""
    timestamp: float
    latitude: float
    longitude: float
    altitude: float  # meters above sea level
    relative_altitude: float  # meters above takeoff point
    heading: float  # degrees (0-360)
    pitch: float  # degrees (-90 to 90)
    roll: float  # degrees (-180 to 180)
    yaw: float  # degrees (-180 to 180)
    velocity_x: float  # m/s
    velocity_y: float  # m/s
    velocity_z: float  # m/s
    gimbal_pitch: float  # degrees
    gimbal_roll: float  # degrees
    gimbal_yaw: float  # degrees
    battery_level: int  # percentage (0-100)
    signal_strength: int  # percentage (0-100)
    gps_satellites: int
    flight_mode: str
    is_recording: bool
    camera_mode: str
    iso: int
    shutter_speed: str
    aperture: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TelemetryPacket':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class FrameData:
    """Frame data with metadata."""
    frame: np.ndarray
    timestamp: float
    frame_id: int
    telemetry: Optional[TelemetryPacket] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self, include_frame: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "telemetry": self.telemetry.to_dict() if self.telemetry else None,
            "metadata": self.metadata
        }
        if include_frame:
            data["frame_shape"] = self.frame.shape
            data["frame_dtype"] = str(self.frame.dtype)
        return data


class FrameStream(ABC):
    """Abstract base class for frame streams."""
    
    @abstractmethod
    async def get_frame(self) -> Optional[FrameData]:
        """Get the next frame from the stream."""
        pass
    
    @abstractmethod
    async def start(self) -> bool:
        """Start the stream."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the stream."""
        pass
    
    @abstractmethod
    def is_active(self) -> bool:
        """Check if stream is active."""
        pass
    
    @abstractmethod
    def get_stream_info(self) -> Dict[str, Any]:
        """Get stream information."""
        pass


class DJILiveStream(FrameStream):
    """Live DJI O4 stream implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_running = False
        self.frame_count = 0
        self.start_time = None
        self.last_telemetry = None
        
        # DJI SDK connection parameters
        self.rtmp_url = config.get("rtmp_url", "rtmp://localhost:1935/live/stream")
        self.telemetry_port = config.get("telemetry_port", 8080)
        self.connection_timeout = config.get("connection_timeout", 10.0)
        
        # Video capture
        self.cap = None
        
        logger.info(f"Initialized DJI Live Stream with RTMP: {self.rtmp_url}")
    
    async def start(self) -> bool:
        """Start the live stream."""
        try:
            logger.info("Starting DJI O4 live stream...")
            
            # Initialize video capture
            self.cap = cv2.VideoCapture(self.rtmp_url)
            if not self.cap.isOpened():
                logger.error(f"Failed to open RTMP stream: {self.rtmp_url}")
                return False
            
            # Configure capture properties
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_running = True
            self.start_time = time.time()
            self.frame_count = 0
            
            logger.info("DJI O4 live stream started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start DJI live stream: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop the live stream."""
        logger.info("Stopping DJI O4 live stream...")
        
        self.is_running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        logger.info("DJI O4 live stream stopped")
    
    async def get_frame(self) -> Optional[FrameData]:
        """Get the next frame from the live stream."""
        if not self.is_running or not self.cap:
            return None
        
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning("Failed to read frame from DJI stream")
                return None
            
            self.frame_count += 1
            timestamp = time.time()
            
            # Get latest telemetry (mock for now - would integrate with DJI SDK)
            telemetry = await self._get_telemetry()
            
            return FrameData(
                frame=frame,
                timestamp=timestamp,
                frame_id=self.frame_count,
                telemetry=telemetry,
                metadata={
                    "source": "dji_o4_live",
                    "stream_url": self.rtmp_url,
                    "frame_size": frame.shape[:2]
                }
            )
            
        except Exception as e:
            logger.error(f"Error getting frame from DJI stream: {e}")
            return None
    
    async def _get_telemetry(self) -> Optional[TelemetryPacket]:
        """Get telemetry data from DJI O4."""
        # TODO: Integrate with actual DJI SDK
        # For now, return mock telemetry data
        return TelemetryPacket(
            timestamp=time.time(),
            latitude=37.7749,  # San Francisco
            longitude=-122.4194,
            altitude=100.0,
            relative_altitude=50.0,
            heading=45.0,
            pitch=-15.0,
            roll=2.0,
            yaw=45.0,
            velocity_x=5.0,
            velocity_y=0.0,
            velocity_z=0.0,
            gimbal_pitch=-30.0,
            gimbal_roll=0.0,
            gimbal_yaw=0.0,
            battery_level=85,
            signal_strength=95,
            gps_satellites=12,
            flight_mode="GPS",
            is_recording=True,
            camera_mode="Video",
            iso=100,
            shutter_speed="1/60",
            aperture="f/2.8"
        )
    
    def is_active(self) -> bool:
        """Check if stream is active."""
        return self.is_running and self.cap is not None and self.cap.isOpened()
    
    def get_stream_info(self) -> Dict[str, Any]:
        """Get stream information."""
        info = {
            "type": "dji_o4_live",
            "rtmp_url": self.rtmp_url,
            "is_active": self.is_active(),
            "frame_count": self.frame_count
        }
        
        if self.start_time:
            info["uptime"] = time.time() - self.start_time
            if self.frame_count > 0:
                info["fps"] = self.frame_count / info["uptime"]
        
        if self.cap:
            info["resolution"] = {
                "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            }
        
        return info


class RecordedDataStream(FrameStream):
    """Recorded data stream for testing without drone."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_running = False
        self.frame_count = 0
        self.start_time = None
        
        # Recorded data parameters
        self.video_path = config.get("video_path")
        self.telemetry_path = config.get("telemetry_path")
        self.playback_speed = config.get("playback_speed", 1.0)
        self.loop = config.get("loop", True)
        
        # Data storage
        self.cap = None
        self.telemetry_data = []
        self.telemetry_index = 0
        
        logger.info(f"Initialized Recorded Data Stream: {self.video_path}")
    
    async def start(self) -> bool:
        """Start the recorded data stream."""
        try:
            logger.info("Starting recorded data stream...")
            
            # Load video file
            if self.video_path and Path(self.video_path).exists():
                self.cap = cv2.VideoCapture(str(self.video_path))
                if not self.cap.isOpened():
                    logger.error(f"Failed to open video file: {self.video_path}")
                    return False
            else:
                logger.error(f"Video file not found: {self.video_path}")
                return False
            
            # Load telemetry data
            await self._load_telemetry_data()
            
            self.is_running = True
            self.start_time = time.time()
            self.frame_count = 0
            self.telemetry_index = 0
            
            logger.info("Recorded data stream started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start recorded data stream: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop the recorded data stream."""
        logger.info("Stopping recorded data stream...")
        
        self.is_running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        logger.info("Recorded data stream stopped")
    
    async def get_frame(self) -> Optional[FrameData]:
        """Get the next frame from the recorded stream."""
        if not self.is_running or not self.cap:
            return None
        
        try:
            ret, frame = self.cap.read()
            
            # Handle end of video
            if not ret or frame is None:
                if self.loop:
                    # Restart from beginning
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.telemetry_index = 0
                    ret, frame = self.cap.read()
                    if not ret:
                        return None
                else:
                    return None
            
            self.frame_count += 1
            timestamp = time.time()
            
            # Get corresponding telemetry
            telemetry = self._get_telemetry_for_frame()
            
            # Simulate playback speed
            if self.playback_speed != 1.0:
                await asyncio.sleep(1.0 / (30.0 * self.playback_speed))  # Assume 30 FPS
            
            return FrameData(
                frame=frame,
                timestamp=timestamp,
                frame_id=self.frame_count,
                telemetry=telemetry,
                metadata={
                    "source": "recorded_data",
                    "video_path": self.video_path,
                    "frame_size": frame.shape[:2],
                    "playback_speed": self.playback_speed
                }
            )
            
        except Exception as e:
            logger.error(f"Error getting frame from recorded stream: {e}")
            return None
    
    async def _load_telemetry_data(self) -> None:
        """Load telemetry data from file."""
        if not self.telemetry_path or not Path(self.telemetry_path).exists():
            logger.warning(f"Telemetry file not found: {self.telemetry_path}")
            # Generate synthetic telemetry
            self._generate_synthetic_telemetry()
            return
        
        try:
            with open(self.telemetry_path, 'r') as f:
                data = json.load(f)
                self.telemetry_data = [
                    TelemetryPacket.from_dict(item) for item in data
                ]
            logger.info(f"Loaded {len(self.telemetry_data)} telemetry packets")
            
        except Exception as e:
            logger.error(f"Failed to load telemetry data: {e}")
            self._generate_synthetic_telemetry()
    
    def _generate_synthetic_telemetry(self) -> None:
        """Generate synthetic telemetry for testing."""
        logger.info("Generating synthetic telemetry data")
        
        # Generate 1000 telemetry packets (about 33 seconds at 30 FPS)
        base_time = time.time()
        for i in range(1000):
            self.telemetry_data.append(TelemetryPacket(
                timestamp=base_time + i * (1.0 / 30.0),
                latitude=37.7749 + (i * 0.0001),  # Simulate movement
                longitude=-122.4194 + (i * 0.0001),
                altitude=100.0 + (i * 0.1),
                relative_altitude=50.0 + (i * 0.1),
                heading=45.0 + (i * 0.5) % 360,
                pitch=-15.0 + np.sin(i * 0.1) * 5,
                roll=np.sin(i * 0.05) * 3,
                yaw=45.0 + (i * 0.5) % 360,
                velocity_x=5.0 + np.sin(i * 0.1),
                velocity_y=np.cos(i * 0.1),
                velocity_z=0.1 * np.sin(i * 0.05),
                gimbal_pitch=-30.0 + np.sin(i * 0.2) * 10,
                gimbal_roll=np.sin(i * 0.1) * 2,
                gimbal_yaw=np.cos(i * 0.15) * 15,
                battery_level=max(20, 100 - i // 10),
                signal_strength=max(50, 100 - i // 20),
                gps_satellites=min(15, 8 + i // 50),
                flight_mode="GPS",
                is_recording=True,
                camera_mode="Video",
                iso=100,
                shutter_speed="1/60",
                aperture="f/2.8"
            ))
    
    def _get_telemetry_for_frame(self) -> Optional[TelemetryPacket]:
        """Get telemetry data for current frame."""
        if not self.telemetry_data:
            return None
        
        if self.telemetry_index < len(self.telemetry_data):
            telemetry = self.telemetry_data[self.telemetry_index]
            self.telemetry_index += 1
            return telemetry
        
        # Loop telemetry if video loops
        if self.loop:
            self.telemetry_index = 0
            return self.telemetry_data[0] if self.telemetry_data else None
        
        return None
    
    def is_active(self) -> bool:
        """Check if stream is active."""
        return self.is_running and self.cap is not None and self.cap.isOpened()
    
    def get_stream_info(self) -> Dict[str, Any]:
        """Get stream information."""
        info = {
            "type": "recorded_data",
            "video_path": self.video_path,
            "telemetry_path": self.telemetry_path,
            "is_active": self.is_active(),
            "frame_count": self.frame_count,
            "playback_speed": self.playback_speed,
            "loop": self.loop
        }
        
        if self.start_time:
            info["uptime"] = time.time() - self.start_time
            if self.frame_count > 0:
                info["fps"] = self.frame_count / info["uptime"]
        
        if self.cap:
            info["total_frames"] = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            info["resolution"] = {
                "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            }
        
        if self.telemetry_data:
            info["telemetry_packets"] = len(self.telemetry_data)
        
        return info


class ConnectionManager:
    """DJI O4 Connection Manager with clean API for stream and telemetry access."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.streams: Dict[str, FrameStream] = {}
        self.active_stream_id: Optional[str] = None
        
        logger.info("Initialized DJI O4 Connection Manager")
    
    async def open_stream(self, stream_id: str, stream_type: str = "live", **kwargs) -> Optional[FrameStream]:
        """Open a new stream.
        
        Args:
            stream_id: Unique identifier for the stream
            stream_type: Type of stream ('live' or 'recorded')
            **kwargs: Additional configuration for the stream
        
        Returns:
            FrameStream instance or None if failed
        """
        try:
            # Merge config with kwargs
            stream_config = {**self.config.get(stream_type, {}), **kwargs}
            
            # Create appropriate stream type
            if stream_type == "live":
                stream = DJILiveStream(stream_config)
            elif stream_type == "recorded":
                stream = RecordedDataStream(stream_config)
            else:
                logger.error(f"Unknown stream type: {stream_type}")
                return None
            
            # Start the stream
            if await stream.start():
                self.streams[stream_id] = stream
                if self.active_stream_id is None:
                    self.active_stream_id = stream_id
                
                logger.info(f"Opened {stream_type} stream: {stream_id}")
                return stream
            else:
                logger.error(f"Failed to start {stream_type} stream: {stream_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error opening stream {stream_id}: {e}")
            return None
    
    async def close_stream(self, stream_id: str) -> bool:
        """Close a stream.
        
        Args:
            stream_id: Stream identifier to close
        
        Returns:
            True if successfully closed
        """
        if stream_id not in self.streams:
            logger.warning(f"Stream not found: {stream_id}")
            return False
        
        try:
            await self.streams[stream_id].stop()
            del self.streams[stream_id]
            
            if self.active_stream_id == stream_id:
                # Set new active stream if available
                self.active_stream_id = next(iter(self.streams.keys()), None)
            
            logger.info(f"Closed stream: {stream_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error closing stream {stream_id}: {e}")
            return False
    
    async def get_frame(self, stream_id: Optional[str] = None) -> Optional[FrameData]:
        """Get a frame from the specified stream or active stream.
        
        Args:
            stream_id: Stream to get frame from (uses active stream if None)
        
        Returns:
            FrameData or None if no frame available
        """
        target_stream_id = stream_id or self.active_stream_id
        
        if not target_stream_id or target_stream_id not in self.streams:
            return None
        
        return await self.streams[target_stream_id].get_frame()
    
    async def get_telemetry(self, stream_id: Optional[str] = None) -> Optional[TelemetryPacket]:
        """Get telemetry from the specified stream or active stream.
        
        Args:
            stream_id: Stream to get telemetry from (uses active stream if None)
        
        Returns:
            TelemetryPacket or None if no telemetry available
        """
        frame_data = await self.get_frame(stream_id)
        return frame_data.telemetry if frame_data else None
    
    def get_stream_info(self, stream_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get information about a stream.
        
        Args:
            stream_id: Stream to get info for (uses active stream if None)
        
        Returns:
            Stream information dictionary or None
        """
        target_stream_id = stream_id or self.active_stream_id
        
        if not target_stream_id or target_stream_id not in self.streams:
            return None
        
        return self.streams[target_stream_id].get_stream_info()
    
    def list_streams(self) -> Dict[str, Dict[str, Any]]:
        """List all active streams.
        
        Returns:
            Dictionary of stream_id -> stream_info
        """
        return {
            stream_id: stream.get_stream_info()
            for stream_id, stream in self.streams.items()
        }
    
    def set_active_stream(self, stream_id: str) -> bool:
        """Set the active stream.
        
        Args:
            stream_id: Stream to set as active
        
        Returns:
            True if successfully set
        """
        if stream_id in self.streams:
            self.active_stream_id = stream_id
            logger.info(f"Set active stream: {stream_id}")
            return True
        else:
            logger.warning(f"Cannot set active stream, not found: {stream_id}")
            return False
    
    def get_active_stream_id(self) -> Optional[str]:
        """Get the active stream ID.
        
        Returns:
            Active stream ID or None
        """
        return self.active_stream_id
    
    async def close_all_streams(self) -> None:
        """Close all active streams."""
        logger.info("Closing all streams...")
        
        for stream_id in list(self.streams.keys()):
            await self.close_stream(stream_id)
        
        self.active_stream_id = None
        logger.info("All streams closed")
    
    def get_manager_status(self) -> Dict[str, Any]:
        """Get connection manager status.
        
        Returns:
            Status information dictionary
        """
        return {
            "active_streams": len(self.streams),
            "active_stream_id": self.active_stream_id,
            "streams": self.list_streams(),
            "timestamp": time.time()
        }


# Example usage and configuration
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Configuration for different stream types
        config = {
            "live": {
                "rtmp_url": "rtmp://localhost:1935/live/stream",
                "telemetry_port": 8080,
                "connection_timeout": 10.0
            },
            "recorded": {
                "video_path": "data/test_video.mp4",
                "telemetry_path": "data/test_telemetry.json",
                "playback_speed": 1.0,
                "loop": True
            }
        }
        
        # Initialize connection manager
        manager = ConnectionManager(config)
        
        try:
            # Open a recorded data stream for testing
            stream = await manager.open_stream(
                "test_stream", 
                "recorded",
                video_path="data/test_video.mp4"
            )
            
            if stream:
                print("Stream opened successfully")
                
                # Get some frames
                for i in range(5):
                    frame_data = await manager.get_frame()
                    if frame_data:
                        print(f"Frame {frame_data.frame_id}: {frame_data.frame.shape}")
                        if frame_data.telemetry:
                            print(f"  Telemetry: lat={frame_data.telemetry.latitude:.6f}, "
                                  f"lon={frame_data.telemetry.longitude:.6f}")
                    
                    await asyncio.sleep(0.1)
                
                # Get stream info
                info = manager.get_stream_info()
                print(f"Stream info: {info}")
            
        finally:
            # Clean up
            await manager.close_all_streams()
    
    # Run example
    asyncio.run(main())