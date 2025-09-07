#!/usr/bin/env python3
"""
Mock implementations for testing frame-telemetry synchronization.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, AsyncGenerator
import cv2
import numpy as np

@dataclass
class TelemetryPacket:
    """Mock telemetry packet."""
    timestamp: float
    data: Dict[str, Any]

@dataclass
class FrameData:
    """Mock frame data."""
    frame: np.ndarray
    timestamp: float
    frame_id: str
    telemetry: Optional[TelemetryPacket] = None
    metadata: Optional[Dict[str, Any]] = None

class ConnectionManager:
    """Mock connection manager."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.streams = {}
        self.active_stream = None
    
    async def get_frame(self, stream_id: Optional[str] = None) -> Optional[FrameData]:
        """Get mock frame."""
        # Create a simple test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (100, 150, 200)  # Fill with color
        
        return FrameData(
            frame=frame,
            timestamp=time.time(),
            frame_id=f"mock_frame_{int(time.time() * 1000)}",
            telemetry=TelemetryPacket(
                timestamp=time.time(),
                data={"lat": 37.7749, "lon": -122.4194}
            )
        )
    
    async def open_stream(self, stream_id: str, stream_type: str, **kwargs) -> bool:
        """Mock stream opening."""
        self.streams[stream_id] = {
            "type": stream_type,
            "config": kwargs
        }
        self.active_stream = stream_id
        return True
    
    async def close_all_streams(self) -> None:
        """Mock stream closing."""
        self.streams.clear()
        self.active_stream = None
    
    def get_stream_info(self) -> Dict[str, Any]:
        """Get mock stream info."""
        return {
            "active_stream": self.active_stream,
            "streams": list(self.streams.keys()),
            "total_frames": 100
        }