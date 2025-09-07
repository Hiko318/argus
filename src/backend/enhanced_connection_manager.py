#!/usr/bin/env python3
"""
Enhanced Connection Manager with Frame-Telemetry Synchronization

Integrates RTCP SR/PTS mapping for accurate frame-to-telemetry synchronization
in SAR operations. Provides enhanced FrameData with synchronized telemetry.
"""

import asyncio
import logging
import time
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, AsyncGenerator, Callable
from datetime import datetime, timezone

import cv2
import numpy as np

# Import existing components
try:
    from .connection_manager import ConnectionManager, FrameData, TelemetryPacket
    from .frame_telemetry_sync import FrameTelemetrySync, create_frame_telemetry_sync
    from .telemetry_service import TelemetryService, get_telemetry_service
except ImportError:
    # Fallback for direct execution
    from connection_manager import ConnectionManager, FrameData, TelemetryPacket
    from frame_telemetry_sync import FrameTelemetrySync, create_frame_telemetry_sync
    from telemetry_service import TelemetryService, get_telemetry_service

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class EnhancedFrameData(FrameData):
    """Enhanced frame data with synchronized telemetry."""
    
    # Synchronization metadata
    pts: Optional[int] = None
    synchronized_telemetry: Optional[Dict[str, Any]] = None
    sync_quality: Optional[str] = None
    time_diff_ms: Optional[float] = None
    wall_clock_time: Optional[float] = None
    sync_confidence: Optional[float] = None
    
    @property
    def has_synchronized_telemetry(self) -> bool:
        """Check if frame has synchronized telemetry data."""
        return self.synchronized_telemetry is not None
    
    @property
    def telemetry_data(self) -> Optional[Any]:
        """Get telemetry data if available."""
        if self.synchronized_telemetry:
            return self.synchronized_telemetry.get('telemetry')
        return None
    
    @property
    def is_well_synchronized(self) -> bool:
        """Check if synchronization quality is good."""
        return (self.sync_quality in ['excellent', 'good'] and 
                self.time_diff_ms is not None and 
                self.time_diff_ms < 50.0)

class RTCPReceiver:
    """Receives and processes RTCP packets for synchronization."""
    
    def __init__(self, sync_manager: FrameTelemetrySync, port: int = 5005):
        self.sync_manager = sync_manager
        self.port = port
        self.is_running = False
        self.socket = None
        self.receive_task = None
        self.logger = logging.getLogger(f"{__name__}.RTCPReceiver")
    
    async def start(self) -> bool:
        """Start RTCP receiver.
        
        Returns:
            True if started successfully
        """
        try:
            import socket
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(('0.0.0.0', self.port))
            self.socket.setblocking(False)
            
            self.is_running = True
            self.receive_task = asyncio.create_task(self._receive_loop())
            
            self.logger.info(f"RTCP receiver started on port {self.port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start RTCP receiver: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop RTCP receiver."""
        self.is_running = False
        
        if self.receive_task:
            self.receive_task.cancel()
            try:
                await self.receive_task
            except asyncio.CancelledError:
                pass
        
        if self.socket:
            self.socket.close()
            self.socket = None
        
        self.logger.info("RTCP receiver stopped")
    
    async def _receive_loop(self) -> None:
        """Main receive loop for RTCP packets."""
        while self.is_running:
            try:
                # Use asyncio to make socket non-blocking
                loop = asyncio.get_event_loop()
                data, addr = await loop.sock_recvfrom(self.socket, 1024)
                
                # Process RTCP packet
                if self.sync_manager.process_rtcp_packet(data):
                    self.logger.debug(f"Processed RTCP packet from {addr}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.is_running:  # Only log if we're supposed to be running
                    self.logger.error(f"Error receiving RTCP packet: {e}")
                await asyncio.sleep(0.1)

class EnhancedConnectionManager(ConnectionManager):
    """Enhanced connection manager with frame-telemetry synchronization."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Synchronization components
        self.telemetry_service = get_telemetry_service()
        self.sync_manager = create_frame_telemetry_sync(
            self.telemetry_service, 
            config.get('sync_config', {})
        )
        
        # RTCP receiver for synchronization
        rtcp_port = config.get('rtcp_port', 5005)
        self.rtcp_receiver = RTCPReceiver(self.sync_manager, rtcp_port)
        
        # Frame processing
        self.frame_counter = 0
        self.pts_base = None
        self.sync_enabled = config.get('enable_sync', True)
        
        # Callbacks
        self.sync_callbacks: List[Callable[[EnhancedFrameData], None]] = []
        
        self.logger = logging.getLogger(f"{__name__}.EnhancedConnectionManager")
        
        # Add sync callback
        self.sync_manager.add_sync_callback(self._on_sync_event)
    
    def add_sync_callback(self, callback: Callable[[EnhancedFrameData], None]) -> None:
        """Add callback for synchronized frames.
        
        Args:
            callback: Function to call with synchronized frame data
        """
        self.sync_callbacks.append(callback)
    
    def _on_sync_event(self, sync_data: Dict[str, Any]) -> None:
        """Handle synchronization events.
        
        Args:
            sync_data: Synchronization data from sync manager
        """
        self.logger.debug(f"Sync event: PTS={sync_data['pts']}, "
                         f"quality={sync_data['sync_quality'].value}, "
                         f"drift={sync_data['time_diff_ms']:.1f}ms")
    
    async def start_sync(self) -> bool:
        """Start synchronization components.
        
        Returns:
            True if started successfully
        """
        if not self.sync_enabled:
            self.logger.info("Synchronization disabled")
            return True
        
        # Start telemetry service
        self.telemetry_service.start_all()
        
        # Start RTCP receiver
        if await self.rtcp_receiver.start():
            self.logger.info("Frame-telemetry synchronization started")
            return True
        else:
            self.logger.error("Failed to start synchronization")
            return False
    
    async def stop_sync(self) -> None:
        """Stop synchronization components."""
        await self.rtcp_receiver.stop()
        self.telemetry_service.stop_all()
        self.logger.info("Frame-telemetry synchronization stopped")
    
    def _generate_pts(self, frame_timestamp: float) -> int:
        """Generate PTS for frame based on timestamp.
        
        Args:
            frame_timestamp: Frame timestamp
            
        Returns:
            Generated PTS value
        """
        if self.pts_base is None:
            self.pts_base = int(frame_timestamp * 90000)  # 90kHz clock
        
        # Generate PTS based on 90kHz clock
        pts = int(frame_timestamp * 90000)
        return pts
    
    async def get_enhanced_frame(self, stream_id: Optional[str] = None) -> Optional[EnhancedFrameData]:
        """Get frame with synchronized telemetry.
        
        Args:
            stream_id: Stream to get frame from (uses active stream if None)
            
        Returns:
            EnhancedFrameData with synchronized telemetry or None
        """
        # Get base frame data
        base_frame = await self.get_frame(stream_id)
        if not base_frame:
            return None
        
        # Generate PTS for this frame
        pts = self._generate_pts(base_frame.timestamp)
        self.frame_counter += 1
        
        # Create enhanced frame data
        enhanced_frame = EnhancedFrameData(
            frame=base_frame.frame,
            timestamp=base_frame.timestamp,
            frame_id=base_frame.frame_id,
            telemetry=base_frame.telemetry,
            metadata=base_frame.metadata,
            pts=pts
        )
        
        # Get synchronized telemetry if sync is enabled
        if self.sync_enabled:
            sync_data = self.sync_manager.get_synchronized_telemetry(pts)
            if sync_data:
                enhanced_frame.synchronized_telemetry = sync_data
                enhanced_frame.sync_quality = sync_data['sync_quality'].value
                enhanced_frame.time_diff_ms = sync_data['time_diff_ms']
                enhanced_frame.wall_clock_time = sync_data['wall_clock_time']
                enhanced_frame.sync_confidence = sync_data['confidence']
                
                self.logger.debug(f"Frame {self.frame_counter} synchronized: "
                                f"PTS={pts}, quality={enhanced_frame.sync_quality}, "
                                f"drift={enhanced_frame.time_diff_ms:.1f}ms")
            else:
                self.logger.debug(f"Frame {self.frame_counter} not synchronized: PTS={pts}")
        
        # Notify callbacks
        for callback in self.sync_callbacks:
            try:
                callback(enhanced_frame)
            except Exception as e:
                self.logger.error(f"Sync callback error: {e}")
        
        return enhanced_frame
    
    async def stream_enhanced_frames(self, stream_id: Optional[str] = None) -> AsyncGenerator[EnhancedFrameData, None]:
        """Stream enhanced frames with synchronized telemetry.
        
        Args:
            stream_id: Stream to get frames from (uses active stream if None)
            
        Yields:
            EnhancedFrameData with synchronized telemetry
        """
        self.logger.info(f"Starting enhanced frame stream for {stream_id or 'active stream'}")
        
        try:
            while True:
                enhanced_frame = await self.get_enhanced_frame(stream_id)
                if enhanced_frame:
                    yield enhanced_frame
                else:
                    # Small delay if no frame available
                    await asyncio.sleep(0.001)
                    
        except asyncio.CancelledError:
            self.logger.info("Enhanced frame stream cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in enhanced frame stream: {e}")
            raise
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get synchronization status.
        
        Returns:
            Dictionary with sync status information
        """
        base_status = self.get_stream_info()
        sync_status = self.sync_manager.get_sync_status()
        
        return {
            **base_status,
            "synchronization": {
                **sync_status,
                "sync_enabled": self.sync_enabled,
                "rtcp_receiver_running": self.rtcp_receiver.is_running,
                "frame_counter": self.frame_counter,
                "pts_base": self.pts_base
            }
        }
    
    async def close_all_streams(self) -> None:
        """Close all streams and stop synchronization."""
        await self.stop_sync()
        await super().close_all_streams()
        self.logger.info("All enhanced streams closed")

# Factory function for easy initialization
def create_enhanced_connection_manager(config: Dict[str, Any]) -> EnhancedConnectionManager:
    """Create and configure EnhancedConnectionManager.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured EnhancedConnectionManager instance
    """
    # Set default sync configuration
    default_config = {
        "enable_sync": True,
        "rtcp_port": 5005,
        "sync_config": {
            "max_history": 5000
        }
    }
    
    # Merge with provided config
    merged_config = {**default_config, **config}
    
    manager = EnhancedConnectionManager(merged_config)
    logger.info(f"Created EnhancedConnectionManager with sync_enabled={merged_config['enable_sync']}")
    
    return manager

if __name__ == "__main__":
    # Demo usage
    import sys
    
    async def demo():
        """Demonstrate enhanced connection manager."""
        config = {
            "enable_sync": True,
            "rtcp_port": 5005,
            "recorded": {
                "video_path": "data/test_video.mp4",
                "telemetry_path": "data/test_telemetry.json",
                "playback_speed": 1.0,
                "loop": True
            }
        }
        
        manager = create_enhanced_connection_manager(config)
        
        # Add sync callback
        def on_sync_frame(frame_data):
            if frame_data.has_synchronized_telemetry:
                print(f"Synchronized frame {frame_data.frame_id}: "
                      f"quality={frame_data.sync_quality}, "
                      f"drift={frame_data.time_diff_ms:.1f}ms")
        
        manager.add_sync_callback(on_sync_frame)
        
        try:
            # Start synchronization
            if await manager.start_sync():
                print("Synchronization started")
            
            # Open test stream
            stream = await manager.open_stream(
                "test_stream", 
                "recorded",
                video_path="data/test_video.mp4"
            )
            
            if stream:
                print("Stream opened, processing frames...")
                
                # Process some frames
                frame_count = 0
                async for enhanced_frame in manager.stream_enhanced_frames():
                    frame_count += 1
                    
                    print(f"Frame {frame_count}: {enhanced_frame.frame.shape}, "
                          f"PTS={enhanced_frame.pts}, "
                          f"sync={enhanced_frame.has_synchronized_telemetry}")
                    
                    if frame_count >= 10:
                        break
                
                # Print final status
                status = manager.get_sync_status()
                print(f"\nFinal status: {status['synchronization']}")
            
        except KeyboardInterrupt:
            print("\nDemo interrupted")
        
        finally:
            await manager.close_all_streams()
            print("Demo completed")
    
    # Run demo
    asyncio.run(demo())