#!/usr/bin/env python3
"""
Foresight Ingest Module

Standardized interface for ingesting video streams from multiple sources:
- DJI O4 UDP/RTSP streams
- Android WebRTC streams
- scrcpy stream fallback
- HTTP MJPEG streams

Usage:
    python -m src.backend.ingest
"""

import asyncio
import logging
import time
import json
import socket
import struct
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, List, Callable
import threading
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import queue
import subprocess
from datetime import datetime

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import JSONResponse
import uvicorn

# Try to import DJI-related libraries
try:
    from djitellopy import Tello
    DJI_AVAILABLE = True
except ImportError:
    DJI_AVAILABLE = False
    logger.warning("DJI Tello library not available. Install with: pip install djitellopy")

# Try to import USB video capture libraries
try:
    import pyudev
    USB_CAPTURE_AVAILABLE = True
except ImportError:
    USB_CAPTURE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IngestStatus(Enum):
    """Status of ingest source"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


@dataclass
class TelemetryData:
    """Telemetry data from drone or other sources"""
    timestamp: float
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None  # meters above sea level
    relative_altitude: Optional[float] = None  # meters above takeoff point
    heading: Optional[float] = None  # degrees (0-360)
    pitch: Optional[float] = None  # degrees
    roll: Optional[float] = None  # degrees
    yaw: Optional[float] = None  # degrees
    speed: Optional[float] = None  # m/s
    battery_level: Optional[int] = None  # percentage
    signal_strength: Optional[int] = None  # percentage
    flight_mode: Optional[str] = None
    is_flying: Optional[bool] = None
    temperature: Optional[float] = None  # celsius
    barometer: Optional[float] = None  # hPa
    additional_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

@dataclass
class FrameMetadata:
    """Metadata for ingested frames"""
    timestamp: float
    frame_id: int
    source_id: str
    width: int
    height: int
    fps: float
    format: str = "BGR"
    telemetry: Optional[TelemetryData] = None


class TelemetryManager:
    """Manages telemetry data from various sources"""
    
    def __init__(self):
        self._telemetry_queue = queue.Queue(maxsize=1000)
        self._latest_telemetry: Optional[TelemetryData] = None
        self._telemetry_callbacks: List[Callable[[TelemetryData], None]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
    def start(self):
        """Start telemetry processing"""
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._process_telemetry, daemon=True)
            self._thread.start()
            logger.info("Telemetry manager started")
            
    def stop(self):
        """Stop telemetry processing"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        logger.info("Telemetry manager stopped")
        
    def add_telemetry(self, telemetry: TelemetryData):
        """Add telemetry data to the queue"""
        try:
            self._telemetry_queue.put_nowait(telemetry)
        except queue.Full:
            # Remove oldest telemetry if queue is full
            try:
                self._telemetry_queue.get_nowait()
                self._telemetry_queue.put_nowait(telemetry)
            except queue.Empty:
                pass
                
    def get_latest_telemetry(self) -> Optional[TelemetryData]:
        """Get the most recent telemetry data"""
        return self._latest_telemetry
        
    def add_callback(self, callback: Callable[[TelemetryData], None]):
        """Add a callback for telemetry updates"""
        self._telemetry_callbacks.append(callback)
        
    def _process_telemetry(self):
        """Process telemetry data in background thread"""
        while self._running:
            try:
                telemetry = self._telemetry_queue.get(timeout=0.1)
                self._latest_telemetry = telemetry
                
                # Call all registered callbacks
                for callback in self._telemetry_callbacks:
                    try:
                        callback(telemetry)
                    except Exception as e:
                        logger.error(f"Error in telemetry callback: {e}")
                        
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing telemetry: {e}")


class IngestSource(ABC):
    """Abstract base class for video ingest sources"""
    
    def __init__(self, source_id: str, config: Dict[str, Any] = None, telemetry_manager: Optional[TelemetryManager] = None):
        self.source_id = source_id
        self.config = config or {}
        self.status = IngestStatus.STOPPED
        self.frame_count = 0
        self.start_time = None
        self.last_frame_time = None
        self.fps_counter = 0
        self.fps_start_time = None
        self.current_fps = 0.0
        self._lock = threading.Lock()
        self._telemetry_manager = telemetry_manager
        
    @abstractmethod
    async def start(self) -> bool:
        """Start the ingest source
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        pass
    
    @abstractmethod
    async def read_frame(self) -> Optional[Tuple[np.ndarray, FrameMetadata]]:
        """Read a frame from the source
        
        Returns:
            Tuple of (frame, metadata) or None if no frame available
        """
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """Stop the ingest source
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        pass
    
    def _update_fps(self):
        """Update FPS calculation"""
        with self._lock:
            current_time = time.time()
            
            if self.fps_start_time is None:
                self.fps_start_time = current_time
                self.fps_counter = 0
            
            self.fps_counter += 1
            elapsed = current_time - self.fps_start_time
            
            if elapsed >= 1.0:  # Update FPS every second
                self.current_fps = self.fps_counter / elapsed
                self.fps_counter = 0
                self.fps_start_time = current_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ingest statistics"""
        with self._lock:
            uptime = time.time() - self.start_time if self.start_time else 0
            stats = {
                "source_id": self.source_id,
                "status": self.status.value,
                "frame_count": self.frame_count,
                "current_fps": round(self.current_fps, 2),
                "uptime_seconds": round(uptime, 2),
                "last_frame_time": self.last_frame_time,
                "reconnect_attempts": getattr(self, 'reconnect_attempts', 0)
            }
            
            # Add telemetry info if available
            if self._telemetry_manager:
                latest_telemetry = self._telemetry_manager.get_latest_telemetry()
                if latest_telemetry:
                    stats["latest_telemetry"] = latest_telemetry.to_dict()
                    
            return stats
            
    def _create_frame_metadata(self, frame: np.ndarray) -> FrameMetadata:
        """Create frame metadata with telemetry if available"""
        current_time = time.time()
        self.frame_count += 1
        self.last_frame_time = current_time
        self._update_fps()
        
        # Get latest telemetry
        telemetry = None
        if self._telemetry_manager:
            telemetry = self._telemetry_manager.get_latest_telemetry()
            
        return FrameMetadata(
            timestamp=current_time,
            frame_id=self.frame_count,
            source_id=self.source_id,
            width=frame.shape[1],
            height=frame.shape[0],
            fps=self.current_fps,
            telemetry=telemetry
        )
        
    def _should_reconnect(self) -> bool:
        """Check if we should attempt reconnection"""
        return getattr(self, 'reconnect_attempts', 0) < getattr(self, 'max_reconnect_attempts', 5)
        
    async def _attempt_reconnect(self) -> bool:
        """Attempt to reconnect to the source"""
        if not self._should_reconnect():
            return False
            
        if not hasattr(self, 'reconnect_attempts'):
            self.reconnect_attempts = 0
        if not hasattr(self, 'max_reconnect_attempts'):
            self.max_reconnect_attempts = 5
        if not hasattr(self, 'reconnect_delay'):
            self.reconnect_delay = 2.0
            
        self.reconnect_attempts += 1
        logger.info(f"Attempting reconnection {self.reconnect_attempts}/{self.max_reconnect_attempts} for {self.source_id}")
        
        await asyncio.sleep(self.reconnect_delay)
        
        try:
            await self.stop()
            return await self.start()
        except Exception as e:
            logger.error(f"Reconnection attempt failed for {self.source_id}: {e}")
            return False


class RTSPIngest(IngestSource):
    """RTSP stream ingest for DJI O4 and other RTSP sources"""
    
    def __init__(self, source_id: str, rtsp_url: str, config: Dict[str, Any] = None):
        super().__init__(source_id, config)
        self.rtsp_url = rtsp_url
        self.cap = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = self.config.get("max_reconnect_attempts", 5)
        
    async def start(self) -> bool:
        """Start RTSP capture"""
        try:
            self.status = IngestStatus.STARTING
            logger.info(f"Starting RTSP ingest from {self.rtsp_url}")
            
            # OpenCV VideoCapture for RTSP
            self.cap = cv2.VideoCapture(self.rtsp_url)
            
            # Set buffer size to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open RTSP stream: {self.rtsp_url}")
                self.status = IngestStatus.ERROR
                return False
            
            self.status = IngestStatus.RUNNING
            self.start_time = time.time()
            logger.info(f"RTSP ingest started successfully: {self.source_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting RTSP ingest: {e}")
            self.status = IngestStatus.ERROR
            return False
    
    async def read_frame(self) -> Optional[Tuple[np.ndarray, FrameMetadata]]:
        """Read frame from RTSP stream"""
        if self.status != IngestStatus.RUNNING or not self.cap:
            return None
        
        try:
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                logger.warning(f"Failed to read frame from RTSP: {self.source_id}")
                # Attempt reconnection
                if self.reconnect_attempts < self.max_reconnect_attempts:
                    self.reconnect_attempts += 1
                    logger.info(f"Attempting reconnection {self.reconnect_attempts}/{self.max_reconnect_attempts}")
                    await self.stop()
                    await asyncio.sleep(1)
                    await self.start()
                return None
            
            # Reset reconnect attempts on successful frame
            self.reconnect_attempts = 0
            
            # Update frame statistics
            self.frame_count += 1
            self.last_frame_time = time.time()
            self._update_fps()
            
            # Create metadata
            height, width = frame.shape[:2]
            metadata = FrameMetadata(
                timestamp=self.last_frame_time,
                frame_id=self.frame_count,
                source_id=self.source_id,
                width=width,
                height=height,
                fps=self.current_fps,
                format="BGR"
            )
            
            return frame, metadata
            
        except Exception as e:
            logger.error(f"Error reading RTSP frame: {e}")
            self.status = IngestStatus.ERROR
            return None
    
    async def stop(self) -> bool:
        """Stop RTSP capture"""
        try:
            if self.cap:
                self.cap.release()
                self.cap = None
            
            self.status = IngestStatus.STOPPED
            logger.info(f"RTSP ingest stopped: {self.source_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping RTSP ingest: {e}")
            return False


class MJPEGIngest(IngestSource):
    """MJPEG stream ingest for scrcpy and phone webservers"""
    
    def __init__(self, source_id: str, mjpeg_url: str, config: Dict[str, Any] = None):
        super().__init__(source_id, config)
        self.mjpeg_url = mjpeg_url
        self.cap = None
        
    async def start(self) -> bool:
        """Start MJPEG capture"""
        try:
            self.status = IngestStatus.STARTING
            logger.info(f"Starting MJPEG ingest from {self.mjpeg_url}")
            
            # OpenCV VideoCapture for MJPEG
            self.cap = cv2.VideoCapture(self.mjpeg_url)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open MJPEG stream: {self.mjpeg_url}")
                self.status = IngestStatus.ERROR
                return False
            
            self.status = IngestStatus.RUNNING
            self.start_time = time.time()
            logger.info(f"MJPEG ingest started successfully: {self.source_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting MJPEG ingest: {e}")
            self.status = IngestStatus.ERROR
            return False
    
    async def read_frame(self) -> Optional[Tuple[np.ndarray, FrameMetadata]]:
        """Read frame from MJPEG stream"""
        if self.status != IngestStatus.RUNNING or not self.cap:
            return None
        
        try:
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                logger.warning(f"Failed to read frame from MJPEG: {self.source_id}")
                return None
            
            # Update frame statistics
            self.frame_count += 1
            self.last_frame_time = time.time()
            self._update_fps()
            
            # Create metadata
            height, width = frame.shape[:2]
            metadata = FrameMetadata(
                timestamp=self.last_frame_time,
                frame_id=self.frame_count,
                source_id=self.source_id,
                width=width,
                height=height,
                fps=self.current_fps,
                format="BGR"
            )
            
            return frame, metadata
            
        except Exception as e:
            logger.error(f"Error reading MJPEG frame: {e}")
            self.status = IngestStatus.ERROR
            return None
    
    async def stop(self) -> bool:
        """Stop MJPEG capture"""
        try:
            if self.cap:
                self.cap.release()
                self.cap = None
            
            self.status = IngestStatus.STOPPED
            logger.info(f"MJPEG ingest stopped: {self.source_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping MJPEG ingest: {e}")
            return False


class DJITelloIngest(IngestSource):
    """DJI Tello drone video ingest with telemetry"""
    
    def __init__(self, source_id: str, config: Dict[str, Any] = None, telemetry_manager: Optional[TelemetryManager] = None):
        super().__init__(source_id, config, telemetry_manager)
        self.tello: Optional[Tello] = None
        self.telemetry_thread: Optional[threading.Thread] = None
        self.telemetry_running = False
        
        if not DJI_AVAILABLE:
            raise ImportError("DJI Tello library not available. Install with: pip install djitellopy")
            
    async def start(self) -> bool:
        try:
            self.status = IngestStatus.STARTING
            self.tello = Tello()
            self.tello.connect()
            
            # Start video stream
            self.tello.streamon()
            
            # Start telemetry collection
            if self._telemetry_manager:
                self.telemetry_running = True
                self.telemetry_thread = threading.Thread(target=self._collect_telemetry, daemon=True)
                self.telemetry_thread.start()
            
            self.status = IngestStatus.RUNNING
            self.start_time = time.time()
            logger.info(f"DJI Tello ingest started for {self.source_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start DJI Tello ingest for {self.source_id}: {e}")
            self.status = IngestStatus.ERROR
            return False
            
    async def read_frame(self) -> Optional[Tuple[np.ndarray, FrameMetadata]]:
        if self.status != IngestStatus.RUNNING or not self.tello:
            return None
            
        try:
            frame = self.tello.get_frame_read().frame
            if frame is not None:
                metadata = self._create_frame_metadata(frame)
                return frame, metadata
            else:
                # Try to reconnect if frame is None
                if self._should_reconnect():
                    if await self._attempt_reconnect():
                        return await self.read_frame()
                return None
                
        except Exception as e:
            logger.error(f"Error reading frame from DJI Tello {self.source_id}: {e}")
            if self._should_reconnect():
                if await self._attempt_reconnect():
                    return await self.read_frame()
            return None
            
    async def stop(self) -> bool:
        self.status = IngestStatus.STOPPED
        self.telemetry_running = False
        
        if self.telemetry_thread:
            self.telemetry_thread.join(timeout=1.0)
            
        if self.tello:
            try:
                self.tello.streamoff()
                self.tello.end()
            except Exception as e:
                logger.error(f"Error stopping DJI Tello {self.source_id}: {e}")
                
        logger.info(f"DJI Tello ingest stopped for {self.source_id}")
        return True
        
    def _collect_telemetry(self):
        """Collect telemetry data from Tello drone"""
        while self.telemetry_running and self.tello:
            try:
                telemetry = TelemetryData(
                    timestamp=time.time(),
                    battery_level=self.tello.get_battery(),
                    temperature=self.tello.get_temperature(),
                    altitude=self.tello.get_height(),
                    speed=self.tello.get_speed_x(),  # Simplified - could combine x,y,z
                    pitch=self.tello.get_pitch(),
                    roll=self.tello.get_roll(),
                    yaw=self.tello.get_yaw(),
                    flight_mode="auto",  # Tello doesn't provide detailed flight modes
                    is_flying=True,  # Assume flying if connected
                    barometer=self.tello.get_barometer()
                )
                
                if self._telemetry_manager:
                    self._telemetry_manager.add_telemetry(telemetry)
                    
                time.sleep(0.1)  # 10Hz telemetry rate
                
            except Exception as e:
                logger.error(f"Error collecting telemetry from Tello: {e}")
                time.sleep(1.0)


class USBCaptureIngest(IngestSource):
    """USB video capture device ingest (for FPV cameras)"""
    
    def __init__(self, source_id: str, config: Dict[str, Any] = None, telemetry_manager: Optional[TelemetryManager] = None):
        super().__init__(source_id, config, telemetry_manager)
        self.device_index = config.get('device_index', 0) if config else 0
        self.cap: Optional[cv2.VideoCapture] = None
        
    async def start(self) -> bool:
        try:
            self.status = IngestStatus.STARTING
            self.cap = cv2.VideoCapture(self.device_index)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open USB capture device {self.device_index}")
                self.status = IngestStatus.ERROR
                return False
                
            # Set capture properties if specified in config
            if self.config:
                if 'width' in self.config:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['width'])
                if 'height' in self.config:
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['height'])
                if 'fps' in self.config:
                    self.cap.set(cv2.CAP_PROP_FPS, self.config['fps'])
                    
            self.status = IngestStatus.RUNNING
            self.start_time = time.time()
            logger.info(f"USB capture ingest started for device {self.device_index}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start USB capture ingest: {e}")
            self.status = IngestStatus.ERROR
            return False
            
    async def read_frame(self) -> Optional[Tuple[np.ndarray, FrameMetadata]]:
        if self.status != IngestStatus.RUNNING or not self.cap:
            return None
            
        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                metadata = self._create_frame_metadata(frame)
                return frame, metadata
            else:
                # Try to reconnect
                if self._should_reconnect():
                    if await self._attempt_reconnect():
                        return await self.read_frame()
                return None
                
        except Exception as e:
            logger.error(f"Error reading frame from USB capture: {e}")
            if self._should_reconnect():
                if await self._attempt_reconnect():
                    return await self.read_frame()
            return None
            
    async def stop(self) -> bool:
        self.status = IngestStatus.STOPPED
        if self.cap:
            self.cap.release()
        logger.info(f"USB capture ingest stopped for device {self.device_index}")
        return True


class FileIngest(IngestSource):
    """Video file ingest for testing with recorded videos"""
    
    def __init__(self, source_id: str, config: Dict[str, Any] = None, telemetry_manager: Optional[TelemetryManager] = None):
        super().__init__(source_id, config, telemetry_manager)
        self.file_path = config.get('file_path') if config else None
        self.loop = config.get('loop', False) if config else False
        self.cap: Optional[cv2.VideoCapture] = None
        self.simulated_telemetry = config.get('simulated_telemetry', False) if config else False
        
        if not self.file_path:
            raise ValueError("file_path must be specified in config")
            
    async def start(self) -> bool:
        try:
            self.status = IngestStatus.STARTING
            self.cap = cv2.VideoCapture(self.file_path)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open video file {self.file_path}")
                self.status = IngestStatus.ERROR
                return False
                
            self.status = IngestStatus.RUNNING
            self.start_time = time.time()
            logger.info(f"File ingest started for {self.file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start file ingest: {e}")
            self.status = IngestStatus.ERROR
            return False
            
    async def read_frame(self) -> Optional[Tuple[np.ndarray, FrameMetadata]]:
        if self.status != IngestStatus.RUNNING or not self.cap:
            return None
            
        try:
            ret, frame = self.cap.read()
            
            if not ret:
                if self.loop:
                    # Restart from beginning
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    
                if not ret:
                    return None
                    
            if frame is not None:
                # Generate simulated telemetry if enabled
                if self.simulated_telemetry and self._telemetry_manager:
                    self._generate_simulated_telemetry()
                    
                metadata = self._create_frame_metadata(frame)
                return frame, metadata
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error reading frame from file: {e}")
            return None
            
    async def stop(self) -> bool:
        self.status = IngestStatus.STOPPED
        if self.cap:
            self.cap.release()
        logger.info(f"File ingest stopped for {self.file_path}")
        return True
        
    def _generate_simulated_telemetry(self):
        """Generate simulated telemetry data for testing"""
        import random
        import math
        
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Simulate a circular flight pattern
        radius = 100  # meters
        angular_velocity = 0.1  # rad/s
        angle = angular_velocity * elapsed
        
        telemetry = TelemetryData(
            timestamp=current_time,
            latitude=37.7749 + (radius * math.cos(angle)) / 111320,  # San Francisco base
            longitude=-122.4194 + (radius * math.sin(angle)) / (111320 * math.cos(math.radians(37.7749))),
            altitude=100 + 20 * math.sin(elapsed * 0.2),  # Varying altitude
            relative_altitude=50 + 20 * math.sin(elapsed * 0.2),
            heading=math.degrees(angle) % 360,
            pitch=random.uniform(-5, 5),
            roll=random.uniform(-10, 10),
            yaw=math.degrees(angle) % 360,
            speed=random.uniform(5, 15),
            battery_level=max(10, 100 - int(elapsed / 60)),  # Decrease over time
            signal_strength=random.randint(70, 100),
            flight_mode="auto",
            is_flying=True,
            temperature=random.uniform(20, 35),
            barometer=random.uniform(1010, 1020)
        )
        
        self._telemetry_manager.add_telemetry(telemetry)


class WebRTCIngest(IngestSource):
    """WebRTC ingest for OvenMediaEngine and Android WebRTC"""
    
    def __init__(self, source_id: str, webrtc_url: str, config: Dict[str, Any] = None):
        super().__init__(source_id, config)
        self.webrtc_url = webrtc_url
        self.connection = None
        
    async def start(self) -> bool:
        """Start WebRTC connection"""
        try:
            self.status = IngestStatus.STARTING
            logger.info(f"Starting WebRTC ingest from {self.webrtc_url}")
            
            # Note: This is a placeholder implementation
            # Real WebRTC implementation would require aiortc or similar library
            logger.warning("WebRTC ingest is not fully implemented yet")
            logger.info("Falling back to RTSP/MJPEG for now")
            
            self.status = IngestStatus.ERROR
            return False
            
        except Exception as e:
            logger.error(f"Error starting WebRTC ingest: {e}")
            self.status = IngestStatus.ERROR
            return False
    
    async def read_frame(self) -> Optional[Tuple[np.ndarray, FrameMetadata]]:
        """Read frame from WebRTC connection"""
        # Placeholder implementation
        return None
    
    async def stop(self) -> bool:
        """Stop WebRTC connection"""
        try:
            self.status = IngestStatus.STOPPED
            logger.info(f"WebRTC ingest stopped: {self.source_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping WebRTC ingest: {e}")
            return False


class IngestManager:
    """Manager for multiple ingest sources with telemetry integration"""
    
    def __init__(self):
        self.sources: Dict[str, IngestSource] = {}
        self.telemetry_manager = TelemetryManager()
        self.app = FastAPI(title="Foresight Ingest API")
        self._setup_routes()
        self._frame_callbacks: List[Callable[[str, np.ndarray, FrameMetadata], None]] = []
        
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            total_sources = len(self.sources)
            running_sources = sum(1 for s in self.sources.values() if s.status == IngestStatus.RUNNING)
            
            return JSONResponse({
                "status": "healthy" if running_sources > 0 else "no_sources",
                "total_sources": total_sources,
                "running_sources": running_sources,
                "telemetry_active": self.telemetry_manager._running,
                "timestamp": time.time()
            })
        
        @self.app.get("/sources")
        async def list_sources():
            """List all ingest sources and their stats"""
            return {
                "sources": [source.get_stats() for source in self.sources.values()]
            }
        
        @self.app.get("/sources/{source_id}")
        async def get_source(source_id: str):
            """Get specific source stats"""
            if source_id not in self.sources:
                raise HTTPException(status_code=404, detail="Source not found")
            
            return self.sources[source_id].get_stats()
            
        @self.app.get("/telemetry")
        async def get_latest_telemetry():
            """Get latest telemetry data"""
            telemetry = self.telemetry_manager.get_latest_telemetry()
            if telemetry:
                return telemetry.to_dict()
            return {"message": "No telemetry data available"}
            
        @self.app.websocket("/ws/frames/{source_id}")
        async def websocket_frames(websocket: WebSocket, source_id: str):
            """WebSocket endpoint for streaming frames"""
            await websocket.accept()
            
            if source_id not in self.sources:
                await websocket.send_json({"error": "Source not found"})
                await websocket.close()
                return
                
            source = self.sources[source_id]
            
            try:
                while True:
                    result = await source.read_frame()
                    if result:
                        frame, metadata = result
                        
                        # Encode frame as JPEG for transmission
                        _, buffer = cv2.imencode('.jpg', frame)
                        frame_data = buffer.tobytes()
                        
                        await websocket.send_json({
                            "frame_data": frame_data.hex(),
                            "metadata": {
                                "timestamp": metadata.timestamp,
                                "frame_id": metadata.frame_id,
                                "width": metadata.width,
                                "height": metadata.height,
                                "fps": metadata.fps,
                                "telemetry": metadata.telemetry.to_dict() if metadata.telemetry else None
                            }
                        })
                    else:
                        await asyncio.sleep(0.033)  # ~30 FPS
                        
            except Exception as e:
                logger.error(f"WebSocket error for {source_id}: {e}")
            finally:
                await websocket.close()
    
    def add_source(self, source: IngestSource):
        """Add an ingest source"""
        self.sources[source.source_id] = source
        logger.info(f"Added ingest source: {source.source_id}")
    
    def remove_source(self, source_id: str):
        """Remove an ingest source"""
        if source_id in self.sources:
            del self.sources[source_id]
            logger.info(f"Removed ingest source: {source_id}")
    
    async def start_all(self):
        """Start all ingest sources and telemetry manager"""
        self.telemetry_manager.start()
        for source in self.sources.values():
            await source.start()
    
    async def stop_all(self):
        """Stop all ingest sources and telemetry manager"""
        for source in self.sources.values():
            await source.stop()
        self.telemetry_manager.stop()
    
    def get_total_fps(self) -> float:
        """Get total FPS across all sources"""
        return sum(source.current_fps for source in self.sources.values())
        
    def add_frame_callback(self, callback: Callable[[str, np.ndarray, FrameMetadata], None]):
        """Add a callback for frame processing"""
        self._frame_callbacks.append(callback)
        
    def add_telemetry_callback(self, callback: Callable[[TelemetryData], None]):
        """Add a callback for telemetry updates"""
        self.telemetry_manager.add_callback(callback)
        
    async def process_frames(self):
        """Process frames from all sources and call callbacks"""
        while True:
            for source_id, source in self.sources.items():
                if source.status == IngestStatus.RUNNING:
                    result = await source.read_frame()
                    if result:
                        frame, metadata = result
                        
                        # Call all frame callbacks
                        for callback in self._frame_callbacks:
                            try:
                                callback(source_id, frame, metadata)
                            except Exception as e:
                                logger.error(f"Error in frame callback: {e}")
                                
            await asyncio.sleep(0.001)  # Small delay to prevent busy waiting


# Global ingest manager instance
ingest_manager = IngestManager()


async def demo_ingest():
    """Demo function to test video ingest with telemetry"""
    logger.info("Starting enhanced video ingest demo...")
    
    # Add file source with simulated telemetry for testing
    file_config = {
        "file_path": "test_video.mp4",  # You can replace with actual video file
        "loop": True,
        "simulated_telemetry": True
    }
    
    try:
        # Try to create a file source
        file_source = FileIngest("test_video", file_config, ingest_manager.telemetry_manager)
        ingest_manager.add_source(file_source)
        logger.info("Added file ingest source with simulated telemetry")
    except Exception as e:
        logger.warning(f"Could not add file source: {e}")
    
    # Add USB capture source (for FPV cameras)
    usb_config = {
        "device_index": 0,
        "width": 640,
        "height": 480,
        "fps": 30
    }
    
    try:
        usb_source = USBCaptureIngest("fpv_camera", usb_config, ingest_manager.telemetry_manager)
        ingest_manager.add_source(usb_source)
        logger.info("Added USB capture source")
    except Exception as e:
        logger.warning(f"Could not add USB source: {e}")
    
    # Add DJI Tello source if available
    if DJI_AVAILABLE:
        try:
            tello_source = DJITelloIngest("tello_drone", {}, ingest_manager.telemetry_manager)
            ingest_manager.add_source(tello_source)
            logger.info("Added DJI Tello source")
        except Exception as e:
            logger.warning(f"Could not add Tello source: {e}")
    
    # Add RTSP source (DJI O4 simulation)
    rtsp_source = RTSPIngest("dji_o4", "rtsp://192.168.1.100:8554/live")
    ingest_manager.add_source(rtsp_source)
    logger.info("Added RTSP source (DJI O4 simulation)")
    
    # Add frame processing callback
    def frame_callback(source_id: str, frame: np.ndarray, metadata: FrameMetadata):
        """Process incoming frames"""
        if metadata.telemetry:
            logger.debug(f"Frame from {source_id}: {frame.shape}, Telemetry: Lat={metadata.telemetry.latitude}, Alt={metadata.telemetry.altitude}")
        else:
            logger.debug(f"Frame from {source_id}: {frame.shape}, No telemetry")
    
    # Add telemetry callback
    def telemetry_callback(telemetry: TelemetryData):
        """Process telemetry updates"""
        logger.info(f"Telemetry update: Lat={telemetry.latitude}, Lon={telemetry.longitude}, Alt={telemetry.altitude}, Battery={telemetry.battery_level}%")
    
    ingest_manager.add_frame_callback(frame_callback)
    ingest_manager.add_telemetry_callback(telemetry_callback)
    
    try:
        await ingest_manager.start_all()
        logger.info("All sources started, beginning frame processing...")
        
        # Start frame processing task
        frame_task = asyncio.create_task(ingest_manager.process_frames())
        
        # Run for 60 seconds
        for i in range(600):
            await asyncio.sleep(0.1)
            
            # Print stats every 10 seconds
            if i % 100 == 0:
                logger.info(f"=== Status Update (t={i/10:.1f}s) ===")
                logger.info(f"Total FPS: {ingest_manager.get_total_fps():.2f}")
                
                for source_id, source in ingest_manager.sources.items():
                    stats = source.get_stats()
                    logger.info(f"{source_id}: {stats['status']} - {stats['current_fps']:.2f} FPS")
                    
                # Show latest telemetry
                latest_telemetry = ingest_manager.telemetry_manager.get_latest_telemetry()
                if latest_telemetry:
                    logger.info(f"Latest telemetry: {latest_telemetry.latitude}, {latest_telemetry.longitude}, {latest_telemetry.altitude}m")
                    
        frame_task.cancel()
                    
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    finally:
        await ingest_manager.stop_all()
        logger.info("Enhanced demo completed")


async def create_test_video():
    """Create a simple test video for demo purposes"""
    try:
        import cv2
        import numpy as np
        
        # Create a simple test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('test_video.mp4', fourcc, 20.0, (640, 480))
        
        for i in range(200):  # 10 seconds at 20 FPS
            # Create a frame with moving circle
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            center_x = int(320 + 200 * np.sin(i * 0.1))
            center_y = int(240 + 100 * np.cos(i * 0.1))
            cv2.circle(frame, (center_x, center_y), 50, (0, 255, 0), -1)
            cv2.putText(frame, f'Frame {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out.write(frame)
            
        out.release()
        logger.info("Created test_video.mp4 for demo")
        return True
    except Exception as e:
        logger.warning(f"Could not create test video: {e}")
        return False


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Foresight Ingest Module")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    parser.add_argument("--port", type=int, default=8001, help="API server port")
    parser.add_argument("--host", default="0.0.0.0", help="API server host")
    parser.add_argument("--create-test-video", action="store_true", help="Create test video for demo")
    
    args = parser.parse_args()
    
    if args.create_test_video:
        asyncio.run(create_test_video())
    elif args.demo:
        # Create test video if it doesn't exist
        if not Path('test_video.mp4').exists():
            asyncio.run(create_test_video())
        asyncio.run(demo_ingest())
    else:
        # Run API server
        logger.info(f"Starting Foresight Ingest API on {args.host}:{args.port}")
        logger.info("Available endpoints:")
        logger.info("  GET /health - Health check")
        logger.info("  GET /sources - List all sources")
        logger.info("  GET /sources/{source_id} - Get source stats")
        logger.info("  GET /telemetry - Get latest telemetry")
        logger.info("  WS /ws/frames/{source_id} - Stream frames via WebSocket")
        uvicorn.run(ingest_manager.app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()