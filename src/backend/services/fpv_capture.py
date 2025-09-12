#!/usr/bin/env python3
"""
FPV Video Capture Service

Supports multiple FPV video sources:
- DJI O4 Lite/Goggles 3 via HDMI capture cards
- USB UVC capture devices
- Analog FPV via composite/HDMI converters
- DJI SDK streams (if available)
- Standard webcams and IP cameras

Author: Foresight AI Team
Date: 2024
"""

import cv2
import numpy as np
import threading
import time
import logging
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class CaptureSourceType(Enum):
    """Types of video capture sources"""
    WEBCAM = "webcam"
    UVC_CAPTURE_CARD = "uvc_capture"
    DJI_HDMI = "dji_hdmi"
    ANALOG_FPV = "analog_fpv"
    DJI_SDK = "dji_sdk"
    IP_CAMERA = "ip_camera"
    RTSP_STREAM = "rtsp"
    UDP_STREAM = "udp"
    FILE = "file"

@dataclass
class CaptureConfig:
    """Configuration for video capture"""
    source_type: CaptureSourceType
    source_id: str  # Device index, URL, or path
    width: int = 1920
    height: int = 1080
    fps: int = 30
    buffer_size: int = 1
    auto_exposure: bool = True
    brightness: float = 0.5
    contrast: float = 0.5
    saturation: float = 0.5
    
class FPVCaptureService:
    """
    FPV Video Capture Service
    
    Handles multiple video sources with automatic detection and optimization
    for real-time FPV feeds from drones and capture cards.
    """
    
    def __init__(self, config: Optional[CaptureConfig] = None):
        self.config = config or self._get_default_config()
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.frame_lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.actual_fps = 0.0
        self.frame_callbacks: List[Callable[[np.ndarray], None]] = []
        
        # Performance metrics
        self.stats = {
            'frames_captured': 0,
            'frames_dropped': 0,
            'avg_fps': 0.0,
            'capture_latency_ms': 0.0,
            'source_connected': False
        }
        
    def _get_default_config(self) -> CaptureConfig:
        """Get default capture configuration"""
        # Try to auto-detect best available source
        source_type, source_id = self._auto_detect_source()
        return CaptureConfig(
            source_type=source_type,
            source_id=source_id,
            width=1920,
            height=1080,
            fps=30
        )
    
    def _auto_detect_source(self) -> Tuple[CaptureSourceType, str]:
        """Auto-detect the best available video source"""
        logger.info("Auto-detecting video sources...")
        
        # Check for UVC capture devices (capture cards)
        for i in range(10):  # Check first 10 device indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Get device name if possible
                backend = cap.getBackendName()
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                
                cap.release()
                
                # Heuristic: if resolution is 1080p or higher, likely a capture card
                if width >= 1920 and height >= 1080:
                    logger.info(f"Found capture card at index {i} ({width}x{height})")
                    return CaptureSourceType.UVC_CAPTURE_CARD, str(i)
                elif width > 0:  # Valid webcam
                    logger.info(f"Found webcam at index {i} ({width}x{height})")
                    if i == 0:  # Prefer index 0 for webcam
                        return CaptureSourceType.WEBCAM, str(i)
        
        # Check environment variables for custom sources
        if 'FPV_SOURCE' in os.environ:
            source = os.environ['FPV_SOURCE']
            if source.startswith('rtsp://'):
                return CaptureSourceType.RTSP_STREAM, source
            elif source.startswith('udp://'):
                return CaptureSourceType.UDP_STREAM, source
            elif source.startswith('http://'):
                return CaptureSourceType.IP_CAMERA, source
            elif source.isdigit():
                return CaptureSourceType.WEBCAM, source
        
        # Default fallback
        logger.warning("No video sources detected, using default webcam")
        return CaptureSourceType.WEBCAM, "0"
    
    def start(self) -> bool:
        """Start video capture"""
        if self.is_running:
            logger.warning("Capture already running")
            return True
            
        logger.info(f"Starting FPV capture: {self.config.source_type.value} - {self.config.source_id}")
        
        # Initialize capture device
        if not self._initialize_capture():
            return False
            
        # Start capture thread
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        logger.info("FPV capture started successfully")
        return True
    
    def stop(self):
        """Stop video capture"""
        if not self.is_running:
            return
            
        logger.info("Stopping FPV capture...")
        self.is_running = False
        
        if self.thread:
            self.thread.join(timeout=2.0)
            
        if self.cap:
            self.cap.release()
            self.cap = None
            
        logger.info("FPV capture stopped")
    
    def _initialize_capture(self) -> bool:
        """Initialize the video capture device"""
        try:
            source_id = self.config.source_id
            
            # Handle different source types
            if self.config.source_type in [CaptureSourceType.WEBCAM, CaptureSourceType.UVC_CAPTURE_CARD]:
                # Numeric device index
                device_id = int(source_id) if source_id.isdigit() else 0
                self.cap = cv2.VideoCapture(device_id)
                
            elif self.config.source_type in [CaptureSourceType.RTSP_STREAM, CaptureSourceType.UDP_STREAM, CaptureSourceType.IP_CAMERA]:
                # Network streams
                self.cap = cv2.VideoCapture(source_id, cv2.CAP_FFMPEG)
                
            elif self.config.source_type == CaptureSourceType.FILE:
                # Video file
                self.cap = cv2.VideoCapture(source_id)
                
            else:
                logger.error(f"Unsupported source type: {self.config.source_type}")
                return False
            
            if not self.cap or not self.cap.isOpened():
                logger.error(f"Failed to open video source: {source_id}")
                return False
            
            # Configure capture properties
            self._configure_capture_properties()
            
            # Test frame capture
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.error("Failed to capture test frame")
                return False
                
            logger.info(f"Capture initialized: {frame.shape[1]}x{frame.shape[0]} @ {self.config.fps}fps")
            self.stats['source_connected'] = True
            return True
            
        except Exception as e:
            logger.error(f"Error initializing capture: {e}")
            return False
    
    def _configure_capture_properties(self):
        """Configure capture device properties for optimal FPV performance"""
        if not self.cap:
            return
            
        try:
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            
            # Set FPS
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            # Minimize buffer for low latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
            
            # Auto exposure for varying light conditions
            if self.config.auto_exposure:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            
            # Set image properties
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.config.brightness)
            self.cap.set(cv2.CAP_PROP_CONTRAST, self.config.contrast)
            self.cap.set(cv2.CAP_PROP_SATURATION, self.config.saturation)
            
            # For capture cards, try to set format to MJPG for better performance
            if self.config.source_type == CaptureSourceType.UVC_CAPTURE_CARD:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            
            # Log actual properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Capture configured: {actual_width}x{actual_height} @ {actual_fps}fps")
            
        except Exception as e:
            logger.warning(f"Error configuring capture properties: {e}")
    
    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        logger.info("Capture loop started")
        
        while self.is_running and self.cap:
            try:
                start_time = time.time()
                
                # Capture frame
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    logger.warning("Failed to capture frame")
                    self.stats['frames_dropped'] += 1
                    time.sleep(0.01)  # Brief pause before retry
                    continue
                
                # Update frame safely
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                    self.frame_count += 1
                    self.stats['frames_captured'] += 1
                
                # Calculate performance metrics
                capture_time = (time.time() - start_time) * 1000
                self.stats['capture_latency_ms'] = capture_time
                
                # Update FPS counter
                self.fps_counter += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.actual_fps = self.fps_counter / (current_time - self.last_fps_time)
                    self.stats['avg_fps'] = self.actual_fps
                    self.fps_counter = 0
                    self.last_fps_time = current_time
                
                # Call frame callbacks
                for callback in self.frame_callbacks:
                    try:
                        callback(frame)
                    except Exception as e:
                        logger.error(f"Error in frame callback: {e}")
                
                # Control frame rate
                target_delay = 1.0 / self.config.fps
                elapsed = time.time() - start_time
                if elapsed < target_delay:
                    time.sleep(target_delay - elapsed)
                    
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)
        
        logger.info("Capture loop ended")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest captured frame"""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def add_frame_callback(self, callback: Callable[[np.ndarray], None]):
        """Add a callback function to be called for each frame"""
        self.frame_callbacks.append(callback)
    
    def remove_frame_callback(self, callback: Callable[[np.ndarray], None]):
        """Remove a frame callback"""
        if callback in self.frame_callbacks:
            self.frame_callbacks.remove(callback)
    
    def get_stats(self) -> Dict:
        """Get capture statistics"""
        return self.stats.copy()
    
    def is_connected(self) -> bool:
        """Check if video source is connected"""
        return self.stats['source_connected'] and self.is_running
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get current capture resolution"""
        if self.cap:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return width, height
        return 0, 0
    
    def set_config(self, config: CaptureConfig):
        """Update capture configuration (requires restart)"""
        was_running = self.is_running
        if was_running:
            self.stop()
        
        self.config = config
        
        if was_running:
            self.start()

def create_fpv_capture(source_type: str = None, source_id: str = None) -> FPVCaptureService:
    """Factory function to create FPV capture service"""
    if source_type and source_id:
        config = CaptureConfig(
            source_type=CaptureSourceType(source_type),
            source_id=source_id
        )
        return FPVCaptureService(config)
    else:
        return FPVCaptureService()  # Auto-detect

def list_available_sources() -> List[Dict]:
    """List all available video sources"""
    sources = []
    
    # Check webcams and capture cards
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            source_type = "capture_card" if width >= 1920 else "webcam"
            
            sources.append({
                'id': str(i),
                'type': source_type,
                'name': f"Device {i}",
                'resolution': f"{width}x{height}",
                'fps': fps
            })
            cap.release()
    
    return sources

if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    print("Available video sources:")
    for source in list_available_sources():
        print(f"  {source}")
    
    # Create and start capture
    capture = create_fpv_capture()
    
    if capture.start():
        print("\nCapture started. Press Ctrl+C to stop.")
        try:
            while True:
                frame = capture.get_frame()
                if frame is not None:
                    cv2.imshow('FPV Feed', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                time.sleep(0.033)  # ~30fps display
        except KeyboardInterrupt:
            pass
        finally:
            capture.stop()
            cv2.destroyAllWindows()
    else:
        print("Failed to start capture")