#!/usr/bin/env python3
"""
Telemetry Service for Drone Data Collection

Collects and manages drone telemetry data including:
- GPS coordinates (latitude, longitude)
- Altitude above ground level (AGL)
- Orientation angles (yaw, pitch, roll)
- Gimbal tilt angles
- Timestamp synchronization

Supports multiple drone platforms and telemetry sources.
"""

import time
import json
import logging
import threading
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Callable, List
from datetime import datetime, timezone
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TelemetryData:
    """Complete telemetry data structure for a single timestamp."""
    
    # GPS coordinates
    latitude: float  # degrees
    longitude: float  # degrees
    altitude_msl: float  # meters above sea level
    altitude_agl: float  # meters above ground level
    
    # Drone orientation (Euler angles in degrees)
    yaw: float    # rotation around Z-axis (heading)
    pitch: float  # rotation around Y-axis (nose up/down)
    roll: float   # rotation around X-axis (bank left/right)
    
    # Gimbal orientation (degrees)
    gimbal_yaw: float    # gimbal yaw relative to drone
    gimbal_pitch: float  # gimbal pitch (tilt)
    gimbal_roll: float   # gimbal roll
    
    # Metadata
    timestamp: float  # Unix timestamp
    frame_id: Optional[int] = None  # Associated video frame ID
    quality: float = 1.0  # Data quality indicator (0-1)
    source: str = "unknown"  # Data source identifier
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TelemetryData':
        """Create from dictionary."""
        return cls(**data)
    
    def is_valid(self) -> bool:
        """Check if telemetry data is valid."""
        # Check GPS bounds
        if not (-90 <= self.latitude <= 90):
            return False
        if not (-180 <= self.longitude <= 180):
            return False
        
        # Check altitude reasonableness
        if not (-1000 <= self.altitude_msl <= 50000):  # -1km to 50km
            return False
        if not (0 <= self.altitude_agl <= 10000):  # 0 to 10km AGL
            return False
        
        # Check angle ranges
        if not (-180 <= self.yaw <= 180):
            return False
        if not (-90 <= self.pitch <= 90):
            return False
        if not (-180 <= self.roll <= 180):
            return False
        
        return True

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters for DJI O4 and other cameras."""
    
    # Principal point (optical center)
    cx: float  # x-coordinate of principal point (pixels)
    cy: float  # y-coordinate of principal point (pixels)
    
    # Focal lengths
    fx: float  # focal length in x direction (pixels)
    fy: float  # focal length in y direction (pixels)
    
    # Image dimensions
    width: int   # image width (pixels)
    height: int  # image height (pixels)
    
    # Distortion coefficients (optional)
    k1: float = 0.0  # radial distortion
    k2: float = 0.0  # radial distortion
    p1: float = 0.0  # tangential distortion
    p2: float = 0.0  # tangential distortion
    k3: float = 0.0  # radial distortion
    
    # Camera model metadata
    model: str = "pinhole"
    camera_name: str = "unknown"
    
    def get_camera_matrix(self) -> List[List[float]]:
        """Get 3x3 camera intrinsic matrix."""
        return [
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ]
    
    def get_distortion_coeffs(self) -> List[float]:
        """Get distortion coefficients array."""
        return [self.k1, self.k2, self.p1, self.p2, self.k3]
    
    @classmethod
    def dji_o4_default(cls) -> 'CameraIntrinsics':
        """Default intrinsics for DJI O4 camera (approximate values)."""
        return cls(
            cx=1920.0,  # Assuming 4K center
            cy=1080.0,
            fx=2800.0,  # Approximate focal length
            fy=2800.0,
            width=3840,
            height=2160,
            camera_name="DJI_O4",
            model="pinhole"
        )
    
    @classmethod
    def from_config(cls, config_path: str) -> 'CameraIntrinsics':
        """Load camera intrinsics from configuration file."""
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    data = json.load(f)
                else:
                    import yaml
                    data = yaml.safe_load(f)
            
            return cls(**data)
        except Exception as e:
            logger.warning(f"Failed to load camera config from {config_path}: {e}")
            return cls.dji_o4_default()

class TelemetryCollector:
    """Base class for telemetry data collection."""
    
    def __init__(self, source_name: str = "unknown"):
        self.source_name = source_name
        self.is_running = False
        self.callbacks: List[Callable[[TelemetryData], None]] = []
        self.latest_data: Optional[TelemetryData] = None
        self.data_lock = threading.Lock()
        
    def add_callback(self, callback: Callable[[TelemetryData], None]):
        """Add callback for new telemetry data."""
        self.callbacks.append(callback)
    
    def _notify_callbacks(self, data: TelemetryData):
        """Notify all callbacks of new data."""
        for callback in self.callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Telemetry callback error: {e}")
    
    def get_latest_data(self) -> Optional[TelemetryData]:
        """Get the most recent telemetry data."""
        with self.data_lock:
            return self.latest_data
    
    def start(self):
        """Start telemetry collection."""
        self.is_running = True
        logger.info(f"Started telemetry collector: {self.source_name}")
    
    def stop(self):
        """Stop telemetry collection."""
        self.is_running = False
        logger.info(f"Stopped telemetry collector: {self.source_name}")
    
    def collect_data(self) -> Optional[TelemetryData]:
        """Collect telemetry data (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement collect_data")

class DJITelemetryCollector(TelemetryCollector):
    """DJI drone telemetry collector using DJI SDK."""
    
    def __init__(self, connection_string: Optional[str] = None):
        super().__init__("DJI_SDK")
        self.connection_string = connection_string
        self.drone_connection = None
        
    def connect(self) -> bool:
        """Connect to DJI drone."""
        try:
            # This would integrate with actual DJI SDK
            # For now, we'll simulate the connection
            logger.info(f"Connecting to DJI drone: {self.connection_string}")
            self.drone_connection = True  # Simulated connection
            return True
        except Exception as e:
            logger.error(f"Failed to connect to DJI drone: {e}")
            return False
    
    def collect_data(self) -> Optional[TelemetryData]:
        """Collect telemetry from DJI drone."""
        if not self.drone_connection:
            return None
        
        try:
            # This would use actual DJI SDK calls
            # For now, we'll return simulated data
            current_time = time.time()
            
            # Simulated telemetry data
            data = TelemetryData(
                latitude=37.7749,  # San Francisco coordinates
                longitude=-122.4194,
                altitude_msl=100.0,
                altitude_agl=50.0,
                yaw=45.0,
                pitch=0.0,
                roll=0.0,
                gimbal_yaw=0.0,
                gimbal_pitch=-30.0,  # Looking down
                gimbal_roll=0.0,
                timestamp=current_time,
                quality=0.95,
                source=self.source_name
            )
            
            if data.is_valid():
                with self.data_lock:
                    self.latest_data = data
                self._notify_callbacks(data)
                return data
            else:
                logger.warning("Invalid telemetry data received")
                return None
                
        except Exception as e:
            logger.error(f"Error collecting DJI telemetry: {e}")
            return None

class MAVLinkTelemetryCollector(TelemetryCollector):
    """MAVLink telemetry collector for ArduPilot/PX4 drones."""
    
    def __init__(self, connection_string: str = "udp:127.0.0.1:14550"):
        super().__init__("MAVLink")
        self.connection_string = connection_string
        self.mavlink_connection = None
        
    def connect(self) -> bool:
        """Connect to MAVLink vehicle."""
        try:
            # This would use pymavlink
            logger.info(f"Connecting to MAVLink: {self.connection_string}")
            self.mavlink_connection = True  # Simulated connection
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MAVLink: {e}")
            return False
    
    def collect_data(self) -> Optional[TelemetryData]:
        """Collect telemetry from MAVLink vehicle."""
        if not self.mavlink_connection:
            return None
        
        try:
            # This would use actual MAVLink message parsing
            current_time = time.time()
            
            # Simulated telemetry data
            data = TelemetryData(
                latitude=37.7749,
                longitude=-122.4194,
                altitude_msl=120.0,
                altitude_agl=70.0,
                yaw=90.0,
                pitch=5.0,
                roll=-2.0,
                gimbal_yaw=0.0,
                gimbal_pitch=-45.0,
                gimbal_roll=0.0,
                timestamp=current_time,
                quality=0.90,
                source=self.source_name
            )
            
            if data.is_valid():
                with self.data_lock:
                    self.latest_data = data
                self._notify_callbacks(data)
                return data
            else:
                logger.warning("Invalid MAVLink telemetry data")
                return None
                
        except Exception as e:
            logger.error(f"Error collecting MAVLink telemetry: {e}")
            return None

class SimulatedTelemetryCollector(TelemetryCollector):
    """Simulated telemetry collector for testing."""
    
    def __init__(self, update_rate: float = 10.0):
        super().__init__("Simulated")
        self.update_rate = update_rate
        self.start_time = time.time()
        self.thread = None
        
    def start(self):
        """Start simulated telemetry generation."""
        super().start()
        self.thread = threading.Thread(target=self._generate_data, daemon=True)
        self.thread.start()
    
    def _generate_data(self):
        """Generate simulated telemetry data."""
        while self.is_running:
            try:
                data = self.collect_data()
                if data:
                    time.sleep(1.0 / self.update_rate)
            except Exception as e:
                logger.error(f"Error in simulated telemetry: {e}")
                time.sleep(1.0)
    
    def collect_data(self) -> Optional[TelemetryData]:
        """Generate simulated telemetry data."""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Simulate circular flight pattern
        radius = 100.0  # meters
        angular_velocity = 0.1  # rad/s
        angle = angular_velocity * elapsed
        
        # Base coordinates (San Francisco)
        base_lat = 37.7749
        base_lon = -122.4194
        
        # Convert meters to degrees (approximate)
        lat_offset = (radius * math.cos(angle)) / 111320.0
        lon_offset = (radius * math.sin(angle)) / (111320.0 * math.cos(math.radians(base_lat)))
        
        data = TelemetryData(
            latitude=base_lat + lat_offset,
            longitude=base_lon + lon_offset,
            altitude_msl=150.0 + 20.0 * math.sin(elapsed * 0.2),
            altitude_agl=100.0 + 20.0 * math.sin(elapsed * 0.2),
            yaw=math.degrees(angle) % 360,
            pitch=5.0 * math.sin(elapsed * 0.3),
            roll=3.0 * math.cos(elapsed * 0.4),
            gimbal_yaw=0.0,
            gimbal_pitch=-30.0 + 10.0 * math.sin(elapsed * 0.1),
            gimbal_roll=0.0,
            timestamp=current_time,
            quality=0.95 + 0.05 * math.sin(elapsed),
            source=self.source_name
        )
        
        if data.is_valid():
            with self.data_lock:
                self.latest_data = data
            self._notify_callbacks(data)
            return data
        
        return None

class TelemetryService:
    """Main telemetry service that manages multiple collectors."""
    
    def __init__(self):
        self.collectors: Dict[str, TelemetryCollector] = {}
        self.primary_collector: Optional[str] = None
        self.data_history: List[TelemetryData] = []
        self.max_history = 1000
        self.history_lock = threading.Lock()
        
    def add_collector(self, name: str, collector: TelemetryCollector, is_primary: bool = False):
        """Add a telemetry collector."""
        self.collectors[name] = collector
        collector.add_callback(self._on_telemetry_data)
        
        if is_primary or self.primary_collector is None:
            self.primary_collector = name
            
        logger.info(f"Added telemetry collector: {name}")
    
    def _on_telemetry_data(self, data: TelemetryData):
        """Handle new telemetry data."""
        with self.history_lock:
            self.data_history.append(data)
            if len(self.data_history) > self.max_history:
                self.data_history.pop(0)
    
    def get_latest_telemetry(self) -> Optional[TelemetryData]:
        """Get the latest telemetry from primary collector."""
        if self.primary_collector and self.primary_collector in self.collectors:
            return self.collectors[self.primary_collector].get_latest_data()
        return None
    
    def get_telemetry_at_time(self, timestamp: float, tolerance: float = 0.1) -> Optional[TelemetryData]:
        """Get telemetry data closest to specified timestamp."""
        with self.history_lock:
            if not self.data_history:
                return None
            
            # Find closest timestamp
            closest_data = None
            min_diff = float('inf')
            
            for data in self.data_history:
                diff = abs(data.timestamp - timestamp)
                if diff < min_diff and diff <= tolerance:
                    min_diff = diff
                    closest_data = data
            
            return closest_data
    
    def start_all(self):
        """Start all telemetry collectors."""
        for name, collector in self.collectors.items():
            try:
                collector.start()
                logger.info(f"Started collector: {name}")
            except Exception as e:
                logger.error(f"Failed to start collector {name}: {e}")
    
    def stop_all(self):
        """Stop all telemetry collectors."""
        for name, collector in self.collectors.items():
            try:
                collector.stop()
                logger.info(f"Stopped collector: {name}")
            except Exception as e:
                logger.error(f"Failed to stop collector {name}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all collectors."""
        status = {
            "primary_collector": self.primary_collector,
            "collectors": {},
            "data_history_count": len(self.data_history)
        }
        
        for name, collector in self.collectors.items():
            latest = collector.get_latest_data()
            status["collectors"][name] = {
                "running": collector.is_running,
                "source": collector.source_name,
                "latest_timestamp": latest.timestamp if latest else None,
                "data_quality": latest.quality if latest else 0.0
            }
        
        return status

# Global telemetry service instance
telemetry_service = TelemetryService()

def get_telemetry_service() -> TelemetryService:
    """Get the global telemetry service instance."""
    return telemetry_service

def initialize_telemetry(config: Optional[Dict[str, Any]] = None) -> TelemetryService:
    """Initialize telemetry service with configuration."""
    global telemetry_service
    
    if config is None:
        config = {}
    
    # Add simulated collector by default
    sim_collector = SimulatedTelemetryCollector(update_rate=config.get("sim_rate", 10.0))
    telemetry_service.add_collector("simulated", sim_collector, is_primary=True)
    
    # Add DJI collector if configured
    if config.get("enable_dji", False):
        dji_collector = DJITelemetryCollector(config.get("dji_connection"))
        if dji_collector.connect():
            telemetry_service.add_collector("dji", dji_collector, is_primary=True)
    
    # Add MAVLink collector if configured
    if config.get("enable_mavlink", False):
        mavlink_collector = MAVLinkTelemetryCollector(config.get("mavlink_connection", "udp:127.0.0.1:14550"))
        if mavlink_collector.connect():
            telemetry_service.add_collector("mavlink", mavlink_collector, is_primary=True)
    
    logger.info("Telemetry service initialized")
    return telemetry_service

if __name__ == "__main__":
    # Demo usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Telemetry Service Demo")
    parser.add_argument("--duration", type=int, default=30, help="Demo duration in seconds")
    parser.add_argument("--rate", type=float, default=5.0, help="Telemetry update rate")
    args = parser.parse_args()
    
    # Initialize with simulated data
    config = {
        "sim_rate": args.rate,
        "enable_dji": False,
        "enable_mavlink": False
    }
    
    service = initialize_telemetry(config)
    service.start_all()
    
    try:
        print(f"Running telemetry demo for {args.duration} seconds...")
        start_time = time.time()
        
        while time.time() - start_time < args.duration:
            latest = service.get_latest_telemetry()
            if latest:
                print(f"Telemetry: Lat={latest.latitude:.6f}, Lon={latest.longitude:.6f}, "
                      f"Alt={latest.altitude_agl:.1f}m, Yaw={latest.yaw:.1f}°, "
                      f"Gimbal={latest.gimbal_pitch:.1f}°")
            
            time.sleep(2.0)
    
    except KeyboardInterrupt:
        print("\nDemo interrupted")
    
    finally:
        service.stop_all()
        print("Telemetry service stopped")