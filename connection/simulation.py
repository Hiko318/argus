"""Simulation connection for testing and development.

Provides synthetic video frames and telemetry data for testing the SAR pipeline
without requiring actual hardware.
"""

import asyncio
import logging
import time
import math
from typing import AsyncGenerator, Optional, Tuple, Dict, Any, List

import cv2
import numpy as np
from .dji_o4 import FrameData, TelemetryData, DJIO4ConnectionBase


class SimulationConnection(DJIO4ConnectionBase):
    """Simulation connection for testing and development.
    
    Generates synthetic video frames with overlaid information and
    realistic telemetry data following predefined flight patterns.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.frame_width = self.config.get('frame_width', 1920)
        self.frame_height = self.config.get('frame_height', 1080)
        self.fps = self.config.get('fps', 30)
        self.flight_pattern = self.config.get('flight_pattern', 'circle')
        self.base_lat = self.config.get('base_latitude', 37.7749)
        self.base_lon = self.config.get('base_longitude', -122.4194)
        self.flight_altitude = self.config.get('flight_altitude', 100.0)
        self.flight_speed = self.config.get('flight_speed', 5.0)  # m/s
        self.add_synthetic_targets = self.config.get('add_synthetic_targets', True)
        
        # Flight pattern parameters
        self.pattern_radius = self.config.get('pattern_radius', 200.0)  # meters
        self.pattern_speed = self.config.get('pattern_speed', 0.02)  # rad/s
        
        # Synthetic target parameters
        self.target_positions = self._generate_target_positions()
        
    def _generate_target_positions(self) -> List[Tuple[int, int]]:
        """Generate positions for synthetic human targets in the frame."""
        positions = []
        if self.add_synthetic_targets:
            # Add some random target positions
            np.random.seed(42)  # Reproducible targets
            for _ in range(np.random.randint(2, 6)):
                x = np.random.randint(100, self.frame_width - 100)
                y = np.random.randint(100, self.frame_height - 100)
                positions.append((x, y))
        return positions
    
    async def connect(self) -> bool:
        """Connect to simulation (always succeeds)."""
        try:
            self.logger.info("Initializing simulation connection...")
            await asyncio.sleep(0.5)  # Simulate connection time
            
            self._connected = True
            self.logger.info(f"Simulation connected: {self.frame_width}x{self.frame_height}@{self.fps}fps")
            self.logger.info(f"Flight pattern: {self.flight_pattern}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize simulation: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from simulation."""
        if self._connected:
            self.logger.info("Disconnecting simulation...")
            self._connected = False
            self.logger.info("Simulation disconnected")
    
    async def stream_frames(self) -> AsyncGenerator[Tuple[FrameData, TelemetryData], None]:
        """Stream synthetic frames with telemetry."""
        if not self._connected:
            raise RuntimeError("Simulation not connected")
        
        self.logger.info("Starting simulation stream...")
        start_time = time.time()
        
        try:
            while self._connected:
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Generate frame
                frame = self._generate_frame(elapsed)
                
                # Generate telemetry
                telemetry = self._generate_telemetry(elapsed)
                
                frame_data = FrameData(
                    frame=frame,
                    timestamp=current_time,
                    pts=self._frame_count,
                    frame_number=self._frame_count,
                    width=self.frame_width,
                    height=self.frame_height
                )
                
                self._frame_count += 1
                self._last_telemetry = telemetry
                
                yield frame_data, telemetry
                
                # Maintain target FPS
                await asyncio.sleep(1.0 / self.fps)
                
        except Exception as e:
            self.logger.error(f"Error in simulation stream: {e}")
            raise
    
    def _generate_frame(self, elapsed_time: float) -> np.ndarray:
        """Generate synthetic video frame."""
        # Create base frame (aerial view simulation)
        frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        # Add ground texture (simple grid pattern)
        self._add_ground_texture(frame)
        
        # Add synthetic targets
        if self.add_synthetic_targets:
            self._add_synthetic_targets(frame, elapsed_time)
        
        # Add HUD overlay
        self._add_hud_overlay(frame, elapsed_time)
        
        # Add some noise for realism
        noise = np.random.normal(0, 5, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return frame
    
    def _add_ground_texture(self, frame: np.ndarray) -> None:
        """Add ground texture to simulate aerial view."""
        # Create a simple ground pattern
        h, w = frame.shape[:2]
        
        # Base ground color (brownish)
        frame[:, :] = [101, 67, 33]
        
        # Add grid lines to simulate fields/roads
        grid_size = 100
        for i in range(0, h, grid_size):
            cv2.line(frame, (0, i), (w, i), (120, 80, 40), 2)
        for j in range(0, w, grid_size):
            cv2.line(frame, (j, 0), (j, h), (120, 80, 40), 2)
        
        # Add some random vegetation patches
        np.random.seed(42)
        for _ in range(20):
            x = np.random.randint(0, w - 50)
            y = np.random.randint(0, h - 50)
            cv2.circle(frame, (x, y), np.random.randint(10, 30), (34, 139, 34), -1)
    
    def _add_synthetic_targets(self, frame: np.ndarray, elapsed_time: float) -> None:
        """Add synthetic human targets to the frame."""
        for i, (base_x, base_y) in enumerate(self.target_positions):
            # Add some movement to targets
            movement_x = 20 * math.sin(elapsed_time * 0.1 + i)
            movement_y = 15 * math.cos(elapsed_time * 0.15 + i)
            
            x = int(base_x + movement_x)
            y = int(base_y + movement_y)
            
            # Ensure target stays in frame
            x = max(20, min(self.frame_width - 20, x))
            y = max(20, min(self.frame_height - 20, y))
            
            # Draw human-like shape (simple rectangle for now)
            target_width = 12
            target_height = 20
            
            # Body (darker color)
            cv2.rectangle(frame, 
                         (x - target_width//2, y - target_height//2),
                         (x + target_width//2, y + target_height//2),
                         (50, 50, 150), -1)
            
            # Head (lighter color)
            cv2.circle(frame, (x, y - target_height//2 - 5), 4, (80, 80, 180), -1)
            
            # Add slight shadow for realism
            cv2.ellipse(frame, (x + 2, y + target_height//2 + 3), (8, 3), 0, 0, 360, (30, 30, 30), -1)
    
    def _add_hud_overlay(self, frame: np.ndarray, elapsed_time: float) -> None:
        """Add HUD overlay with flight information."""
        # Timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"SIM: {timestamp}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Frame counter
        cv2.putText(frame, f"Frame: {self._frame_count:06d}", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Flight time
        cv2.putText(frame, f"Flight Time: {elapsed_time:.1f}s", 
                   (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Crosshair in center
        center_x, center_y = self.frame_width // 2, self.frame_height // 2
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 2)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 2)
        cv2.circle(frame, (center_x, center_y), 50, (0, 255, 0), 2)
        
        # Altitude and speed indicators (top right)
        cv2.putText(frame, f"ALT: {self.flight_altitude:.0f}m", 
                   (self.frame_width - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"SPD: {self.flight_speed:.1f}m/s", 
                   (self.frame_width - 200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Battery indicator (bottom right)
        battery = 100 - (elapsed_time * 0.1)  # Simulate battery drain
        battery = max(0, battery)
        color = (0, 255, 0) if battery > 50 else (0, 255, 255) if battery > 20 else (0, 0, 255)
        cv2.putText(frame, f"BAT: {battery:.0f}%", 
                   (self.frame_width - 150, self.frame_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def _generate_telemetry(self, elapsed_time: float) -> TelemetryData:
        """Generate realistic telemetry data based on flight pattern."""
        if self.flight_pattern == 'circle':
            # Circular flight pattern
            angle = elapsed_time * self.pattern_speed
            
            # Convert to lat/lon offset (approximate)
            lat_offset = (self.pattern_radius * math.cos(angle)) / 111320  # meters to degrees
            lon_offset = (self.pattern_radius * math.sin(angle)) / (111320 * math.cos(math.radians(self.base_lat)))
            
            latitude = self.base_lat + lat_offset
            longitude = self.base_lon + lon_offset
            heading = math.degrees(angle + math.pi/2) % 360
            
            velocity_x = self.flight_speed * math.cos(angle + math.pi/2)
            velocity_y = self.flight_speed * math.sin(angle + math.pi/2)
            
        elif self.flight_pattern == 'linear':
            # Linear flight pattern
            distance = elapsed_time * self.flight_speed
            lat_offset = distance / 111320  # meters to degrees (north)
            
            latitude = self.base_lat + lat_offset
            longitude = self.base_lon
            heading = 0.0  # North
            
            velocity_x = self.flight_speed
            velocity_y = 0.0
            
        else:  # stationary
            latitude = self.base_lat
            longitude = self.base_lon
            heading = 0.0
            velocity_x = 0.0
            velocity_y = 0.0
        
        # Add some realistic variations
        altitude_variation = 5 * math.sin(elapsed_time * 0.1)
        pitch_variation = 3 * math.sin(elapsed_time * 0.2)
        roll_variation = 2 * math.cos(elapsed_time * 0.3)
        
        return TelemetryData(
            timestamp=time.time(),
            latitude=latitude,
            longitude=longitude,
            altitude_msl=self.flight_altitude + altitude_variation,
            altitude_agl=self.flight_altitude + altitude_variation - 10,  # Assume 10m ground elevation
            heading=heading,
            pitch=pitch_variation,
            roll=roll_variation,
            yaw=0,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            velocity_z=0.0,
            gimbal_pitch=-45 + 5 * math.sin(elapsed_time * 0.05),
            gimbal_roll=0,
            gimbal_yaw=0,
            battery_percentage=max(0, 100 - elapsed_time * 0.1),
            signal_strength=5,
            gps_satellite_count=12,
            flight_mode="SIMULATION"
        )


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_simulation():
        """Test simulation connection."""
        config = {
            'frame_width': 1280,
            'frame_height': 720,
            'fps': 30,
            'flight_pattern': 'circle',
            'add_synthetic_targets': True
        }
        
        sim = SimulationConnection(config)
        
        try:
            if await sim.connect():
                print("Simulation started!")
                
                frame_count = 0
                async for frame_data, telemetry in sim.stream_frames():
                    print(f"Frame {frame_count}: {frame_data.shape}, "
                          f"Pos: ({telemetry.latitude:.6f}, {telemetry.longitude:.6f}), "
                          f"Alt: {telemetry.altitude_agl:.1f}m")
                    
                    # Optionally save frame for inspection
                    if frame_count == 0:
                        cv2.imwrite('simulation_frame.jpg', frame_data.frame)
                        print("Saved simulation_frame.jpg")
                    
                    frame_count += 1
                    if frame_count >= 100:  # Run for ~3 seconds
                        break
                        
        finally:
            await sim.disconnect()
    
    # Run test
    asyncio.run(test_simulation())