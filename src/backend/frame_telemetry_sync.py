#!/usr/bin/env python3
"""
Frame-Telemetry Synchronization Module

Implements RTCP SR (Sender Report) / PTS (Presentation Time Stamp) mapping
for accurate synchronization between video frames and telemetry data.

This module provides:
- RTCP SR packet parsing and timing extraction
- PTS to wall-clock time mapping
- Frame-to-telemetry temporal alignment
- Drift compensation and clock synchronization
- Quality metrics for sync accuracy
"""

import time
import struct
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Callable
from datetime import datetime, timezone
from collections import deque
import numpy as np
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

class SyncQuality(Enum):
    """Synchronization quality levels."""
    EXCELLENT = "excellent"  # < 10ms drift
    GOOD = "good"           # < 50ms drift
    FAIR = "fair"           # < 100ms drift
    POOR = "poor"           # > 100ms drift
    UNKNOWN = "unknown"     # No sync data

@dataclass
class RTCPSenderReport:
    """RTCP Sender Report packet data."""
    ssrc: int                    # Synchronization source identifier
    ntp_timestamp_msw: int       # NTP timestamp most significant word
    ntp_timestamp_lsw: int       # NTP timestamp least significant word
    rtp_timestamp: int           # RTP timestamp
    packet_count: int            # Sender's packet count
    octet_count: int             # Sender's octet count
    wall_clock_time: float       # Converted wall clock time
    received_time: float         # Time when SR was received
    
    @property
    def ntp_timestamp(self) -> float:
        """Convert NTP timestamp to seconds since epoch."""
        # NTP epoch is Jan 1, 1900; Unix epoch is Jan 1, 1970
        # Difference is 70 years = 2208988800 seconds
        ntp_seconds = (self.ntp_timestamp_msw << 32) | self.ntp_timestamp_lsw
        ntp_seconds = ntp_seconds / (2**32)  # Convert from fixed-point
        return ntp_seconds - 2208988800  # Convert to Unix timestamp

@dataclass
class PTSMapping:
    """PTS to wall-clock time mapping."""
    pts: int                     # Presentation timestamp
    wall_clock_time: float       # Corresponding wall clock time
    rtp_timestamp: int           # RTP timestamp from RTCP SR
    confidence: float            # Mapping confidence (0.0-1.0)
    source: str                  # Source of mapping (rtcp_sr, interpolation, etc.)

@dataclass
class SyncMetrics:
    """Synchronization quality metrics."""
    avg_drift_ms: float          # Average drift in milliseconds
    max_drift_ms: float          # Maximum drift in milliseconds
    std_drift_ms: float          # Standard deviation of drift
    sync_quality: SyncQuality    # Overall sync quality
    rtcp_sr_count: int           # Number of RTCP SR packets received
    pts_mappings_count: int      # Number of PTS mappings created
    last_sync_time: float        # Last successful sync timestamp
    clock_drift_rate: float      # Estimated clock drift rate (ppm)

class RTCPParser:
    """RTCP packet parser for extracting Sender Reports."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.RTCPParser")
    
    def parse_rtcp_packet(self, packet_data: bytes) -> Optional[RTCPSenderReport]:
        """Parse RTCP packet and extract Sender Report if present.
        
        Args:
            packet_data: Raw RTCP packet bytes
            
        Returns:
            RTCPSenderReport if SR packet found, None otherwise
        """
        try:
            if len(packet_data) < 8:
                return None
            
            # Parse RTCP header
            header = struct.unpack('!BBH', packet_data[:4])
            version = (header[0] >> 6) & 0x3
            padding = (header[0] >> 5) & 0x1
            reception_count = header[0] & 0x1f
            packet_type = header[1]
            length = header[2]
            
            # Check for RTCP version 2 and Sender Report (PT=200)
            if version != 2 or packet_type != 200:
                return None
            
            # Parse SR payload
            if len(packet_data) < 28:  # Minimum SR size
                return None
            
            sr_data = struct.unpack('!IIIIII', packet_data[4:28])
            
            return RTCPSenderReport(
                ssrc=sr_data[0],
                ntp_timestamp_msw=sr_data[1],
                ntp_timestamp_lsw=sr_data[2],
                rtp_timestamp=sr_data[3],
                packet_count=sr_data[4],
                octet_count=sr_data[5],
                wall_clock_time=0.0,  # Will be calculated
                received_time=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing RTCP packet: {e}")
            return None

class PTSMapper:
    """Maps PTS values to wall-clock time using RTCP SR data."""
    
    def __init__(self, max_mappings: int = 1000):
        self.max_mappings = max_mappings
        self.mappings: deque = deque(maxlen=max_mappings)
        self.rtcp_reports: deque = deque(maxlen=100)
        self.clock_offset = 0.0
        self.clock_drift_rate = 0.0  # ppm
        self.last_calibration_time = 0.0
        self.logger = logging.getLogger(f"{__name__}.PTSMapper")
        self._lock = threading.Lock()
    
    def add_rtcp_sr(self, sr: RTCPSenderReport) -> None:
        """Add RTCP Sender Report for PTS mapping.
        
        Args:
            sr: RTCP Sender Report data
        """
        with self._lock:
            # Calculate wall clock time from NTP timestamp
            sr.wall_clock_time = sr.ntp_timestamp
            
            self.rtcp_reports.append(sr)
            
            # Create PTS mapping
            mapping = PTSMapping(
                pts=sr.rtp_timestamp,  # Use RTP timestamp as PTS reference
                wall_clock_time=sr.wall_clock_time,
                rtp_timestamp=sr.rtp_timestamp,
                confidence=1.0,
                source="rtcp_sr"
            )
            
            self.mappings.append(mapping)
            
            # Update clock calibration
            self._update_clock_calibration()
            
            self.logger.debug(f"Added RTCP SR mapping: PTS={sr.rtp_timestamp}, "
                            f"time={sr.wall_clock_time:.3f}")
    
    def _update_clock_calibration(self) -> None:
        """Update clock offset and drift rate estimation."""
        if len(self.rtcp_reports) < 2:
            return
        
        # Use last two reports for drift calculation
        recent_reports = list(self.rtcp_reports)[-2:]
        
        time_diff = recent_reports[1].wall_clock_time - recent_reports[0].wall_clock_time
        rtp_diff = recent_reports[1].rtp_timestamp - recent_reports[0].rtp_timestamp
        
        if time_diff > 0 and rtp_diff > 0:
            # Estimate clock drift (assuming 90kHz RTP clock)
            expected_rtp_diff = time_diff * 90000  # 90kHz
            drift_ratio = rtp_diff / expected_rtp_diff
            self.clock_drift_rate = (drift_ratio - 1.0) * 1e6  # ppm
            
            # Update offset
            self.clock_offset = recent_reports[1].wall_clock_time - \
                              (recent_reports[1].rtp_timestamp / 90000.0)
        
        self.last_calibration_time = time.time()
    
    def pts_to_wall_clock(self, pts: int) -> Optional[PTSMapping]:
        """Convert PTS to wall-clock time.
        
        Args:
            pts: Presentation timestamp
            
        Returns:
            PTSMapping with wall-clock time or None if no mapping available
        """
        with self._lock:
            if not self.mappings:
                return None
            
            # Find closest PTS mapping
            closest_mapping = min(self.mappings, 
                                key=lambda m: abs(m.pts - pts))
            
            pts_diff = pts - closest_mapping.pts
            time_diff = pts_diff / 90000.0  # Assume 90kHz clock
            
            # Apply drift compensation
            if self.clock_drift_rate != 0.0:
                drift_correction = time_diff * (self.clock_drift_rate / 1e6)
                time_diff += drift_correction
            
            wall_clock_time = closest_mapping.wall_clock_time + time_diff
            
            # Calculate confidence based on time distance
            time_distance = abs(time_diff)
            confidence = max(0.1, 1.0 - (time_distance / 10.0))  # Decay over 10 seconds
            
            return PTSMapping(
                pts=pts,
                wall_clock_time=wall_clock_time,
                rtp_timestamp=pts,
                confidence=confidence,
                source="interpolation" if pts != closest_mapping.pts else "direct"
            )

class FrameTelemetrySync:
    """Main synchronization class for frame-telemetry alignment."""
    
    def __init__(self, telemetry_service, max_history: int = 5000):
        self.telemetry_service = telemetry_service
        self.max_history = max_history
        
        # Components
        self.rtcp_parser = RTCPParser()
        self.pts_mapper = PTSMapper()
        
        # Sync history and metrics
        self.sync_history: deque = deque(maxlen=max_history)
        self.metrics = SyncMetrics(
            avg_drift_ms=0.0,
            max_drift_ms=0.0,
            std_drift_ms=0.0,
            sync_quality=SyncQuality.UNKNOWN,
            rtcp_sr_count=0,
            pts_mappings_count=0,
            last_sync_time=0.0,
            clock_drift_rate=0.0
        )
        
        # Threading
        self._lock = threading.Lock()
        self.logger = logging.getLogger(f"{__name__}.FrameTelemetrySync")
        
        # Callbacks
        self.sync_callbacks: List[Callable[[Dict[str, Any]], None]] = []
    
    def add_sync_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback for sync events.
        
        Args:
            callback: Function to call with sync data
        """
        self.sync_callbacks.append(callback)
    
    def process_rtcp_packet(self, packet_data: bytes) -> bool:
        """Process incoming RTCP packet.
        
        Args:
            packet_data: Raw RTCP packet bytes
            
        Returns:
            True if Sender Report was processed
        """
        sr = self.rtcp_parser.parse_rtcp_packet(packet_data)
        if sr:
            self.pts_mapper.add_rtcp_sr(sr)
            
            with self._lock:
                self.metrics.rtcp_sr_count += 1
                self.metrics.last_sync_time = time.time()
                self.metrics.clock_drift_rate = self.pts_mapper.clock_drift_rate
            
            self.logger.debug(f"Processed RTCP SR: SSRC={sr.ssrc}, "
                            f"RTP={sr.rtp_timestamp}")
            return True
        
        return False
    
    def get_synchronized_telemetry(self, pts: int, 
                                 tolerance_ms: float = 100.0) -> Optional[Dict[str, Any]]:
        """Get telemetry data synchronized to frame PTS.
        
        Args:
            pts: Frame presentation timestamp
            tolerance_ms: Maximum time difference tolerance in milliseconds
            
        Returns:
            Dictionary with telemetry data and sync info
        """
        # Convert PTS to wall-clock time
        pts_mapping = self.pts_mapper.pts_to_wall_clock(pts)
        if not pts_mapping:
            self.logger.warning(f"No PTS mapping available for PTS={pts}")
            return None
        
        # Get telemetry at the mapped time
        tolerance_sec = tolerance_ms / 1000.0
        telemetry = self.telemetry_service.get_telemetry_at_time(
            pts_mapping.wall_clock_time, tolerance_sec
        )
        
        if not telemetry:
            self.logger.warning(f"No telemetry found for time={pts_mapping.wall_clock_time:.3f}")
            return None
        
        # Calculate sync quality
        time_diff_ms = abs(telemetry.timestamp - pts_mapping.wall_clock_time) * 1000
        
        sync_data = {
            "telemetry": telemetry,
            "pts": pts,
            "pts_mapping": pts_mapping,
            "time_diff_ms": time_diff_ms,
            "sync_quality": self._calculate_sync_quality(time_diff_ms),
            "wall_clock_time": pts_mapping.wall_clock_time,
            "confidence": pts_mapping.confidence
        }
        
        # Update metrics
        self._update_sync_metrics(time_diff_ms)
        
        # Store in history
        with self._lock:
            self.sync_history.append(sync_data)
            self.metrics.pts_mappings_count += 1
        
        # Notify callbacks
        for callback in self.sync_callbacks:
            try:
                callback(sync_data)
            except Exception as e:
                self.logger.error(f"Sync callback error: {e}")
        
        return sync_data
    
    def _calculate_sync_quality(self, time_diff_ms: float) -> SyncQuality:
        """Calculate sync quality based on time difference.
        
        Args:
            time_diff_ms: Time difference in milliseconds
            
        Returns:
            SyncQuality enum value
        """
        if time_diff_ms < 10:
            return SyncQuality.EXCELLENT
        elif time_diff_ms < 50:
            return SyncQuality.GOOD
        elif time_diff_ms < 100:
            return SyncQuality.FAIR
        else:
            return SyncQuality.POOR
    
    def _update_sync_metrics(self, time_diff_ms: float) -> None:
        """Update synchronization metrics.
        
        Args:
            time_diff_ms: Latest time difference in milliseconds
        """
        with self._lock:
            # Collect recent time differences
            recent_diffs = []
            for sync_data in list(self.sync_history)[-100:]:  # Last 100 syncs
                recent_diffs.append(sync_data["time_diff_ms"])
            
            if recent_diffs:
                self.metrics.avg_drift_ms = np.mean(recent_diffs)
                self.metrics.max_drift_ms = np.max(recent_diffs)
                self.metrics.std_drift_ms = np.std(recent_diffs)
                self.metrics.sync_quality = self._calculate_sync_quality(
                    self.metrics.avg_drift_ms
                )
    
    def get_sync_metrics(self) -> SyncMetrics:
        """Get current synchronization metrics.
        
        Returns:
            Current SyncMetrics
        """
        with self._lock:
            return SyncMetrics(
                avg_drift_ms=self.metrics.avg_drift_ms,
                max_drift_ms=self.metrics.max_drift_ms,
                std_drift_ms=self.metrics.std_drift_ms,
                sync_quality=self.metrics.sync_quality,
                rtcp_sr_count=self.metrics.rtcp_sr_count,
                pts_mappings_count=self.metrics.pts_mappings_count,
                last_sync_time=self.metrics.last_sync_time,
                clock_drift_rate=self.metrics.clock_drift_rate
            )
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get detailed synchronization status.
        
        Returns:
            Dictionary with sync status information
        """
        metrics = self.get_sync_metrics()
        
        return {
            "sync_quality": metrics.sync_quality.value,
            "avg_drift_ms": metrics.avg_drift_ms,
            "max_drift_ms": metrics.max_drift_ms,
            "std_drift_ms": metrics.std_drift_ms,
            "rtcp_sr_count": metrics.rtcp_sr_count,
            "pts_mappings_count": metrics.pts_mappings_count,
            "last_sync_time": metrics.last_sync_time,
            "clock_drift_rate_ppm": metrics.clock_drift_rate,
            "sync_history_size": len(self.sync_history),
            "is_synchronized": metrics.last_sync_time > 0 and 
                             (time.time() - metrics.last_sync_time) < 30.0
        }

# Factory function for easy initialization
def create_frame_telemetry_sync(telemetry_service, config: Optional[Dict[str, Any]] = None) -> FrameTelemetrySync:
    """Create and configure FrameTelemetrySync instance.
    
    Args:
        telemetry_service: Telemetry service instance
        config: Optional configuration dictionary
        
    Returns:
        Configured FrameTelemetrySync instance
    """
    if config is None:
        config = {}
    
    max_history = config.get("max_history", 5000)
    sync = FrameTelemetrySync(telemetry_service, max_history)
    
    logger.info(f"Created FrameTelemetrySync with max_history={max_history}")
    return sync

if __name__ == "__main__":
    # Demo usage
    import sys
    import asyncio
    from telemetry_service import initialize_telemetry
    
    async def demo():
        """Demonstrate frame-telemetry synchronization."""
        # Initialize telemetry service
        telemetry_service = initialize_telemetry({
            "sim_rate": 10.0,
            "enable_dji": False
        })
        telemetry_service.start_all()
        
        # Create sync instance
        sync = create_frame_telemetry_sync(telemetry_service)
        
        # Add sync callback
        def on_sync(sync_data):
            print(f"Sync: PTS={sync_data['pts']}, "
                  f"drift={sync_data['time_diff_ms']:.1f}ms, "
                  f"quality={sync_data['sync_quality'].value}")
        
        sync.add_sync_callback(on_sync)
        
        try:
            print("Running frame-telemetry sync demo...")
            
            # Simulate RTCP SR packets
            for i in range(10):
                # Create mock RTCP SR packet
                current_time = time.time()
                ntp_time = current_time + 2208988800  # Convert to NTP
                ntp_msw = int(ntp_time) & 0xFFFFFFFF
                ntp_lsw = int((ntp_time - int(ntp_time)) * (2**32)) & 0xFFFFFFFF
                rtp_timestamp = int(current_time * 90000) & 0xFFFFFFFF
                
                # Pack RTCP SR packet
                packet = struct.pack('!BBHIIIIII',
                    0x80,  # V=2, P=0, RC=0
                    200,   # PT=SR
                    6,     # Length
                    0x12345678,  # SSRC
                    ntp_msw,
                    ntp_lsw,
                    rtp_timestamp,
                    i + 1,  # Packet count
                    (i + 1) * 1000  # Octet count
                )
                
                sync.process_rtcp_packet(packet)
                
                # Simulate frame with PTS
                pts = rtp_timestamp + (i * 3000)  # Simulate 30fps
                sync_data = sync.get_synchronized_telemetry(pts)
                
                if sync_data:
                    print(f"Frame {i}: Synchronized successfully")
                else:
                    print(f"Frame {i}: Sync failed")
                
                await asyncio.sleep(0.1)
            
            # Print final metrics
            status = sync.get_sync_status()
            print(f"\nFinal sync status: {status}")
            
        except KeyboardInterrupt:
            print("\nDemo interrupted")
        
        finally:
            telemetry_service.stop_all()
            print("Demo completed")
    
    # Run demo
    asyncio.run(demo())