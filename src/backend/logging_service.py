import json
import csv
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from queue import Queue
import time

@dataclass
class DetectionEvent:
    """Data structure for detection events"""
    timestamp: str
    event_type: str  # 'detection', 'suspect_match', 'handoff', 'mode_change'
    drone_gps: Dict[str, float]  # {'lat': float, 'lon': float, 'alt': float}
    bounding_box: Dict[str, float]  # {'x': float, 'y': float, 'width': float, 'height': float}
    geo_coordinates: Optional[Dict[str, float]]  # {'lat': float, 'lon': float}
    detection_confidence: float
    suspect_lock_status: bool
    target_id: Optional[str]  # Masked target ID for privacy
    additional_data: Optional[Dict[str, Any]] = None

@dataclass
class TelemetryEvent:
    """Data structure for telemetry events"""
    timestamp: str
    drone_gps: Dict[str, float]
    orientation: Dict[str, float]  # {'roll': float, 'pitch': float, 'yaw': float}
    camera_intrinsics: Dict[str, Any]
    flight_mode: str
    battery_level: Optional[float] = None
    signal_strength: Optional[float] = None

@dataclass
class SystemEvent:
    """Data structure for system events"""
    timestamp: str
    event_type: str  # 'mode_change', 'target_set', 'tracking_start', 'tracking_stop'
    description: str
    user_action: bool
    additional_data: Optional[Dict[str, Any]] = None

class LoggingService:
    """Comprehensive logging service for SAR operations"""
    
    def __init__(self, base_log_dir: str = "out/logs"):
        self.base_log_dir = Path(base_log_dir)
        self.base_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session-specific directory
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_log_dir / f"session_{self.session_id}"
        self.session_dir.mkdir(exist_ok=True)
        
        # Initialize log files
        self.detection_log_file = self.session_dir / "detections.json"
        self.telemetry_log_file = self.session_dir / "telemetry.json"
        self.system_log_file = self.session_dir / "system_events.json"
        self.csv_export_file = self.session_dir / "mission_summary.csv"
        
        # Initialize log queues for async processing
        self.detection_queue = Queue()
        self.telemetry_queue = Queue()
        self.system_queue = Queue()
        
        # Start background logging threads
        self.logging_active = True
        self.detection_thread = threading.Thread(target=self._process_detection_logs, daemon=True)
        self.telemetry_thread = threading.Thread(target=self._process_telemetry_logs, daemon=True)
        self.system_thread = threading.Thread(target=self._process_system_logs, daemon=True)
        
        self.detection_thread.start()
        self.telemetry_thread.start()
        self.system_thread.start()
        
        # Initialize CSV file with headers
        self._initialize_csv_export()
        
        print(f"Logging service initialized for session: {self.session_id}")
        print(f"Log directory: {self.session_dir}")
    
    def log_detection(self, 
                     drone_gps: Dict[str, float],
                     bounding_box: Dict[str, float],
                     detection_confidence: float,
                     geo_coordinates: Optional[Dict[str, float]] = None,
                     suspect_lock_status: bool = False,
                     target_id: Optional[str] = None,
                     event_type: str = "detection",
                     additional_data: Optional[Dict[str, Any]] = None):
        """Log a detection event"""
        event = DetectionEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            drone_gps=drone_gps,
            bounding_box=bounding_box,
            geo_coordinates=geo_coordinates,
            detection_confidence=detection_confidence,
            suspect_lock_status=suspect_lock_status,
            target_id=target_id,
            additional_data=additional_data
        )
        
        self.detection_queue.put(event)
    
    def log_telemetry(self,
                     drone_gps: Dict[str, float],
                     orientation: Dict[str, float],
                     camera_intrinsics: Dict[str, Any],
                     flight_mode: str,
                     battery_level: Optional[float] = None,
                     signal_strength: Optional[float] = None):
        """Log telemetry data"""
        event = TelemetryEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            drone_gps=drone_gps,
            orientation=orientation,
            camera_intrinsics=camera_intrinsics,
            flight_mode=flight_mode,
            battery_level=battery_level,
            signal_strength=signal_strength
        )
        
        self.telemetry_queue.put(event)
    
    def log_system_event(self,
                        event_type: str,
                        description: str,
                        user_action: bool = False,
                        additional_data: Optional[Dict[str, Any]] = None):
        """Log system events"""
        event = SystemEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            description=description,
            user_action=user_action,
            additional_data=additional_data
        )
        
        self.system_queue.put(event)
    
    def log_suspect_match(self,
                         drone_gps: Dict[str, float],
                         bounding_box: Dict[str, float],
                         detection_confidence: float,
                         match_confidence: float,
                         target_id: str,
                         geo_coordinates: Optional[Dict[str, float]] = None):
        """Log a suspect match event"""
        additional_data = {
            "match_confidence": match_confidence,
            "match_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        self.log_detection(
            drone_gps=drone_gps,
            bounding_box=bounding_box,
            detection_confidence=detection_confidence,
            geo_coordinates=geo_coordinates,
            suspect_lock_status=True,
            target_id=target_id,
            event_type="suspect_match",
            additional_data=additional_data
        )
    
    def log_handoff_event(self,
                         drone_gps: Dict[str, float],
                         target_location: Dict[str, float],
                         handoff_type: str,
                         recipient: str,
                         additional_data: Optional[Dict[str, Any]] = None):
        """Log handoff events"""
        handoff_data = {
            "handoff_type": handoff_type,
            "recipient": recipient,
            "target_location": target_location
        }
        
        if additional_data:
            handoff_data.update(additional_data)
        
        self.log_system_event(
            event_type="handoff",
            description=f"Handoff to {recipient} ({handoff_type})",
            user_action=True,
            additional_data=handoff_data
        )
    
    def _process_detection_logs(self):
        """Background thread to process detection logs"""
        while self.logging_active:
            try:
                if not self.detection_queue.empty():
                    event = self.detection_queue.get(timeout=1)
                    self._write_json_log(self.detection_log_file, event)
                    self._update_csv_export(event)
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error processing detection log: {e}")
    
    def _process_telemetry_logs(self):
        """Background thread to process telemetry logs"""
        while self.logging_active:
            try:
                if not self.telemetry_queue.empty():
                    event = self.telemetry_queue.get(timeout=1)
                    self._write_json_log(self.telemetry_log_file, event)
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error processing telemetry log: {e}")
    
    def _process_system_logs(self):
        """Background thread to process system logs"""
        while self.logging_active:
            try:
                if not self.system_queue.empty():
                    event = self.system_queue.get(timeout=1)
                    self._write_json_log(self.system_log_file, event)
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error processing system log: {e}")
    
    def _write_json_log(self, file_path: Path, event):
        """Write event to JSON log file"""
        try:
            # Read existing data
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
            else:
                data = []
            
            # Append new event
            data.append(asdict(event))
            
            # Write back to file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Error writing to {file_path}: {e}")
    
    def _initialize_csv_export(self):
        """Initialize CSV export file with headers"""
        headers = [
            'timestamp', 'event_type', 'drone_lat', 'drone_lon', 'drone_alt',
            'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height',
            'geo_lat', 'geo_lon', 'detection_confidence', 'suspect_lock_status',
            'target_id', 'additional_info'
        ]
        
        with open(self.csv_export_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def _update_csv_export(self, event: DetectionEvent):
        """Update CSV export with detection event"""
        try:
            row = [
                event.timestamp,
                event.event_type,
                event.drone_gps.get('lat', ''),
                event.drone_gps.get('lon', ''),
                event.drone_gps.get('alt', ''),
                event.bounding_box.get('x', ''),
                event.bounding_box.get('y', ''),
                event.bounding_box.get('width', ''),
                event.bounding_box.get('height', ''),
                event.geo_coordinates.get('lat', '') if event.geo_coordinates else '',
                event.geo_coordinates.get('lon', '') if event.geo_coordinates else '',
                event.detection_confidence,
                event.suspect_lock_status,
                event.target_id or '',
                json.dumps(event.additional_data) if event.additional_data else ''
            ]
            
            with open(self.csv_export_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
                
        except Exception as e:
            print(f"Error updating CSV export: {e}")
    
    def export_mission_summary(self) -> Dict[str, Any]:
        """Export comprehensive mission summary"""
        summary = {
            "session_id": self.session_id,
            "start_time": self.session_id,
            "end_time": datetime.now(timezone.utc).isoformat(),
            "log_files": {
                "detections": str(self.detection_log_file),
                "telemetry": str(self.telemetry_log_file),
                "system_events": str(self.system_log_file),
                "csv_export": str(self.csv_export_file)
            },
            "statistics": self._calculate_mission_statistics()
        }
        
        # Save summary to file
        summary_file = self.session_dir / "mission_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def _calculate_mission_statistics(self) -> Dict[str, Any]:
        """Calculate mission statistics"""
        stats = {
            "total_detections": 0,
            "suspect_matches": 0,
            "handoff_events": 0,
            "telemetry_points": 0
        }
        
        try:
            # Count detections
            if self.detection_log_file.exists():
                with open(self.detection_log_file, 'r') as f:
                    detections = json.load(f)
                    stats["total_detections"] = len(detections)
                    stats["suspect_matches"] = len([d for d in detections if d.get("suspect_lock_status")])
            
            # Count telemetry points
            if self.telemetry_log_file.exists():
                with open(self.telemetry_log_file, 'r') as f:
                    telemetry = json.load(f)
                    stats["telemetry_points"] = len(telemetry)
            
            # Count handoff events
            if self.system_log_file.exists():
                with open(self.system_log_file, 'r') as f:
                    system_events = json.load(f)
                    stats["handoff_events"] = len([e for e in system_events if e.get("event_type") == "handoff"])
                    
        except Exception as e:
            print(f"Error calculating statistics: {e}")
        
        return stats
    
    def shutdown(self):
        """Shutdown logging service and export final summary"""
        print("Shutting down logging service...")
        self.logging_active = False
        
        # Wait for threads to finish processing queues
        time.sleep(1)
        
        # Export final mission summary
        summary = self.export_mission_summary()
        print(f"Mission summary exported: {summary['log_files']}")
        
        return summary

# Global logging service instance
_logging_service: Optional[LoggingService] = None

def get_logging_service() -> LoggingService:
    """Get or create global logging service instance"""
    global _logging_service
    if _logging_service is None:
        _logging_service = LoggingService()
    return _logging_service

def shutdown_logging():
    """Shutdown global logging service"""
    global _logging_service
    if _logging_service:
        return _logging_service.shutdown()
    return None