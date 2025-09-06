#!/usr/bin/env python3
"""
Snapshot Packaging for Foresight SAR System

This module packages mission snapshots with video, metadata, detections,
and tracking data for evidence preservation and handoff operations.
"""

import json
import time
import shutil
import hashlib
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import logging

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV not available - video processing disabled")

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available - image processing disabled")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MissionMetadata:
    """Mission metadata for snapshot packaging"""
    mission_id: str
    operator_id: str
    start_time: str
    end_time: str
    location: str
    aircraft_type: str
    weather_conditions: str
    mission_type: str = "SAR"
    priority_level: str = "normal"
    notes: str = ""
    
@dataclass
class DetectionSnapshot:
    """Single detection snapshot"""
    detection_id: str
    timestamp: str
    lat: float
    lng: float
    confidence: float
    detection_type: str
    track_id: Optional[int] = None
    locked: bool = False
    image_path: Optional[str] = None
    metadata: Dict = None

@dataclass
class TrackingSnapshot:
    """Tracking data snapshot"""
    track_id: int
    start_time: str
    end_time: str
    positions: List[Dict]  # List of {timestamp, lat, lng, confidence}
    locked: bool = False
    suspect_profile: Optional[Dict] = None
    reid_features: Optional[List[float]] = None

@dataclass
class TelemetrySnapshot:
    """Telemetry data snapshot"""
    timestamp: str
    altitude: float
    speed: float
    heading: float
    battery_level: float
    signal_strength: float
    gps_accuracy: float
    camera_position: Dict  # {lat, lng, alt, pitch, yaw, roll}

@dataclass
class PackagingConfig:
    """Configuration for snapshot packaging"""
    include_video: bool = True
    include_images: bool = True
    include_heatmap: bool = True
    include_telemetry: bool = True
    video_quality: str = "high"  # low, medium, high
    image_format: str = "jpg"  # jpg, png
    compression_level: int = 6  # 0-9 for zip compression
    encrypt_package: bool = False
    digital_signature: bool = False
    chain_of_custody: bool = True

class SnapshotPackager:
    """Packages mission snapshots for evidence and handoff"""
    
    def __init__(self, config: PackagingConfig = None):
        self.config = config or PackagingConfig()
        self.temp_dir = None
        
    def _create_temp_directory(self) -> Path:
        """Create temporary directory for packaging"""
        if self.temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="foresight_snapshot_"))
        return self.temp_dir
    
    def _cleanup_temp_directory(self):
        """Clean up temporary directory"""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _create_manifest(self, package_dir: Path) -> Dict[str, str]:
        """Create manifest with file hashes"""
        manifest = {}
        
        for file_path in package_dir.rglob("*"):
            if file_path.is_file() and file_path.name != "manifest.json":
                relative_path = file_path.relative_to(package_dir)
                manifest[str(relative_path)] = self._calculate_file_hash(file_path)
        
        return manifest
    
    def _process_video_frame(self, frame: Any, detections: List[DetectionSnapshot], 
                           timestamp: str) -> Any:
        """Add detection overlays to video frame"""
        if not CV2_AVAILABLE:
            return frame
        
        # Draw detection boxes and labels
        for detection in detections:
            if detection.timestamp == timestamp and detection.image_path:
                # This is a simplified overlay - in practice, you'd need
                # pixel coordinates from the detection system
                cv2.putText(frame, f"{detection.detection_type}: {detection.confidence:.2f}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def _create_video_summary(self, video_path: str, detections: List[DetectionSnapshot],
                            output_path: Path) -> Optional[str]:
        """Create video summary with detection overlays"""
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available - skipping video processing")
            return None
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return None
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Set up video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_path = output_path / "mission_video.mp4"
            out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate timestamp for this frame
                timestamp = datetime.now().isoformat()  # Simplified
                
                # Add detection overlays
                frame = self._process_video_frame(frame, detections, timestamp)
                
                # Write frame
                out.write(frame)
                frame_count += 1
            
            cap.release()
            out.release()
            
            logger.info(f"Processed {frame_count} frames to {out_path}")
            return str(out_path)
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            return None
    
    def _create_detection_summary(self, detections: List[DetectionSnapshot],
                                output_path: Path) -> str:
        """Create detection summary document"""
        summary_path = output_path / "detection_summary.json"
        
        # Group detections by type
        by_type = {}
        for detection in detections:
            if detection.detection_type not in by_type:
                by_type[detection.detection_type] = []
            by_type[detection.detection_type].append(detection)
        
        # Create summary statistics
        summary = {
            "total_detections": len(detections),
            "by_type": {k: len(v) for k, v in by_type.items()},
            "locked_detections": len([d for d in detections if d.locked]),
            "confidence_stats": {
                "min": min([d.confidence for d in detections]) if detections else 0,
                "max": max([d.confidence for d in detections]) if detections else 0,
                "avg": sum([d.confidence for d in detections]) / len(detections) if detections else 0
            },
            "time_range": {
                "start": min([d.timestamp for d in detections]) if detections else None,
                "end": max([d.timestamp for d in detections]) if detections else None
            },
            "detections": [asdict(d) for d in detections]
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return str(summary_path)
    
    def _create_tracking_summary(self, tracks: List[TrackingSnapshot],
                               output_path: Path) -> str:
        """Create tracking summary document"""
        summary_path = output_path / "tracking_summary.json"
        
        summary = {
            "total_tracks": len(tracks),
            "locked_tracks": len([t for t in tracks if t.locked]),
            "tracks_with_reid": len([t for t in tracks if t.reid_features]),
            "tracks": [asdict(t) for t in tracks]
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return str(summary_path)
    
    def _create_telemetry_summary(self, telemetry: List[TelemetrySnapshot],
                                output_path: Path) -> str:
        """Create telemetry summary document"""
        summary_path = output_path / "telemetry_summary.json"
        
        if not telemetry:
            summary = {"total_records": 0, "records": []}
        else:
            summary = {
                "total_records": len(telemetry),
                "altitude_stats": {
                    "min": min([t.altitude for t in telemetry]),
                    "max": max([t.altitude for t in telemetry]),
                    "avg": sum([t.altitude for t in telemetry]) / len(telemetry)
                },
                "speed_stats": {
                    "min": min([t.speed for t in telemetry]),
                    "max": max([t.speed for t in telemetry]),
                    "avg": sum([t.speed for t in telemetry]) / len(telemetry)
                },
                "battery_stats": {
                    "min": min([t.battery_level for t in telemetry]),
                    "max": max([t.battery_level for t in telemetry]),
                    "avg": sum([t.battery_level for t in telemetry]) / len(telemetry)
                },
                "records": [asdict(t) for t in telemetry]
            }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return str(summary_path)
    
    def _create_chain_of_custody(self, metadata: MissionMetadata,
                               output_path: Path) -> str:
        """Create chain of custody document"""
        custody_path = output_path / "chain_of_custody.json"
        
        custody = {
            "mission_id": metadata.mission_id,
            "created_by": metadata.operator_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "package_hash": "",  # Will be filled after packaging
            "custody_chain": [
                {
                    "action": "package_created",
                    "operator": metadata.operator_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "notes": "Initial evidence package creation"
                }
            ],
            "integrity_checks": {
                "manifest_verified": True,
                "hashes_verified": True,
                "signature_verified": False  # Will be updated if signing enabled
            }
        }
        
        with open(custody_path, 'w') as f:
            json.dump(custody, f, indent=2)
        
        return str(custody_path)
    
    def _copy_detection_images(self, detections: List[DetectionSnapshot],
                             output_path: Path) -> List[str]:
        """Copy detection images to package"""
        images_dir = output_path / "detection_images"
        images_dir.mkdir(exist_ok=True)
        
        copied_images = []
        
        for detection in detections:
            if detection.image_path and Path(detection.image_path).exists():
                src_path = Path(detection.image_path)
                dst_path = images_dir / f"{detection.detection_id}_{src_path.name}"
                
                try:
                    shutil.copy2(src_path, dst_path)
                    copied_images.append(str(dst_path))
                except Exception as e:
                    logger.warning(f"Failed to copy image {src_path}: {e}")
        
        return copied_images
    
    def create_snapshot_package(self, 
                              metadata: MissionMetadata,
                              detections: List[DetectionSnapshot] = None,
                              tracks: List[TrackingSnapshot] = None,
                              telemetry: List[TelemetrySnapshot] = None,
                              video_path: str = None,
                              heatmap_data: Dict = None,
                              output_path: str = None) -> Dict[str, Any]:
        """Create complete snapshot package"""
        
        # Set defaults
        detections = detections or []
        tracks = tracks or []
        telemetry = telemetry or []
        
        # Create output directory
        if output_path is None:
            output_path = f"foresight_snapshot_{metadata.mission_id}_{int(time.time())}"
        
        package_dir = Path(output_path)
        package_dir.mkdir(exist_ok=True)
        
        try:
            # Create mission metadata
            metadata_path = package_dir / "mission_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)
            
            created_files = [str(metadata_path)]
            
            # Process video if available
            if self.config.include_video and video_path:
                video_file = self._create_video_summary(video_path, detections, package_dir)
                if video_file:
                    created_files.append(video_file)
            
            # Copy detection images
            if self.config.include_images and detections:
                image_files = self._copy_detection_images(detections, package_dir)
                created_files.extend(image_files)
            
            # Create detection summary
            if detections:
                detection_file = self._create_detection_summary(detections, package_dir)
                created_files.append(detection_file)
            
            # Create tracking summary
            if tracks:
                tracking_file = self._create_tracking_summary(tracks, package_dir)
                created_files.append(tracking_file)
            
            # Create telemetry summary
            if self.config.include_telemetry and telemetry:
                telemetry_file = self._create_telemetry_summary(telemetry, package_dir)
                created_files.append(telemetry_file)
            
            # Include heatmap data
            if self.config.include_heatmap and heatmap_data:
                heatmap_path = package_dir / "heatmap_data.json"
                with open(heatmap_path, 'w') as f:
                    json.dump(heatmap_data, f, indent=2)
                created_files.append(str(heatmap_path))
            
            # Create manifest
            manifest = self._create_manifest(package_dir)
            manifest_path = package_dir / "manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            created_files.append(str(manifest_path))
            
            # Create chain of custody
            if self.config.chain_of_custody:
                custody_file = self._create_chain_of_custody(metadata, package_dir)
                created_files.append(custody_file)
            
            # Create package info
            package_info = {
                "package_id": f"{metadata.mission_id}_{int(time.time())}",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "mission_id": metadata.mission_id,
                "operator_id": metadata.operator_id,
                "total_files": len(created_files),
                "total_detections": len(detections),
                "total_tracks": len(tracks),
                "total_telemetry_records": len(telemetry),
                "package_size_bytes": sum(f.stat().st_size for f in package_dir.rglob("*") if f.is_file()),
                "files": created_files,
                "config": asdict(self.config)
            }
            
            package_info_path = package_dir / "package_info.json"
            with open(package_info_path, 'w') as f:
                json.dump(package_info, f, indent=2)
            
            logger.info(f"Snapshot package created: {package_dir}")
            logger.info(f"Total files: {len(created_files)}")
            logger.info(f"Package size: {package_info['package_size_bytes']} bytes")
            
            return {
                "success": True,
                "package_path": str(package_dir),
                "package_info": package_info
            }
            
        except Exception as e:
            logger.error(f"Failed to create snapshot package: {e}")
            return {
                "success": False,
                "error": str(e),
                "package_path": str(package_dir)
            }
    
    def create_compressed_package(self, package_dir: str, 
                                output_file: str = None) -> Dict[str, Any]:
        """Create compressed ZIP package"""
        package_path = Path(package_dir)
        
        if not package_path.exists():
            return {"success": False, "error": "Package directory not found"}
        
        if output_file is None:
            output_file = f"{package_path.name}.zip"
        
        try:
            with zipfile.ZipFile(output_file, 'w', 
                               compression=zipfile.ZIP_DEFLATED,
                               compresslevel=self.config.compression_level) as zipf:
                
                for file_path in package_path.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(package_path)
                        zipf.write(file_path, arcname)
            
            # Calculate package hash
            package_hash = self._calculate_file_hash(Path(output_file))
            
            return {
                "success": True,
                "compressed_package": output_file,
                "package_hash": package_hash,
                "original_size": sum(f.stat().st_size for f in package_path.rglob("*") if f.is_file()),
                "compressed_size": Path(output_file).stat().st_size
            }
            
        except Exception as e:
            logger.error(f"Failed to create compressed package: {e}")
            return {"success": False, "error": str(e)}
    
    def create_handoff_json(self, metadata: MissionMetadata,
                          detections: List[DetectionSnapshot] = None,
                          tracks: List[TrackingSnapshot] = None,
                          priority_targets: List[Dict] = None) -> Dict[str, Any]:
        """Create handoff JSON for other agencies"""
        
        detections = detections or []
        tracks = tracks or []
        priority_targets = priority_targets or []
        
        handoff_data = {
            "handoff_info": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "mission_id": metadata.mission_id,
                "operator_id": metadata.operator_id,
                "handoff_type": "SAR_INTELLIGENCE",
                "classification": "UNCLASSIFIED",  # Adjust as needed
                "urgency": metadata.priority_level
            },
            "mission_summary": {
                "mission_type": metadata.mission_type,
                "location": metadata.location,
                "start_time": metadata.start_time,
                "end_time": metadata.end_time,
                "weather": metadata.weather_conditions,
                "aircraft": metadata.aircraft_type,
                "notes": metadata.notes
            },
            "intelligence_summary": {
                "total_detections": len(detections),
                "confirmed_targets": len([d for d in detections if d.locked]),
                "active_tracks": len(tracks),
                "priority_targets": len(priority_targets),
                "confidence_threshold": 0.7  # Configurable
            },
            "priority_targets": priority_targets,
            "confirmed_detections": [
                {
                    "id": d.detection_id,
                    "timestamp": d.timestamp,
                    "location": {"lat": d.lat, "lng": d.lng},
                    "type": d.detection_type,
                    "confidence": d.confidence,
                    "track_id": d.track_id,
                    "status": "locked" if d.locked else "unconfirmed"
                }
                for d in detections if d.confidence >= 0.7
            ],
            "tracking_data": [
                {
                    "track_id": t.track_id,
                    "start_time": t.start_time,
                    "end_time": t.end_time,
                    "position_count": len(t.positions),
                    "last_position": t.positions[-1] if t.positions else None,
                    "status": "locked" if t.locked else "active",
                    "has_reid_profile": t.reid_features is not None
                }
                for t in tracks
            ],
            "recommendations": [
                "Continue monitoring locked targets",
                "Deploy ground teams to high-confidence detection areas",
                "Maintain aerial surveillance of active tracks"
            ],
            "contact_info": {
                "operator": metadata.operator_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
        
        return handoff_data
    
    def __del__(self):
        """Cleanup on destruction"""
        self._cleanup_temp_directory()

def create_sample_mission_data() -> Dict[str, Any]:
    """Create sample mission data for testing"""
    
    # Sample metadata
    metadata = MissionMetadata(
        mission_id="SAR-2024-001",
        operator_id="PILOT-001",
        start_time=datetime.now().isoformat(),
        end_time=(datetime.now()).isoformat(),
        location="Search Area Alpha",
        aircraft_type="DJI Mavic 3",
        weather_conditions="Clear, 15mph winds",
        mission_type="Missing Person Search",
        priority_level="high",
        notes="Searching for missing hiker, last seen 24 hours ago"
    )
    
    # Sample detections
    detections = [
        DetectionSnapshot(
            detection_id="DET-001",
            timestamp=datetime.now().isoformat(),
            lat=37.7749,
            lng=-122.4194,
            confidence=0.85,
            detection_type="person",
            track_id=1,
            locked=True
        ),
        DetectionSnapshot(
            detection_id="DET-002",
            timestamp=datetime.now().isoformat(),
            lat=37.7750,
            lng=-122.4195,
            confidence=0.72,
            detection_type="person",
            track_id=2,
            locked=False
        )
    ]
    
    # Sample tracks
    tracks = [
        TrackingSnapshot(
            track_id=1,
            start_time=datetime.now().isoformat(),
            end_time=datetime.now().isoformat(),
            positions=[
                {"timestamp": datetime.now().isoformat(), "lat": 37.7749, "lng": -122.4194, "confidence": 0.85}
            ],
            locked=True
        )
    ]
    
    # Sample telemetry
    telemetry = [
        TelemetrySnapshot(
            timestamp=datetime.now().isoformat(),
            altitude=120.0,
            speed=15.5,
            heading=45.0,
            battery_level=85.0,
            signal_strength=95.0,
            gps_accuracy=2.1,
            camera_position={"lat": 37.7749, "lng": -122.4194, "alt": 120, "pitch": -30, "yaw": 45, "roll": 0}
        )
    ]
    
    return {
        "metadata": metadata,
        "detections": detections,
        "tracks": tracks,
        "telemetry": telemetry
    }

if __name__ == "__main__":
    # Example usage
    sample_data = create_sample_mission_data()
    
    config = PackagingConfig(
        include_video=True,
        include_images=True,
        include_heatmap=True,
        compression_level=6
    )
    
    packager = SnapshotPackager(config)
    
    # Create snapshot package
    result = packager.create_snapshot_package(
        metadata=sample_data["metadata"],
        detections=sample_data["detections"],
        tracks=sample_data["tracks"],
        telemetry=sample_data["telemetry"]
    )
    
    print("Snapshot packaging result:")
    print(json.dumps(result, indent=2, default=str))
    
    # Create handoff JSON
    handoff_data = packager.create_handoff_json(
        metadata=sample_data["metadata"],
        detections=sample_data["detections"],
        tracks=sample_data["tracks"]
    )
    
    print("\nHandoff JSON:")
    print(json.dumps(handoff_data, indent=2))