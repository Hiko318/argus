import os
import json
import shutil
import zipfile
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import threading
from queue import Queue
import cv2
import numpy as np
from dataclasses import dataclass, asdict
import hashlib
import time

@dataclass
class VideoSnippet:
    """Data structure for video snippets"""
    snippet_id: str
    timestamp: str
    duration_seconds: float
    file_path: str
    detection_events: List[Dict[str, Any]]
    metadata: Dict[str, Any]

@dataclass
class StorageManifest:
    """Manifest for stored data"""
    session_id: str
    created_at: str
    total_size_bytes: int
    video_snippets: List[VideoSnippet]
    log_files: List[str]
    telemetry_files: List[str]
    checksum: str

class OfflineStorageService:
    """Offline storage service for video, logs, and telemetry data"""
    
    def __init__(self, base_storage_dir: str = "out/storage", max_storage_gb: float = 10.0):
        self.base_storage_dir = Path(base_storage_dir)
        self.base_storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage limits
        self.max_storage_bytes = int(max_storage_gb * 1024 * 1024 * 1024)
        
        # Create session-specific directory
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_storage_dir / f"session_{self.session_id}"
        self.session_dir.mkdir(exist_ok=True)
        
        # Storage subdirectories
        self.video_dir = self.session_dir / "video_snippets"
        self.logs_dir = self.session_dir / "logs"
        self.telemetry_dir = self.session_dir / "telemetry"
        self.exports_dir = self.session_dir / "exports"
        
        for dir_path in [self.video_dir, self.logs_dir, self.telemetry_dir, self.exports_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Video recording settings
        self.video_codec = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_fps = 30
        self.video_quality = 85  # JPEG quality for frames
        
        # Storage queues for async processing
        self.video_queue = Queue()
        self.storage_active = True
        
        # Background storage thread
        self.storage_thread = threading.Thread(target=self._process_storage_queue, daemon=True)
        self.storage_thread.start()
        
        # Storage manifest
        self.manifest = StorageManifest(
            session_id=self.session_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            total_size_bytes=0,
            video_snippets=[],
            log_files=[],
            telemetry_files=[],
            checksum=""
        )
        
        print(f"Offline storage service initialized for session: {self.session_id}")
        print(f"Storage directory: {self.session_dir}")
        print(f"Max storage: {max_storage_gb:.1f} GB")
    
    def record_video_snippet(self, 
                           frames: List[np.ndarray],
                           detection_events: List[Dict[str, Any]],
                           duration_seconds: float,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """Record a video snippet with associated detection events"""
        snippet_id = f"snippet_{int(time.time() * 1000)}"
        timestamp = datetime.now(timezone.utc).isoformat()
        
        snippet_data = {
            'snippet_id': snippet_id,
            'timestamp': timestamp,
            'frames': frames,
            'detection_events': detection_events,
            'duration_seconds': duration_seconds,
            'metadata': metadata or {}
        }
        
        self.video_queue.put(snippet_data)
        return snippet_id
    
    def _process_storage_queue(self):
        """Background thread to process storage operations"""
        while self.storage_active:
            try:
                if not self.video_queue.empty():
                    snippet_data = self.video_queue.get(timeout=1)
                    self._save_video_snippet(snippet_data)
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error processing storage queue: {e}")
    
    def _save_video_snippet(self, snippet_data: Dict[str, Any]):
        """Save video snippet to disk"""
        try:
            snippet_id = snippet_data['snippet_id']
            frames = snippet_data['frames']
            
            if not frames:
                return
            
            # Create video file
            video_file = self.video_dir / f"{snippet_id}.mp4"
            height, width = frames[0].shape[:2]
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(video_file), 
                fourcc, 
                self.video_fps, 
                (width, height)
            )
            
            # Write frames
            for frame in frames:
                video_writer.write(frame)
            
            video_writer.release()
            
            # Create video snippet object
            snippet = VideoSnippet(
                snippet_id=snippet_id,
                timestamp=snippet_data['timestamp'],
                duration_seconds=snippet_data['duration_seconds'],
                file_path=str(video_file),
                detection_events=snippet_data['detection_events'],
                metadata=snippet_data['metadata']
            )
            
            # Add to manifest
            self.manifest.video_snippets.append(snippet)
            
            # Update storage size
            file_size = video_file.stat().st_size
            self.manifest.total_size_bytes += file_size
            
            # Check storage limits
            self._check_storage_limits()
            
            print(f"Video snippet saved: {snippet_id} ({file_size / 1024 / 1024:.1f} MB)")
            
        except Exception as e:
            print(f"Error saving video snippet: {e}")
    
    def _check_storage_limits(self):
        """Check and enforce storage limits"""
        if self.manifest.total_size_bytes > self.max_storage_bytes:
            print("Storage limit exceeded, cleaning up old files...")
            self._cleanup_old_files()
    
    def _cleanup_old_files(self):
        """Remove oldest files to free up space"""
        try:
            # Sort snippets by timestamp
            sorted_snippets = sorted(
                self.manifest.video_snippets, 
                key=lambda x: x.timestamp
            )
            
            # Remove oldest snippets until under limit
            while (self.manifest.total_size_bytes > self.max_storage_bytes * 0.8 and 
                   sorted_snippets):
                
                oldest_snippet = sorted_snippets.pop(0)
                file_path = Path(oldest_snippet.file_path)
                
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    self.manifest.total_size_bytes -= file_size
                    print(f"Removed old snippet: {oldest_snippet.snippet_id}")
                
                # Remove from manifest
                self.manifest.video_snippets = [
                    s for s in self.manifest.video_snippets 
                    if s.snippet_id != oldest_snippet.snippet_id
                ]
                
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def store_log_file(self, source_file: str, log_type: str = "general") -> str:
        """Store a log file in the session directory"""
        try:
            source_path = Path(source_file)
            if not source_path.exists():
                raise FileNotFoundError(f"Source file not found: {source_file}")
            
            # Create destination filename with timestamp
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            dest_filename = f"{log_type}_{timestamp}_{source_path.name}"
            dest_path = self.logs_dir / dest_filename
            
            # Copy file
            shutil.copy2(source_path, dest_path)
            
            # Add to manifest
            self.manifest.log_files.append(str(dest_path))
            
            # Update storage size
            file_size = dest_path.stat().st_size
            self.manifest.total_size_bytes += file_size
            
            print(f"Log file stored: {dest_filename} ({file_size / 1024:.1f} KB)")
            return str(dest_path)
            
        except Exception as e:
            print(f"Error storing log file: {e}")
            return ""
    
    def store_telemetry_data(self, telemetry_data: Dict[str, Any], filename: str = None) -> str:
        """Store telemetry data as JSON file"""
        try:
            if filename is None:
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                filename = f"telemetry_{timestamp}.json"
            
            file_path = self.telemetry_dir / filename
            
            # Save telemetry data
            with open(file_path, 'w') as f:
                json.dump(telemetry_data, f, indent=2)
            
            # Add to manifest
            self.manifest.telemetry_files.append(str(file_path))
            
            # Update storage size
            file_size = file_path.stat().st_size
            self.manifest.total_size_bytes += file_size
            
            print(f"Telemetry data stored: {filename} ({file_size / 1024:.1f} KB)")
            return str(file_path)
            
        except Exception as e:
            print(f"Error storing telemetry data: {e}")
            return ""
    
    def create_mission_archive(self, include_videos: bool = True) -> str:
        """Create a compressed archive of all mission data"""
        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            archive_name = f"mission_archive_{self.session_id}_{timestamp}.zip"
            archive_path = self.exports_dir / archive_name
            
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add manifest
                self._update_manifest_checksum()
                manifest_file = self.session_dir / "manifest.json"
                with open(manifest_file, 'w') as f:
                    json.dump(asdict(self.manifest), f, indent=2)
                zipf.write(manifest_file, "manifest.json")
                
                # Add log files
                for log_file in self.manifest.log_files:
                    if Path(log_file).exists():
                        zipf.write(log_file, f"logs/{Path(log_file).name}")
                
                # Add telemetry files
                for telemetry_file in self.manifest.telemetry_files:
                    if Path(telemetry_file).exists():
                        zipf.write(telemetry_file, f"telemetry/{Path(telemetry_file).name}")
                
                # Add video snippets if requested
                if include_videos:
                    for snippet in self.manifest.video_snippets:
                        if Path(snippet.file_path).exists():
                            zipf.write(snippet.file_path, f"videos/{Path(snippet.file_path).name}")
                        
                        # Add snippet metadata
                        metadata_file = f"videos/{snippet.snippet_id}_metadata.json"
                        zipf.writestr(metadata_file, json.dumps(asdict(snippet), indent=2))
            
            archive_size = archive_path.stat().st_size
            print(f"Mission archive created: {archive_name} ({archive_size / 1024 / 1024:.1f} MB)")
            return str(archive_path)
            
        except Exception as e:
            print(f"Error creating mission archive: {e}")
            return ""
    
    def _update_manifest_checksum(self):
        """Update manifest checksum"""
        try:
            # Create checksum of all files
            hasher = hashlib.sha256()
            
            # Hash log files
            for log_file in self.manifest.log_files:
                if Path(log_file).exists():
                    with open(log_file, 'rb') as f:
                        hasher.update(f.read())
            
            # Hash telemetry files
            for telemetry_file in self.manifest.telemetry_files:
                if Path(telemetry_file).exists():
                    with open(telemetry_file, 'rb') as f:
                        hasher.update(f.read())
            
            # Hash video files
            for snippet in self.manifest.video_snippets:
                if Path(snippet.file_path).exists():
                    with open(snippet.file_path, 'rb') as f:
                        # Hash first and last 1KB for large files
                        data = f.read(1024)
                        hasher.update(data)
                        f.seek(-1024, 2)
                        data = f.read(1024)
                        hasher.update(data)
            
            self.manifest.checksum = hasher.hexdigest()
            
        except Exception as e:
            print(f"Error updating checksum: {e}")
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get current storage statistics"""
        stats = {
            "session_id": self.session_id,
            "total_size_mb": self.manifest.total_size_bytes / 1024 / 1024,
            "max_size_mb": self.max_storage_bytes / 1024 / 1024,
            "usage_percentage": (self.manifest.total_size_bytes / self.max_storage_bytes) * 100,
            "video_snippets_count": len(self.manifest.video_snippets),
            "log_files_count": len(self.manifest.log_files),
            "telemetry_files_count": len(self.manifest.telemetry_files),
            "storage_directory": str(self.session_dir)
        }
        return stats
    
    def export_data_for_transfer(self, export_format: str = "zip") -> Tuple[str, Dict[str, Any]]:
        """Export data in format suitable for transfer"""
        try:
            if export_format == "zip":
                archive_path = self.create_mission_archive(include_videos=True)
                transfer_info = {
                    "format": "zip",
                    "file_path": archive_path,
                    "size_mb": Path(archive_path).stat().st_size / 1024 / 1024,
                    "checksum": self.manifest.checksum,
                    "session_id": self.session_id
                }
                return archive_path, transfer_info
            
            elif export_format == "directory":
                # Copy entire session directory
                export_dir = self.exports_dir / f"export_{int(time.time())}"
                shutil.copytree(self.session_dir, export_dir)
                
                transfer_info = {
                    "format": "directory",
                    "directory_path": str(export_dir),
                    "size_mb": self._get_directory_size(export_dir) / 1024 / 1024,
                    "checksum": self.manifest.checksum,
                    "session_id": self.session_id
                }
                return str(export_dir), transfer_info
            
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
                
        except Exception as e:
            print(f"Error exporting data: {e}")
            return "", {}
    
    def _get_directory_size(self, directory: Path) -> int:
        """Calculate total size of directory"""
        total_size = 0
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    def cleanup_session(self, keep_exports: bool = True):
        """Clean up session data"""
        try:
            if keep_exports:
                # Only remove non-export directories
                for dir_path in [self.video_dir, self.logs_dir, self.telemetry_dir]:
                    if dir_path.exists():
                        shutil.rmtree(dir_path)
            else:
                # Remove entire session directory
                if self.session_dir.exists():
                    shutil.rmtree(self.session_dir)
            
            print(f"Session cleanup completed: {self.session_id}")
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def shutdown(self) -> Dict[str, Any]:
        """Shutdown storage service and create final archive"""
        print("Shutting down storage service...")
        self.storage_active = False
        
        # Wait for storage thread to finish
        time.sleep(1)
        
        # Create final archive
        archive_path = self.create_mission_archive(include_videos=True)
        
        # Get final statistics
        stats = self.get_storage_statistics()
        stats["final_archive"] = archive_path
        
        print(f"Storage service shutdown complete. Archive: {archive_path}")
        return stats

# Global storage service instance
_storage_service: Optional[OfflineStorageService] = None

def get_storage_service() -> OfflineStorageService:
    """Get or create global storage service instance"""
    global _storage_service
    if _storage_service is None:
        _storage_service = OfflineStorageService()
    return _storage_service

def shutdown_storage():
    """Shutdown global storage service"""
    global _storage_service
    if _storage_service:
        return _storage_service.shutdown()
    return None