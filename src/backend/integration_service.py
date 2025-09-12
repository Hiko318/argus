#!/usr/bin/env python3
"""
Integration Service Module

Integrates the human detection pipeline with the video ingest system
for end-to-end real-time human detection and tracking.

Features:
- Real-time video stream processing
- Detection result streaming
- Performance monitoring
- Alert generation
- Data persistence
- WebSocket communication

Author: Foresight AI Team
Date: 2024
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import sys

# Import our modules
sys.path.append('.')
from src.backend.detection_pipeline import DetectionPipeline, HumanDetection, DetectionFrame
from src.backend.edge_optimizer import EdgeOptimizer, OptimizationConfig

logger = logging.getLogger(__name__)

@dataclass
class StreamConfig:
    """Configuration for video stream processing"""
    stream_id: str
    source_url: str
    detection_enabled: bool = True
    tracking_enabled: bool = True
    alert_enabled: bool = True
    
    # Detection settings
    confidence_threshold: float = 0.5
    human_only: bool = True
    aerial_optimized: bool = False
    
    # Performance settings
    target_fps: float = 30.0
    max_queue_size: int = 100
    
    # Alert settings
    min_detection_confidence: float = 0.7
    alert_cooldown_seconds: float = 5.0
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class Alert:
    """Alert for detected events"""
    alert_id: str
    stream_id: str
    timestamp: datetime
    alert_type: str  # "human_detected", "multiple_humans", "high_confidence"
    confidence: float
    human_count: int
    frame_id: int
    bounding_boxes: List[Dict]
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

class StreamProcessor:
    """Processes individual video streams"""
    
    def __init__(self, config: StreamConfig, detection_pipeline: DetectionPipeline):
        self.config = config
        self.pipeline = detection_pipeline
        self.is_running = False
        self.frame_queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.result_queue = asyncio.Queue()
        
        # Statistics
        self.frames_processed = 0
        self.humans_detected = 0
        self.alerts_generated = 0
        self.last_alert_time = 0
        self.start_time = time.time()
        
        # Performance tracking
        self.fps_history = []
        self.processing_times = []
        
        logger.info(f"StreamProcessor initialized for {config.stream_id}")
    
    async def start_processing(self):
        """Start stream processing"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info(f"Starting stream processing for {self.config.stream_id}")
        
        # Start processing tasks
        tasks = [
            asyncio.create_task(self._capture_frames()),
            asyncio.create_task(self._process_frames()),
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Stream processing error: {e}")
        finally:
            self.is_running = False
    
    async def stop_processing(self):
        """Stop stream processing"""
        self.is_running = False
        logger.info(f"Stopping stream processing for {self.config.stream_id}")
    
    async def _capture_frames(self):
        """Capture frames from video source"""
        cap = cv2.VideoCapture(self.config.source_url)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video source: {self.config.source_url}")
            return
        
        frame_id = 0
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame from {self.config.source_url}")
                    break
                
                # Add frame to queue (non-blocking)
                try:
                    self.frame_queue.put_nowait((frame_id, frame, time.time()))
                    frame_id += 1
                except asyncio.QueueFull:
                    # Drop oldest frame if queue is full
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait((frame_id, frame, time.time()))
                        frame_id += 1
                    except asyncio.QueueEmpty:
                        pass
                
                # Control frame rate
                await asyncio.sleep(1.0 / self.config.target_fps)
        
        finally:
            cap.release()
    
    async def _process_frames(self):
        """Process frames for detection"""
        while self.is_running:
            try:
                # Get frame from queue
                frame_id, frame, capture_time = await asyncio.wait_for(
                    self.frame_queue.get(), timeout=1.0
                )
                
                # Process frame
                start_time = time.time()
                
                detection_frame = self.pipeline.process_frame(
                    frame, frame_id=frame_id, timestamp=datetime.now()
                )
                
                processing_time = time.time() - start_time
                
                # Update statistics
                self.frames_processed += 1
                self.humans_detected += detection_frame.total_humans
                self.processing_times.append(processing_time * 1000)  # ms
                
                # Calculate FPS
                if len(self.processing_times) > 30:
                    self.processing_times.pop(0)
                
                current_fps = 1.0 / processing_time if processing_time > 0 else 0
                self.fps_history.append(current_fps)
                if len(self.fps_history) > 30:
                    self.fps_history.pop(0)
                
                # Generate alerts if needed
                alerts = self._generate_alerts(detection_frame)
                
                # Add result to queue
                result = {
                    'stream_id': self.config.stream_id,
                    'detection_frame': detection_frame,
                    'processing_time_ms': processing_time * 1000,
                    'alerts': alerts,
                    'timestamp': datetime.now().isoformat()
                }
                
                try:
                    self.result_queue.put_nowait(result)
                except asyncio.QueueFull:
                    # Drop oldest result if queue is full
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait(result)
                    except asyncio.QueueEmpty:
                        pass
            
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
    
    def _generate_alerts(self, detection_frame: DetectionFrame) -> List[Alert]:
        """Generate alerts based on detection results"""
        if not self.config.alert_enabled:
            return []
        
        alerts = []
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_alert_time < self.config.alert_cooldown_seconds:
            return alerts
        
        # Human detected alert
        if detection_frame.total_humans > 0:
            high_confidence_humans = [
                h for h in detection_frame.humans 
                if h.confidence >= self.config.min_detection_confidence
            ]
            
            if high_confidence_humans:
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    stream_id=self.config.stream_id,
                    timestamp=detection_frame.timestamp,
                    alert_type="human_detected" if len(high_confidence_humans) == 1 else "multiple_humans",
                    confidence=max(h.confidence for h in high_confidence_humans),
                    human_count=len(high_confidence_humans),
                    frame_id=detection_frame.frame_id,
                    bounding_boxes=[{
                        'x': h.bbox[0], 'y': h.bbox[1],
                        'width': h.bbox[2], 'height': h.bbox[3],
                        'confidence': h.confidence,
                        'track_id': h.track_id
                    } for h in high_confidence_humans]
                )
                
                alerts.append(alert)
                self.alerts_generated += 1
                self.last_alert_time = current_time
        
        return alerts
    
    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        runtime = time.time() - self.start_time
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        
        return {
            'stream_id': self.config.stream_id,
            'runtime_seconds': runtime,
            'frames_processed': self.frames_processed,
            'humans_detected': self.humans_detected,
            'alerts_generated': self.alerts_generated,
            'average_fps': avg_fps,
            'average_processing_time_ms': avg_processing_time,
            'detection_rate': self.humans_detected / max(self.frames_processed, 1),
            'is_running': self.is_running
        }

class IntegrationService:
    """Main integration service"""
    
    def __init__(self, model_path: str = "models/yolo11s.pt"):
        self.model_path = model_path
        self.detection_pipeline = None
        self.stream_processors: Dict[str, StreamProcessor] = {}
        self.websocket_connections: List[WebSocket] = []
        
        # Initialize FastAPI app
        self.app = FastAPI(title="Foresight Integration Service", version="1.0.0")
        self._setup_routes()
        self._setup_middleware()
        
        logger.info("IntegrationService initialized")
    
    def _setup_middleware(self):
        """Setup CORS middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.on_event("startup")
        async def startup_event():
            await self._initialize_detection_pipeline()
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "active_streams": len(self.stream_processors),
                "websocket_connections": len(self.websocket_connections)
            }
        
        @self.app.post("/streams")
        async def create_stream(config: dict):
            """Create new stream processor"""
            try:
                stream_config = StreamConfig(**config)
                
                if stream_config.stream_id in self.stream_processors:
                    raise HTTPException(status_code=400, detail="Stream already exists")
                
                # Create stream processor
                processor = StreamProcessor(stream_config, self.detection_pipeline)
                self.stream_processors[stream_config.stream_id] = processor
                
                # Start processing
                asyncio.create_task(processor.start_processing())
                
                return {
                    "message": "Stream created successfully",
                    "stream_id": stream_config.stream_id
                }
            
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/streams/{stream_id}")
        async def delete_stream(stream_id: str):
            """Delete stream processor"""
            if stream_id not in self.stream_processors:
                raise HTTPException(status_code=404, detail="Stream not found")
            
            # Stop processing
            await self.stream_processors[stream_id].stop_processing()
            del self.stream_processors[stream_id]
            
            return {"message": "Stream deleted successfully"}
        
        @self.app.get("/streams")
        async def list_streams():
            """List all streams"""
            return {
                "streams": [
                    processor.get_statistics() 
                    for processor in self.stream_processors.values()
                ]
            }
        
        @self.app.get("/streams/{stream_id}/stats")
        async def get_stream_stats(stream_id: str):
            """Get stream statistics"""
            if stream_id not in self.stream_processors:
                raise HTTPException(status_code=404, detail="Stream not found")
            
            return self.stream_processors[stream_id].get_statistics()
        
        @self.app.websocket("/ws/detections")
        async def websocket_detections(websocket: WebSocket):
            """WebSocket endpoint for real-time detection results"""
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                # Send detection results to client
                await self._stream_detection_results(websocket)
            except WebSocketDisconnect:
                pass
            finally:
                if websocket in self.websocket_connections:
                    self.websocket_connections.remove(websocket)
        
        @self.app.websocket("/ws/alerts")
        async def websocket_alerts(websocket: WebSocket):
            """WebSocket endpoint for real-time alerts"""
            await websocket.accept()
            
            try:
                # Send alerts to client
                await self._stream_alerts(websocket)
            except WebSocketDisconnect:
                pass
    
    async def _initialize_detection_pipeline(self):
        """Initialize detection pipeline"""
        try:
            self.detection_pipeline = DetectionPipeline(
                model_path=self.model_path,
                human_only=True,
                confidence_threshold=0.5
            )
            logger.info("Detection pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize detection pipeline: {e}")
            raise
    
    async def _stream_detection_results(self, websocket: WebSocket):
        """Stream detection results to WebSocket client"""
        while True:
            try:
                # Collect results from all stream processors
                results = []
                
                for processor in self.stream_processors.values():
                    try:
                        result = processor.result_queue.get_nowait()
                        
                        # Convert DetectionFrame to dict for JSON serialization
                        detection_frame = result['detection_frame']
                        result['detection_frame'] = {
                            'frame_id': detection_frame.frame_id,
                            'timestamp': detection_frame.timestamp.isoformat(),
                            'total_humans': detection_frame.total_humans,
                            'processing_time_ms': detection_frame.processing_time_ms,
                            'frame_shape': detection_frame.frame_shape,
                            'humans': [{
                                'track_id': h.track_id,
                                'bbox': h.bbox,
                                'confidence': h.confidence,
                                'center': h.center,
                                'area': h.area,
                                'is_new': h.is_new,
                                'is_lost': h.is_lost
                            } for h in detection_frame.humans]
                        }
                        
                        results.append(result)
                    except asyncio.QueueEmpty:
                        continue
                
                if results:
                    await websocket.send_json({
                        'type': 'detection_results',
                        'data': results,
                        'timestamp': datetime.now().isoformat()
                    })
                
                await asyncio.sleep(0.1)  # 10 FPS update rate
                
            except Exception as e:
                logger.error(f"WebSocket streaming error: {e}")
                break
    
    async def _stream_alerts(self, websocket: WebSocket):
        """Stream alerts to WebSocket client"""
        while True:
            try:
                # Collect alerts from all stream processors
                alerts = []
                
                for processor in self.stream_processors.values():
                    try:
                        result = processor.result_queue.get_nowait()
                        if result.get('alerts'):
                            alerts.extend([alert.to_dict() for alert in result['alerts']])
                    except asyncio.QueueEmpty:
                        continue
                
                if alerts:
                    await websocket.send_json({
                        'type': 'alerts',
                        'data': alerts,
                        'timestamp': datetime.now().isoformat()
                    })
                
                await asyncio.sleep(1.0)  # 1 FPS update rate for alerts
                
            except Exception as e:
                logger.error(f"Alert streaming error: {e}")
                break
    
    def run(self, host: str = "0.0.0.0", port: int = 8004):
        """Run the integration service"""
        logger.info(f"Starting Integration Service on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)

def create_integration_service(model_path: str = "models/yolo11s.pt") -> IntegrationService:
    """Create integration service instance"""
    return IntegrationService(model_path)

async def demo_integration():
    """Demo function for integration service"""
    logger.info("Starting integration service demo")
    
    # Create service
    service = IntegrationService()
    
    # Initialize detection pipeline
    await service._initialize_detection_pipeline()
    
    # Create demo stream configuration
    demo_config = StreamConfig(
        stream_id="demo_stream",
        source_url=0,  # Webcam
        detection_enabled=True,
        tracking_enabled=True,
        alert_enabled=True,
        confidence_threshold=0.5
    )
    
    # Create and start stream processor
    processor = StreamProcessor(demo_config, service.detection_pipeline)
    
    try:
        # Run for 30 seconds
        await asyncio.wait_for(processor.start_processing(), timeout=30.0)
    except asyncio.TimeoutError:
        logger.info("Demo completed")
    finally:
        await processor.stop_processing()
        
        # Print statistics
        stats = processor.get_statistics()
        logger.info(f"Demo results: {stats}")

if __name__ == "__main__":
    import argparse
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Foresight Integration Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8004, help="Port number")
    parser.add_argument("--model", default="models/yolo11s.pt", help="YOLO model path")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    
    args = parser.parse_args()
    
    if args.demo:
        asyncio.run(demo_integration())
    else:
        service = IntegrationService(args.model)
        service.run(args.host, args.port)