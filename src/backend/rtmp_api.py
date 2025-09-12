"""RTMP API endpoints for DJI Fly app integration.

Provides REST API endpoints for:
- RTMP server configuration
- Stream status monitoring
- DJI Fly app setup instructions
- Stream quality management
"""

import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from connection.rtmp_stream import RTMPConnection
from connection.phone_stream import PhoneStreamConnection

# Configure logging
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/api/rtmp", tags=["RTMP"])

# Global RTMP connection instance
rtmp_connection: Optional[RTMPConnection] = None
phone_connection: Optional[PhoneStreamConnection] = None


class RTMPConfig(BaseModel):
    """RTMP configuration model."""
    port: int = 1935
    app_name: str = "live"
    stream_key: str = "dji_stream"
    target_fps: int = 30
    target_width: int = 1920
    target_height: int = 1080


class StreamStatus(BaseModel):
    """Stream status model."""
    connected: bool
    stream_active: bool
    rtmp_url: str
    frame_count: int = 0
    last_frame_time: Optional[float] = None
    error_message: Optional[str] = None


@router.get("/status")
async def get_rtmp_status() -> StreamStatus:
    """Get current RTMP stream status."""
    global rtmp_connection, phone_connection
    
    if not rtmp_connection and not phone_connection:
        return StreamStatus(
            connected=False,
            stream_active=False,
            rtmp_url="",
            error_message="RTMP service not initialized"
        )
    
    # Check phone connection with RTMP
    if phone_connection and phone_connection.video_source == 'rtmp':
        conn = phone_connection.rtmp_connection
    else:
        conn = rtmp_connection
    
    if not conn:
        return StreamStatus(
            connected=False,
            stream_active=False,
            rtmp_url="",
            error_message="RTMP connection not available"
        )
    
    return StreamStatus(
        connected=conn._connected,
        stream_active=conn.cap is not None and conn.cap.isOpened() if hasattr(conn, 'cap') else False,
        rtmp_url=conn.rtmp_url,
        frame_count=conn._frame_count,
        last_frame_time=getattr(conn, '_last_frame_time', None)
    )


@router.post("/start")
async def start_rtmp_server(config: RTMPConfig, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Start RTMP server with specified configuration."""
    global rtmp_connection, phone_connection
    
    try:
        # Create phone connection with RTMP configuration
        rtmp_config = {
            'video_source': 'rtmp',
            'rtmp_port': config.port,
            'rtmp_app': config.app_name,
            'rtmp_stream_key': config.stream_key,
            'target_fps': config.target_fps,
            'target_width': config.target_width,
            'target_height': config.target_height
        }
        
        phone_connection = PhoneStreamConnection(rtmp_config)
        
        # Start the connection
        success = await phone_connection.connect()
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to start RTMP server")
        
        # Get stream info for response
        stream_info = phone_connection.get_stream_info()
        
        logger.info(f"RTMP server started on port {config.port}")
        
        return {
            "status": "success",
            "message": "RTMP server started successfully",
            "stream_info": stream_info
        }
        
    except Exception as e:
        logger.error(f"Failed to start RTMP server: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start RTMP server: {str(e)}")


@router.post("/stop")
async def stop_rtmp_server() -> Dict[str, str]:
    """Stop RTMP server."""
    global rtmp_connection, phone_connection
    
    try:
        if phone_connection:
            await phone_connection.disconnect()
            phone_connection = None
        
        if rtmp_connection:
            await rtmp_connection.disconnect()
            rtmp_connection = None
        
        logger.info("RTMP server stopped")
        
        return {
            "status": "success",
            "message": "RTMP server stopped successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to stop RTMP server: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop RTMP server: {str(e)}")


@router.get("/config")
async def get_rtmp_config() -> Dict[str, Any]:
    """Get current RTMP configuration."""
    try:
        # Load configuration from file
        config_path = Path("config/dji_config.json")
        
        if not config_path.exists():
            raise HTTPException(status_code=404, detail="Configuration file not found")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        rtmp_config = config.get('dji_connection', {}).get('rtmp_server', {})
        phone_bridge_config = config.get('dji_connection', {}).get('phone_bridge', {})
        
        return {
            "rtmp_server": rtmp_config,
            "phone_bridge": phone_bridge_config
        }
        
    except Exception as e:
        logger.error(f"Failed to load RTMP configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load configuration: {str(e)}")


@router.get("/setup-instructions")
async def get_setup_instructions() -> Dict[str, Any]:
    """Get DJI Fly app setup instructions."""
    global phone_connection
    
    # Get current stream info
    stream_info = {}
    if phone_connection and phone_connection.video_source == 'rtmp':
        stream_info = phone_connection.get_stream_info()
    else:
        # Default configuration
        stream_info = {
            'rtmp_url': 'rtmp://localhost:1935/live/dji_stream',
            'rtmp_port': 1935,
            'rtmp_app': 'live',
            'stream_key': 'dji_stream'
        }
    
    return {
        "title": "DJI Fly App RTMP Setup",
        "description": "Configure DJI Fly app to stream video to Foresight via RTMP",
        "requirements": [
            "DJI drone with DJI Fly app support",
            "Smartphone with DJI Fly app installed",
            "Wi-Fi connection between phone and PC",
            "Foresight application running on PC"
        ],
        "steps": [
            {
                "step": 1,
                "title": "Connect Drone",
                "description": "Connect your DJI drone to the smartphone via DJI RC or DJI Goggles",
                "details": [
                    "Power on your DJI drone",
                    "Connect smartphone to DJI RC or DJI Goggles",
                    "Open DJI Fly app",
                    "Verify drone connection"
                ]
            },
            {
                "step": 2,
                "title": "Configure Live Streaming",
                "description": "Set up custom RTMP streaming in DJI Fly app",
                "details": [
                    "In DJI Fly app, go to Settings",
                    "Navigate to 'Live Streaming' or 'Transmission'",
                    "Select 'Custom RTMP'",
                    f"Enter RTMP URL: {stream_info.get('rtmp_url', 'rtmp://localhost:1935/live/dji_stream')}",
                    "Save the configuration"
                ]
            },
            {
                "step": 3,
                "title": "Start Foresight RTMP Server",
                "description": "Ensure Foresight is ready to receive the stream",
                "details": [
                    "Open Foresight application",
                    "Go to Connection Settings",
                    "Select 'Phone Bridge (RTMP)'",
                    "Click 'Start RTMP Server'",
                    "Wait for 'Server Ready' status"
                ]
            },
            {
                "step": 4,
                "title": "Start Streaming",
                "description": "Begin live streaming from DJI Fly app",
                "details": [
                    "In DJI Fly app, start live streaming",
                    "Select the custom RTMP configuration",
                    "Verify stream starts successfully",
                    "Check Foresight for incoming video"
                ]
            }
        ],
        "troubleshooting": [
            {
                "issue": "Connection Failed",
                "solutions": [
                    "Check Wi-Fi connectivity between phone and PC",
                    "Verify RTMP server is running in Foresight",
                    "Ensure firewall allows port 1935",
                    "Try restarting both DJI Fly app and Foresight"
                ]
            },
            {
                "issue": "Poor Video Quality",
                "solutions": [
                    "Check Wi-Fi signal strength",
                    "Reduce video resolution in DJI Fly app",
                    "Move closer to Wi-Fi router",
                    "Close other apps using network bandwidth"
                ]
            },
            {
                "issue": "Stream Drops",
                "solutions": [
                    "Ensure stable Wi-Fi connection",
                    "Check for interference from other devices",
                    "Verify sufficient battery on smartphone",
                    "Restart RTMP server if needed"
                ]
            }
        ],
        "stream_info": stream_info,
        "network_requirements": {
            "minimum_bandwidth": "2 Mbps upload",
            "recommended_bandwidth": "5+ Mbps upload",
            "latency": "< 100ms preferred",
            "ports": [1935, 8554]
        }
    }


@router.get("/test-connection")
async def test_rtmp_connection() -> Dict[str, Any]:
    """Test RTMP server connectivity."""
    global phone_connection
    
    try:
        if not phone_connection or phone_connection.video_source != 'rtmp':
            return {
                "status": "error",
                "message": "RTMP server not running",
                "connected": False
            }
        
        rtmp_conn = phone_connection.rtmp_connection
        if not rtmp_conn:
            return {
                "status": "error",
                "message": "RTMP connection not initialized",
                "connected": False
            }
        
        # Test if we can wait for a stream (with short timeout)
        stream_available = await rtmp_conn.wait_for_stream(timeout=5)
        
        return {
            "status": "success" if stream_available else "waiting",
            "message": "Stream detected" if stream_available else "Waiting for DJI Fly app stream",
            "connected": rtmp_conn._connected,
            "stream_available": stream_available,
            "rtmp_url": rtmp_conn.rtmp_url
        }
        
    except Exception as e:
        logger.error(f"RTMP connection test failed: {e}")
        return {
            "status": "error",
            "message": f"Connection test failed: {str(e)}",
            "connected": False
        }