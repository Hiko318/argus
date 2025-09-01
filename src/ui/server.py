from fastapi import FastAPI, WebSocket, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import asyncio
import json
import time
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List
import threading

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
clients: List[WebSocket] = []
latest_frame = None
frame_lock = threading.Lock()
video_thread_started = False
model = None

print("Starting Foresight backend...")

@app.get("/")
def root():
    return {"message": "Foresight backend is running!"}

@app.get("/status")
def get_status():
    global video_thread_started, latest_frame, model
    return {
        "server": "running",
        "video_thread_started": video_thread_started,
        "has_frame": latest_frame is not None,
        "model_loaded": model is not None,
        "timestamp": time.time()
    }

def create_test_frame():
    """Create a test frame to show system is working"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (50, 100, 50)  # Green background
    
    # Add text
    cv2.putText(frame, "Foresight Live Feed", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.putText(frame, f"Time: {time.strftime('%H:%M:%S')}", (50, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "Waiting for video...", (50, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    # Animated rectangle
    x = int((time.time() * 100) % 590)
    cv2.rectangle(frame, (x, 300), (x + 50, 350), (0, 255, 255), -1)
    
    return frame

def video_processor():
    """Background thread to process video frames"""
    global latest_frame, model
    print("Video processor thread started")
    
    # Load YOLO model
    try:
        MODEL_PATH = "yolov8n.pt"
        model = YOLO(MODEL_PATH)
        print(f"YOLO model loaded: {MODEL_PATH}")
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        model = None
    
    # Try to connect to video source
    VIDEO_SOURCE = "rtsp://127.0.0.1:8554/live.sdp"
    cap = None
    video_connected = False
    
    try:
        print(f"Attempting to connect to: {VIDEO_SOURCE}")
        cap = cv2.VideoCapture(VIDEO_SOURCE)
        if cap.isOpened():
            print("Video source connected!")
            video_connected = True
        else:
            print("Video source not available, using test frames")
    except Exception as e:
        print(f"Video connection error: {e}")
    
    frame_count = 0
    
    while True:
        try:
            frame = None
            
            # Try to get frame from video source
            if cap and cap.isOpened() and video_connected:
                ret, frame = cap.read()
                if ret:
                    frame_count += 1
                    if frame_count % 60 == 0:  # Log every 60 frames
                        print(f"Processed {frame_count} video frames")
                else:
                    print("Lost video connection, switching to test frames")
                    video_connected = False
            
            # Use test frame if no video
            if frame is None:
                frame = create_test_frame()
            
            # Apply YOLO if available and we have a real video frame
            if model is not None and video_connected and frame is not None:
                try:
                    results = model(frame)
                    frame = results[0].plot()
                    
                    # Get detections for broadcasting
                    dets = []
                    for box in results[0].boxes.data.tolist():
                        x1, y1, x2, y2, conf, cls = box
                        dets.append({
                            "cls": model.names[int(cls)],
                            "conf": float(conf),
                            "bbox": [float(x1), float(y1), float(x2), float(y2)]
                        })
                    
                except Exception as e:
                    print(f"YOLO processing error: {e}")
            
            # Store frame safely
            with frame_lock:
                latest_frame = frame.copy()
            
            time.sleep(1/30)  # 30 FPS
            
        except Exception as e:
            print(f"Video processor error: {e}")
            time.sleep(1)

def get_current_frame():
    """Safely get the current frame"""
    with frame_lock:
        if latest_frame is not None:
            return latest_frame.copy()
    return create_test_frame()

@app.get("/preview")
def get_preview(t: int = 1):
    """Get a single frame (what your frontend is requesting)"""
    global video_thread_started
    
    # Start video processing thread on first request
    if not video_thread_started:
        thread = threading.Thread(target=video_processor, daemon=True)
        thread.start()
        video_thread_started = True
        print("Started video processing thread")
        # Give it a moment to initialize
        time.sleep(0.1)
    
    try:
        frame = get_current_frame()
        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        if not ret:
            return Response(status_code=500, content="Failed to encode frame")
        
        return Response(content=jpeg.tobytes(), media_type="image/jpeg")
    
    except Exception as e:
        print(f"Preview endpoint error: {e}")
        return Response(status_code=500, content=str(e))

@app.get("/video_feed")
def video_feed():
    """Streaming video endpoint"""
    def generate():
        while True:
            try:
                frame = get_current_frame()
                ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                
                if ret:
                    frame_bytes = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(1/30)
                
            except Exception as e:
                print(f"Video feed error: {e}")
                time.sleep(1)
    
    return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    print(f"WebSocket client connected. Total clients: {len(clients)}")
    try:
        while True:
            await asyncio.sleep(0.1)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if ws in clients:
            clients.remove(ws)
        print(f"WebSocket client disconnected. Total clients: {len(clients)}")

async def broadcast(obj):
    """Broadcast message to all connected WebSocket clients"""
    if not clients:
        return
    
    dead = []
    for client in clients:
        try:
            await client.send_text(json.dumps(obj))
        except Exception:
            dead.append(client)
    
    # Remove dead connections
    for dead_client in dead:
        if dead_client in clients:
            clients.remove(dead_client)
