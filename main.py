from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import pytesseract
import loguru
from PIL import ImageGrab
import time
import mss
import threading
from typing import Optional
from pydantic import BaseModel

# Ultralytics YOLO (pip install ultralytics)
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    print("Ultralytics not available:", e)

RTSP_URL = "rtsp://127.0.0.1:8554/scrcpy"

# -----------------------------
# App + CORS setup
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all origins (frontend on :5173 can connect)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = loguru.logger

# -----------------------------
# Shared state
# -----------------------------
state = {
    "heading": 0,
    "home": {"lat": 6.1164, "lon": 125.1716}
}

# -----------------------------
# YOLO Detector Class
# -----------------------------
class ToggleReq(BaseModel):
    enabled: bool

class Detector:
    def __init__(self, rtsp: str):
        self.rtsp = rtsp
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.sar_enabled = True
        self.lock_enabled = False
        self.last_jpeg = None
        self.lock = threading.Lock()
        self.model = YOLO("yolov8n.pt") if YOLO else None

    def annotate(self, frame, results):
        # draw simple boxes & labels
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for b in boxes:
                c = int(b.cls)
                conf = float(b.conf)
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                label = f"{self.model.names.get(c,'?')} {conf:.2f}"
                # If suspect lock is on, you might filter class/person etc. For now, draw all.
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, max(y1-6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        return frame

    def loop(self):
        self.cap = cv2.VideoCapture(self.rtsp, cv2.CAP_FFMPEG)
        # small open retry loop
        open_deadline = time.time() + 10
        while self.cap is None or not self.cap.isOpened():
            if time.time() > open_deadline:
                print("Failed to open RTSP within 10s")
                break
            print("Waiting for RTSP...")
            time.sleep(1)
            self.cap = cv2.VideoCapture(self.rtsp, cv2.CAP_FFMPEG)

        while self.running and self.cap and self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            # If SAR disabled → passthrough only
            if self.sar_enabled and self.model is not None:
                results = self.model.predict(source=frame, imgsz=640, conf=0.25, verbose=False)
                frame = self.annotate(frame, results)
                # (Optional) apply "suspect lock" heuristic here

            # encode to jpeg for MJPEG output
            ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                with self.lock:
                    self.last_jpeg = jpeg.tobytes()
            else:
                time.sleep(0.01)

        if self.cap:
            self.cap.release()

    def start(self):
        if self.running:
            return
        self.running = True
        t = threading.Thread(target=self.loop, daemon=True)
        t.start()

    def stop(self):
        self.running = False

det = Detector(RTSP_URL)

@app.on_event("startup")
def on_start():
    det.start()

@app.on_event("shutdown")
def on_stop():
    det.stop()

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def root():
    return {"status": "server is running", "message": "Welcome to foresight!"}

# -----------------------------
# YOLO Toggle Routes
# -----------------------------
@app.get("/state")
def get_state():
    return {"sar": det.sar_enabled, "lock": det.lock_enabled}

@app.post("/toggle/sar")
def toggle_sar(req: ToggleReq):
    det.sar_enabled = bool(req.enabled)
    return {"sar": det.sar_enabled}

@app.post("/toggle/lock")
def toggle_lock(req: ToggleReq):
    det.lock_enabled = bool(req.enabled)
    return {"lock": det.lock_enabled}

@app.get("/video.mjpg")
def video_mjpeg():
    boundary = "frame"
    def gen():
        while True:
            with det.lock:
                jpeg = det.last_jpeg
            if jpeg is not None:
                yield (b"--" + boundary.encode() + b"\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                       + jpeg + b"\r\n")
            else:
                time.sleep(0.02)
    return StreamingResponse(gen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")

# -----------------------------
# Your existing OCR routes
# -----------------------------
@app.get("/capture")
def capture_screen():
    try:
        img = ImageGrab.grab()  # grab full screen
        frame = np.array(img)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        text = pytesseract.image_to_string(gray)
        return {"captured_text": text.strip()}
    except Exception as e:
        logger.error(f"Error during capture: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/camera")
def capture_camera():
    try:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return JSONResponse(status_code=500, content={"error": "Camera not available"})

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        return {"camera_text": text.strip()}

    except Exception as e:
        logger.error(f"Camera error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/geolocate")
def geolocate_example():
    return {
        "lat": state["home"]["lat"],
        "lon": state["home"]["lon"],
        "confidence": "placeholder"
    }

@app.post("/heading")
async def set_heading(req: Request):
    data = await req.json()
    state["heading"] = data.get("heading", 0)
    return {"ok": True, "heading": state["heading"]}

@app.post("/home")
async def set_home(req: Request):
    data = await req.json()
    state["home"] = {"lat": data.get("lat", 0), "lon": data.get("lon", 0)}
    return {"ok": True, "home": state["home"]}

@app.get("/mjpg")
def mjpg_stream():
    def generate():
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # capture primary monitor
            while True:
                img = np.array(sct.grab(monitor))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                ret, jpeg = cv2.imencode(".jpg", frame)
                if not ret:
                    continue

                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
                time.sleep(0.1)

    print("🚀 DESKTOP CAPTURE VERSION LOADED")
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")