import asyncio, json, os, subprocess, time
from typing import List, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from loguru import logger
import cv2
import numpy as np
from ultralytics import YOLO
from math import radians
from pyproj import Geod

# ---------- Config ----------
SCRCPY_WIN_TITLE = os.environ.get("FS_WINDOW_TITLE", "FORESIGHT_FEED")
FFMPEG = os.environ.get("FS_FFMPEG", "ffmpeg")
CAP_FPS = int(os.environ.get("FS_CAPTURE_FPS", "10"))
MODEL_PATH = os.environ.get("FS_MODEL", "yolov8n.pt")  # swap to your own
MISSION_ID = os.environ.get("FS_MISSION", "SAR-2025-001")
SAVE_DIR = os.path.join("missions", MISSION_ID)
os.makedirs(SAVE_DIR, exist_ok=True)

# Manual/telemetry (stub; replace with real DJI telemetry feed later)
DRONE_LAT, DRONE_LON = 6.1164, 125.1716  # GenSan approx
DRONE_ALT_M = 35.0
CAM_FOV_DEG = 78.0   # phone-ish FOV
GIMBAL_PITCH_DEG = 60.0
YAW_DEG = 0.0        # north

geod = Geod(ellps="WGS84")

app = FastAPI() 

app.mount("/public", StaticFiles(directory="public"), name="public")

latest_frame = None
latest_annotated = None
latest_stats: Dict[str, Any] = {"fps": 0, "latency_ms": 0, "geo_error_m": 6.0}
clients: List[WebSocket] = []

# ---------- Simple IOU tracker ----------
class IOUTracker:
    def __init__(self, iou_thr=0.4, max_age=20):
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.tracks = {}  # id -> dict
        self._next_id = 1

    @staticmethod
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0
        inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / max(area_a + area_b - inter, 1e-6)

    def update(self, boxes):
        # boxes: list of (x1,y1,x2,y2,conf,cls)
        assigned = set()
        # match existing
        for tid, tr in list(self.tracks.items()):
            tr["age"] += 1
            best_iou, best_j = 0.0, -1
            for j, b in enumerate(boxes):
                if j in assigned:
                    continue
                i = self.iou(tr["bbox"], b[:4])
                if i > best_iou:
                    best_iou, best_j = i, j
            if best_iou >= self.iou_thr and best_j != -1:
                b = boxes[best_j]
                tr["bbox"] = b[:4]
                tr["conf"] = float(b[4])
                tr["cls"] = int(b[5])
                tr["age"] = 0
                assigned.add(best_j)

        # new tracks
        for j, b in enumerate(boxes):
            if j in assigned:
                continue
            self.tracks[self._next_id] = {
                "id": self._next_id,
                "bbox": b[:4],
                "conf": float(b[4]),
                "cls": int(b[5]),
                "age": 0,
            }
            self._next_id += 1

        # prune old
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]["age"] > self.max_age:
                del self.tracks[tid]

        return list(self.tracks.values())

tracker = IOUTracker()

# ---------- Geolocation (very simple nadir-ish) ----------
def estimate_ground_point(cx_norm, cy_norm, width, height) -> Dict[str, float]:
    # cx, cy in [0,1]; crude ray→ground-plane intersection using FOV + pitch + altitude
    pitch = radians(GIMBAL_PITCH_DEG)
    fov = radians(CAM_FOV_DEG)
    alt = DRONE_ALT_M
    if pitch < 1e-3:
        pitch = radians(1.0)

    forward_m = alt / np.tan(pitch)
    half_fov = fov / 2.0
    lateral_span = 2.0 * forward_m * np.tan(half_fov)

    dx = (cx_norm - 0.5) * lateral_span
    dy = forward_m * (1 - cy_norm)  # crude

    # rotate by yaw
    theta = radians(YAW_DEG)
    x_east = dx * np.cos(theta) - dy * np.sin(theta)
    y_north = dx * np.sin(theta) + dy * np.cos(theta)

    # meters→lat/lon
    az = (np.degrees(np.arctan2(x_east, y_north)) + 360) % 360
    dist = float(np.hypot(x_east, y_north))
    lon2, lat2, _ = geod.fwd(DRONE_LON, DRONE_LAT, az, dist)
    return {"lat": lat2, "lon": lon2, "alt": DRONE_ALT_M, "error_m": 8.0}

# ---------- FFmpeg reader ----------
def spawn_ffmpeg():
    cmd = [
        FFMPEG,
        "-f", "gdigrab", "-framerate", str(CAP_FPS),
        "-i", f"title={SCRCPY_WIN_TITLE}",
        "-vf", "scale=960:-1",
        "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "3", "-"
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)

def read_mjpeg_stream(proc):
    global latest_frame
    data = b""
    SOI, EOI = b"\xff\xd8", b"\xff\xd9"
    while True:
        chunk = proc.stdout.read(4096)
        if not chunk:
            break
        data += chunk
        while True:
            i = data.find(SOI)
            j = data.find(EOI, i + 2)
            if i != -1 and j != -1:
                jpg = data[i : j + 2]
                data = data[j + 2 :]
                img_array = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                latest_frame = frame
            else:
                break

# ---------- YOLO loop ----------
model = YOLO(MODEL_PATH)

def annotate_and_detect(frame):
    h, w = frame.shape[:2]
    t0 = time.time()
    res = model.predict(frame, imgsz=640, conf=0.3, verbose=False)[0]
    boxes = []
    for b in res.boxes:
        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
        conf = float(b.conf[0])
        cls = int(b.cls[0])
        boxes.append((x1, y1, x2, y2, conf, cls))

    tracks = tracker.update(boxes)

    # draw + geo
    dets = []
    for tr in tracks:
        x1, y1, x2, y2 = tr["bbox"]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        geo = estimate_ground_point(cx / w, cy / h, w, h)

        color = (16, 185, 129)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        name = res.names.get(tr["cls"], "obj") if hasattr(res, "names") else "obj"
        label = f"id{tr['id']} {name} {tr['conf']:.2f}"
        cv2.putText(
            frame, label, (int(x1), int(y1) - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )

        dets.append({
            "id": tr["id"],
            "cls": int(tr["cls"]),
            "name": name,
            "conf": float(tr["conf"]),
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "lat": geo["lat"], "lon": geo["lon"], "alt": geo["alt"], "error": geo["error_m"],
        })

    latency = int((time.time() - t0) * 1000)
    latest_stats.update({
        "fps": int(1000 / max(latency, 1)),
        "latency_ms": latency,
        "geo_error_m": float(np.mean([d["error"] for d in dets])) if dets else 6.0
    })
    return frame, dets

async def ws_broadcast(payload: Dict[str, Any]):
    if not clients:
        return
    msg = json.dumps(payload)
    dead = []
    for ws in clients:
        try:
            await ws.send_text(msg)
        except:
            dead.append(ws)
    for d in dead:
        try:
            clients.remove(d)
        except:
            pass

async def pipeline_loop():
    logger.info("Starting FFmpeg and pipeline...")
    proc = spawn_ffmpeg()
    reader = asyncio.to_thread(read_mjpeg_stream, proc)
    asyncio.create_task(reader)

    last_push = 0.0
    while True:
        await asyncio.sleep(0.001)
        if latest_frame is None:
            continue
        frame = latest_frame.copy()
        annotated, dets = annotate_and_detect(frame)

        global latest_annotated
        latest_annotated = annotated

        now = time.time()
        if now - last_push > 0.2:  # 5 Hz metadata
            last_push = now
            await ws_broadcast({
                "type": "stats",
                "fps": latest_stats["fps"],
                "latency": latest_stats["latency_ms"],
                "geo_error": latest_stats["geo_error_m"],
                "detections": dets
            })

bg_task_started = False

@app.on_event("startup")
async def on_start():
    # Start only when first WS connects (saves CPU)
    pass

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    global bg_task_started
    await ws.accept()
    clients.append(ws)
    if not bg_task_started:
        asyncio.create_task(pipeline_loop())
        bg_task_started = True
    try:
        await ws.send_text(json.dumps({"type": "hello", "mission": MISSION_ID}))
        while True:
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        try:
            clients.remove(ws)
        except:
            pass

@app.get("/")
def landing():
    return FileResponse("public/foresight-webapp.html")

@app.get("/frame.jpg")
def frame_jpg():
    if latest_annotated is None:
        return Response(status_code=204)
    ok, buf = cv2.imencode(".jpg", latest_annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return Response(status_code=500)
    return Response(content=buf.tobytes(), media_type="image/jpeg")
