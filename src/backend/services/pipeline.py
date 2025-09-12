from __future__ import annotations
import threading, time, os
from typing import List, Dict, Optional
import cv2
import numpy as np

class SarPipeline:
    """
    Simple background video pipeline:
      - reads frames from webcam (0) or UDP (e.g. udp://127.0.0.1:5555)
      - optional YOLO (if ultralytics/torch available) else mock detections
      - face blur when sar_blur=True
      - exposes latest annotated JPEG and rolling stats
    """
    def __init__(self, source: Optional[str] = None):
        self.source = source or os.environ.get("FORESIGHT_SOURCE", "0")  # "0" -> webcam
        self.running = False
        self.thread: Optional[threading.Thread] = None

        self._cap = None
        self._lock = threading.Lock()
        self._last_jpeg: Optional[bytes] = None
        self._fps = 0.0
        self._latency_ms = 0
        self._geo_error_m = 2.5
        self._detections: List[Dict] = []

        # mode state
        self.mode = "sar"     # "sar" or "suspect"
        self.sar_blur = True  # blur faces in SAR mode

        # optional YOLO
        self._yolo = None
        try:
            from ultralytics import YOLO
            model_path = os.environ.get("FORESIGHT_YOLO", "yolo11n.pt")
            self._yolo = YOLO(model_path)
        except Exception:
            self._yolo = None  # ok: we’ll simulate detections

        # OpenCV frontal face detector
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            self._face_cascade = None

    # ---------- public API ----------
    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass

    def set_mode(self, mode: str):
        self.mode = "sar" if mode.lower() == "sar" else "suspect"

    def set_blur(self, enabled: bool):
        self.sar_blur = bool(enabled)

    def snapshot_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._last_jpeg

    def stats(self) -> Dict:
        with self._lock:
            return {
                "fps": round(self._fps, 1),
                "latency": int(self._latency_ms),
                "geo_error": float(self._geo_error_m),
                "detections": list(self._detections),
                "mode": self.mode,
                "blur": self.sar_blur,
            }

    # ---------- internals ----------
    def _open_capture(self):
        src = self.source
        if src == "0" or src.isdigit():
            self._cap = cv2.VideoCapture(int(src))
        else:
            # UDP or file
            self._cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if not self._cap or not self._cap.isOpened():
            # fallback to a synthetic generator if no camera/stream is found
            self._cap = None

    def _synthesize_frame(self, t):
        # gray background + moving box (so UI shows *something*)
        img = np.full((360, 640, 3), 36, dtype=np.uint8)
        x = int((np.sin(t) * 0.4 + 0.5) * (640 - 100))
        cv2.rectangle(img, (x, 120), (x+100, 220), (0, 160, 255), 2)
        cv2.putText(img, "Synthetic feed", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
        return img

    def _maybe_yolo(self, frame):
        dets = []
        if self._yolo is None:
            # mock: one moving "person" box
            h, w = frame.shape[:2]
            t = time.time()
            x = int((np.cos(t) * 0.4 + 0.5) * (w - 120))
            y = int((np.sin(t) * 0.2 + 0.5) * (h - 200))
            dets.append({"name": "person", "conf": 0.76, "xyxy": [x, y, x+120, y+200], "id": 1})
            return dets

        # real YOLO
        try:
            results = self._yolo.predict(frame, verbose=False)
            for r in results:
                for b in r.boxes:
                    xyxy = b.xyxy.cpu().numpy().astype(int)[0].tolist()
                    name = self._yolo.names.get(int(b.cls[0]), "obj")
                    dets.append({"name": name, "conf": float(b.conf[0]), "xyxy": xyxy})
        except Exception:
            pass
        return dets

    def _apply_face_blur(self, frame):
        if not self.sar_blur or self._face_cascade is None:
            return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0: 
                continue
            roi = cv2.GaussianBlur(roi, (31, 31), 0)
            frame[y:y+h, x:x+w] = roi
        return frame

    def _annotate(self, frame, dets):
        for d in dets:
            if "xyxy" in d:
                x1, y1, x2, y2 = map(int, d["xyxy"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (37, 140, 255), 2)
                label = f"{d.get('name','obj')} {int(d.get('conf',0)*100)}%"
                cv2.putText(frame, label, (x1, max(20, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 2, cv2.LINE_AA)
        return frame

    def _encode_jpeg(self, frame) -> Optional[bytes]:
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return buf.tobytes() if ok else None

    def _loop(self):
        self._open_capture()
        last = time.time()
        frame = None

        while self.running:
            t0 = time.time()

            if self._cap is not None:
                ok, frame = self._cap.read()
                if not ok:
                    # stream hiccup: brief sleep and try again
                    time.sleep(0.01)
                    continue
            else:
                frame = self._synthesize_frame(time.time())

            dets = self._maybe_yolo(frame)
            frame = self._apply_face_blur(frame)
            frame = self._annotate(frame, dets)

            jpeg = self._encode_jpeg(frame)

            now = time.time()
            dt = now - last
            last = now
            fps = 1.0 / dt if dt > 0 else 0

            with self._lock:
                self._last_jpeg = jpeg
                self._fps = 0.9*self._fps + 0.1*fps if self._fps > 0 else fps
                self._latency_ms = int((time.time() - t0) * 1000)
                self._detections = dets

            # keep CPU reasonable
            time.sleep(0.01)
