import asyncio, json, time, cv2, requests
import websockets
from loguru import logger

from src.ingest.capture_screen import ScreenCapture
from src.util.hud_ocr import read_hud
from src.geo.approx_pos import dest_from_bearing
from src.detect.yolo_infer import YoloDetector
from src.track.iou_tracker import IoUTracker

WS_URL = "ws://localhost:8000/ws"
PUSH_URL = "http://localhost:8000/push_frame"

async def main():
    # Connect WS for telemetry/detections
    ws = await websockets.connect(WS_URL)
    cap = ScreenCapture(title="DJI_MIRROR", target_fps=12)
    det = YoloDetector(conf=0.25)
    trk = IoUTracker()

    # Day-1 defaults
    home_lat, home_lon = 14.5995, 120.9842  # Manila
    heading_deg = 0.0
    D = 0.0; H = 30.0

    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok: break

        # HUD OCR (stub): returns None, we keep defaults
        hud = read_hud(frame)
        if hud.get("D") is not None: D = float(hud["D"])
        if hud.get("H") is not None: H = float(hud["H"])

        # Approximate drone position (home + distance along heading)
        drone_lat, drone_lon = dest_from_bearing(home_lat, home_lon, float(D), heading_deg)

        # Detect + track (stub)
        dets = det.infer(frame)
        dets_tr = trk.update(dets)

        # Annotate frame for big video
        vis = frame.copy()
        cv2.putText(vis, f"D={D:.1f}m H={H:.1f}m Heading={heading_deg:.1f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
        for d in dets_tr:
            x,y,w,h = d["bbox"]
            cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(vis, f"{d['cls']} {d['conf']:.2f}", (x, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Push frame (JPEG) to server for /mjpg
        ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            try:
                requests.post(PUSH_URL, data=buf.tobytes(), headers={"Content-Type":"image/jpeg"}, timeout=1.0)
            except Exception as e:
                logger.warning(f"push_frame failed: {e}")

        # Prepare detections for map pins (for Day-1, pin at drone pos)
        dets_out = []
        for d in dets_tr:
            dets_out.append({
                "id": d["id"],
                "cls": d["cls"],
                "conf": d["conf"],
                "bbox": [d["bbox"][0]/frame.shape[1], d["bbox"][1]/frame.shape[0],
                         d["bbox"][2]/frame.shape[1], d["bbox"][3]/frame.shape[0]],
                "geo": {"lat": drone_lat, "lon": drone_lon, "err_m": max(10.0, 0.3*D + 0.2*H)}
            })

        payload = {
            "type":"tick",
            "time": round(time.time()-t0,2),
            "telemetry": {"lat": drone_lat, "lon": drone_lon, "alt": H},
            "detections": dets_out
        }
        try:
            await ws.send(json.dumps(payload))
        except Exception as e:
            logger.error(f"WS send error: {e}")
            break

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    try:
        await ws.close()
    except:
        pass

if __name__=="__main__":
    asyncio.run(main())
