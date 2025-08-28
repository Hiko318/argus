from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio, json, time
from typing import List

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

clients: List[WebSocket] = []

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    try:
        while True:
            # keep alive receive (optional)
            await asyncio.sleep(0.1)
    except Exception:
        pass
    finally:
        clients.remove(ws)

async def broadcast(obj):
    dead = []
    for c in clients:
        try:
            await c.send_text(json.dumps(obj))
        except Exception:
            dead.append(c)
    for d in dead:
        if d in clients: 
            clients.remove(d)

# Demo publisher loop (launch with uvicorn below)
async def demo_stream():
    t0 = time.time()
    while True:
        t = time.time() - t0
        payload = {
            "type": "tick",
            "time": round(t, 2),
            "telemetry": { "lat": 14.5995, "lon": 120.9842, "alt": 30 + 2.0 },
            "detections": [
                { "id": 1, "cls": "person", "conf": 0.91,
                  "bbox": [0.4,0.3,0.2,0.35], "geo": {"lat":14.5996,"lon":120.9843, "err_m": 8.5} }
            ]
        }
        await broadcast(payload)
        await asyncio.sleep(0.5)

@app.on_event("startup")
async def on_start():
    asyncio.create_task(demo_stream())
