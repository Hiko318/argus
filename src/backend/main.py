from fastapi import FastAPI
from fastapi.responses import RedirectResponse

app = FastAPI()

@app.get("/frame.jpg")
async def proxy():
    # Redirect FastAPI request to ffmpeg’s MJPEG stream
    return RedirectResponse("http://127.0.0.1:8090/feed.mjpg")
