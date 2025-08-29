from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# --- Static files setup ---
# Go up to foresight/ root (repo root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent  
PUBLIC_DIR = BASE_DIR / "public"

# Mount /public route to serve files in foresight/public
app.mount("/public", StaticFiles(directory=str(PUBLIC_DIR)), name="public")

# --- Example route to auto-load webapp ---
@app.get("/")
async def root():
    return {"message": "Backend is running. Visit /public/foresight-webapp.html for the UI"}
