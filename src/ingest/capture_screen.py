import time
import mss
import numpy as np
import pygetwindow as gw
import cv2

def get_window_bbox(title="DJI_MIRROR"):
    wins = [w for w in gw.getAllTitles() if title in w]
    if not wins:
        raise RuntimeError(f"Window '{title}' not found. Launch scrcpy with --window-title {title}")
    w = gw.getWindowsWithTitle(wins[0])[0]
    left, top, width, height = w.left, w.top, w.width, w.height
    if width <= 0 or height <= 0:
        w.maximize()
        time.sleep(0.3)
        left, top, width, height = w.left, w.top, w.width, w.height
    return {"left": left, "top": top, "width": width, "height": height}

class ScreenCapture:
    def __init__(self, title="DJI_MIRROR", target_fps=12):
        self.title = title
        self.box = get_window_bbox(title)
        self.sct = mss.mss()
        self.period = 1.0/float(target_fps)
        self._t = time.time()

    def read(self):
        now = time.time()
        delay = self.period - (now - self._t)
        if delay > 0:
            time.sleep(delay)
        self._t = time.time()
        img = np.asarray(self.sct.grab(self.box))  # BGRA
        frame = img[...,:3]  # BGR
        return True, frame

    def release(self):
        self.sct.close()
