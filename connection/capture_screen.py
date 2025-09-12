import time
import mss
import numpy as np
import pygetwindow as gw
import cv2
import win32gui
import win32con

def get_window_bbox(title="FORESIGHT_FEED"):
    wins = [w for w in gw.getAllTitles() if title in w]
    if not wins:
        raise RuntimeError(f"Window '{title}' not found. Launch scrcpy with --window-title {title}")
    
    # Store the window handle for later use
    hwnd = win32gui.FindWindow(None, wins[0])
    
    # Get window dimensions even if minimized or not in focus
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bottom - top
    
    if width <= 0 or height <= 0:
        # Try to restore the window if it's minimized
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        time.sleep(0.3)
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = right - left
        height = bottom - top
    
    # Ensure window is visible but don't bring to foreground
    if not win32gui.IsWindowVisible(hwnd):
        win32gui.ShowWindow(hwnd, win32con.SW_SHOWNOACTIVATE)
    
    # Adjust for window borders and title bar (approximate values)
    border_x = 8  # Window border width
    title_bar = 31  # Title bar height
    
    # Return the content area of the window
    return {
        "left": left + border_x, 
        "top": top + title_bar, 
        "width": width - 2 * border_x, 
        "height": height - title_bar - border_x
    }

class ScreenCapture:
    def __init__(self, title="FORESIGHT_FEED", target_fps=12):
        self.title = title
        self.sct = mss.mss()
        self.period = 1.0/float(target_fps)
        self._t = time.time()
        self.last_update = 0
        self.update_interval = 0.5  # Update box position every 0.5 seconds
        
        # Initialize box with default values
        self.box = {"left": 0, "top": 0, "width": 1280, "height": 720}
        
        # Try to get the actual window position on initialization
        try:
            self.box = get_window_bbox(self.title)
        except Exception as e:
            print(f"Warning: Could not find window '{self.title}' on initialization: {e}")
            print("Using default screen capture area. Launch scrcpy with --window-title FORESIGHT_FEED for proper capture.")

    def read(self):
        now = time.time()
        
        # Update the window position periodically to handle window movement
        if now - self.last_update > self.update_interval:
            try:
                self.box = get_window_bbox(self.title)
                self.last_update = now
            except Exception as e:
                print(f"Warning: Could not update window position: {e}")
        
        # Maintain target FPS
        delay = self.period - (now - self._t)
        if delay > 0:
            time.sleep(delay)
        self._t = time.time()
        
        try:
            if hasattr(self, 'box') and self.box:
                img = np.asarray(self.sct.grab(self.box))  # BGRA
                frame = img[...,:3]  # BGR
                return True, frame
            else:
                # No valid box, return placeholder frame
                return False, np.zeros((720, 1280, 3), dtype=np.uint8)
        except Exception as e:
            print(f"Error capturing screen: {e}")
            # Return a black frame as fallback
            h, w = self.box.get("height", 720), self.box.get("width", 1280)
            return False, np.zeros((h, w, 3), dtype=np.uint8)

    def release(self):
        self.sct.close()
