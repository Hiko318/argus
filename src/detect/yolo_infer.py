class YoloDetector:
    def __init__(self, onnx_path=None, conf=0.25):
        self.conf = conf
    def infer(self, frame_bgr):
        h, w, _ = frame_bgr.shape
        cx, cy, bw, bh = int(w*0.5), int(h*0.5), int(w*0.2), int(h*0.3)
        x = max(0, cx - bw//2); y = max(0, cy - bh//2)
        bw = min(bw, w - x); bh = min(bh, h - y)
        return [{"cls":"person","conf":0.9,"bbox":[x, y, bw, bh]}]
