import itertools
class IoUTracker:
    def __init__(self, iou_thr=0.3, ttl=15):
        self.next_id = itertools.count(1)
    def update(self, dets):
        out=[]
        for d in dets:
            d = dict(d)
            d["id"] = next(self.next_id)
            out.append(d)
        return out
