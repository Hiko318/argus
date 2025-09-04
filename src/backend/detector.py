# src/detector.py
import os, sys
import cv2
import numpy as np

class DummyDetector:
    """Quick detector for Day One: simple foreground blob detection.
       Replace with a YOLO model when ready (ultralytics.YOLO or torch)."""

    def __init__(self, min_area=500):
        self.min_area = min_area

    def detect(self, image):
        # image: BGR
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, th = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area: continue
            x,y,w,h = cv2.boundingRect(c)
            dets.append({"bbox":[int(x),int(y),int(w),int(h)], "score": 0.5, "class":"person"})
        return dets
