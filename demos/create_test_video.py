#!/usr/bin/env python3
"""
Create a test video file for simulation mode
"""

import cv2
import numpy as np
from pathlib import Path

def create_test_video():
    """Create a simple test video with moving objects."""
    output_path = "stream.mp4"
    
    print(f"Creating test video: {output_path}")
    
    # Video properties
    width, height = 1280, 720
    fps = 30
    duration = 30  # seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_num in range(total_frames):
        # Create background (sky gradient)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Sky gradient (blue to light blue)
        for y in range(height):
            intensity = int(100 + (155 * y / height))
            frame[y, :] = [intensity, intensity//2, 50]
        
        # Add ground (green)
        ground_start = int(height * 0.7)
        frame[ground_start:, :] = [34, 139, 34]  # Forest green
        
        # Add moving "person" (red rectangle)
        person_x = int(50 + (frame_num * 3) % (width - 100))
        person_y = int(ground_start - 60)
        cv2.rectangle(frame, (person_x, person_y), (person_x + 30, person_y + 60), (0, 0, 255), -1)
        
        # Add another moving object (blue circle - could be vehicle)
        vehicle_x = int(200 + (frame_num * 2) % (width - 200))
        vehicle_y = int(ground_start - 30)
        cv2.circle(frame, (vehicle_x, vehicle_y), 15, (255, 0, 0), -1)
        
        # Add timestamp
        timestamp = f"Time: {frame_num/fps:.1f}s"
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add frame number
        frame_text = f"Frame: {frame_num}"
        cv2.putText(frame, frame_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add search area indicator
        search_text = "SAR SIMULATION - Test Objects Moving"
        cv2.putText(frame, search_text, (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        out.write(frame)
        
        if frame_num % 100 == 0:
            print(f"Progress: {frame_num}/{total_frames} frames ({frame_num/total_frames*100:.1f}%)")
    
    out.release()
    print(f"✅ Test video created: {output_path}")
    print(f"📊 Video specs: {width}x{height} @ {fps}fps, {duration}s duration")
    print(f"📦 File size: {Path(output_path).stat().st_size / (1024*1024):.1f} MB")

if __name__ == "__main__":
    create_test_video()