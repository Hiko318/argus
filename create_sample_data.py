#!/usr/bin/env python3
"""
Sample Data Generator for YOLOv11 Training
Creates minimal sample dataset for demonstration purposes
"""

import os
import numpy as np
from PIL import Image, ImageDraw
import random

def create_sample_image_with_annotations(image_path, label_path, class_id, image_size=(640, 640)):
    """
    Create a sample image with a simple colored rectangle and corresponding YOLO annotation
    """
    # Create a simple image with random background
    img = Image.new('RGB', image_size, color=(random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)))
    draw = ImageDraw.Draw(img)
    
    # Draw a simple rectangle representing the object
    width, height = image_size
    
    # Random object position and size
    obj_width = random.randint(50, 150)
    obj_height = random.randint(50, 150)
    x1 = random.randint(0, width - obj_width)
    y1 = random.randint(0, height - obj_height)
    x2 = x1 + obj_width
    y2 = y1 + obj_height
    
    # Different colors for different classes
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red for person, Green for dog, Blue for cat
    draw.rectangle([x1, y1, x2, y2], fill=colors[class_id], outline=(255, 255, 255), width=2)
    
    # Save image
    img.save(image_path)
    
    # Create YOLO format annotation
    # YOLO format: class_id center_x center_y width height (all normalized 0-1)
    center_x = (x1 + x2) / 2 / width
    center_y = (y1 + y2) / 2 / height
    norm_width = obj_width / width
    norm_height = obj_height / height
    
    with open(label_path, 'w') as f:
        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")

def create_sample_dataset():
    """
    Create a minimal sample dataset for training demonstration
    """
    print("Creating sample dataset for YOLOv11 training...")
    
    # Paths
    train_images_dir = "data/training/images/train"
    train_labels_dir = "data/training/labels/train"
    val_images_dir = "data/training/images/val"
    val_labels_dir = "data/training/labels/val"
    
    # Create training samples
    print("Creating training samples...")
    for i in range(30):  # 30 training samples
        class_id = i % 3  # Cycle through classes 0, 1, 2
        image_path = os.path.join(train_images_dir, f"sample_{i:03d}.jpg")
        label_path = os.path.join(train_labels_dir, f"sample_{i:03d}.txt")
        create_sample_image_with_annotations(image_path, label_path, class_id)
    
    # Create validation samples
    print("Creating validation samples...")
    for i in range(10):  # 10 validation samples
        class_id = i % 3  # Cycle through classes 0, 1, 2
        image_path = os.path.join(val_images_dir, f"val_{i:03d}.jpg")
        label_path = os.path.join(val_labels_dir, f"val_{i:03d}.txt")
        create_sample_image_with_annotations(image_path, label_path, class_id)
    
    print("Sample dataset created successfully!")
    print(f"Training samples: 30 images in {train_images_dir}")
    print(f"Validation samples: 10 images in {val_images_dir}")
    print("Classes: 0=person, 1=dog, 2=cat")
    print("\nNote: This is a minimal synthetic dataset for demonstration.")
    print("For real training, replace with actual SAR imagery.")

if __name__ == "__main__":
    create_sample_dataset()