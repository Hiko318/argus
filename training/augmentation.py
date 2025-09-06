"""SAR-specific data augmentation for aerial human detection.

Provides specialized augmentations for Search and Rescue scenarios including
altitude simulation, weather effects, lighting variations, and small target enhancement.
"""

import cv2
import numpy as np
import random
from typing import Tuple, Dict, List, Optional, Union
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SARAugmentation:
    """SAR-specific augmentation pipeline for aerial imagery."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.enable_sar_augmentations = self.config.get('enable_sar_augmentations', True)
        self.altitude_simulation = self.config.get('altitude_simulation', True)
        self.weather_simulation = self.config.get('weather_simulation', True)
        self.lighting_variation = self.config.get('lighting_variation', True)
        self.small_target_focus = self.config.get('small_target_focus', True)
        
        # Augmentation probabilities
        self.prob_altitude = self.config.get('prob_altitude', 0.3)
        self.prob_weather = self.config.get('prob_weather', 0.4)
        self.prob_lighting = self.config.get('prob_lighting', 0.5)
        self.prob_small_target = self.config.get('prob_small_target', 0.6)
        
        # Initialize augmentation pipeline
        self.transform = self._create_augmentation_pipeline()
    
    def _create_augmentation_pipeline(self) -> A.Compose:
        """Create the main augmentation pipeline."""
        transforms = []
        
        # Basic geometric augmentations (conservative for aerial view)
        transforms.extend([
            A.HorizontalFlip(p=0.5),  # OK for aerial view
            A.RandomRotate90(p=0.2),  # Occasional 90-degree rotation
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.3,
                rotate_limit=15,  # Small rotation only
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.3
            ),
        ])
        
        # SAR-specific augmentations
        if self.enable_sar_augmentations:
            if self.altitude_simulation:
                transforms.append(A.Lambda(
                    image=self._simulate_altitude_effects,
                    p=self.prob_altitude
                ))
            
            if self.weather_simulation:
                transforms.append(A.Lambda(
                    image=self._simulate_weather_effects,
                    p=self.prob_weather
                ))
            
            if self.lighting_variation:
                transforms.append(A.Lambda(
                    image=self._simulate_lighting_variations,
                    p=self.prob_lighting
                ))
        
        # Color and contrast augmentations
        transforms.extend([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.4
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        ])
        
        # Noise and blur (atmospheric effects)
        transforms.extend([
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ], p=0.3),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.2),
        ])
        
        # Small target enhancement
        if self.small_target_focus:
            transforms.append(A.Lambda(
                image=self._enhance_small_targets,
                p=self.prob_small_target
            ))
        
        return A.Compose(transforms)
    
    def _simulate_altitude_effects(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Simulate effects of different altitudes on image quality."""
        # Simulate atmospheric haze at higher altitudes
        altitude_factor = random.uniform(0.1, 0.8)  # 0.1 = low altitude, 0.8 = high altitude
        
        # Add haze effect
        haze_intensity = altitude_factor * 0.3
        haze = np.ones_like(image) * 255 * haze_intensity
        image = cv2.addWeighted(image, 1 - haze_intensity, haze.astype(np.uint8), haze_intensity, 0)
        
        # Reduce contrast at higher altitudes
        contrast_reduction = altitude_factor * 0.4
        image = cv2.convertScaleAbs(image, alpha=1 - contrast_reduction, beta=0)
        
        # Add slight blur for atmospheric distortion
        if altitude_factor > 0.5:
            kernel_size = int(altitude_factor * 3) * 2 + 1  # Ensure odd kernel size
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        return image
    
    def _simulate_weather_effects(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Simulate various weather conditions."""
        weather_type = random.choice(['clear', 'cloudy', 'hazy', 'overcast', 'light_rain'])
        
        if weather_type == 'cloudy':
            # Add cloud shadows
            shadow_intensity = random.uniform(0.1, 0.3)
            shadow_mask = self._generate_cloud_shadows(image.shape[:2])
            image = image * (1 - shadow_mask * shadow_intensity)
        
        elif weather_type == 'hazy':
            # Reduce visibility and add uniform haze
            haze_intensity = random.uniform(0.2, 0.5)
            haze_color = random.choice([200, 220, 240])  # Different haze colors
            haze = np.full_like(image, haze_color)
            image = cv2.addWeighted(image, 1 - haze_intensity, haze, haze_intensity, 0)
        
        elif weather_type == 'overcast':
            # Reduce overall brightness and contrast
            image = cv2.convertScaleAbs(image, alpha=0.7, beta=-20)
        
        elif weather_type == 'light_rain':
            # Add rain effect (slight blur + noise)
            image = cv2.GaussianBlur(image, (3, 3), 0)
            noise = np.random.normal(0, 5, image.shape).astype(np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def _simulate_lighting_variations(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Simulate different lighting conditions throughout the day."""
        lighting_type = random.choice(['dawn', 'morning', 'noon', 'afternoon', 'dusk', 'overcast'])
        
        if lighting_type == 'dawn' or lighting_type == 'dusk':
            # Warm, low-angle lighting
            # Increase red/orange channels
            image[:, :, 2] = np.clip(image[:, :, 2] * 1.2, 0, 255)  # Red
            image[:, :, 1] = np.clip(image[:, :, 1] * 1.1, 0, 255)  # Green
            # Add directional shadows
            shadow_direction = random.choice(['left', 'right', 'top', 'bottom'])
            image = self._add_directional_shadows(image, shadow_direction, intensity=0.2)
        
        elif lighting_type == 'noon':
            # Harsh, high-contrast lighting
            image = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
            # Add strong shadows
            image = self._add_directional_shadows(image, 'bottom', intensity=0.3)
        
        elif lighting_type == 'overcast':
            # Soft, diffused lighting
            image = cv2.convertScaleAbs(image, alpha=0.9, beta=5)
            # Slight blue tint
            image[:, :, 0] = np.clip(image[:, :, 0] * 1.1, 0, 255)  # Blue
        
        return image
    
    def _enhance_small_targets(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Enhance small targets to improve detection."""
        # Apply unsharp masking to enhance edges
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        unsharp_mask = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        
        # Enhance local contrast using CLAHE
        lab = cv2.cvtColor(unsharp_mask, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _generate_cloud_shadows(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generate realistic cloud shadow patterns."""
        h, w = shape
        shadow_mask = np.zeros((h, w), dtype=np.float32)
        
        # Generate multiple cloud shadows
        num_shadows = random.randint(1, 4)
        for _ in range(num_shadows):
            # Random cloud position and size
            center_x = random.randint(0, w)
            center_y = random.randint(0, h)
            radius_x = random.randint(w // 8, w // 3)
            radius_y = random.randint(h // 8, h // 3)
            
            # Create elliptical shadow
            y, x = np.ogrid[:h, :w]
            mask = ((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2 <= 1
            shadow_mask[mask] = random.uniform(0.3, 0.7)
        
        # Smooth the shadow edges
        shadow_mask = cv2.GaussianBlur(shadow_mask, (21, 21), 0)
        
        return shadow_mask
    
    def _add_directional_shadows(self, image: np.ndarray, direction: str, intensity: float = 0.2) -> np.ndarray:
        """Add directional shadows to simulate sun angle."""
        h, w = image.shape[:2]
        shadow_mask = np.ones((h, w), dtype=np.float32)
        
        if direction == 'left':
            for i in range(w):
                shadow_mask[:, i] = 1 - (i / w) * intensity
        elif direction == 'right':
            for i in range(w):
                shadow_mask[:, i] = 1 - ((w - i) / w) * intensity
        elif direction == 'top':
            for i in range(h):
                shadow_mask[i, :] = 1 - (i / h) * intensity
        elif direction == 'bottom':
            for i in range(h):
                shadow_mask[i, :] = 1 - ((h - i) / h) * intensity
        
        # Apply shadow
        for c in range(image.shape[2]):
            image[:, :, c] = image[:, :, c] * shadow_mask
        
        return image.astype(np.uint8)
    
    def augment_image(self, image: np.ndarray, bboxes: Optional[List] = None) -> Tuple[np.ndarray, Optional[List]]:
        """Apply augmentations to image and bounding boxes."""
        if bboxes is not None:
            # Convert bboxes to albumentations format if needed
            transformed = self.transform(image=image, bboxes=bboxes)
            return transformed['image'], transformed.get('bboxes')
        else:
            transformed = self.transform(image=image)
            return transformed['image'], None
    
    def create_mosaic_augmentation(self, images: List[np.ndarray], bboxes_list: List[List]) -> Tuple[np.ndarray, List]:
        """Create mosaic augmentation for multi-target training."""
        if len(images) != 4:
            raise ValueError("Mosaic augmentation requires exactly 4 images")
        
        # Determine output size
        target_size = 640
        mosaic_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        mosaic_bboxes = []
        
        # Resize and place images in quadrants
        half_size = target_size // 2
        positions = [(0, 0), (half_size, 0), (0, half_size), (half_size, half_size)]
        
        for i, (image, bboxes) in enumerate(zip(images, bboxes_list)):
            # Resize image to half size
            resized = cv2.resize(image, (half_size, half_size))
            
            # Place in mosaic
            y, x = positions[i]
            mosaic_image[y:y+half_size, x:x+half_size] = resized
            
            # Adjust bounding boxes
            for bbox in bboxes:
                # Assuming bbox format: [x_center, y_center, width, height] (normalized)
                adj_bbox = [
                    (bbox[0] * half_size + x) / target_size,  # x_center
                    (bbox[1] * half_size + y) / target_size,  # y_center
                    bbox[2] * half_size / target_size,        # width
                    bbox[3] * half_size / target_size         # height
                ]
                mosaic_bboxes.append(adj_bbox)
        
        return mosaic_image, mosaic_bboxes
    
    def create_mixup_augmentation(self, image1: np.ndarray, image2: np.ndarray, 
                                 bboxes1: List, bboxes2: List, alpha: float = 0.2) -> Tuple[np.ndarray, List]:
        """Create mixup augmentation for improved generalization."""
        # Ensure images are same size
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        
        # Mix images
        mixed_image = cv2.addWeighted(image1, 1 - alpha, image2, alpha, 0)
        
        # Combine bounding boxes (keep all boxes but adjust confidence if needed)
        mixed_bboxes = bboxes1 + bboxes2
        
        return mixed_image, mixed_bboxes
    
    def get_validation_transform(self) -> A.Compose:
        """Get validation-only transforms (no augmentation)."""
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def visualize_augmentation(self, image: np.ndarray, save_path: str = None) -> np.ndarray:
        """Visualize the effect of augmentations."""
        augmented, _ = self.augment_image(image.copy())
        
        # Create side-by-side comparison
        comparison = np.hstack([image, augmented])
        
        if save_path:
            cv2.imwrite(save_path, comparison)
        
        return comparison


# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create sample image
    sample_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Initialize augmentation
    config = {
        'enable_sar_augmentations': True,
        'altitude_simulation': True,
        'weather_simulation': True,
        'lighting_variation': True,
        'small_target_focus': True
    }
    
    aug = SARAugmentation(config)
    
    # Test augmentations
    augmented_image, _ = aug.augment_image(sample_image)
    
    print(f"Original shape: {sample_image.shape}")
    print(f"Augmented shape: {augmented_image.shape}")
    print("SAR augmentation pipeline created successfully!")