"""Training module for Foresight SAR System.

Provides training scripts, data augmentation, and validation tools
for fine-tuning YOLO models on SAR-specific datasets.
"""

from .train_sar_model import SARTrainer
from .augmentation import SARAugmentation
from .validation import SARValidator
from .dataset_manager import DatasetManager

__all__ = [
    'SARTrainer',
    'SARAugmentation', 
    'SARValidator',
    'DatasetManager'
]