#!/usr/bin/env python3
"""
ONNX Model Export Tool for Foresight SAR System

This script exports PyTorch models to ONNX format for deployment optimization.
Supports YOLOv8, custom detection models, and re-identification models.

Usage:
    python export_to_onnx.py --model yolov8n.pt --output yolov8n.onnx
    python export_to_onnx.py --model custom_model.pt --input-size 640 640 --batch-size 1
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torchvision
except ImportError:
    print("Error: PyTorch not found. Please install PyTorch.")
    sys.exit(1)

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    print("Warning: ONNX packages not found. Install with: pip install onnx onnxruntime")
    onnx = None
    ort = None

try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: Ultralytics not found. YOLOv8 export will not be available.")
    YOLO = None

import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelExporter:
    """Handles export of various model types to ONNX format."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        if verbose:
            logger.setLevel(logging.DEBUG)
    
    def detect_model_type(self, model_path: str) -> str:
        """Detect the type of model from file path and contents."""
        model_path = Path(model_path)
        
        # Check file extension
        if model_path.suffix == '.pt':
            # Try to load and inspect the model
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                
                # Check for YOLOv8 model
                if 'model' in checkpoint and hasattr(checkpoint.get('model'), 'yaml'):
                    return 'yolov8'
                
                # Check for Ultralytics format
                if 'train_args' in checkpoint or 'epoch' in checkpoint:
                    return 'yolov8'
                
                # Check for standard PyTorch model
                if 'state_dict' in checkpoint or isinstance(checkpoint, dict):
                    return 'pytorch'
                
                # If it's a model object directly
                if hasattr(checkpoint, 'forward'):
                    return 'pytorch'
                    
            except Exception as e:
                logger.warning(f"Could not inspect model file: {e}")
        
        # Fallback based on filename
        filename = model_path.name.lower()
        if 'yolo' in filename:
            return 'yolov8'
        elif 'reid' in filename or 'embedding' in filename:
            return 'reid'
        else:
            return 'pytorch'
    
    def export_yolov8(self, model_path: str, output_path: str, 
                     input_size: Tuple[int, int] = (640, 640),
                     batch_size: int = 1, **kwargs) -> bool:
        """Export YOLOv8 model to ONNX."""
        if YOLO is None:
            logger.error("Ultralytics not available. Cannot export YOLOv8 model.")
            return False
        
        try:
            logger.info(f"Loading YOLOv8 model: {model_path}")
            model = YOLO(model_path)
            
            # Export to ONNX
            logger.info(f"Exporting to ONNX: {output_path}")
            success = model.export(
                format='onnx',
                imgsz=input_size,
                batch=batch_size,
                simplify=True,
                opset=kwargs.get('opset', 11),
                dynamic=kwargs.get('dynamic', False)
            )
            
            if success:
                # Move the exported file to the desired location
                exported_file = str(model_path).replace('.pt', '.onnx')
                if os.path.exists(exported_file) and exported_file != output_path:
                    os.rename(exported_file, output_path)
                
                logger.info(f"YOLOv8 model exported successfully: {output_path}")
                return True
            else:
                logger.error("YOLOv8 export failed")
                return False
                
        except Exception as e:
            logger.error(f"Error exporting YOLOv8 model: {e}")
            return False
    
    def export_pytorch(self, model_path: str, output_path: str,
                      input_size: Tuple[int, int] = (640, 640),
                      batch_size: int = 1, **kwargs) -> bool:
        """Export generic PyTorch model to ONNX."""
        try:
            logger.info(f"Loading PyTorch model: {model_path}")
            
            # Load model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(model_path, map_location=device)
            
            # Extract model from checkpoint
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    model = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    # Need to reconstruct model architecture
                    logger.error("Model architecture reconstruction not implemented for state_dict")
                    return False
                else:
                    logger.error("Unknown checkpoint format")
                    return False
            else:
                model = checkpoint
            
            # Ensure model is in eval mode
            model.eval()
            
            # Create dummy input
            if len(input_size) == 2:
                dummy_input = torch.randn(batch_size, 3, input_size[0], input_size[1], device=device)
            else:
                dummy_input = torch.randn(batch_size, *input_size, device=device)
            
            # Export to ONNX
            logger.info(f"Exporting to ONNX: {output_path}")
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=kwargs.get('opset', 11),
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'} if kwargs.get('dynamic', False) else {},
                    'output': {0: 'batch_size'} if kwargs.get('dynamic', False) else {}
                }
            )
            
            logger.info(f"PyTorch model exported successfully: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting PyTorch model: {e}")
            return False
    
    def export_reid_model(self, model_path: str, output_path: str,
                         input_size: Tuple[int, int] = (256, 128),
                         batch_size: int = 1, **kwargs) -> bool:
        """Export re-identification model to ONNX."""
        try:
            logger.info(f"Loading ReID model: {model_path}")
            
            # Load model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = torch.load(model_path, map_location=device)
            
            if isinstance(model, dict):
                if 'model' in model:
                    model = model['model']
                elif 'state_dict' in model:
                    logger.error("ReID model architecture reconstruction not implemented")
                    return False
            
            model.eval()
            
            # Create dummy input for ReID (typically smaller images)
            dummy_input = torch.randn(batch_size, 3, input_size[0], input_size[1], device=device)
            
            # Export to ONNX
            logger.info(f"Exporting ReID model to ONNX: {output_path}")
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=kwargs.get('opset', 11),
                do_constant_folding=True,
                input_names=['input'],
                output_names=['embedding'],
                dynamic_axes={
                    'input': {0: 'batch_size'} if kwargs.get('dynamic', False) else {},
                    'embedding': {0: 'batch_size'} if kwargs.get('dynamic', False) else {}
                }
            )
            
            logger.info(f"ReID model exported successfully: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting ReID model: {e}")
            return False
    
    def validate_onnx(self, onnx_path: str, original_model_path: str = None) -> bool:
        """Validate the exported ONNX model."""
        if onnx is None or ort is None:
            logger.warning("ONNX validation skipped (packages not available)")
            return True
        
        try:
            logger.info(f"Validating ONNX model: {onnx_path}")
            
            # Load and check ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # Create ONNX Runtime session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(onnx_path, providers=providers)
            
            # Get input/output info
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]
            
            logger.info(f"Input: {input_info.name} {input_info.shape} {input_info.type}")
            logger.info(f"Output: {output_info.name} {output_info.shape} {output_info.type}")
            
            # Test inference with dummy data
            if input_info.shape[0] == 'batch_size' or input_info.shape[0] is None:
                batch_size = 1
            else:
                batch_size = input_info.shape[0]
            
            if len(input_info.shape) == 4:  # Image input
                dummy_input = np.random.randn(batch_size, input_info.shape[1], 
                                            input_info.shape[2], input_info.shape[3]).astype(np.float32)
            else:
                dummy_input = np.random.randn(*input_info.shape).astype(np.float32)
            
            # Run inference
            start_time = time.time()
            outputs = session.run(None, {input_info.name: dummy_input})
            inference_time = time.time() - start_time
            
            logger.info(f"ONNX inference successful. Time: {inference_time:.4f}s")
            logger.info(f"Output shape: {outputs[0].shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"ONNX validation failed: {e}")
            return False
    
    def get_model_info(self, model_path: str) -> dict:
        """Get information about the model."""
        info = {
            'path': model_path,
            'size_mb': os.path.getsize(model_path) / (1024 * 1024),
            'type': self.detect_model_type(model_path)
        }
        
        try:
            if info['type'] == 'yolov8' and YOLO is not None:
                model = YOLO(model_path)
                info['model_info'] = str(model.model)
            else:
                checkpoint = torch.load(model_path, map_location='cpu')
                if isinstance(checkpoint, dict):
                    info['keys'] = list(checkpoint.keys())
                    if 'epoch' in checkpoint:
                        info['epoch'] = checkpoint['epoch']
        except Exception as e:
            logger.warning(f"Could not get detailed model info: {e}")
        
        return info
    
    def export_model(self, model_path: str, output_path: str, 
                    model_type: str = None, **kwargs) -> bool:
        """Main export function that routes to appropriate exporter."""
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
        
        # Auto-detect model type if not specified
        if model_type is None:
            model_type = self.detect_model_type(model_path)
        
        logger.info(f"Detected model type: {model_type}")
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Route to appropriate exporter
        if model_type == 'yolov8':
            success = self.export_yolov8(model_path, output_path, **kwargs)
        elif model_type == 'reid':
            success = self.export_reid_model(model_path, output_path, **kwargs)
        else:
            success = self.export_pytorch(model_path, output_path, **kwargs)
        
        if success and kwargs.get('validate', True):
            self.validate_onnx(output_path, model_path)
        
        return success

def main():
    parser = argparse.ArgumentParser(
        description='Export PyTorch models to ONNX format for Foresight SAR System'
    )
    
    parser.add_argument('--model', '-m', required=True,
                       help='Path to input PyTorch model (.pt file)')
    parser.add_argument('--output', '-o', required=True,
                       help='Path to output ONNX model (.onnx file)')
    parser.add_argument('--type', '-t', choices=['yolov8', 'pytorch', 'reid'],
                       help='Model type (auto-detected if not specified)')
    parser.add_argument('--input-size', nargs=2, type=int, default=[640, 640],
                       help='Input image size (height width)')
    parser.add_argument('--batch-size', '-b', type=int, default=1,
                       help='Batch size for export')
    parser.add_argument('--opset', type=int, default=11,
                       help='ONNX opset version')
    parser.add_argument('--dynamic', action='store_true',
                       help='Enable dynamic batch size')
    parser.add_argument('--no-validate', action='store_true',
                       help='Skip ONNX model validation')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--info', action='store_true',
                       help='Show model information and exit')
    
    args = parser.parse_args()
    
    # Create exporter
    exporter = ModelExporter(verbose=args.verbose)
    
    # Show model info if requested
    if args.info:
        info = exporter.get_model_info(args.model)
        print("\nModel Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        return
    
    # Export model
    logger.info(f"Starting export: {args.model} -> {args.output}")
    
    export_kwargs = {
        'input_size': tuple(args.input_size),
        'batch_size': args.batch_size,
        'opset': args.opset,
        'dynamic': args.dynamic,
        'validate': not args.no_validate
    }
    
    success = exporter.export_model(
        args.model, 
        args.output, 
        args.type,
        **export_kwargs
    )
    
    if success:
        logger.info("Export completed successfully!")
        
        # Show file sizes
        original_size = os.path.getsize(args.model) / (1024 * 1024)
        onnx_size = os.path.getsize(args.output) / (1024 * 1024)
        
        print(f"\nExport Summary:")
        print(f"  Original model: {original_size:.2f} MB")
        print(f"  ONNX model: {onnx_size:.2f} MB")
        print(f"  Size ratio: {onnx_size/original_size:.2f}x")
        
        sys.exit(0)
    else:
        logger.error("Export failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()