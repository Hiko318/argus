#!/usr/bin/env python3
"""
Active Learning and Retraining Loop for Foresight SAR System

This module provides functionality for:
- Flagging difficult scenes for retraining
- Managing active learning workflows
- Uploading flagged data to training buckets
- Triggering retraining pipelines
- Confidence-based sample selection
"""

import json
import os
import shutil
import datetime
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlagReason(Enum):
    """Reasons for flagging a scene for retraining."""
    LOW_CONFIDENCE = "low_confidence"
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
    OPERATOR_CORRECTION = "operator_correction"
    NEW_SCENARIO = "new_scenario"
    EDGE_CASE = "edge_case"
    POOR_WEATHER = "poor_weather"
    UNUSUAL_TERRAIN = "unusual_terrain"
    EQUIPMENT_ISSUE = "equipment_issue"
    OTHER = "other"


class SampleStatus(Enum):
    """Status of a flagged sample."""
    FLAGGED = "flagged"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    REJECTED = "rejected"
    UPLOADED = "uploaded"
    TRAINED = "trained"


@dataclass
class FlaggedSample:
    """Represents a flagged sample for retraining."""
    id: str
    scene_id: str
    timestamp: str
    operator_id: str
    flag_reason: FlagReason
    confidence_score: float
    predicted_class: str
    corrected_class: Optional[str]
    notes: str
    status: SampleStatus
    image_path: str
    metadata_path: str
    annotations_path: Optional[str]
    priority: int  # 1-5, 5 being highest priority
    review_date: Optional[str]
    reviewer_id: Optional[str]
    upload_date: Optional[str]
    training_batch_id: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class TrainingBatch:
    """Represents a batch of samples for training."""
    id: str
    created_date: str
    samples: List[str]  # Sample IDs
    status: str
    upload_path: str
    size_mb: float
    num_samples: int
    priority: int
    notes: str
    uploaded_by: str
    metadata: Dict[str, Any]


class ActiveLearningManager:
    """Manages active learning and retraining workflows."""
    
    def __init__(self, 
                 data_dir: str = "./active_learning",
                 confidence_threshold: float = 0.7,
                 max_samples_per_batch: int = 1000,
                 training_bucket: str = None):
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.confidence_threshold = confidence_threshold
        self.max_samples_per_batch = max_samples_per_batch
        self.training_bucket = training_bucket or os.getenv('TRAINING_BUCKET', 'foresight-training-data')
        
        # Data directories
        self.flagged_dir = self.data_dir / "flagged"
        self.batches_dir = self.data_dir / "batches"
        self.uploaded_dir = self.data_dir / "uploaded"
        
        for dir_path in [self.flagged_dir, self.batches_dir, self.uploaded_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Registry files
        self.samples_file = self.data_dir / "flagged_samples.json"
        self.batches_file = self.data_dir / "training_batches.json"
        
        # Initialize registries
        self._init_registries()
    
    def _init_registries(self):
        """Initialize registry files if they don't exist."""
        for file_path in [self.samples_file, self.batches_file]:
            if not file_path.exists():
                with open(file_path, 'w') as f:
                    json.dump([], f, indent=2)
    
    def _generate_sample_id(self, scene_id: str) -> str:
        """Generate a unique sample ID."""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        hash_input = f"{scene_id}_{timestamp}"
        hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"sample_{timestamp}_{hash_suffix}"
    
    def flag_scene_for_retraining(self,
                                 scene_id: str,
                                 operator_id: str,
                                 flag_reason: FlagReason,
                                 confidence_score: float,
                                 predicted_class: str,
                                 image_path: str,
                                 metadata_path: str,
                                 corrected_class: str = None,
                                 notes: str = "",
                                 priority: int = 3,
                                 annotations_path: str = None) -> str:
        """Flag a scene for retraining."""
        
        sample_id = self._generate_sample_id(scene_id)
        
        # Copy files to flagged directory
        sample_dir = self.flagged_dir / sample_id
        sample_dir.mkdir(exist_ok=True)
        
        # Copy image and metadata
        flagged_image_path = sample_dir / f"image{Path(image_path).suffix}"
        flagged_metadata_path = sample_dir / "metadata.json"
        flagged_annotations_path = None
        
        try:
            shutil.copy2(image_path, flagged_image_path)
            shutil.copy2(metadata_path, flagged_metadata_path)
            
            if annotations_path and os.path.exists(annotations_path):
                flagged_annotations_path = sample_dir / f"annotations{Path(annotations_path).suffix}"
                shutil.copy2(annotations_path, flagged_annotations_path)
        
        except Exception as e:
            logger.error(f"Error copying files for sample {sample_id}: {e}")
            return None
        
        # Create flagged sample record
        flagged_sample = FlaggedSample(
            id=sample_id,
            scene_id=scene_id,
            timestamp=datetime.datetime.now().isoformat(),
            operator_id=operator_id,
            flag_reason=flag_reason,
            confidence_score=confidence_score,
            predicted_class=predicted_class,
            corrected_class=corrected_class,
            notes=notes,
            status=SampleStatus.FLAGGED,
            image_path=str(flagged_image_path),
            metadata_path=str(flagged_metadata_path),
            annotations_path=str(flagged_annotations_path) if flagged_annotations_path else None,
            priority=priority,
            review_date=None,
            reviewer_id=None,
            upload_date=None,
            training_batch_id=None,
            metadata={
                "original_image_path": image_path,
                "original_metadata_path": metadata_path,
                "original_annotations_path": annotations_path
            }
        )
        
        # Load existing samples
        with open(self.samples_file, 'r') as f:
            samples = json.load(f)
        
        # Add new sample
        samples.append(asdict(flagged_sample))
        
        # Save updated samples
        with open(self.samples_file, 'w') as f:
            json.dump(samples, f, indent=2)
        
        logger.info(f"Scene {scene_id} flagged for retraining as sample {sample_id}")
        return sample_id
    
    def auto_flag_low_confidence(self,
                                scene_id: str,
                                predictions: List[Dict[str, Any]],
                                image_path: str,
                                metadata_path: str,
                                operator_id: str = "system") -> List[str]:
        """Automatically flag scenes with low confidence predictions."""
        
        flagged_samples = []
        
        for prediction in predictions:
            confidence = prediction.get('confidence', 0.0)
            predicted_class = prediction.get('class', 'unknown')
            
            if confidence < self.confidence_threshold:
                sample_id = self.flag_scene_for_retraining(
                    scene_id=scene_id,
                    operator_id=operator_id,
                    flag_reason=FlagReason.LOW_CONFIDENCE,
                    confidence_score=confidence,
                    predicted_class=predicted_class,
                    image_path=image_path,
                    metadata_path=metadata_path,
                    notes=f"Auto-flagged: confidence {confidence:.3f} below threshold {self.confidence_threshold}",
                    priority=2  # Lower priority for auto-flagged
                )
                
                if sample_id:
                    flagged_samples.append(sample_id)
        
        return flagged_samples
    
    def review_sample(self,
                     sample_id: str,
                     reviewer_id: str,
                     approved: bool,
                     corrected_class: str = None,
                     review_notes: str = "") -> bool:
        """Review a flagged sample."""
        
        with open(self.samples_file, 'r') as f:
            samples = json.load(f)
        
        for sample in samples:
            if sample['id'] == sample_id:
                sample['status'] = SampleStatus.APPROVED.value if approved else SampleStatus.REJECTED.value
                sample['review_date'] = datetime.datetime.now().isoformat()
                sample['reviewer_id'] = reviewer_id
                
                if corrected_class:
                    sample['corrected_class'] = corrected_class
                
                if review_notes:
                    sample['notes'] += f" | Review: {review_notes}"
                
                break
        else:
            logger.error(f"Sample {sample_id} not found")
            return False
        
        with open(self.samples_file, 'w') as f:
            json.dump(samples, f, indent=2)
        
        logger.info(f"Sample {sample_id} reviewed by {reviewer_id}: {'approved' if approved else 'rejected'}")
        return True
    
    def create_training_batch(self,
                             batch_name: str = None,
                             max_samples: int = None,
                             min_priority: int = 1,
                             uploaded_by: str = "system",
                             notes: str = "") -> str:
        """Create a training batch from approved samples."""
        
        max_samples = max_samples or self.max_samples_per_batch
        
        # Load approved samples
        with open(self.samples_file, 'r') as f:
            samples = json.load(f)
        
        approved_samples = [
            s for s in samples 
            if s['status'] == SampleStatus.APPROVED.value and s['priority'] >= min_priority
        ]
        
        if not approved_samples:
            logger.warning("No approved samples available for training batch")
            return None
        
        # Sort by priority (highest first) and take up to max_samples
        approved_samples.sort(key=lambda x: x['priority'], reverse=True)
        batch_samples = approved_samples[:max_samples]
        
        # Generate batch ID
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_id = batch_name or f"batch_{timestamp}"
        
        # Create batch directory
        batch_dir = self.batches_dir / batch_id
        batch_dir.mkdir(exist_ok=True)
        
        # Copy samples to batch directory
        total_size = 0
        sample_ids = []
        
        for sample in batch_samples:
            sample_id = sample['id']
            sample_ids.append(sample_id)
            
            # Create sample subdirectory in batch
            sample_batch_dir = batch_dir / sample_id
            sample_batch_dir.mkdir(exist_ok=True)
            
            # Copy files
            try:
                shutil.copy2(sample['image_path'], sample_batch_dir / f"image{Path(sample['image_path']).suffix}")
                shutil.copy2(sample['metadata_path'], sample_batch_dir / "metadata.json")
                
                if sample['annotations_path']:
                    shutil.copy2(sample['annotations_path'], sample_batch_dir / f"annotations{Path(sample['annotations_path']).suffix}")
                
                # Calculate size
                for file_path in sample_batch_dir.iterdir():
                    total_size += file_path.stat().st_size
                
                # Create sample info file
                sample_info = {
                    'original_sample': sample,
                    'batch_id': batch_id,
                    'added_to_batch': datetime.datetime.now().isoformat()
                }
                
                with open(sample_batch_dir / "sample_info.json", 'w') as f:
                    json.dump(sample_info, f, indent=2)
            
            except Exception as e:
                logger.error(f"Error copying sample {sample_id} to batch: {e}")
                continue
        
        # Create training batch record
        training_batch = TrainingBatch(
            id=batch_id,
            created_date=datetime.datetime.now().isoformat(),
            samples=sample_ids,
            status="created",
            upload_path=str(batch_dir),
            size_mb=total_size / (1024 * 1024),
            num_samples=len(sample_ids),
            priority=max(s['priority'] for s in batch_samples),
            notes=notes,
            uploaded_by=uploaded_by,
            metadata={
                'min_priority': min_priority,
                'max_samples_requested': max_samples,
                'creation_timestamp': datetime.datetime.now().isoformat()
            }
        )
        
        # Save batch record
        with open(self.batches_file, 'r') as f:
            batches = json.load(f)
        
        batches.append(asdict(training_batch))
        
        with open(self.batches_file, 'w') as f:
            json.dump(batches, f, indent=2)
        
        # Update sample statuses
        for sample in samples:
            if sample['id'] in sample_ids:
                sample['training_batch_id'] = batch_id
                sample['status'] = SampleStatus.UPLOADED.value
        
        with open(self.samples_file, 'w') as f:
            json.dump(samples, f, indent=2)
        
        logger.info(f"Training batch {batch_id} created with {len(sample_ids)} samples ({total_size/(1024*1024):.1f} MB)")
        return batch_id
    
    def upload_batch_to_cloud(self, batch_id: str) -> bool:
        """Upload a training batch to cloud storage."""
        
        # Load batch info
        with open(self.batches_file, 'r') as f:
            batches = json.load(f)
        
        batch = None
        for b in batches:
            if b['id'] == batch_id:
                batch = b
                break
        
        if not batch:
            logger.error(f"Batch {batch_id} not found")
            return False
        
        batch_dir = Path(batch['upload_path'])
        if not batch_dir.exists():
            logger.error(f"Batch directory {batch_dir} not found")
            return False
        
        try:
            # Create archive
            archive_path = self.uploaded_dir / f"{batch_id}.tar.gz"
            shutil.make_archive(str(archive_path).replace('.tar.gz', ''), 'gztar', batch_dir)
            
            # TODO: Implement actual cloud upload
            # This would typically use boto3 for S3, azure-storage-blob for Azure, etc.
            logger.info(f"Batch {batch_id} archived to {archive_path}")
            logger.info(f"TODO: Upload {archive_path} to {self.training_bucket}")
            
            # Update batch status
            batch['status'] = 'uploaded'
            batch['upload_date'] = datetime.datetime.now().isoformat()
            
            with open(self.batches_file, 'w') as f:
                json.dump(batches, f, indent=2)
            
            return True
        
        except Exception as e:
            logger.error(f"Error uploading batch {batch_id}: {e}")
            return False
    
    def get_flagged_samples(self, status: SampleStatus = None) -> List[Dict]:
        """Get flagged samples, optionally filtered by status."""
        with open(self.samples_file, 'r') as f:
            samples = json.load(f)
        
        if status:
            return [s for s in samples if s['status'] == status.value]
        return samples
    
    def get_training_batches(self, status: str = None) -> List[Dict]:
        """Get training batches, optionally filtered by status."""
        with open(self.batches_file, 'r') as f:
            batches = json.load(f)
        
        if status:
            return [b for b in batches if b['status'] == status]
        return batches
    
    def generate_report(self) -> str:
        """Generate a summary report of active learning activities."""
        samples = self.get_flagged_samples()
        batches = self.get_training_batches()
        
        report = []
        report.append("=" * 50)
        report.append("ACTIVE LEARNING REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.datetime.now().isoformat()}")
        report.append("")
        
        # Sample summary
        report.append("FLAGGED SAMPLES:")
        report.append(f"  Total samples: {len(samples)}")
        for status in SampleStatus:
            count = len([s for s in samples if s['status'] == status.value])
            report.append(f"  {status.value}: {count}")
        report.append("")
        
        # Flag reason summary
        report.append("FLAG REASONS:")
        for reason in FlagReason:
            count = len([s for s in samples if s['flag_reason'] == reason.value])
            if count > 0:
                report.append(f"  {reason.value}: {count}")
        report.append("")
        
        # Batch summary
        report.append("TRAINING BATCHES:")
        report.append(f"  Total batches: {len(batches)}")
        total_samples_in_batches = sum(b['num_samples'] for b in batches)
        total_size_mb = sum(b['size_mb'] for b in batches)
        report.append(f"  Total samples in batches: {total_samples_in_batches}")
        report.append(f"  Total size: {total_size_mb:.1f} MB")
        
        # Recent activity
        recent_samples = [s for s in samples if 
                         (datetime.datetime.now() - datetime.datetime.fromisoformat(s['timestamp'])).days <= 7]
        report.append("")
        report.append(f"RECENT ACTIVITY (last 7 days): {len(recent_samples)} samples flagged")
        
        return "\n".join(report)


def main():
    """CLI interface for active learning management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Foresight SAR Active Learning Manager")
    parser.add_argument("--data-dir", default="./active_learning", help="Active learning data directory")
    parser.add_argument("--confidence-threshold", type=float, default=0.7, help="Confidence threshold for auto-flagging")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Flag scene command
    flag_parser = subparsers.add_parser("flag", help="Flag a scene for retraining")
    flag_parser.add_argument("--scene-id", required=True, help="Scene ID")
    flag_parser.add_argument("--operator-id", required=True, help="Operator ID")
    flag_parser.add_argument("--reason", required=True, choices=[r.value for r in FlagReason], help="Flag reason")
    flag_parser.add_argument("--confidence", type=float, required=True, help="Confidence score")
    flag_parser.add_argument("--predicted-class", required=True, help="Predicted class")
    flag_parser.add_argument("--image-path", required=True, help="Path to image")
    flag_parser.add_argument("--metadata-path", required=True, help="Path to metadata")
    flag_parser.add_argument("--corrected-class", help="Corrected class")
    flag_parser.add_argument("--notes", default="", help="Additional notes")
    flag_parser.add_argument("--priority", type=int, default=3, help="Priority (1-5)")
    
    # Review sample command
    review_parser = subparsers.add_parser("review", help="Review a flagged sample")
    review_parser.add_argument("--sample-id", required=True, help="Sample ID")
    review_parser.add_argument("--reviewer-id", required=True, help="Reviewer ID")
    review_parser.add_argument("--approved", action="store_true", help="Approve the sample")
    review_parser.add_argument("--corrected-class", help="Corrected class")
    review_parser.add_argument("--notes", default="", help="Review notes")
    
    # Create batch command
    batch_parser = subparsers.add_parser("create-batch", help="Create a training batch")
    batch_parser.add_argument("--batch-name", help="Batch name")
    batch_parser.add_argument("--max-samples", type=int, help="Maximum samples in batch")
    batch_parser.add_argument("--min-priority", type=int, default=1, help="Minimum priority")
    batch_parser.add_argument("--uploaded-by", default="system", help="Uploader ID")
    batch_parser.add_argument("--notes", default="", help="Batch notes")
    
    # Upload batch command
    upload_parser = subparsers.add_parser("upload-batch", help="Upload a training batch")
    upload_parser.add_argument("--batch-id", required=True, help="Batch ID")
    
    # List commands
    subparsers.add_parser("list-samples", help="List flagged samples")
    subparsers.add_parser("list-batches", help="List training batches")
    subparsers.add_parser("report", help="Generate active learning report")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = ActiveLearningManager(
        data_dir=args.data_dir,
        confidence_threshold=args.confidence_threshold
    )
    
    if args.command == "flag":
        sample_id = manager.flag_scene_for_retraining(
            scene_id=args.scene_id,
            operator_id=args.operator_id,
            flag_reason=FlagReason(args.reason),
            confidence_score=args.confidence,
            predicted_class=args.predicted_class,
            image_path=args.image_path,
            metadata_path=args.metadata_path,
            corrected_class=args.corrected_class,
            notes=args.notes,
            priority=args.priority
        )
        print(f"Sample flagged: {sample_id}")
    
    elif args.command == "review":
        success = manager.review_sample(
            sample_id=args.sample_id,
            reviewer_id=args.reviewer_id,
            approved=args.approved,
            corrected_class=args.corrected_class,
            review_notes=args.notes
        )
        print(f"Review {'successful' if success else 'failed'}")
    
    elif args.command == "create-batch":
        batch_id = manager.create_training_batch(
            batch_name=args.batch_name,
            max_samples=args.max_samples,
            min_priority=args.min_priority,
            uploaded_by=args.uploaded_by,
            notes=args.notes
        )
        print(f"Training batch created: {batch_id}")
    
    elif args.command == "upload-batch":
        success = manager.upload_batch_to_cloud(args.batch_id)
        print(f"Upload {'successful' if success else 'failed'}")
    
    elif args.command == "list-samples":
        samples = manager.get_flagged_samples()
        print(f"Found {len(samples)} flagged samples:")
        for sample in samples:
            print(f"  {sample['id']}: {sample['scene_id']} ({sample['status']}) - {sample['flag_reason']}")
    
    elif args.command == "list-batches":
        batches = manager.get_training_batches()
        print(f"Found {len(batches)} training batches:")
        for batch in batches:
            print(f"  {batch['id']}: {batch['num_samples']} samples ({batch['status']}) - {batch['size_mb']:.1f} MB")
    
    elif args.command == "report":
        print(manager.generate_report())


if __name__ == "__main__":
    main()