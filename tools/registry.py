#!/usr/bin/env python3
"""
Dataset and Model Registry for Foresight SAR System

This module provides functionality to track and manage:
- Dataset metadata (source, license, augmentation)
- Model versions and their performance metrics
- Training runs and evaluation results
- Model deployment history
"""

import json
import csv
import os
import hashlib
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class DatasetType(Enum):
    """Types of datasets in the SAR system."""
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"
    PRODUCTION = "production"
    ACTIVE_LEARNING = "active_learning"


class ModelStatus(Enum):
    """Model deployment status."""
    TRAINING = "training"
    VALIDATION = "validation"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class DatasetMetadata:
    """Metadata for a dataset entry."""
    id: str
    name: str
    version: str
    dataset_type: DatasetType
    source: str
    license: str
    description: str
    created_date: str
    size_mb: float
    num_samples: int
    num_classes: int
    augmentation_applied: List[str]
    preprocessing_steps: List[str]
    file_path: str
    checksum: str
    tags: List[str]
    metadata: Dict[str, Any]


@dataclass
class ModelMetadata:
    """Metadata for a model entry."""
    id: str
    name: str
    version: str
    architecture: str
    framework: str
    status: ModelStatus
    created_date: str
    trained_on_dataset: str
    training_duration_hours: float
    model_size_mb: float
    file_path: str
    checksum: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    deployment_date: Optional[str]
    tags: List[str]
    metadata: Dict[str, Any]


@dataclass
class TrainingRun:
    """Metadata for a training run."""
    id: str
    model_id: str
    dataset_id: str
    start_time: str
    end_time: str
    duration_hours: float
    status: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    logs_path: str
    artifacts_path: str
    gpu_hours: float
    notes: str


class Registry:
    """Main registry class for managing datasets and models."""
    
    def __init__(self, registry_dir: str = "./registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True)
        
        # Registry files
        self.datasets_file = self.registry_dir / "datasets.json"
        self.models_file = self.registry_dir / "models.json"
        self.training_runs_file = self.registry_dir / "training_runs.json"
        
        # CSV exports for easy viewing
        self.datasets_csv = self.registry_dir / "datasets.csv"
        self.models_csv = self.registry_dir / "models.csv"
        self.training_runs_csv = self.registry_dir / "training_runs.csv"
        
        # Initialize empty registries if files don't exist
        self._init_registries()
    
    def _init_registries(self):
        """Initialize registry files if they don't exist."""
        for file_path in [self.datasets_file, self.models_file, self.training_runs_file]:
            if not file_path.exists():
                with open(file_path, 'w') as f:
                    json.dump([], f, indent=2)
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except FileNotFoundError:
            return "file_not_found"
    
    def _get_file_size_mb(self, file_path: str) -> float:
        """Get file size in MB."""
        try:
            return os.path.getsize(file_path) / (1024 * 1024)
        except FileNotFoundError:
            return 0.0
    
    def register_dataset(self, 
                        name: str,
                        version: str,
                        dataset_type: DatasetType,
                        source: str,
                        license: str,
                        description: str,
                        file_path: str,
                        num_samples: int,
                        num_classes: int,
                        augmentation_applied: List[str] = None,
                        preprocessing_steps: List[str] = None,
                        tags: List[str] = None,
                        metadata: Dict[str, Any] = None) -> str:
        """Register a new dataset."""
        
        dataset_id = f"{name}_{version}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        dataset = DatasetMetadata(
            id=dataset_id,
            name=name,
            version=version,
            dataset_type=dataset_type,
            source=source,
            license=license,
            description=description,
            created_date=datetime.datetime.now().isoformat(),
            size_mb=self._get_file_size_mb(file_path),
            num_samples=num_samples,
            num_classes=num_classes,
            augmentation_applied=augmentation_applied or [],
            preprocessing_steps=preprocessing_steps or [],
            file_path=file_path,
            checksum=self._calculate_checksum(file_path),
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # Load existing datasets
        with open(self.datasets_file, 'r') as f:
            datasets = json.load(f)
        
        # Add new dataset
        datasets.append(asdict(dataset))
        
        # Save updated datasets
        with open(self.datasets_file, 'w') as f:
            json.dump(datasets, f, indent=2)
        
        # Update CSV export
        self._export_datasets_to_csv()
        
        print(f"Dataset registered: {dataset_id}")
        return dataset_id
    
    def register_model(self,
                      name: str,
                      version: str,
                      architecture: str,
                      framework: str,
                      trained_on_dataset: str,
                      training_duration_hours: float,
                      file_path: str,
                      hyperparameters: Dict[str, Any],
                      metrics: Dict[str, float],
                      status: ModelStatus = ModelStatus.TRAINING,
                      tags: List[str] = None,
                      metadata: Dict[str, Any] = None) -> str:
        """Register a new model."""
        
        model_id = f"{name}_{version}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model = ModelMetadata(
            id=model_id,
            name=name,
            version=version,
            architecture=architecture,
            framework=framework,
            status=status,
            created_date=datetime.datetime.now().isoformat(),
            trained_on_dataset=trained_on_dataset,
            training_duration_hours=training_duration_hours,
            model_size_mb=self._get_file_size_mb(file_path),
            file_path=file_path,
            checksum=self._calculate_checksum(file_path),
            hyperparameters=hyperparameters,
            metrics=metrics,
            deployment_date=None,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # Load existing models
        with open(self.models_file, 'r') as f:
            models = json.load(f)
        
        # Add new model
        models.append(asdict(model))
        
        # Save updated models
        with open(self.models_file, 'w') as f:
            json.dump(models, f, indent=2)
        
        # Update CSV export
        self._export_models_to_csv()
        
        print(f"Model registered: {model_id}")
        return model_id
    
    def register_training_run(self,
                             model_id: str,
                             dataset_id: str,
                             start_time: str,
                             end_time: str,
                             status: str,
                             hyperparameters: Dict[str, Any],
                             metrics: Dict[str, float],
                             logs_path: str,
                             artifacts_path: str,
                             gpu_hours: float = 0.0,
                             notes: str = "") -> str:
        """Register a training run."""
        
        run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate duration
        start_dt = datetime.datetime.fromisoformat(start_time)
        end_dt = datetime.datetime.fromisoformat(end_time)
        duration_hours = (end_dt - start_dt).total_seconds() / 3600
        
        training_run = TrainingRun(
            id=run_id,
            model_id=model_id,
            dataset_id=dataset_id,
            start_time=start_time,
            end_time=end_time,
            duration_hours=duration_hours,
            status=status,
            hyperparameters=hyperparameters,
            metrics=metrics,
            logs_path=logs_path,
            artifacts_path=artifacts_path,
            gpu_hours=gpu_hours,
            notes=notes
        )
        
        # Load existing training runs
        with open(self.training_runs_file, 'r') as f:
            runs = json.load(f)
        
        # Add new run
        runs.append(asdict(training_run))
        
        # Save updated runs
        with open(self.training_runs_file, 'w') as f:
            json.dump(runs, f, indent=2)
        
        # Update CSV export
        self._export_training_runs_to_csv()
        
        print(f"Training run registered: {run_id}")
        return run_id
    
    def update_model_status(self, model_id: str, status: ModelStatus, deployment_date: str = None):
        """Update model status and deployment date."""
        with open(self.models_file, 'r') as f:
            models = json.load(f)
        
        for model in models:
            if model['id'] == model_id:
                model['status'] = status.value
                if deployment_date:
                    model['deployment_date'] = deployment_date
                break
        
        with open(self.models_file, 'w') as f:
            json.dump(models, f, indent=2)
        
        self._export_models_to_csv()
        print(f"Model {model_id} status updated to {status.value}")
    
    def get_dataset(self, dataset_id: str) -> Optional[Dict]:
        """Get dataset by ID."""
        with open(self.datasets_file, 'r') as f:
            datasets = json.load(f)
        
        for dataset in datasets:
            if dataset['id'] == dataset_id:
                return dataset
        return None
    
    def get_model(self, model_id: str) -> Optional[Dict]:
        """Get model by ID."""
        with open(self.models_file, 'r') as f:
            models = json.load(f)
        
        for model in models:
            if model['id'] == model_id:
                return model
        return None
    
    def list_datasets(self, dataset_type: DatasetType = None) -> List[Dict]:
        """List all datasets, optionally filtered by type."""
        with open(self.datasets_file, 'r') as f:
            datasets = json.load(f)
        
        if dataset_type:
            return [d for d in datasets if d['dataset_type'] == dataset_type.value]
        return datasets
    
    def list_models(self, status: ModelStatus = None) -> List[Dict]:
        """List all models, optionally filtered by status."""
        with open(self.models_file, 'r') as f:
            models = json.load(f)
        
        if status:
            return [m for m in models if m['status'] == status.value]
        return models
    
    def get_best_model(self, metric: str = "accuracy") -> Optional[Dict]:
        """Get the best performing model based on a metric."""
        models = self.list_models()
        best_model = None
        best_score = -1
        
        for model in models:
            if metric in model['metrics']:
                score = model['metrics'][metric]
                if score > best_score:
                    best_score = score
                    best_model = model
        
        return best_model
    
    def _export_datasets_to_csv(self):
        """Export datasets to CSV for easy viewing."""
        with open(self.datasets_file, 'r') as f:
            datasets = json.load(f)
        
        if not datasets:
            return
        
        with open(self.datasets_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=datasets[0].keys())
            writer.writeheader()
            for dataset in datasets:
                # Convert lists and dicts to strings for CSV
                row = dataset.copy()
                for key, value in row.items():
                    if isinstance(value, (list, dict)):
                        row[key] = json.dumps(value)
                writer.writerow(row)
    
    def _export_models_to_csv(self):
        """Export models to CSV for easy viewing."""
        with open(self.models_file, 'r') as f:
            models = json.load(f)
        
        if not models:
            return
        
        with open(self.models_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=models[0].keys())
            writer.writeheader()
            for model in models:
                # Convert lists and dicts to strings for CSV
                row = model.copy()
                for key, value in row.items():
                    if isinstance(value, (list, dict)):
                        row[key] = json.dumps(value)
                writer.writerow(row)
    
    def _export_training_runs_to_csv(self):
        """Export training runs to CSV for easy viewing."""
        with open(self.training_runs_file, 'r') as f:
            runs = json.load(f)
        
        if not runs:
            return
        
        with open(self.training_runs_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=runs[0].keys())
            writer.writeheader()
            for run in runs:
                # Convert lists and dicts to strings for CSV
                row = run.copy()
                for key, value in row.items():
                    if isinstance(value, (list, dict)):
                        row[key] = json.dumps(value)
                writer.writerow(row)
    
    def generate_report(self) -> str:
        """Generate a summary report of the registry."""
        datasets = self.list_datasets()
        models = self.list_models()
        
        report = []
        report.append("=" * 50)
        report.append("FORESIGHT SAR REGISTRY REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.datetime.now().isoformat()}")
        report.append("")
        
        # Dataset summary
        report.append("DATASETS:")
        report.append(f"  Total datasets: {len(datasets)}")
        for dtype in DatasetType:
            count = len([d for d in datasets if d['dataset_type'] == dtype.value])
            report.append(f"  {dtype.value}: {count}")
        report.append("")
        
        # Model summary
        report.append("MODELS:")
        report.append(f"  Total models: {len(models)}")
        for status in ModelStatus:
            count = len([m for m in models if m['status'] == status.value])
            report.append(f"  {status.value}: {count}")
        report.append("")
        
        # Best models
        best_accuracy = self.get_best_model("accuracy")
        if best_accuracy:
            report.append(f"Best accuracy: {best_accuracy['name']} v{best_accuracy['version']} ({best_accuracy['metrics']['accuracy']:.3f})")
        
        best_f1 = self.get_best_model("f1_score")
        if best_f1:
            report.append(f"Best F1 score: {best_f1['name']} v{best_f1['version']} ({best_f1['metrics']['f1_score']:.3f})")
        
        return "\n".join(report)


def main():
    """CLI interface for the registry."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Foresight SAR Dataset and Model Registry")
    parser.add_argument("--registry-dir", default="./registry", help="Registry directory")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Register dataset command
    dataset_parser = subparsers.add_parser("register-dataset", help="Register a new dataset")
    dataset_parser.add_argument("--name", required=True, help="Dataset name")
    dataset_parser.add_argument("--version", required=True, help="Dataset version")
    dataset_parser.add_argument("--type", required=True, choices=[t.value for t in DatasetType], help="Dataset type")
    dataset_parser.add_argument("--source", required=True, help="Dataset source")
    dataset_parser.add_argument("--license", required=True, help="Dataset license")
    dataset_parser.add_argument("--description", required=True, help="Dataset description")
    dataset_parser.add_argument("--file-path", required=True, help="Path to dataset file")
    dataset_parser.add_argument("--num-samples", type=int, required=True, help="Number of samples")
    dataset_parser.add_argument("--num-classes", type=int, required=True, help="Number of classes")
    dataset_parser.add_argument("--tags", nargs="*", help="Dataset tags")
    
    # Register model command
    model_parser = subparsers.add_parser("register-model", help="Register a new model")
    model_parser.add_argument("--name", required=True, help="Model name")
    model_parser.add_argument("--version", required=True, help="Model version")
    model_parser.add_argument("--architecture", required=True, help="Model architecture")
    model_parser.add_argument("--framework", required=True, help="ML framework")
    model_parser.add_argument("--dataset", required=True, help="Training dataset ID")
    model_parser.add_argument("--duration", type=float, required=True, help="Training duration (hours)")
    model_parser.add_argument("--file-path", required=True, help="Path to model file")
    model_parser.add_argument("--accuracy", type=float, help="Model accuracy")
    model_parser.add_argument("--f1-score", type=float, help="Model F1 score")
    model_parser.add_argument("--tags", nargs="*", help="Model tags")
    
    # List commands
    subparsers.add_parser("list-datasets", help="List all datasets")
    subparsers.add_parser("list-models", help="List all models")
    subparsers.add_parser("report", help="Generate registry report")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    registry = Registry(args.registry_dir)
    
    if args.command == "register-dataset":
        registry.register_dataset(
            name=args.name,
            version=args.version,
            dataset_type=DatasetType(args.type),
            source=args.source,
            license=args.license,
            description=args.description,
            file_path=args.file_path,
            num_samples=args.num_samples,
            num_classes=args.num_classes,
            tags=args.tags or []
        )
    
    elif args.command == "register-model":
        metrics = {}
        if args.accuracy:
            metrics["accuracy"] = args.accuracy
        if args.f1_score:
            metrics["f1_score"] = args.f1_score
        
        registry.register_model(
            name=args.name,
            version=args.version,
            architecture=args.architecture,
            framework=args.framework,
            trained_on_dataset=args.dataset,
            training_duration_hours=args.duration,
            file_path=args.file_path,
            hyperparameters={},
            metrics=metrics,
            tags=args.tags or []
        )
    
    elif args.command == "list-datasets":
        datasets = registry.list_datasets()
        print(f"Found {len(datasets)} datasets:")
        for dataset in datasets:
            print(f"  {dataset['id']}: {dataset['name']} v{dataset['version']} ({dataset['dataset_type']})")
    
    elif args.command == "list-models":
        models = registry.list_models()
        print(f"Found {len(models)} models:")
        for model in models:
            print(f"  {model['id']}: {model['name']} v{model['version']} ({model['status']})")
    
    elif args.command == "report":
        print(registry.generate_report())


if __name__ == "__main__":
    main()