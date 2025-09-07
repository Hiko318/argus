"""Test script for Victim-Lock re-identification pipeline.

Runs comprehensive tests including consistency tests, performance benchmarks,
and True Positive Rate measurements at various thresholds.
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from reid.victim_lock_pipeline import VictimLockPipeline, DetectionInput, PipelineMode, AlertLevel
from reid.face_embedder import FaceModel
from reid.body_reid import BodyModel


class VictimLockTester:
    """Comprehensive tester for Victim-Lock pipeline"""
    
    def __init__(self, 
                 test_data_dir: str = "test_data",
                 results_dir: str = "test_results",
                 face_model: FaceModel = FaceModel.ARCFACE,
                 body_model: BodyModel = BodyModel.OSNET_X1_0):
        
        self.test_data_dir = Path(test_data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.results_dir / 'test.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize pipeline
        self.pipeline = VictimLockPipeline(
            gallery_dir=str(self.results_dir / "test_gallery"),
            face_model=face_model,
            body_model=body_model,
            mode=PipelineMode.AUTO
        )
        
        self.logger.info(f"VictimLock tester initialized with {face_model.value} + {body_model.value}")
    
    def create_synthetic_dataset(self, num_persons: int = 10, images_per_person: int = 5) -> Dict[str, Any]:
        """Create synthetic test dataset
        
        Args:
            num_persons: Number of synthetic persons to create
            images_per_person: Number of images per person
            
        Returns:
            Dataset information
        """
        self.logger.info(f"Creating synthetic dataset: {num_persons} persons, {images_per_person} images each")
        
        dataset_dir = self.test_data_dir / "synthetic"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_info = {
            'persons': {},
            'total_images': 0,
            'creation_time': time.time()
        }
        
        for person_id in range(num_persons):
            person_name = f"person_{person_id:03d}"
            person_dir = dataset_dir / person_name
            person_dir.mkdir(exist_ok=True)
            
            person_images = []
            
            for img_id in range(images_per_person):
                # Create synthetic image (colored rectangle with noise)
                # Different colors for different persons
                base_color = (
                    (person_id * 30) % 255,
                    (person_id * 50) % 255,
                    (person_id * 70) % 255
                )
                
                # Random variations
                height = np.random.randint(200, 400)
                width = np.random.randint(100, 200)
                
                # Create base image
                image = np.full((height, width, 3), base_color, dtype=np.uint8)
                
                # Add noise and patterns
                noise = np.random.randint(-30, 30, (height, width, 3))
                image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                
                # Add some geometric patterns to make it more distinctive
                cv2.rectangle(image, 
                            (width//4, height//4), 
                            (3*width//4, 3*height//4), 
                            (255-base_color[0], 255-base_color[1], 255-base_color[2]), 
                            2)
                
                # Add person-specific pattern
                for i in range(person_id % 5 + 1):
                    y = height // (i + 2)
                    cv2.line(image, (0, y), (width, y), (255, 255, 255), 1)
                
                # Save image
                img_path = person_dir / f"{person_name}_{img_id:02d}.jpg"
                cv2.imwrite(str(img_path), image)
                person_images.append(str(img_path))
                dataset_info['total_images'] += 1
            
            dataset_info['persons'][person_name] = {
                'person_id': person_id,
                'images': person_images,
                'num_images': len(person_images)
            }
        
        # Save dataset info
        with open(dataset_dir / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        self.logger.info(f"Synthetic dataset created: {dataset_info['total_images']} total images")
        return dataset_info
    
    def load_test_images(self, dataset_path: str) -> Dict[str, List[np.ndarray]]:
        """Load test images from directory structure
        
        Expected structure:
        dataset_path/
        ├── person_001/
        │   ├── image_01.jpg
        │   └── image_02.jpg
        └── person_002/
            ├── image_01.jpg
            └── image_02.jpg
        """
        dataset_path = Path(dataset_path)
        images_dict = {}
        
        if not dataset_path.exists():
            self.logger.warning(f"Dataset path does not exist: {dataset_path}")
            return images_dict
        
        for person_dir in dataset_path.iterdir():
            if person_dir.is_dir():
                person_id = person_dir.name
                person_images = []
                
                for img_file in person_dir.glob("*.jpg"):
                    try:
                        image = cv2.imread(str(img_file))
                        if image is not None:
                            person_images.append(image)
                    except Exception as e:
                        self.logger.warning(f"Failed to load image {img_file}: {e}")
                
                if person_images:
                    images_dict[person_id] = person_images
                    self.logger.debug(f"Loaded {len(person_images)} images for {person_id}")
        
        self.logger.info(f"Loaded images for {len(images_dict)} persons")
        return images_dict
    
    def run_consistency_test(self, 
                           gallery_split: float = 0.6,
                           use_synthetic: bool = True,
                           dataset_path: str = None) -> Dict[str, Any]:
        """Run consistency test: gallery vs probe
        
        Args:
            gallery_split: Fraction of images to use for gallery (rest for probe)
            use_synthetic: Whether to use synthetic dataset
            dataset_path: Path to real dataset (if not using synthetic)
            
        Returns:
            Test results with TPR, FPR, and other metrics
        """
        self.logger.info("Starting consistency test...")
        
        # Load or create dataset
        if use_synthetic:
            dataset_info = self.create_synthetic_dataset(num_persons=8, images_per_person=6)
            images_dict = self.load_test_images(self.test_data_dir / "synthetic")
        else:
            if dataset_path is None:
                raise ValueError("dataset_path must be provided when use_synthetic=False")
            images_dict = self.load_test_images(dataset_path)
        
        if not images_dict:
            raise ValueError("No test images loaded")
        
        # Split into gallery and probe sets
        gallery_images = {}
        probe_images = {}
        
        for person_id, images in images_dict.items():
            num_gallery = max(1, int(len(images) * gallery_split))
            
            # Shuffle images
            shuffled_images = images.copy()
            np.random.shuffle(shuffled_images)
            
            gallery_images[person_id] = shuffled_images[:num_gallery]
            probe_images[person_id] = shuffled_images[num_gallery:]
        
        # Add some unknown persons to probe set (for false positive testing)
        if use_synthetic:
            unknown_dataset = self.create_synthetic_dataset(num_persons=3, images_per_person=2)
            unknown_images = self.load_test_images(self.test_data_dir / "synthetic")
            
            for person_id, images in unknown_images.items():
                if person_id not in gallery_images:  # Only add truly unknown persons
                    probe_images[f"unknown_{person_id}"] = images
        
        self.logger.info(
            f"Gallery: {len(gallery_images)} persons, "
            f"Probe: {len(probe_images)} persons (including unknowns)"
        )
        
        # Run consistency test
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        results = self.pipeline.run_consistency_test(
            gallery_images=gallery_images,
            probe_images=probe_images,
            similarity_thresholds=thresholds
        )
        
        # Add metadata
        test_metadata = {
            'test_type': 'consistency_test',
            'gallery_split': gallery_split,
            'use_synthetic': use_synthetic,
            'num_gallery_persons': len(gallery_images),
            'num_probe_persons': len(probe_images),
            'total_gallery_images': sum(len(imgs) for imgs in gallery_images.values()),
            'total_probe_images': sum(len(imgs) for imgs in probe_images.values()),
            'timestamp': time.time()
        }
        
        results['metadata'] = test_metadata
        
        # Save results
        results_file = self.results_dir / f"consistency_test_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Consistency test completed. Results saved to {results_file}")
        return results
    
    def run_performance_benchmark(self, num_iterations: int = 100) -> Dict[str, Any]:
        """Run performance benchmark
        
        Args:
            num_iterations: Number of iterations to run
            
        Returns:
            Performance metrics
        """
        self.logger.info(f"Starting performance benchmark ({num_iterations} iterations)...")
        
        # Create test images
        test_images = []
        for i in range(num_iterations):
            # Create random test image
            height = np.random.randint(200, 400)
            width = np.random.randint(100, 200)
            image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            test_images.append(image)
        
        # Add some target persons to gallery
        for i in range(5):
            ref_images = [test_images[i * 2], test_images[i * 2 + 1]]
            self.pipeline.add_target_person(
                name=f"benchmark_person_{i}",
                description=f"Benchmark person {i}",
                reference_images=ref_images,
                priority=1
            )
        
        # Reset stats
        self.pipeline.reset_stats()
        
        # Run benchmark
        processing_times = []
        
        for i, image in enumerate(tqdm(test_images, desc="Processing images")):
            start_time = time.time()
            
            # Create detection
            h, w = image.shape[:2]
            detection = DetectionInput(
                image=image,
                bbox=(0, 0, w, h),
                confidence=0.9,
                track_id=i
            )
            
            # Process detection
            result = self.pipeline.process_detection(detection)
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
        
        # Calculate metrics
        stats = self.pipeline.get_performance_stats()
        
        benchmark_results = {
            'num_iterations': num_iterations,
            'total_time': sum(processing_times),
            'avg_processing_time': np.mean(processing_times),
            'min_processing_time': np.min(processing_times),
            'max_processing_time': np.max(processing_times),
            'p50_processing_time': np.percentile(processing_times, 50),
            'p95_processing_time': np.percentile(processing_times, 95),
            'p99_processing_time': np.percentile(processing_times, 99),
            'fps': 1.0 / np.mean(processing_times),
            'pipeline_stats': stats,
            'timestamp': time.time()
        }
        
        # Save results
        results_file = self.results_dir / f"performance_benchmark_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        self.logger.info(
            f"Performance benchmark completed. "
            f"Avg: {benchmark_results['avg_processing_time']:.3f}s, "
            f"FPS: {benchmark_results['fps']:.1f}"
        )
        
        return benchmark_results
    
    def plot_consistency_results(self, results: Dict[str, Any], save_plot: bool = True):
        """Plot consistency test results
        
        Args:
            results: Results from consistency test
            save_plot: Whether to save plot to file
        """
        # Extract threshold results
        threshold_data = []
        for threshold, metrics in results.items():
            if isinstance(threshold, float):
                threshold_data.append(metrics)
        
        threshold_data.sort(key=lambda x: x['threshold'])
        
        thresholds = [d['threshold'] for d in threshold_data]
        tpr_values = [d['tpr'] for d in threshold_data]
        fpr_values = [d['fpr'] for d in threshold_data]
        precision_values = [d['precision'] for d in threshold_data]
        f1_values = [d['f1_score'] for d in threshold_data]
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # TPR vs Threshold
        ax1.plot(thresholds, tpr_values, 'b-o', label='True Positive Rate')
        ax1.set_xlabel('Similarity Threshold')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('TPR vs Similarity Threshold')
        ax1.grid(True)
        ax1.legend()
        
        # ROC Curve
        ax2.plot(fpr_values, tpr_values, 'r-o', label='ROC Curve')
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.grid(True)
        ax2.legend()
        
        # Precision vs Threshold
        ax3.plot(thresholds, precision_values, 'g-o', label='Precision')
        ax3.set_xlabel('Similarity Threshold')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision vs Similarity Threshold')
        ax3.grid(True)
        ax3.legend()
        
        # F1 Score vs Threshold
        ax4.plot(thresholds, f1_values, 'm-o', label='F1 Score')
        ax4.set_xlabel('Similarity Threshold')
        ax4.set_ylabel('F1 Score')
        ax4.set_title('F1 Score vs Similarity Threshold')
        ax4.grid(True)
        ax4.legend()
        
        plt.tight_layout()
        
        if save_plot:
            plot_file = self.results_dir / f"consistency_results_{int(time.time())}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {plot_file}")
        
        plt.show()
    
    def generate_report(self, 
                       consistency_results: Dict[str, Any] = None,
                       performance_results: Dict[str, Any] = None) -> str:
        """Generate comprehensive test report
        
        Args:
            consistency_results: Results from consistency test
            performance_results: Results from performance benchmark
            
        Returns:
            Report content as string
        """
        report_lines = [
            "# Victim-Lock Re-identification Pipeline Test Report",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Pipeline Configuration",
            f"- Face Model: {self.pipeline.face_embedder.model_type.value}",
            f"- Body Model: {self.pipeline.body_embedder.model_type.value}",
            f"- Pipeline Mode: {self.pipeline.mode.value}",
            ""
        ]
        
        # Consistency test results
        if consistency_results:
            report_lines.extend([
                "## Consistency Test Results",
                "",
                f"- Gallery Persons: {consistency_results['metadata']['num_gallery_persons']}",
                f"- Probe Persons: {consistency_results['metadata']['num_probe_persons']}",
                f"- Gallery Images: {consistency_results['metadata']['total_gallery_images']}",
                f"- Probe Images: {consistency_results['metadata']['total_probe_images']}",
                "",
                "### Performance at Different Thresholds",
                "",
                "| Threshold | TPR | FPR | Precision | F1 Score |",
                "|-----------|-----|-----|-----------|----------|"
            ])
            
            for threshold, metrics in consistency_results.items():
                if isinstance(threshold, float):
                    report_lines.append(
                        f"| {threshold:.1f} | {metrics['tpr']:.3f} | {metrics['fpr']:.3f} | "
                        f"{metrics['precision']:.3f} | {metrics['f1_score']:.3f} |"
                    )
            
            report_lines.append("")
        
        # Performance benchmark results
        if performance_results:
            report_lines.extend([
                "## Performance Benchmark Results",
                "",
                f"- Total Iterations: {performance_results['num_iterations']}",
                f"- Average Processing Time: {performance_results['avg_processing_time']:.3f}s",
                f"- P95 Processing Time: {performance_results['p95_processing_time']:.3f}s",
                f"- Average FPS: {performance_results['fps']:.1f}",
                "",
                "### Processing Time Distribution",
                f"- Min: {performance_results['min_processing_time']:.3f}s",
                f"- P50: {performance_results['p50_processing_time']:.3f}s",
                f"- P95: {performance_results['p95_processing_time']:.3f}s",
                f"- P99: {performance_results['p99_processing_time']:.3f}s",
                f"- Max: {performance_results['max_processing_time']:.3f}s",
                ""
            ])
        
        # Pipeline statistics
        pipeline_stats = self.pipeline.get_performance_stats()
        report_lines.extend([
            "## Pipeline Statistics",
            "",
            f"- Total Processed: {pipeline_stats['total_processed']}",
            f"- Face Extractions: {pipeline_stats['face_extractions']}",
            f"- Body Extractions: {pipeline_stats['body_extractions']}",
            f"- Matches Found: {pipeline_stats['matches_found']}",
            ""
        ])
        
        if 'avg_face_quality' in pipeline_stats:
            report_lines.append(f"- Average Face Quality: {pipeline_stats['avg_face_quality']:.3f}")
        
        if 'avg_body_quality' in pipeline_stats:
            report_lines.append(f"- Average Body Quality: {pipeline_stats['avg_body_quality']:.3f}")
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_file = self.results_dir / f"test_report_{int(time.time())}.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"Test report saved to {report_file}")
        return report_content
    
    def run_full_test_suite(self) -> Dict[str, Any]:
        """Run complete test suite
        
        Returns:
            Combined results from all tests
        """
        self.logger.info("Starting full test suite...")
        
        results = {
            'start_time': time.time(),
            'tests': {}
        }
        
        try:
            # Run consistency test
            self.logger.info("Running consistency test...")
            consistency_results = self.run_consistency_test()
            results['tests']['consistency'] = consistency_results
            
            # Plot consistency results
            self.plot_consistency_results(consistency_results)
            
            # Run performance benchmark
            self.logger.info("Running performance benchmark...")
            performance_results = self.run_performance_benchmark()
            results['tests']['performance'] = performance_results
            
            # Generate report
            report = self.generate_report(consistency_results, performance_results)
            results['report'] = report
            
        except Exception as e:
            self.logger.error(f"Test suite failed: {e}")
            results['error'] = str(e)
        
        results['end_time'] = time.time()
        results['total_duration'] = results['end_time'] - results['start_time']
        
        # Save combined results
        results_file = self.results_dir / f"full_test_suite_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Full test suite completed in {results['total_duration']:.1f}s")
        return results


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Test Victim-Lock re-identification pipeline")
    parser.add_argument("--test-data-dir", default="test_data", help="Test data directory")
    parser.add_argument("--results-dir", default="test_results", help="Results output directory")
    parser.add_argument("--face-model", default="arcface", choices=["arcface", "facenet"], help="Face model")
    parser.add_argument("--body-model", default="osnet", choices=["osnet", "resnet50", "mobilenet"], help="Body model")
    parser.add_argument("--test-type", default="full", choices=["consistency", "performance", "full"], help="Test type")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset")
    parser.add_argument("--dataset-path", help="Path to real dataset")
    parser.add_argument("--iterations", type=int, default=100, help="Number of benchmark iterations")
    
    args = parser.parse_args()
    
    # Map model names
    face_model_map = {
        "arcface": FaceModel.ARCFACE,
        "facenet": FaceModel.FACENET
    }
    
    body_model_map = {
        "osnet": BodyModel.OSNET_X1_0,
        "resnet50": BodyModel.RESNET50,
        "mobilenet": BodyModel.MOBILENET_V2
    }
    
    # Initialize tester
    tester = VictimLockTester(
        test_data_dir=args.test_data_dir,
        results_dir=args.results_dir,
        face_model=face_model_map[args.face_model],
        body_model=body_model_map[args.body_model]
    )
    
    # Run tests
    if args.test_type == "consistency":
        results = tester.run_consistency_test(
            use_synthetic=args.synthetic,
            dataset_path=args.dataset_path
        )
        tester.plot_consistency_results(results)
        
    elif args.test_type == "performance":
        results = tester.run_performance_benchmark(num_iterations=args.iterations)
        
    elif args.test_type == "full":
        results = tester.run_full_test_suite()
    
    print("\nTest completed successfully!")
    print(f"Results saved to: {tester.results_dir}")


if __name__ == "__main__":
    main()