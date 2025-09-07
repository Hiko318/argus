#!/usr/bin/env python3
"""
Performance Profiler & Thermal Monitor for Foresight SAR System

Runs 30-60 minute soak tests on Jetson platforms:
- Monitors CPU/GPU temperatures
- Tracks FPS performance
- Logs power consumption
- Implements thermal throttling watchdog
- Generates performance reports
"""

import os
import sys
import time
import json
import threading
import subprocess
import psutil
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
import cv2


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: float
    cpu_temp: Optional[float]
    gpu_temp: Optional[float]
    cpu_usage: float
    gpu_usage: Optional[float]
    memory_usage: float
    gpu_memory_usage: Optional[float]
    fps: Optional[float]
    power_draw: Optional[float]
    thermal_throttling: bool
    frequency_cpu: Optional[float]
    frequency_gpu: Optional[float]


class JetsonMonitor:
    """Hardware monitoring for NVIDIA Jetson platforms"""
    
    def __init__(self):
        self.is_jetson = self._detect_jetson()
        self.tegrastats_process = None
        self.tegrastats_data = {}
        
    def _detect_jetson(self) -> bool:
        """Detect if running on Jetson platform"""
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip()
                return 'jetson' in model.lower() or 'tegra' in model.lower()
        except FileNotFoundError:
            return False
    
    def start_tegrastats(self):
        """Start tegrastats monitoring"""
        if not self.is_jetson:
            return
        
        try:
            self.tegrastats_process = subprocess.Popen(
                ['tegrastats', '--interval', '1000'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
        except FileNotFoundError:
            logging.warning("tegrastats not found, using alternative monitoring")
    
    def stop_tegrastats(self):
        """Stop tegrastats monitoring"""
        if self.tegrastats_process:
            self.tegrastats_process.terminate()
            self.tegrastats_process.wait()
    
    def get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature"""
        try:
            # Try thermal zones
            for zone_path in Path('/sys/class/thermal').glob('thermal_zone*'):
                zone_type_path = zone_path / 'type'
                if zone_type_path.exists():
                    with open(zone_type_path, 'r') as f:
                        zone_type = f.read().strip()
                    
                    if 'cpu' in zone_type.lower() or 'soc' in zone_type.lower():
                        temp_path = zone_path / 'temp'
                        if temp_path.exists():
                            with open(temp_path, 'r') as f:
                                temp_millicelsius = int(f.read().strip())
                                return temp_millicelsius / 1000.0
            
            # Fallback: try first thermal zone
            temp_path = Path('/sys/class/thermal/thermal_zone0/temp')
            if temp_path.exists():
                with open(temp_path, 'r') as f:
                    temp_millicelsius = int(f.read().strip())
                    return temp_millicelsius / 1000.0
        except Exception:
            pass
        
        return None
    
    def get_gpu_temperature(self) -> Optional[float]:
        """Get GPU temperature"""
        try:
            # Try nvidia-smi
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception:
            pass
        
        try:
            # Try thermal zones for GPU
            for zone_path in Path('/sys/class/thermal').glob('thermal_zone*'):
                zone_type_path = zone_path / 'type'
                if zone_type_path.exists():
                    with open(zone_type_path, 'r') as f:
                        zone_type = f.read().strip()
                    
                    if 'gpu' in zone_type.lower():
                        temp_path = zone_path / 'temp'
                        if temp_path.exists():
                            with open(temp_path, 'r') as f:
                                temp_millicelsius = int(f.read().strip())
                                return temp_millicelsius / 1000.0
        except Exception:
            pass
        
        return None
    
    def get_gpu_usage(self) -> Optional[float]:
        """Get GPU utilization percentage"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception:
            pass
        
        return None
    
    def get_gpu_memory_usage(self) -> Optional[float]:
        """Get GPU memory usage percentage"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                used, total = map(int, result.stdout.strip().split(', '))
                return (used / total) * 100.0
        except Exception:
            pass
        
        return None
    
    def get_power_draw(self) -> Optional[float]:
        """Get power consumption in watts"""
        try:
            # Try nvidia-smi for GPU power
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception:
            pass
        
        try:
            # Try Jetson power monitoring
            power_paths = [
                '/sys/bus/i2c/drivers/ina3221x/*/iio:device*/in_power0_input',
                '/sys/bus/i2c/drivers/ina3221x/*/iio:device*/in_power1_input'
            ]
            
            for pattern in power_paths:
                for power_file in Path('/').glob(pattern.lstrip('/')):
                    if power_file.exists():
                        with open(power_file, 'r') as f:
                            power_mw = int(f.read().strip())
                            return power_mw / 1000.0  # Convert to watts
        except Exception:
            pass
        
        return None
    
    def get_cpu_frequency(self) -> Optional[float]:
        """Get current CPU frequency in MHz"""
        try:
            freq_path = Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq')
            if freq_path.exists():
                with open(freq_path, 'r') as f:
                    freq_khz = int(f.read().strip())
                    return freq_khz / 1000.0  # Convert to MHz
        except Exception:
            pass
        
        return None
    
    def get_gpu_frequency(self) -> Optional[float]:
        """Get current GPU frequency in MHz"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=clocks.current.graphics', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception:
            pass
        
        return None
    
    def check_thermal_throttling(self) -> bool:
        """Check if thermal throttling is active"""
        try:
            # Check CPU frequency scaling
            max_freq_path = Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq')
            cur_freq_path = Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq')
            
            if max_freq_path.exists() and cur_freq_path.exists():
                with open(max_freq_path, 'r') as f:
                    max_freq = int(f.read().strip())
                with open(cur_freq_path, 'r') as f:
                    cur_freq = int(f.read().strip())
                
                # Consider throttling if current freq is significantly below max
                if cur_freq < max_freq * 0.8:
                    return True
        except Exception:
            pass
        
        return False


class PerformanceProfiler:
    """Main performance profiling and monitoring class"""
    
    def __init__(self, output_dir: str = "performance_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.monitor = JetsonMonitor()
        self.metrics_history: List[PerformanceMetrics] = []
        self.running = False
        self.monitor_thread = None
        self.fps_counter = FPSCounter()
        
        # Setup logging
        log_file = self.output_dir / f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Thermal thresholds
        self.cpu_temp_threshold = 85.0  # °C
        self.gpu_temp_threshold = 87.0  # °C
        self.thermal_warnings = 0
        
    def collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        timestamp = time.time()
        
        # CPU metrics
        cpu_temp = self.monitor.get_cpu_temperature()
        cpu_usage = psutil.cpu_percent(interval=None)
        cpu_freq = self.monitor.get_cpu_frequency()
        
        # GPU metrics
        gpu_temp = self.monitor.get_gpu_temperature()
        gpu_usage = self.monitor.get_gpu_usage()
        gpu_memory = self.monitor.get_gpu_memory_usage()
        gpu_freq = self.monitor.get_gpu_frequency()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Power and thermal
        power_draw = self.monitor.get_power_draw()
        thermal_throttling = self.monitor.check_thermal_throttling()
        
        # FPS
        fps = self.fps_counter.get_fps()
        
        return PerformanceMetrics(
            timestamp=timestamp,
            cpu_temp=cpu_temp,
            gpu_temp=gpu_temp,
            cpu_usage=cpu_usage,
            gpu_usage=gpu_usage,
            memory_usage=memory_usage,
            gpu_memory_usage=gpu_memory,
            fps=fps,
            power_draw=power_draw,
            thermal_throttling=thermal_throttling,
            frequency_cpu=cpu_freq,
            frequency_gpu=gpu_freq
        )
    
    def monitor_loop(self):
        """Main monitoring loop"""
        self.monitor.start_tegrastats()
        
        while self.running:
            try:
                metrics = self.collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check thermal thresholds
                self.check_thermal_warnings(metrics)
                
                # Log metrics periodically
                if len(self.metrics_history) % 60 == 0:  # Every minute
                    self.log_current_status(metrics)
                
                time.sleep(1.0)  # 1 second interval
                
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {e}")
                time.sleep(1.0)
        
        self.monitor.stop_tegrastats()
    
    def check_thermal_warnings(self, metrics: PerformanceMetrics):
        """Check for thermal warnings and throttling"""
        warnings = []
        
        if metrics.cpu_temp and metrics.cpu_temp > self.cpu_temp_threshold:
            warnings.append(f"CPU temperature high: {metrics.cpu_temp:.1f}°C")
        
        if metrics.gpu_temp and metrics.gpu_temp > self.gpu_temp_threshold:
            warnings.append(f"GPU temperature high: {metrics.gpu_temp:.1f}°C")
        
        if metrics.thermal_throttling:
            warnings.append("Thermal throttling detected")
        
        if warnings:
            self.thermal_warnings += 1
            for warning in warnings:
                self.logger.warning(warning)
    
    def log_current_status(self, metrics: PerformanceMetrics):
        """Log current system status"""
        status_parts = []
        
        if metrics.cpu_temp:
            status_parts.append(f"CPU: {metrics.cpu_temp:.1f}°C")
        if metrics.gpu_temp:
            status_parts.append(f"GPU: {metrics.gpu_temp:.1f}°C")
        
        status_parts.append(f"CPU Usage: {metrics.cpu_usage:.1f}%")
        
        if metrics.gpu_usage:
            status_parts.append(f"GPU Usage: {metrics.gpu_usage:.1f}%")
        
        status_parts.append(f"Memory: {metrics.memory_usage:.1f}%")
        
        if metrics.fps:
            status_parts.append(f"FPS: {metrics.fps:.1f}")
        
        if metrics.power_draw:
            status_parts.append(f"Power: {metrics.power_draw:.1f}W")
        
        self.logger.info(" | ".join(status_parts))
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if self.running:
            return
        
        self.logger.info("Starting performance monitoring...")
        self.running = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop)
        self.monitor_thread.start()
        
        # Start FPS counter
        self.fps_counter.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        if not self.running:
            return
        
        self.logger.info("Stopping performance monitoring...")
        self.running = False
        
        if self.monitor_thread:
            self.monitor_thread.join()
        
        self.fps_counter.stop()
    
    def run_soak_test(self, duration_minutes: int = 30, workload_type: str = "detection"):
        """Run soak test for specified duration"""
        self.logger.info(f"Starting {duration_minutes}-minute soak test ({workload_type} workload)")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        # Start monitoring
        self.start_monitoring()
        
        # Start workload
        workload_thread = threading.Thread(
            target=self._run_workload,
            args=(workload_type, duration_minutes * 60)
        )
        workload_thread.start()
        
        try:
            # Wait for completion
            while time.time() < end_time and self.running:
                remaining = end_time - time.time()
                if remaining > 0:
                    self.logger.info(f"Soak test running... {remaining/60:.1f} minutes remaining")
                    time.sleep(60)  # Log every minute
        
        except KeyboardInterrupt:
            self.logger.info("Soak test interrupted by user")
        
        finally:
            # Stop monitoring and workload
            self.stop_monitoring()
            workload_thread.join(timeout=10)
            
            # Generate report
            report = self.generate_report()
            self.save_report(report)
            
            self.logger.info("Soak test completed")
            return report
    
    def _run_workload(self, workload_type: str, duration_seconds: int):
        """Run specified workload for testing"""
        if workload_type == "detection":
            self._run_detection_workload(duration_seconds)
        elif workload_type == "video":
            self._run_video_workload(duration_seconds)
        elif workload_type == "cpu":
            self._run_cpu_workload(duration_seconds)
        else:
            self.logger.warning(f"Unknown workload type: {workload_type}")
    
    def _run_detection_workload(self, duration_seconds: int):
        """Run object detection workload"""
        try:
            import torch
            
            # Create dummy model for testing
            model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 128, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                torch.nn.Linear(128, 10)
            )
            
            if torch.cuda.is_available():
                model = model.cuda()
            
            model.eval()
            
            start_time = time.time()
            frame_count = 0
            
            while time.time() - start_time < duration_seconds and self.running:
                # Generate dummy input
                if torch.cuda.is_available():
                    dummy_input = torch.randn(1, 3, 640, 480).cuda()
                else:
                    dummy_input = torch.randn(1, 3, 640, 480)
                
                # Run inference
                with torch.no_grad():
                    _ = model(dummy_input)
                
                frame_count += 1
                self.fps_counter.tick()
                
                # Small delay to prevent overwhelming
                time.sleep(0.01)
            
            self.logger.info(f"Detection workload completed: {frame_count} frames processed")
            
        except ImportError:
            self.logger.warning("PyTorch not available, using CPU workload instead")
            self._run_cpu_workload(duration_seconds)
        except Exception as e:
            self.logger.error(f"Detection workload error: {e}")
    
    def _run_video_workload(self, duration_seconds: int):
        """Run video processing workload"""
        try:
            # Create dummy video frames
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            start_time = time.time()
            frame_count = 0
            
            while time.time() - start_time < duration_seconds and self.running:
                # Simulate video processing
                processed = cv2.GaussianBlur(frame, (15, 15), 0)
                processed = cv2.Canny(processed, 50, 150)
                
                frame_count += 1
                self.fps_counter.tick()
                
                time.sleep(0.033)  # ~30 FPS
            
            self.logger.info(f"Video workload completed: {frame_count} frames processed")
            
        except Exception as e:
            self.logger.error(f"Video workload error: {e}")
    
    def _run_cpu_workload(self, duration_seconds: int):
        """Run CPU-intensive workload"""
        start_time = time.time()
        iterations = 0
        
        while time.time() - start_time < duration_seconds and self.running:
            # CPU-intensive computation
            _ = sum(i * i for i in range(10000))
            iterations += 1
            
            if iterations % 1000 == 0:
                self.fps_counter.tick()
        
        self.logger.info(f"CPU workload completed: {iterations} iterations")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        if not self.metrics_history:
            return {"error": "No metrics collected"}
        
        # Convert metrics to arrays for analysis
        timestamps = [m.timestamp for m in self.metrics_history]
        cpu_temps = [m.cpu_temp for m in self.metrics_history if m.cpu_temp is not None]
        gpu_temps = [m.gpu_temp for m in self.metrics_history if m.gpu_temp is not None]
        cpu_usage = [m.cpu_usage for m in self.metrics_history]
        gpu_usage = [m.gpu_usage for m in self.metrics_history if m.gpu_usage is not None]
        memory_usage = [m.memory_usage for m in self.metrics_history]
        fps_values = [m.fps for m in self.metrics_history if m.fps is not None]
        power_values = [m.power_draw for m in self.metrics_history if m.power_draw is not None]
        
        # Calculate statistics
        report = {
            "test_info": {
                "start_time": datetime.fromtimestamp(timestamps[0]).isoformat(),
                "end_time": datetime.fromtimestamp(timestamps[-1]).isoformat(),
                "duration_minutes": (timestamps[-1] - timestamps[0]) / 60,
                "total_samples": len(self.metrics_history),
                "thermal_warnings": self.thermal_warnings
            },
            "temperature": {
                "cpu": self._calculate_stats(cpu_temps, "°C") if cpu_temps else None,
                "gpu": self._calculate_stats(gpu_temps, "°C") if gpu_temps else None
            },
            "utilization": {
                "cpu": self._calculate_stats(cpu_usage, "%"),
                "gpu": self._calculate_stats(gpu_usage, "%") if gpu_usage else None,
                "memory": self._calculate_stats(memory_usage, "%")
            },
            "performance": {
                "fps": self._calculate_stats(fps_values, "fps") if fps_values else None
            },
            "power": {
                "consumption": self._calculate_stats(power_values, "W") if power_values else None
            },
            "thermal_events": {
                "throttling_detected": any(m.thermal_throttling for m in self.metrics_history),
                "cpu_over_threshold": len([t for t in cpu_temps if t > self.cpu_temp_threshold]),
                "gpu_over_threshold": len([t for t in gpu_temps if t > self.gpu_temp_threshold])
            }
        }
        
        return report
    
    def _calculate_stats(self, values: List[float], unit: str) -> Dict[str, Any]:
        """Calculate statistics for a list of values"""
        if not values:
            return None
        
        values_array = np.array(values)
        
        return {
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "mean": float(np.mean(values_array)),
            "median": float(np.median(values_array)),
            "std": float(np.std(values_array)),
            "unit": unit,
            "samples": len(values)
        }
    
    def save_report(self, report: Dict[str, Any]):
        """Save performance report to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON report
        json_file = self.output_dir / f"performance_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save CSV data
        csv_file = self.output_dir / f"performance_data_{timestamp}.csv"
        self._save_csv_data(csv_file)
        
        self.logger.info(f"Performance report saved: {json_file}")
        self.logger.info(f"Performance data saved: {csv_file}")
    
    def _save_csv_data(self, csv_file: Path):
        """Save metrics history as CSV"""
        import csv
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'timestamp', 'cpu_temp', 'gpu_temp', 'cpu_usage', 'gpu_usage',
                'memory_usage', 'gpu_memory_usage', 'fps', 'power_draw',
                'thermal_throttling', 'frequency_cpu', 'frequency_gpu'
            ])
            
            # Data
            for metrics in self.metrics_history:
                writer.writerow([
                    metrics.timestamp,
                    metrics.cpu_temp,
                    metrics.gpu_temp,
                    metrics.cpu_usage,
                    metrics.gpu_usage,
                    metrics.memory_usage,
                    metrics.gpu_memory_usage,
                    metrics.fps,
                    metrics.power_draw,
                    metrics.thermal_throttling,
                    metrics.frequency_cpu,
                    metrics.frequency_gpu
                ])


class FPSCounter:
    """FPS counter for performance monitoring"""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = []
        self.last_time = None
        self.running = False
    
    def start(self):
        """Start FPS counting"""
        self.running = True
        self.last_time = time.time()
    
    def stop(self):
        """Stop FPS counting"""
        self.running = False
    
    def tick(self):
        """Record a frame"""
        if not self.running:
            return
        
        current_time = time.time()
        if self.last_time is not None:
            frame_time = current_time - self.last_time
            self.frame_times.append(frame_time)
            
            # Keep only recent frames
            if len(self.frame_times) > self.window_size:
                self.frame_times.pop(0)
        
        self.last_time = current_time
    
    def get_fps(self) -> Optional[float]:
        """Get current FPS"""
        if len(self.frame_times) < 2:
            return None
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        if avg_frame_time > 0:
            return 1.0 / avg_frame_time
        
        return None


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Foresight SAR Performance Profiler")
    parser.add_argument("--duration", type=int, default=30,
                       help="Soak test duration in minutes (default: 30)")
    parser.add_argument("--workload", choices=["detection", "video", "cpu"],
                       default="detection", help="Workload type for testing")
    parser.add_argument("--output-dir", default="performance_logs",
                       help="Output directory for logs and reports")
    parser.add_argument("--monitor-only", action="store_true",
                       help="Monitor only, no workload")
    
    args = parser.parse_args()
    
    profiler = PerformanceProfiler(args.output_dir)
    
    try:
        if args.monitor_only:
            print(f"Starting monitoring (Ctrl+C to stop)...")
            profiler.start_monitoring()
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            
            profiler.stop_monitoring()
            report = profiler.generate_report()
            profiler.save_report(report)
        else:
            profiler.run_soak_test(args.duration, args.workload)
    
    except KeyboardInterrupt:
        print("\nProfiler interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()