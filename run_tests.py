#!/usr/bin/env python3
"""Test runner script for Foresight SAR test suite."""

import sys
import os
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional


def run_command(cmd: List[str], cwd: Optional[str] = None) -> int:
    """Run a command and return the exit code."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd, check=False)
        return result.returncode
    except FileNotFoundError:
        print(f"Error: Command not found: {cmd[0]}")
        return 1
    except Exception as e:
        print(f"Error running command: {e}")
        return 1


def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    required_packages = [
        "pytest",
        "pytest-cov",
        "pytest-mock",
        "pytest-asyncio",
        "pytest-timeout"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


def run_unit_tests(verbose: bool = False, coverage: bool = True) -> int:
    """Run unit tests."""
    cmd = ["python", "-m", "pytest", "tests/unit", "-m", "unit"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing"])
    
    return run_command(cmd)


def run_integration_tests(verbose: bool = False, coverage: bool = True) -> int:
    """Run integration tests."""
    cmd = ["python", "-m", "pytest", "tests/integration", "-m", "integration"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing"])
    
    return run_command(cmd)


def run_e2e_tests(verbose: bool = False) -> int:
    """Run end-to-end tests."""
    cmd = ["python", "-m", "pytest", "tests/e2e", "-m", "e2e"]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd)


def run_smoke_tests(verbose: bool = False) -> int:
    """Run smoke tests."""
    cmd = ["python", "-m", "pytest", "-m", "smoke"]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd)


def run_performance_tests(verbose: bool = False) -> int:
    """Run performance tests."""
    cmd = ["python", "-m", "pytest", "-m", "performance"]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd)


def run_gpu_tests(verbose: bool = False) -> int:
    """Run GPU-specific tests."""
    cmd = ["python", "-m", "pytest", "-m", "gpu"]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd)


def run_all_tests(verbose: bool = False, coverage: bool = True, fast: bool = False) -> int:
    """Run all tests."""
    cmd = ["python", "-m", "pytest", "tests"]
    
    if fast:
        cmd.extend(["-m", "not slow"])
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml"
        ])
    
    return run_command(cmd)


def run_specific_test(test_path: str, verbose: bool = False) -> int:
    """Run a specific test file or test function."""
    cmd = ["python", "-m", "pytest", test_path]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd)


def run_tests_with_markers(markers: List[str], verbose: bool = False) -> int:
    """Run tests with specific markers."""
    cmd = ["python", "-m", "pytest"]
    
    if markers:
        marker_expr = " and ".join(markers)
        cmd.extend(["-m", marker_expr])
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd)


def generate_coverage_report() -> int:
    """Generate detailed coverage report."""
    print("Generating coverage report...")
    
    # Run tests with coverage
    exit_code = run_command([
        "python", "-m", "pytest", "tests",
        "--cov=src",
        "--cov-report=html:htmlcov",
        "--cov-report=xml:coverage.xml",
        "--cov-report=term-missing"
    ])
    
    if exit_code == 0:
        print("\nCoverage report generated:")
        print("  HTML: htmlcov/index.html")
        print("  XML: coverage.xml")
    
    return exit_code


def clean_test_artifacts() -> None:
    """Clean test artifacts and cache files."""
    print("Cleaning test artifacts...")
    
    artifacts = [
        ".pytest_cache",
        "htmlcov",
        "coverage.xml",
        ".coverage",
        "__pycache__",
        "*.pyc",
        "*.pyo"
    ]
    
    for artifact in artifacts:
        if "*" in artifact:
            # Handle glob patterns
            import glob
            for file_path in glob.glob(f"**/{artifact}", recursive=True):
                try:
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
                except OSError:
                    pass
        else:
            # Handle directories and files
            path = Path(artifact)
            if path.exists():
                if path.is_dir():
                    import shutil
                    shutil.rmtree(path)
                    print(f"Removed directory: {path}")
                else:
                    path.unlink()
                    print(f"Removed file: {path}")


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Foresight SAR Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --unit                    # Run unit tests
  python run_tests.py --integration             # Run integration tests
  python run_tests.py --e2e                     # Run end-to-end tests
  python run_tests.py --smoke                   # Run smoke tests
  python run_tests.py --all                     # Run all tests
  python run_tests.py --all --fast              # Run all tests except slow ones
  python run_tests.py --markers "unit and not slow"  # Run with custom markers
  python run_tests.py --test tests/unit/test_detector.py  # Run specific test
  python run_tests.py --coverage                # Generate coverage report
  python run_tests.py --clean                   # Clean test artifacts
        """
    )
    
    # Test type options
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument("--unit", action="store_true", help="Run unit tests")
    test_group.add_argument("--integration", action="store_true", help="Run integration tests")
    test_group.add_argument("--e2e", action="store_true", help="Run end-to-end tests")
    test_group.add_argument("--smoke", action="store_true", help="Run smoke tests")
    test_group.add_argument("--performance", action="store_true", help="Run performance tests")
    test_group.add_argument("--gpu", action="store_true", help="Run GPU tests")
    test_group.add_argument("--all", action="store_true", help="Run all tests")
    test_group.add_argument("--coverage", action="store_true", help="Generate coverage report")
    test_group.add_argument("--clean", action="store_true", help="Clean test artifacts")
    
    # Custom options
    parser.add_argument("--test", type=str, help="Run specific test file or function")
    parser.add_argument("--markers", type=str, help="Run tests with specific markers (e.g., 'unit and not slow')")
    
    # General options
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage reporting")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument("--check-deps", action="store_true", help="Check test dependencies")
    
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_deps:
        if check_dependencies():
            print("All test dependencies are installed.")
            return 0
        else:
            return 1
    
    # Clean artifacts if requested
    if args.clean:
        clean_test_artifacts()
        return 0
    
    # Check dependencies before running tests
    if not check_dependencies():
        print("\nPlease install missing dependencies before running tests.")
        return 1
    
    # Set up environment
    os.environ["PYTHONPATH"] = str(Path.cwd())
    
    # Run tests based on arguments
    exit_code = 0
    coverage = not args.no_coverage
    
    if args.unit:
        exit_code = run_unit_tests(args.verbose, coverage)
    elif args.integration:
        exit_code = run_integration_tests(args.verbose, coverage)
    elif args.e2e:
        exit_code = run_e2e_tests(args.verbose)
    elif args.smoke:
        exit_code = run_smoke_tests(args.verbose)
    elif args.performance:
        exit_code = run_performance_tests(args.verbose)
    elif args.gpu:
        exit_code = run_gpu_tests(args.verbose)
    elif args.all:
        exit_code = run_all_tests(args.verbose, coverage, args.fast)
    elif args.coverage:
        exit_code = generate_coverage_report()
    elif args.test:
        exit_code = run_specific_test(args.test, args.verbose)
    elif args.markers:
        markers = [m.strip() for m in args.markers.split("and")]
        exit_code = run_tests_with_markers(markers, args.verbose)
    else:
        # Default: run smoke tests
        print("No test type specified. Running smoke tests...")
        exit_code = run_smoke_tests(args.verbose)
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code: {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())