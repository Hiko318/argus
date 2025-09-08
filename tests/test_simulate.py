#!/usr/bin/env python3
"""
Test suite for simulation mode functionality.

This module tests that main.py --simulate starts successfully and exits cleanly.
"""

import subprocess
import sys
import time
import pytest
from pathlib import Path


class TestSimulateMode:
    """Test cases for simulation mode."""
    
    def test_simulate_flag_exits_cleanly(self):
        """Test that main.py --simulate starts and can be terminated cleanly."""
        # Get the project root directory
        project_root = Path(__file__).parent.parent
        main_py_path = project_root / "main.py"
        
        # Ensure main.py exists
        assert main_py_path.exists(), f"main.py not found at {main_py_path}"
        
        # Start the simulation process
        process = subprocess.Popen(
            [sys.executable, str(main_py_path), "--simulate"],
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        try:
            # Give the process time to start up
            time.sleep(3)
            
            # Check if process is still running (it should be)
            assert process.poll() is None, "Process exited prematurely"
            
            # Terminate the process gracefully
            process.terminate()
            
            # Wait for clean shutdown with timeout
            try:
                stdout, stderr = process.communicate(timeout=10)
                exit_code = process.returncode
                
                # Process should exit cleanly (code 0 or -15 for SIGTERM)
                assert exit_code in [0, -15], f"Unexpected exit code: {exit_code}"
                
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate gracefully
                process.kill()
                stdout, stderr = process.communicate()
                pytest.fail("Process did not terminate gracefully within timeout")
                
        except Exception as e:
            # Ensure cleanup in case of test failure
            if process.poll() is None:
                process.kill()
                process.communicate()
            raise e
    
    def test_simulate_flag_starts_without_crash(self):
        """Test that main.py --simulate starts without immediate crash."""
        project_root = Path(__file__).parent.parent
        main_py_path = project_root / "main.py"
        
        # Start the simulation process
        process = subprocess.Popen(
            [sys.executable, str(main_py_path), "--simulate"],
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        try:
            # Wait a short time to see if it crashes immediately
            time.sleep(2)
            
            # Process should still be running
            assert process.poll() is None, "Simulation mode crashed on startup"
            
        finally:
            # Clean up
            if process.poll() is None:
                process.terminate()
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()