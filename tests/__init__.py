"""Test suite for Foresight SAR System.

This package contains comprehensive tests for all components of the Foresight
Search and Rescue system, including unit tests, integration tests, and
acceptance tests.

Test Structure:
- unit/: Unit tests for individual components
- integration/: Integration tests for component interactions
- acceptance/: End-to-end acceptance tests
- fixtures/: Test data and fixtures
- utils/: Test utilities and helpers

Usage:
    # Run all tests
    pytest
    
    # Run specific test category
    pytest tests/unit/
    pytest tests/integration/
    pytest tests/acceptance/
    
    # Run with coverage
    pytest --cov=src --cov-report=html
    
    # Run performance tests
    pytest -m performance
    
    # Run security tests
    pytest -m security
"""

__version__ = "1.0.0"
__author__ = "Foresight SAR Team"