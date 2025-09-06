"""Packaging Module for Foresight SAR System

This module provides evidence packaging and export capabilities for
SAR operations, including metadata collection, file archiving, and
secure evidence handling.
"""

from .evidence_packager import EvidencePackager

__all__ = [
    'EvidencePackager'
]

__version__ = '1.0.0'