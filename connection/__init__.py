"""Connection Module for Foresight SAR System

This module handles various connection types for drone and device integration,
including DJI O4 integration, phone streaming, simulation modes, stream bridging,
and data ingestion capabilities.
"""

from .dji_o4 import DJI_O4_Connection, DJI_O4_Config
from .phone_stream import PhoneStreamConnection, StreamConfig
from .simulation import SimulationConnection, SimulationConfig
from .stream_bridge import StreamBridge
from .capture_screen import ScreenCapture
from .simulated_feed import SimulatedFeed

__all__ = [
    'DJI_O4_Connection',
    'DJI_O4_Config',
    'PhoneStreamConnection', 
    'StreamConfig',
    'SimulationConnection',
    'SimulationConfig',
    'StreamBridge',
    'ScreenCapture',
    'SimulatedFeed'
]

__version__ = '1.0.0'