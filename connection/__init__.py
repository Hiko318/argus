"""Connection Module for Foresight SAR System

This module handles various connection types for drone and device integration,
including DJI O4 integration, phone streaming, simulation modes, stream bridging,
and data ingestion capabilities.
"""

from .dji_o4 import DJIO4Connection, TelemetryData
from .phone_stream import PhoneStreamConnection
from .capture_screen import ScreenCapture

# Import other modules that exist
try:
    from .simulation import SimulationConnection
except ImportError:
    SimulationConnection = None

try:
    from .stream_bridge import StreamBridge
except ImportError:
    StreamBridge = None

try:
    from .simulated_feed import SimulatedFeed
except ImportError:
    SimulatedFeed = None

__all__ = [
    'DJIO4Connection',
    'TelemetryData',
    'PhoneStreamConnection',
    'ScreenCapture'
]

# Add optional imports if they exist
if SimulationConnection is not None:
    __all__.append('SimulationConnection')
if StreamBridge is not None:
    __all__.append('StreamBridge')
if SimulatedFeed is not None:
    __all__.append('SimulatedFeed')

__version__ = '1.0.0'