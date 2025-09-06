# DJI O4 Air Unit Integration

This module provides comprehensive integration with DJI O4 Air Units for live video streaming and telemetry data collection in SAR operations.

## Overview

The DJI O4 integration supports two primary modes:

1. **Mobile SDK Mode** - For Android devices using DJI Mobile SDK
2. **Payload SDK Mode** - For Jetson/Linux devices using DJI Payload SDK

## Features

- ✅ Live H.264 video streaming (up to 1080p@60fps)
- ✅ Real-time telemetry data (GPS, attitude, gimbal, battery)
- ✅ Hardware-accelerated video decoding
- ✅ Frame-telemetry synchronization
- ✅ Automatic platform detection
- ✅ Recording and playback capabilities
- ✅ Comprehensive error handling and recovery
- ✅ Test suite with synthetic data

## Quick Start

### Basic Usage

```python
import asyncio
from connection.dji_o4 import DJIO4Connection

async def main():
    # Auto-detect platform and create connection
    connection = DJIO4Connection.create('auto')
    
    # Connect to aircraft
    if await connection.connect():
        print("Connected to DJI aircraft!")
        
        # Stream video and telemetry
        async for frame_data, telemetry in connection.stream_frames():
            print(f"Frame: {frame_data.shape}, Alt: {telemetry.altitude_agl:.1f}m")
            
            # Process frame for detection/tracking
            # your_detection_pipeline(frame_data.frame)
            
    await connection.disconnect()

asyncio.run(main())
```

### Configuration

```python
# Custom configuration
config = {
    'video_fps': 30,
    'video_resolution': '1920x1080',
    'telemetry_rate': 10,
    'app_key': 'your_dji_app_key'  # For Mobile SDK
}

connection = DJIO4Connection.create('mobile', config)
```

## Installation & Setup

### Prerequisites

#### For Mobile SDK (Android)

1. **DJI Developer Account**
   - Register at [developer.dji.com](https://developer.dji.com)
   - Create app and obtain App Key
   - Download DJI Mobile SDK v4.16+

2. **Android Requirements**
   - Android API level 21+ (Android 5.0+)
   - USB debugging enabled
   - DJI Fly app installed and aircraft bound

3. **Hardware**
   - DJI O4 Air Unit
   - Compatible DJI aircraft (Mini 3, Air 2S, etc.)
   - Android device with USB-C/OTG support

#### For Payload SDK (Jetson)

1. **DJI Payload SDK**
   - Download DJI Payload SDK v3.8+
   - Register payload device with DJI

2. **Jetson Requirements**
   - JetPack 4.6+ or 5.0+
   - CUDA support enabled
   - GStreamer with hardware acceleration

3. **Hardware**
   - DJI O4 Air Unit with Payload SDK support
   - UART/USB connection to flight controller
   - Jetson Nano/Xavier/Orin

### Installation Steps

#### 1. Install Dependencies

```bash
# Core dependencies
pip install opencv-python numpy asyncio

# For Jetson (GStreamer)
sudo apt-get install gstreamer1.0-tools gstreamer1.0-plugins-*
pip install gi PyGObject

# For testing
pip install pytest pytest-asyncio
```

#### 2. Configure DJI Settings

```bash
# Copy configuration template
cp configs/dji_o4_config.yaml.example configs/dji_o4_config.yaml

# Edit configuration
nano configs/dji_o4_config.yaml
```

#### 3. Set Up App Key (Mobile SDK)

```yaml
# In configs/dji_o4_config.yaml
mobile_sdk:
  app_key: "YOUR_DJI_APP_KEY_HERE"
  app_name: "Foresight SAR"
```

#### 4. Configure Hardware (Payload SDK)

```yaml
# In configs/dji_o4_config.yaml
payload_sdk:
  hardware:
    uart_device: "/dev/ttyUSB0"
    baudrate: 921600
```

## Platform-Specific Setup

### Android (Mobile SDK)

1. **Enable Developer Options**
   ```
   Settings → About Phone → Tap "Build Number" 7 times
   Settings → Developer Options → Enable USB Debugging
   ```

2. **Install DJI Fly App**
   - Download from Google Play Store
   - Bind your aircraft to DJI account
   - Verify connection works in DJI Fly

3. **Connect Hardware**
   ```
   Aircraft → DJI O4 Air Unit → USB-C Cable → Android Device
   ```

4. **Test Connection**
   ```bash
   python -m connection.test_dji_o4
   ```

### Jetson (Payload SDK)

1. **Install JetPack**
   ```bash
   # Flash JetPack 5.0+ to Jetson
   sudo apt update && sudo apt upgrade
   ```

2. **Configure UART**
   ```bash
   # Add user to dialout group
   sudo usermod -a -G dialout $USER
   
   # Configure UART permissions
   sudo chmod 666 /dev/ttyUSB0
   ```

3. **Install GStreamer**
   ```bash
   sudo apt-get install \
       gstreamer1.0-tools \
       gstreamer1.0-plugins-good \
       gstreamer1.0-plugins-bad \
       gstreamer1.0-plugins-ugly \
       gstreamer1.0-libav
   ```

4. **Test Hardware Acceleration**
   ```bash
   gst-launch-1.0 videotestsrc ! nvvidconv ! 'video/x-raw(memory:NVMM)' ! nvv4l2h264enc ! fakesink
   ```

5. **Connect Hardware**
   ```
   Aircraft → Flight Controller → UART/USB → Jetson
   ```

## Configuration Reference

### Video Settings

```yaml
video:
  resolution: "1920x1080"  # 1920x1080, 1280x720, 640x480
  fps: 30                  # Target frame rate
  codec: "H264"           # Video codec
  bitrate: 8000           # kbps
```

### Telemetry Settings

```yaml
telemetry:
  rate: 10                # Hz - update frequency
  enable_gps: true
  enable_attitude: true
  enable_velocity: true
  enable_gimbal: true
  enable_battery: true
```

### Recording Settings

```yaml
recording:
  enable: true
  output_dir: "data/recordings"
  video:
    format: "mp4"
    quality: "high"
  telemetry:
    format: "json"
    compression: true
```

## Testing

### Unit Tests

```bash
# Run all tests
pytest connection/test_dji_o4.py -v

# Run specific test
pytest connection/test_dji_o4.py::TestDJIO4Connection::test_mobile_sdk_simulation -v

# Run with coverage
pytest connection/test_dji_o4.py --cov=connection.dji_o4
```

### Integration Tests

```bash
# Test with recorded data
python connection/test_dji_o4.py

# Test Mobile SDK simulation
python -c "from connection.dji_o4 import DJIO4MobileSDK; import asyncio; asyncio.run(DJIO4MobileSDK().connect())"

# Test Payload SDK simulation
python -c "from connection.dji_o4 import DJIO4PayloadSDK; import asyncio; asyncio.run(DJIO4PayloadSDK().connect())"
```

### Creating Test Data

```python
from connection.test_dji_o4 import TestDataManager

# Create synthetic test data
manager = TestDataManager()
video_path = manager.create_test_video("test_flight.mp4", duration=30)
telemetry_path = manager.create_test_telemetry("test_telemetry.json", duration=30)

print(f"Created: {video_path}, {telemetry_path}")
```

## Troubleshooting

### Common Issues

#### Mobile SDK

**Issue**: "App Key Invalid"
```
Solution: 
1. Verify app key in configs/dji_o4_config.yaml
2. Check DJI developer account status
3. Ensure app is activated for your aircraft
```

**Issue**: "USB Connection Failed"
```
Solution:
1. Enable USB debugging on Android
2. Check USB cable and OTG support
3. Try different USB port
4. Restart DJI Fly app
```

**Issue**: "No Video Stream"
```
Solution:
1. Check aircraft is powered and connected
2. Verify camera gimbal is not obstructed
3. Check video resolution/fps settings
4. Restart connection
```

#### Payload SDK

**Issue**: "UART Connection Failed"
```
Solution:
1. Check UART device path (/dev/ttyUSB0)
2. Verify baudrate (921600)
3. Check user permissions (dialout group)
4. Test with: sudo chmod 666 /dev/ttyUSB0
```

**Issue**: "GStreamer Pipeline Error"
```
Solution:
1. Install missing GStreamer plugins
2. Check hardware acceleration support
3. Verify CUDA/NVENC availability
4. Test with simpler pipeline
```

**Issue**: "High Latency/Frame Drops"
```
Solution:
1. Reduce video resolution/fps
2. Enable hardware acceleration
3. Increase GPU memory allocation
4. Check network bandwidth
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug features
config = {
    'debug': {
        'save_raw_frames': True,
        'telemetry_logging': True,
        'performance_metrics': True
    }
}
```

### Performance Optimization

#### For Jetson Nano
```yaml
platform_overrides:
  jetson_nano:
    video:
      fps: 15
      resolution: "1280x720"
    gstreamer:
      gpu_memory_fraction: 0.5
```

#### For High-Performance Systems
```yaml
video:
  fps: 60
  resolution: "1920x1080"
qos:
  adaptive_bitrate:
    enable: true
    max_bitrate: 15000
```

## API Reference

### Classes

#### `DJIO4Connection`
Factory class for creating DJI O4 connections.

```python
@staticmethod
def create(platform: str = 'auto', config: Dict = None) -> DJIO4ConnectionBase
```

#### `DJIO4MobileSDK`
DJI Mobile SDK implementation.

```python
async def connect() -> bool
async def disconnect() -> None
async def stream_frames() -> AsyncGenerator[Tuple[FrameData, TelemetryData], None]
```

#### `DJIO4PayloadSDK`
DJI Payload SDK implementation.

```python
async def connect() -> bool
async def disconnect() -> None
async def stream_frames() -> AsyncGenerator[Tuple[FrameData, TelemetryData], None]
```

### Data Classes

#### `TelemetryData`
Telemetry information from aircraft.

```python
@dataclass
class TelemetryData:
    timestamp: float
    latitude: float
    longitude: float
    altitude_msl: float
    altitude_agl: float
    heading: float
    # ... additional fields
```

#### `FrameData`
Video frame with metadata.

```python
@dataclass
class FrameData:
    frame: np.ndarray
    timestamp: float
    pts: int
    frame_number: int
    width: int
    height: int
```

## Contributing

1. **Adding New Features**
   - Follow existing code patterns
   - Add comprehensive tests
   - Update documentation

2. **Testing**
   - Test on both platforms (Android/Jetson)
   - Use recorded data for reproducible tests
   - Verify hardware acceleration works

3. **Documentation**
   - Update this README for new features
   - Add docstrings to all public methods
   - Include configuration examples

## License

This module is part of the Foresight SAR system. See LICENSE file for details.

## Support

For issues and questions:
1. Check troubleshooting section above
2. Review test output and logs
3. Create issue with full error details and configuration