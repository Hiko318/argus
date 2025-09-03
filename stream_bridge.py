#!/usr/bin/env python3
"""
Stream Bridge - Connects ADB H.264 stream to Electron WebSocket
This script captures your phone's screen via ADB and forwards it to the Electron app
"""

import subprocess
import websocket
import threading
import time
import sys

class StreamBridge:
    def __init__(self):
        self.ws_url = "ws://127.0.0.1:9998"
        self.ws = None
        self.adb_process = None
        self.running = False
        
    def connect_websocket(self):
        """Connect to Electron WebSocket server"""
        try:
            self.ws = websocket.WebSocket()
            self.ws.connect(self.ws_url)
            print("✅ Connected to Electron WebSocket server")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to WebSocket: {e}")
            return False
    
    def start_adb_stream(self):
        """Start ADB screen recording stream"""
        try:
            cmd = [
                "adb", "exec-out", 
                "screenrecord", 
                "--bit-rate=8000000",
                "--output-format=h264",
                "--size=720x1280",
                "-"
            ]
            
            self.adb_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            print("✅ Started ADB screen recording")
            return True
        except Exception as e:
            print(f"❌ Failed to start ADB stream: {e}")
            return False
    
    def stream_data(self):
        """Stream H.264 data from ADB to WebSocket"""
        if not self.adb_process or not self.ws:
            return
            
        print("🔄 Starting H.264 stream bridge...")
        chunk_size = 4096
        
        try:
            while self.running and self.adb_process.poll() is None:
                data = self.adb_process.stdout.read(chunk_size)
                if data:
                    # Send binary H.264 data to WebSocket
                    self.ws.send_binary(data)
                    print(f"📡 Sent {len(data)} bytes to Electron")
                else:
                    time.sleep(0.01)  # Small delay if no data
                    
        except Exception as e:
            print(f"❌ Stream error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        
        if self.adb_process:
            self.adb_process.terminate()
            self.adb_process = None
            
        if self.ws:
            self.ws.close()
            self.ws = None
            
        print("🧹 Cleaned up resources")
    
    def run(self):
        """Main execution method"""
        print("🚀 Starting Phone Stream Bridge...")
        
        # Connect to WebSocket
        if not self.connect_websocket():
            return False
            
        # Start ADB stream
        if not self.start_adb_stream():
            return False
            
        # Start streaming
        self.running = True
        
        try:
            self.stream_data()
        except KeyboardInterrupt:
            print("\n⏹️ Stopping stream bridge...")
        finally:
            self.cleanup()
            
        return True

if __name__ == "__main__":
    bridge = StreamBridge()
    
    try:
        success = bridge.run()
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)