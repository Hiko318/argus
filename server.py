#!/usr/bin/env python3
import asyncio
import websockets
import subprocess
import threading
import time
import json
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os

class StreamServer:
    def __init__(self):
        self.clients = set()
        self.adb_process = None
        self.streaming = False
        
    async def handle_websocket(self, websocket, path):
        """Handle WebSocket connections"""
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")
        
        # Send initial device info
        try:
            device_info = self.get_device_info()
            await websocket.send(json.dumps({
                'type': 'device_info',
                'data': device_info
            }).encode())
        except Exception as e:
            print(f"Failed to send device info: {e}")
        
        try:
            # Start ADB streaming when first client connects
            if len(self.clients) == 1 and not self.streaming:
                self.start_adb_stream()
            
            # Keep connection alive
            await websocket.wait_closed()
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            print(f"Client disconnected. Total clients: {len(self.clients)}")
            
            # Stop streaming when no clients
            if len(self.clients) == 0 and self.streaming:
                self.stop_adb_stream()
    
    def start_adb_stream(self):
        """Start ADB H.264 screen recording"""
        if self.streaming:
            return
            
        try:
            print("Starting ADB H.264 stream...")
            
            # Start ADB screen recording process
            self.adb_process = subprocess.Popen([
                'adb', 'exec-out', 'screenrecord',
                '--bit-rate=8000000',
                '--output-format=h264',
                '--size=720x1280',
                '-'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.streaming = True
            
            # Start reading H.264 data in a separate thread
            threading.Thread(target=self.read_h264_stream, daemon=True).start()
            
            print("ADB stream started successfully")
            
        except Exception as e:
            print(f"Failed to start ADB stream: {e}")
            self.streaming = False
    
    async def stop_adb_stream(self):
        """Stop ADB H.264 stream"""
        if not self.streaming:
            return
            
        print("Stopping ADB stream...")
        self.streaming = False
        
        if self.adb_process:
            try:
                self.adb_process.terminate()
                self.adb_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.adb_process.kill()
            except Exception as e:
                print(f"Error stopping ADB process: {e}")
            finally:
                self.adb_process = None
        
        print("ADB stream stopped")
    
    def read_h264_stream(self):
        """Read H.264 data from ADB and broadcast to clients"""
        if not self.adb_process:
            return
            
        print("Reading H.264 stream data...")
        buffer_size = 8192
        
        try:
            while self.streaming and self.adb_process and self.adb_process.poll() is None:
                data = self.adb_process.stdout.read(buffer_size)
                if not data:
                    break
                    
                # Broadcast to all connected clients
                if self.clients:
                    asyncio.run_coroutine_threadsafe(
                        self.broadcast_data(data), 
                        asyncio.get_event_loop()
                    )
                    
        except Exception as e:
            print(f"Error reading H.264 stream: {e}")
        finally:
            print("H.264 stream reading stopped")
            self.streaming = False
    
    async def broadcast_data(self, data):
        """Broadcast H.264 data to all connected clients"""
        if not self.clients:
            return
            
        # Send data to all clients
        disconnected = set()
        for client in self.clients.copy():
            try:
                await client.send(data)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                print(f"Error sending data to client: {e}")
                disconnected.add(client)
        
        # Remove disconnected clients
        for client in disconnected:
            self.clients.discard(client)
    
    def get_device_info(self):
        """Get connected ADB device information"""
        try:
            result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                devices = [line.split('\t')[0] for line in lines if '\tdevice' in line]
                if devices:
                    # Get device model
                    model_result = subprocess.run(['adb', 'shell', 'getprop', 'ro.product.model'], 
                                                capture_output=True, text=True)
                    model = model_result.stdout.strip() if model_result.returncode == 0 else 'Unknown'
                    return {
                        'connected': True,
                        'device_id': devices[0],
                        'model': model,
                        'count': len(devices)
                    }
            return {'connected': False, 'device_id': None, 'model': None, 'count': 0}
        except Exception as e:
            print(f"Error getting device info: {e}")
            return {'connected': False, 'device_id': None, 'model': None, 'count': 0}

class CustomHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler to serve files from the correct directory"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

def start_http_server():
    """Start HTTP server for serving the web interface"""
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, CustomHTTPRequestHandler)
    print(f"HTTP server running on http://localhost:8000")
    httpd.serve_forever()

async def main():
    """Main function to start both HTTP and WebSocket servers"""
    print("Starting Foresight Phone Stream Server...")
    
    # Check if ADB is available
    try:
        result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
        if 'device' not in result.stdout:
            print("Warning: No ADB devices found. Make sure your phone is connected.")
        else:
            print("ADB device detected")
    except FileNotFoundError:
        print("Error: ADB not found. Please install Android SDK Platform Tools.")
        return
    
    # Create stream server
    stream_server = StreamServer()
    
    # Start HTTP server in a separate thread
    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()
    
    # Create WebSocket handler wrapper
    async def websocket_handler(websocket, path=None):
        await stream_server.handle_websocket(websocket, path or "/")
    
    # Start WebSocket server
    print("WebSocket server starting on ws://localhost:9998")
    async with websockets.serve(websocket_handler, "localhost", 9998):
        print("\n[*] Foresight Server Ready!")
        print("[*] Open http://localhost:8000 in your browser")
        print("[*] WebSocket server: ws://localhost:9998")
        print("\nPress Ctrl+C to stop the server")
        
        try:
            await asyncio.Future()  # Run forever
        except KeyboardInterrupt:
            print("\nShutting down server...")
            await stream_server.stop_adb_stream()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped")