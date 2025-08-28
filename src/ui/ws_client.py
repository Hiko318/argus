import asyncio
import websockets

async def listen():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        print(f"Connected to {uri}")
        try:
            while True:
                msg = await websocket.recv()
                print("Received:", msg)
        except websockets.ConnectionClosed:
            print("Connection closed")

if __name__ == "__main__":
    asyncio.run(listen())
