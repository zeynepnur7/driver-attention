
import asyncio
import json
import time
from websockets.asyncio.server import serve

SAMPLES = [
    {
        "pitch": 0,
        "yaw": 0,
        "roll": 0,
        "ear": 0.30,
        "perclos": 0.02,
        "faceDetected": True,
        "alertState": "Normal"
    },
    {
        "pitch": 12,
        "yaw": 25,
        "roll": 5,
        "ear": 0.21,
        "perclos": 0.15,
        "faceDetected": True,
        "alertState": "Caution"
    },
    {
        "pitch": 30,
        "yaw": -15,
        "roll": 10,
        "ear": 0.14,
        "perclos": 0.45,
        "faceDetected": True,
        "alertState": "Critical"
    }
]

async def handler(websocket):
    print("Unity client connected.")

    index = 0

    try:
        while True:
            data = SAMPLES[index].copy()
            data["timestamp"] = int(time.time())

            await websocket.send(json.dumps(data))

            print("Sent:", data["alertState"])

            index = (index + 1) % len(SAMPLES)
            await asyncio.sleep(2)

    except Exception as e:
        print("Client disconnected:", e)

async def main():
    async with serve(handler, "127.0.0.1", 8765):
        print("WebSocket server running on ws://127.0.0.1:8765")
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nWebSocket server stopped.")
