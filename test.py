import asyncio
import websockets

async def timer():
    while True:
        print("Timer event")
        await asyncio.sleep(1)

async def connect():
    uri = "wss://demo.piesocket.com/v3/channel_123?api_key=VCXCEuvhGcBDP7XhiJJUDvR1e1D3eiVjgZ9VRiaV&notify_self"
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            print(f"Received message: {message}")

async def main():
    timer_task = asyncio.create_task(timer())
    connect_task = asyncio.create_task(connect())
    await asyncio.gather(timer_task, connect_task)

if __name__ == "__main__":
    asyncio.run(main())