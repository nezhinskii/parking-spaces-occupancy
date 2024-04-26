import asyncio

ws = None
stream_mask = None
thread = None

def close_connection():
    ws.close()
    stream_mask = None