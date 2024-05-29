ws = None
stream_mask = None
thread = None

def close_connection():
    ws.close()
    global stream_mask 
    stream_mask = None