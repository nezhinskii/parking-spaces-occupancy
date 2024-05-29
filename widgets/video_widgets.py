import threading
import cv2
import numpy as np
import streamlit as st
import asyncio
import widgets.shared as shared

def video_results(stream_url, model):
    del st.session_state['stream_mask']
    if shared.ws is not None:
        shared.close_connection()
    cap = cv2.VideoCapture(stream_url)
    if (not cap.isOpened()):
        st.markdown('<strong>Something went wrong, check the link you entered</strong>', unsafe_allow_html=True)
        return
    
    shared.thread = threading.Thread(target=model.process_stream, args = [stream_url])
    shared.thread.start()
    
    result_image = st.image([], width=1000)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Can't receive frame")
            break
        result_frame = frame
        if shared.stream_mask is None:
            shared.stream_mask = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

        result_frame = cv2.addWeighted(frame, 1, shared.stream_mask, 1, 0)
        result_image.image(result_frame, channels="BGR",)