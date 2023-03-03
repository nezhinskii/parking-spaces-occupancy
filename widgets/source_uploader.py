import streamlit as st
import cv2
import numpy as np

def upload_source(model):
    st.markdown('<strong>Upload image for inference</strong>', unsafe_allow_html=True)
    uploaded_source = st.file_uploader("none", type = [".img", ".jpg", ".png"], label_visibility='collapsed')
    if uploaded_source is not None:
        file_bytes = np.asarray(bytearray(uploaded_source.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        model.load_image_source(opencv_image)
        return True
    return False
