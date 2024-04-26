import streamlit as st
import cv2
import numpy as np

def upload_source(model):
    st.markdown('<strong>Upload image for inference</strong>', unsafe_allow_html=True)
    uploaded_source = st.file_uploader("none", type = [".img", ".jpg", ".png"], label_visibility='collapsed')
    if uploaded_source is not None:
        file_bytes = np.asarray(bytearray(uploaded_source.read()), dtype=np.uint8)
        model.load_image_source(file_bytes)
        return True
    return False
