import streamlit as st
import detection.parking_occupancy as po
import pandas as pd
import cv2
import numpy as np
import torch

@st.cache_data(show_spinner=False)
def load_model():
    return po.ParkingOccupancy(torch.hub.load('ultralytics/yolov5', 'custom', path='weights\coco-voc.pt', verbose=False, force_reload=True))

def upload_source():
    uploaded_csv = st.file_uploader("Upload parking spaces annotation", type = [".csv", ".txt"])
    if uploaded_csv is not None:
        parking_occupancy.load_spaces_annotation(pd.read_csv(uploaded_csv))
    
    uploaded_source = st.file_uploader("Upload source file", type = [".img", ".jpg", ".png"])
    if uploaded_source is not None:
        file_bytes = np.asarray(bytearray(uploaded_source.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        parking_occupancy.load_image_source(opencv_image)
    return uploaded_csv is not None and uploaded_source is not None

def show_preview(preview_col):
    with preview_col:
        st.markdown("<h2 style='text-align: center;'>Preview</h1>", unsafe_allow_html=True)
        parking_occupancy.create_prewiew()
        st.image(parking_occupancy.preview, channels="BGR")

def show_result(result_col):
    with result_col:
        st.markdown("<h2 style='text-align: center;'>Result</h1>", unsafe_allow_html=True)
        with st.spinner('Wait for the model to make inference'):
            result = parking_occupancy.process_frame()
        st.image(result, channels="BGR")


st.set_page_config(layout="wide", page_title="Parking spaces occupancy",)
st.title("THE PROBLEM OF MONITORING THE CONGESTION OF PARKING SPACES")
with st.spinner('Wait for the model to load'):
    parking_occupancy = load_model()
is_ready = upload_source()

preview_col, result_col = st.columns(2)
if is_ready:
    show_preview(preview_col)
    show_result(result_col)