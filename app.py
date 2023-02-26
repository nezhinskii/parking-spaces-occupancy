import streamlit as st
import detection.parking_occupancy as po
import pandas as pd
import cv2
import numpy as np

parking_occupancy = po.ParkingOccupancy()

st.set_page_config(layout="wide")
st.title("THE PROBLEM OF MONITORING THE CONGESTION OF PARKING SPACES")
uploaded_csv = st.file_uploader("Upload parking spaces annotation", type = [".csv", ".txt"])
if uploaded_csv is not None:
    parking_occupancy.load_spaces_annotation(pd.read_csv(uploaded_csv))

uploaded_source = st.file_uploader("Upload source file", type = [".img", ".jpg", ".png"])
if uploaded_source is not None:
    file_bytes = np.asarray(bytearray(uploaded_source.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    parking_occupancy.load_image_source(opencv_image)

col1, col2 = st.columns([0.5, 0.5])
with col1:
    st.subheader("Preview")
    if uploaded_source is not None and uploaded_csv is not None:
        parking_occupancy.create_prewiew()
        st.image(parking_occupancy.preview, channels="BGR")
    else:
        st.empty()
with col2:
    st.subheader("Result")