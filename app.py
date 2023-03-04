import streamlit as st
import detection.parking_occupancy as po
import torch
import widgets.annotation_uploader as annotation_uploader
import widgets.source_uploader as source_uploader
import widgets.preview_result as preview_result

@st.cache_resource(show_spinner=False)
def load_model():
    return po.ParkingOccupancy(torch.hub.load('ultralytics/yolov5', 'custom', path='weights\coco-voc.pt', verbose=False, force_reload=True))

st.set_page_config(layout="wide", page_title="Parking spaces occupancy",)
st.title("THE PROBLEM OF MONITORING THE CONGESTION OF PARKING SPACES")
annotation_ready = False
with st.spinner('Wait for the model to load'):
    parking_occupancy = load_model()
annotation_ready = annotation_uploader.upload_annotation(parking_occupancy)
source_ready = source_uploader.upload_source(parking_occupancy)
if annotation_ready and source_ready:
    preview_col, result_col = st.columns(2)
    preview_result.show_preview(parking_occupancy, preview_col)
    preview_result.show_result(parking_occupancy, result_col)