import asyncio
import streamlit as st
import detection.parking_occupancy as po
import widgets.annotation_uploader as annotation_uploader
import widgets.source_uploader as source_uploader
import widgets.preview_result as preview_result
import widgets.video_widgets as video_widgets
import widgets.shared as shared

def load_model():
    return po.ParkingOccupancy(st.secrets["backend_url"])

st.session_state['stream_mask'] = None
st.set_page_config(layout="wide", page_title="Parking spaces occupancy")
st.title("THE PROBLEM OF MONITORING THE CONGESTION OF PARKING SPACES")
annotation_ready = False
with st.spinner('Wait for the model to load'):
    parking_occupancy = load_model()
annotation_ready = annotation_uploader.upload_annotation(parking_occupancy)
source_type = st.radio(
    "Source Type",
    ["Image", "Video stream"]
)
if source_type == "Image":
    if shared.ws is not None:
        shared.close_connection()
    if 'ws' in st.session_state:
        st.session_state['ws'].close()
    source_ready = source_uploader.upload_source(parking_occupancy)
    if annotation_ready and source_ready:
        preview_col, result_col = st.columns(2)
        preview_result.show_preview(parking_occupancy, preview_col)
        preview_result.show_result(parking_occupancy, result_col)
else:
    stream_url = st.text_input('Enter the link to the video stream')
    if ((stream_url is not None) and stream_url and annotation_ready):        
        try:
            video_widgets.video_results(stream_url, parking_occupancy)
        except Exception as ex:
            print(ex)
            st.markdown('<strong>Something went wrong, check the link you entered</strong>', unsafe_allow_html=True)
