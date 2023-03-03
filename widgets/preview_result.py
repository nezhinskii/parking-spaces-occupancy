import streamlit as st

def show_preview(model, preview_col):
    with preview_col:
        st.markdown("<h2 style='text-align: center;'>Preview</h2>", unsafe_allow_html=True)
        model.create_prewiew()
        st.image(model.preview, channels="BGR")

def show_result(model, result_col):
    with result_col:
        st.markdown("<h2 style='text-align: center;'>Result</h2>", unsafe_allow_html=True)
        with st.spinner('Wait for the model to make inference'):
            result = model.process_frame()
        st.image(result, channels="BGR")