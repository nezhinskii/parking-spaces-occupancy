import streamlit as st
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas

def upload_annotation(model):
    st.markdown('<strong>Upload parking spaces annotations as csv file imported from [VGG Imgae Annotator](https://annotate.officialstatistics.org/). '
                'Or you can annotate it right here if you upload an image</strong>', unsafe_allow_html=True)
    annotation_ready = False
    uploaded_file = st.file_uploader('none', type = [".csv", ".img", ".jpg", ".png"], label_visibility = "collapsed")
    if uploaded_file is not None:
        if uploaded_file.type == 'text/csv':
            model.load_spaces_annotation_csv(pd.read_csv(uploaded_file))
            annotation_ready = True
        else:
            st.markdown('<strong>Annotate parking lots using:</strong>'
                '<li>left mouse button to place a point</li>'
                '<li>right mouse button to link the last point to the first one and close the polygon</li>'
                '<li>double-clicking with the right mouse button to cancel the last point</li>'
                '<li>or you can use the additional buttons below the image</li>'
            '<strong>When you are done with annotating just go down and upload image for inference</strong>', unsafe_allow_html=True)
            image = Image.open(uploaded_file)
            w, h = image.size
            scale_factor = 900 / w
            canvas_result = st_canvas(
                background_image=Image.open(uploaded_file),
                height=h*scale_factor,
                width=w*scale_factor,
                drawing_mode='polygon',
                stroke_width=2,
                stroke_color='#FFFFFF',
                fill_color='#ff000055'
            )

            if canvas_result.json_data is not None:
                model.load_spaces_annotation_json(canvas_result.json_data['objects'], scale_factor)
            else:
                model.load_spaces_annotation_json([], scale_factor)

            annotation_ready = True
    return annotation_ready