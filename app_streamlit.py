import streamlit as st
import os
import numpy as np
from PIL import Image
import torch
import base64
import io
from app import EyeNovaApp

# Title and description
st.set_page_config(page_title="Keratoconus Multimodal Support", layout="wide")
st.title("👁️ Live Demo: Clinical AI Decision Support")
st.write("Clinical AI decision support for keratoconus triage, combining multimodal topography and structured ocular parameters.")

# Initialize the inference engine
@st.cache_resource
def get_engine():
    engine = EyeNovaApp()
    engine.load_artifacts()
    return engine

try:
    engine = get_engine()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Layout: two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Clinical Input Form")
    
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=25)
    gender = st.selectbox("Gender", options=["Male", "Female"])
    astigmatism_value = st.number_input("Astigmatism Value (D)", value=0.0, format="%.2f")
    astigmatism_axis = st.number_input("Astigmatism Axis (°)", min_value=0, max_value=180, value=0)
    pachy_x = st.number_input("Pachymetry Thinnest X Coordinate", value=0.0, format="%.2f")
    pachy_y = st.number_input("Pachymetry Thinnest Y Coordinate", value=0.0, format="%.2f")

with col2:
    st.header("Topography Grid Upload")
    
    ant_file = st.file_uploader("Anterior Map", type=["jpg", "jpeg", "png"])
    ax_file = st.file_uploader("Axial Map", type=["jpg", "jpeg", "png"])
    pachy_file = st.file_uploader("Pachymetry Map", type=["jpg", "jpeg", "png"])
    post_file = st.file_uploader("Posterior Map", type=["jpg", "jpeg", "png"])

if st.button("Run Assessment", use_container_width=True):
    if not (ant_file and ax_file and pachy_file and post_file):
        st.warning("Please upload all four topography maps to run the assessment.")
    else:
        with st.spinner("Processing scans and running inference..."):
            # Prepare payload
            payload = {
                'age': age,
                'gender': 1 if gender == "Male" else 0,
                'astigmatism_value': astigmatism_value,
                'astigmatism_axis': astigmatism_axis,
                'pachy_x': pachy_x,
                'pachy_y': pachy_y,
            }
            
            # The engine preprocess_images expects file-like objects with a .read() method
            # Streamlit UploadedFile objects implement .read() and behave as expected.
            image_files = [ant_file, ax_file, pachy_file, post_file]
            
            try:
                status, confidence, original_b64, heatmap_b64 = engine.infer(payload, image_files)
                
                st.success("Assessment Completed!")
                
                # Results layout
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric(label="Diagnosis", value=status)
                with res_col2:
                    st.metric(label="Confidence", value=f"{confidence:.1f}%")
                
                # Decode and display the output images
                orig_img = Image.open(io.BytesIO(base64.b64decode(original_b64)))
                heat_img = Image.open(io.BytesIO(base64.b64decode(heatmap_b64)))
                
                img_col1, img_col2 = st.columns(2)
                with img_col1:
                    st.image(orig_img, caption="Stitched Topography Grid", use_container_width=True)
                with img_col2:
                    st.image(heat_img, caption="AI Attention Heatmap", use_container_width=True)
                    
            except Exception as e:
                st.error(f"Assessment Error: {e}")
