import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Page config
st.set_page_config(
    page_title="Space Station Safety Detector",
    page_icon="🚀",
    layout="wide"
)

# Title and Description
st.title("🚀 Space Station Safety Object Detection")
st.markdown("""
This application uses YOLOv8 to detect safety equipment in the International Space Station (ISS).
Upload an image to detect objects like:
- Oxygen Tanks
- Nitrogen Tanks
- First Aid Boxes
- Fire Alarms
- Safety Switch Panels
- Emergency Phones
- Fire Extinguishers
""")

# Sidebar
st.sidebar.header("Model Configuration")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05
)

# Load Model
@st.cache_resource
def load_model():
    model_path = 'runs/detect/train/weights/best.pt'
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please ensure the model is trained and the weights are available.")
        return None
    return YOLO(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Main Interface
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None and model is not None:
    # Create two columns
    col1, col2 = st.columns(2)
    
    # Display original image
    image = Image.open(uploaded_file)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    
    # Run inference
    if st.button("Detect Objects", type="primary"):
        with st.spinner("Running detection..."):
            # Convert PIL Image to numpy array
            img_array = np.array(image)
            
            # Run prediction
            results = model.predict(
                source=img_array,
                conf=confidence_threshold
            )
            
            # Plot results
            res_plotted = results[0].plot()
            
            # Display result
            with col2:
                st.subheader("Detected Objects")
                st.image(res_plotted, use_container_width=True)
            
            # Show detections table
            st.subheader("Detailed Detections")
            
            detections = []
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = results[0].names[cls]
                detections.append({
                    "Object": name,
                    "Confidence": f"{conf:.2%}",
                    "Coordinates": f"[{box.xyxy[0][0]:.1f}, {box.xyxy[0][1]:.1f}, {box.xyxy[0][2]:.1f}, {box.xyxy[0][3]:.1f}]"
                })
            
            if detections:
                st.table(detections)
            else:
                st.info("No objects detected with current confidence threshold.")

# Footer
st.markdown("---")
st.markdown("Built for Duality AI Space Station Challenge #2")
