import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch

# Page configuration
st.set_page_config(
    page_title="Helmet Detection System",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .header-text {
        font-size: 48px;
        font-weight: bold;
        color: #1f2937;
        text-align: center;
        margin-bottom: 10px;
    }
    .subheader-text {
        font-size: 20px;
        color: #6b7280;
        text-align: center;
        margin-bottom: 30px;
    }
    .info-box {
        background-color: #e0f2fe;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #0284c7;
        margin: 20px 0;
    }
    .success-box {
        background-color: #dcfce7;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #16a34a;
        margin: 20px 0;
    }
    .warning-box {
        background-color: #fef3c7;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #f59e0b;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="header-text">🪖 Helmet Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader-text">AI-Powered Safety Compliance Detection using YOLOv11</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    st.markdown("---")
    st.markdown("### 📊 About")
    st.info("""
    This application uses YOLOv11 to detect whether people are wearing helmets or not.
    
    **Features:**
    - Real-time detection
    - High accuracy
    - Easy to use
    """)
    st.markdown("---")
    st.markdown("### 👨‍💻 Developer")
    st.markdown("**Saifullah Khalid**")
    st.markdown("[GitHub Repository](https://github.com/Saifullah-Khalid-DataScientist/Helmet-Detection-Using-YOLOv11)")

# Load model using torch directly
@st.cache_resource
def load_model():
    try:
        # Try loading with ultralytics first
        try:
            from ultralytics import YOLO
            model = YOLO('best.pt')
            return model, 'ultralytics'
        except:
            # Fallback to torch hub
            model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)
            model.conf = 0.25
            return model, 'torch'
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure 'best.pt' file is in the root directory.")
        return None, None

# Detection function
def detect_objects(image, model, model_type, conf_threshold):
    img_array = np.array(image)
    
    if model_type == 'ultralytics':
        results = model.predict(img_array, conf=conf_threshold)
        annotated_img = results[0].plot()
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        detections = results[0].boxes
        class_names = results[0].names
        
        detection_data = []
        for box in detections:
            class_id = int(box.cls[0])
            class_name = class_names[class_id]
            confidence = float(box.conf[0])
            detection_data.append({'class': class_name, 'conf': confidence})
            
    else:  # torch hub
        model.conf = conf_threshold
        results = model(img_array)
        
        # Get annotated image
        annotated_img_rgb = np.squeeze(results.render())
        
        # Parse detections
        detection_data = []
        df = results.pandas().xyxy[0]
        for _, row in df.iterrows():
            detection_data.append({'class': row['name'], 'conf': row['confidence']})
    
    return annotated_img_rgb, detection_data

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

with col2:
    st.markdown("### 🎯 Detection Results")
    
    if uploaded_file is not None:
        model, model_type = load_model()
        
        if model is not None:
            # Run detection
            with st.spinner('🔍 Analyzing image...'):
                try:
                    annotated_img, detections = detect_objects(image, model, model_type, confidence_threshold)
                    
                    # Display result
                    st.image(annotated_img, caption='Detection Result', use_column_width=True)
                    
                    # Detection statistics
                    num_detections = len(detections)
                    
                    if num_detections > 0:
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.markdown(f"### ✅ Detected {num_detections} object(s)")
                        
                        # Count helmets and no-helmets
                        helmet_count = 0
                        no_helmet_count = 0
                        
                        for det in detections:
                            class_name = det['class'].lower()
                            if 'helmet' in class_name and 'no' not in class_name:
                                helmet_count += 1
                            else:
                                no_helmet_count += 1
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("With Helmet 🟢", helmet_count)
                        with col_b:
                            st.metric("Without Helmet 🔴", no_helmet_count)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Detailed detection info
                        with st.expander("📋 Detailed Detection Information"):
                            for idx, det in enumerate(detections):
                                st.write(f"**Detection {idx+1}:** {det['class']} - Confidence: {det['conf']:.2%}")
                        
                        # Safety compliance check
                        if no_helmet_count > 0:
                            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                            st.warning(f"⚠️ Safety Alert: {no_helmet_count} person(s) detected without helmet!")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.success("✅ All detected persons are wearing helmets!")
                    else:
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.info("No objects detected. Try adjusting the confidence threshold or upload a different image.")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"Error during detection: {str(e)}")
        else:
            st.error("Model could not be loaded. Please check the error message above.")
    else:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.info("👈 Please upload an image to start detection")
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #6b7280;'>
        <p>Helmet Detection System | Powered by YOLOv11 & Streamlit</p>
        <p>Developed by Saifullah Khalid</p>
    </div>
""", unsafe_allow_html=True)
