import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Deepfake Image Detection",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded"
)


# ============================================================
# CLEAN BLUE AND WHITE DESIGN
# ============================================================

st.markdown("""
<style>

/* MAIN APPLICATION */
.stApp { background-color: #f7f9fc; }
[data-testid="stAppViewContainer"] { background-color: #f7f9fc; }
[data-testid="stHeader"] { background-color: #0b1f33; }

.block-container {
    max-width: 950px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #dce6f0;
}
[data-testid="stSidebar"] * { color: #173f67; }
[data-testid="stSidebar"] hr { border-color: #dce6f0; margin: 0.6rem 0; }

.sidebar-heading {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #6b8cae !important;
    margin-top: 18px;
    margin-bottom: 4px;
}
.sidebar-value {
    font-size: 15px;
    font-weight: 500;
    color: #173f67 !important;
    margin-bottom: 2px;
    line-height: 1.4;
}
.sidebar-model-list {
    font-size: 14px;
    color: #34495e !important;
    line-height: 1.9;
    margin-bottom: 4px;
}

/* TITLES */
h1 {
    color: #173f67 !important;
    font-weight: 700 !important;
    text-align: center !important;
}
h2, h3 { color: #173f67 !important; }

/* Streamlit wraps st.title in a flex header row that is left-aligned
   by default; center that row so the title lines up with the
   centered description text underneath it. */
[data-testid="stHeadingWithActionElements"] {
    display: flex;
    justify-content: center;
}

/* DESCRIPTION */
.description {
    text-align: center;
    color: #52657a;
    font-size: 16px;
    margin-top: -10px;
    margin-bottom: 30px;
}

/* PROJECT BOX */
.project-box {
    background-color: #eaf3ff;
    border-left: 5px solid #2d8cff;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 28px;
}
.project-title {
    color: #173f67;
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 8px;
}
.project-topic {
    color: #34495e;
    font-size: 15px;
    line-height: 1.55;
}
.project-highlights {
    color: #34495e;
    font-size: 14.5px;
    line-height: 1.9;
    margin: 0;
    padding-left: 0;
    list-style: none;
}
.project-highlights li {
    margin-bottom: 2px;
}
.project-highlights b {
    color: #173f67;
}

/* SECTION LABEL */
.section-label {
    color: #173f67;
    font-size: 20px;
    font-weight: 700;
    margin-top: 30px;
    margin-bottom: 10px;
}

/* BUTTON */
div.stButton > button {
    background-color: #2d8cff;
    color: white;
    border: none;
    border-radius: 8px;
    height: 48px;
    font-size: 16px;
    font-weight: 600;
    transition: background-color 0.15s ease-in-out;
}
div.stButton > button:hover { background-color: #176fc1; color: white; }

/* RESULT AREA */
.result-box {
    background-color: #ffffff;
    border: 1px solid #dce6f0;
    border-radius: 12px;
    padding: 25px;
    margin-top: 25px;
    margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(23, 63, 103, 0.06);
}
.result-title {
    color: #173f67;
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 18px;
    text-align: center;
}
.prediction {
    text-align: center;
    color: #173f67;
    font-size: 30px;
    font-weight: 700;
    margin-top: 10px;
}
.prediction-text {
    text-align: center;
    color: #52657a;
    font-size: 16px;
    margin-top: 8px;
    margin-bottom: 15px;
}
.prediction-percentage {
    text-align: center;
    color: #173f67;
    font-size: 32px;
    font-weight: 700;
}
.prediction-label {
    text-align: center;
    color: #718096;
    font-size: 14px;
    margin-bottom: 15px;
}
.level {
    text-align: center;
    color: #34495e;
    font-size: 17px;
    margin-top: 8px;
    margin-bottom: 15px;
}
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.03em;
}
.badge-real { background-color: #e3f9ec; color: #1a8a4c; }
.badge-fake { background-color: #fde8e8; color: #c0392b; }

/* FOOTER */
.footer {
    text-align: center;
    color: #7a8795;
    font-size: 13px;
    margin-top: 35px;
    padding-top: 18px;
    border-top: 1px solid #dce6f0;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🔍 Deepfake Detector")
    st.divider()

    st.markdown('<div class="sidebar-heading">Student</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-value">Fatmata Yealie Bangura</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-heading">University</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-value">Wrexham University</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-heading">Programme</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-value">MSc Data Science and Big Data Analytics</div>', unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="sidebar-heading">Models Compared</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-model-list">'
        '• VGG16<br>'
        '• ResNet50<br>'
        '• MobileNetV2<br>'
        '• EfficientNetB0<br>'
        '• Xception'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown('<div class="sidebar-heading">Project Topic</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-value" style="font-weight:400; font-size:13.5px;">'
        'A Comparative Study of Transfer Learning-Based Convolutional Neural '
        'Networks for Deepfake Image Detection.'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# MAIN TITLE
# ============================================================

st.title("🔍 Deepfake Image Detection")

st.markdown(
    '<div class="description">'
    'A transfer learning-based system for classifying images as Real or Deepfake.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PROJECT INFORMATION
# ============================================================
# NOTE: The HTML below is built as a single joined string with no
# leading indentation on each line. Streamlit's markdown renderer
# treats 4+ leading spaces as a code block, which was the cause of
# the raw HTML showing up on the page instead of rendering.

project_box_html = (
    '<div class="project-box">'
    '<div class="project-title">Master\'s Dissertation Project</div>'
    '<ul class="project-highlights">'
    '<li><b>Objective:</b> Classify facial images as Real or Deepfake</li>'
    '<li><b>Approach:</b> Transfer learning with fine-tuned CNN backbones</li>'
    '<li><b>Output:</b> Predicted class with confidence score and '
    'prediction level</li>'
    '</ul>'
    '</div>'
)

st.markdown(project_box_html, unsafe_allow_html=True)


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_DIR = "models"

MODEL_PATHS = {
    "VGG16": os.path.join(MODEL_DIR, "VGG16_best.keras"),
    "ResNet50": os.path.join(MODEL_DIR, "ResNet50_best.keras"),
    "MobileNetV2": os.path.join(MODEL_DIR, "MobileNetV2_best.keras"),
    "EfficientNetB0": os.path.join(MODEL_DIR, "EfficientNetB0_best.keras"),
    "Xception": os.path.join(MODEL_DIR, "Xception_best.keras"),
}

IMAGE_SIZE = (224, 224)


# ============================================================
# SELECT MODEL
# ============================================================

st.markdown('<div class="section-label">Select CNN Model</div>', unsafe_allow_html=True)

selected_model = st.selectbox(
    "Choose a trained model:",
    list(MODEL_PATHS.keys()),
    label_visibility="collapsed"
)


# ============================================================
# CHECK MODEL
# ============================================================

model_path = MODEL_PATHS[selected_model]

if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path, compile=False)


model = load_model(model_path)


# ============================================================
# UPLOAD IMAGE
# ============================================================

st.markdown('<div class="section-label">Upload Image</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a facial image for analysis",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)


# ============================================================
# DISPLAY IMAGE — CENTRED + MEDIUM SIZE
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    left_space, image_column, right_space = st.columns([1, 2, 1])

    with image_column:
        st.image(image, width=450)
        st.caption("Uploaded Image")

    st.write("")

    # --------------------------------------------------------
    # ANALYSE BUTTON
    # --------------------------------------------------------

    analyse = st.button(
        "🔍 Analyse Image",
        type="primary",
        use_container_width=True
    )

    # --------------------------------------------------------
    # ANALYSIS
    # --------------------------------------------------------

    if analyse:

        with st.spinner("Analysing image..."):

            image_resized = image.resize(IMAGE_SIZE)
            image_array = np.asarray(image_resized, dtype=np.float32)
            image_array = image_array / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            output = model.predict(image_array, verbose=0)

            probability_real = float(output[0][0])
            probability_fake = 1.0 - probability_real

        # ---- Classification ----
        if probability_real >= 0.5:
            prediction = "REAL"
            confidence = probability_real
            badge_class = "badge-real"
        else:
            prediction = "DEEPFAKE"
            confidence = probability_fake
            badge_class = "badge-fake"

        # ---- Prediction level ----
        if confidence >= 0.80:
            prediction_level = "HIGH"
        elif confidence >= 0.60:
            prediction_level = "MODERATE"
        else:
            prediction_level = "LOW"

        # ---- Result card ----
        result_html = (
            '<div class="result-box">'
            '<div class="result-title">Detection Result</div>'
            f'<div style="text-align:center; margin-bottom:8px;">'
            f'<span class="badge {badge_class}">{prediction}</span></div>'
            f'<div class="prediction-text">This image is classified as '
            f'<b>{prediction}</b>.</div>'
            f'<div class="prediction-percentage">{confidence * 100:.2f}%</div>'
            '<div class="prediction-label">Prediction Probability</div>'
            f'<div class="level"><b>Prediction Level:</b> {prediction_level}</div>'
            '</div>'
        )

        st.markdown(result_html, unsafe_allow_html=True)

        # ---- Probability breakdown ----
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Real Probability", f"{probability_real * 100:.2f}%")

        with col2:
            st.metric("Deepfake Probability", f"{probability_fake * 100:.2f}%")

        # ---- Model info ----
        st.caption(
            f"Model used: {selected_model} • Input size: 224 × 224 pixels"
        )

else:
    st.info("Upload an image above to run detection.")


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    'Transfer Learning-Based Deepfake Image Detection'
    '<br>'
    'Wrexham University • MSc Data Science and Big Data Analytics'
    '</div>',
    unsafe_allow_html=True
)
