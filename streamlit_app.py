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

st.markdown(
    """
    <style>

    /* ================================
       MAIN APPLICATION
       ================================ */

    .stApp {
        background-color: #f7f9fc;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #f7f9fc;
    }

    [data-testid="stHeader"] {
        background-color: #0b1f33;
    }

    /* Main content width and spacing */
    .block-container {
        max-width: 950px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }


    /* ================================
       SIDEBAR
       ================================ */

    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #dce6f0;
    }

    [data-testid="stSidebar"] * {
        color: #173f67;
    }

    [data-testid="stSidebar"] hr {
        border-color: #dce6f0;
    }


    /* ================================
       TITLES
       ================================ */

    h1 {
        color: #173f67 !important;
        font-weight: 700 !important;
    }

    h2, h3 {
        color: #173f67 !important;
    }


    /* ================================
       DESCRIPTION
       ================================ */

    .description {
        text-align: center;
        color: #52657a;
        font-size: 16px;
        margin-top: -10px;
        margin-bottom: 30px;
    }


    /* ================================
       PROJECT BOX
       ================================ */

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


    /* ================================
       UPLOADED IMAGE AREA
       ================================ */

    .image-title {
        color: #173f67;
        font-size: 20px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 10px;
    }


    /* ================================
       BUTTON
       ================================ */

    div.stButton > button {
        background-color: #2d8cff;
        color: white;
        border: none;
        border-radius: 8px;
        height: 48px;
        font-size: 16px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #176fc1;
        color: white;
    }


    /* ================================
       RESULT AREA
       ================================ */

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


    /* ================================
       FOOTER
       ================================ */

    .footer {
        text-align: center;
        color: #7a8795;
        font-size: 13px;
        margin-top: 35px;
        padding-top: 18px;
        border-top: 1px solid #dce6f0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🔍 Deepfake Detector")

    st.divider()

    st.markdown("**Student**")
    st.write("Fatmata Yealie Bangura")

    st.markdown("**University**")
    st.write("Wrexham University")

    st.markdown("**Programme**")
    st.write("MSc Data Science and Big Data Analytics")

    st.markdown("**Models**")
    st.write(
        "VGG16\n\n"
        "ResNet50\n\n"
        "MobileNetV2\n\n"
        "EfficientNetB0\n\n"
        "Xception"
    )

    st.markdown("**Project Topic**")

    st.write(
        "A Comparative Study of Transfer Learning-Based "
        "Convolutional Neural Networks for Deepfake Image Detection."
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

st.markdown(
    """
    <div class="project-box">
        <div class="project-title">
            Master's Dissertation Project
        </div>

        <div class="project-topic">
            A Comparative Study of Transfer Learning-Based
            Convolutional Neural Networks for Deepfake Image Detection.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_DIR = "models"

MODEL_PATHS = {
    "VGG16": os.path.join(
        MODEL_DIR,
        "VGG16_best.keras"
    ),

    "ResNet50": os.path.join(
        MODEL_DIR,
        "ResNet50_best.keras"
    ),

    "MobileNetV2": os.path.join(
        MODEL_DIR,
        "MobileNetV2_best.keras"
    ),

    "EfficientNetB0": os.path.join(
        MODEL_DIR,
        "EfficientNetB0_best.keras"
    ),

    "Xception": os.path.join(
        MODEL_DIR,
        "Xception_best.keras"
    )
}


# ============================================================
# IMAGE SIZE
# ============================================================

IMAGE_SIZE = (224, 224)


# ============================================================
# SELECT MODEL
# ============================================================

st.subheader("Select CNN Model")

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

    st.error(
        f"Model file not found: {model_path}"
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model(path):

    return tf.keras.models.load_model(
        path,
        compile=False
    )


model = load_model(model_path)


# ============================================================
# UPLOAD IMAGE
# ============================================================

st.subheader("Upload Image")

uploaded_file = st.file_uploader(
    "Upload a facial image for analysis",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ],
    label_visibility="collapsed"
)


# ============================================================
# DISPLAY IMAGE
# CENTRED + MEDIUM SIZE
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # Centre the uploaded image
    left_space, image_column, right_space = st.columns(
        [1, 2, 1]
    )

    with image_column:

        st.image(
            image,
            width=450
        )

        st.caption("Uploaded Image")


    st.write("")


    # ========================================================
    # ANALYSE BUTTON
    # ========================================================

    analyse = st.button(
        "🔍 Analyse Image",
        type="primary",
        use_container_width=True
    )


    # ========================================================
    # ANALYSIS
    # ========================================================

    if analyse:

        with st.spinner("Analysing image..."):

            # Resize image
            image_resized = image.resize(
                IMAGE_SIZE
            )

            # Convert to NumPy array
            image_array = np.asarray(
                image_resized,
                dtype=np.float32
            )

            # Same normalisation used during training
            image_array = image_array / 255.0

            # Add batch dimension
            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # Model prediction
            output = model.predict(
                image_array,
                verbose=0
            )

            probability_real = float(
                output[0][0]
            )

            probability_fake = (
                1.0 - probability_real
            )


        # ====================================================
        # CLASSIFICATION
        # ====================================================

        if probability_real >= 0.5:

            prediction = "REAL"

            confidence = probability_real

        else:

            prediction = "DEEPFAKE"

            confidence = probability_fake


        # ====================================================
        # PREDICTION LEVEL
        # ====================================================

        if confidence >= 0.80:

            prediction_level = "HIGH"

        elif confidence >= 0.60:

            prediction_level = "MODERATE"

        else:

            prediction_level = "LOW"


        # ====================================================
        # RESULT
        # ====================================================

        st.markdown(
            '<div class="result-box">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="result-title">'
            'Detection Result'
            '</div>',
            unsafe_allow_html=True
        )

        # Main prediction
        st.markdown(
            f'<div class="prediction">{prediction}</div>',
            unsafe_allow_html=True
        )

        # Classification sentence
        st.markdown(
            f'<div class="prediction-text">'
            f'This image is classified as <b>{prediction}</b>.'
            f'</div>',
            unsafe_allow_html=True
        )

        # Percentage
        st.markdown(
            f'<div class="prediction-percentage">'
            f'{confidence * 100:.2f}%'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="prediction-label">'
            'Prediction Probability'
            '</div>',
            unsafe_allow_html=True
        )

        # Prediction level
        st.markdown(
            f'<div class="level">'
            f'<b>Prediction Level:</b> {prediction_level}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # ====================================================
        # PROBABILITY INFORMATION
        # ====================================================

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Real Probability",
                f"{probability_real * 100:.2f}%"
            )

        with col2:

            st.metric(
                "Deepfake Probability",
                f"{probability_fake * 100:.2f}%"
            )


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        st.caption(
            f"Model used: {selected_model} • "
            f"Input size: 224 × 224 pixels"
        )


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
