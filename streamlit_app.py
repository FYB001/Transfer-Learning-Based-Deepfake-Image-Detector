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
# CUSTOM STYLE
# ============================================================

st.markdown(
    """
    <style>

    /* Main page background */
    .stApp {
        background-color: #f7f9fc;
    }

    /* Main content width */
    .main .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Main title */
    .main-title {
        text-align: center;
        color: #173f67;
        font-size: 40px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #52657a;
        font-size: 16px;
        margin-bottom: 28px;
    }

    /* Project information box */
    .project-box {
        background-color: #eaf3ff;
        border-left: 5px solid #2d8cff;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 30px;
    }

    .project-title {
        color: #173f67;
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .project-text {
        color: #34495e;
        font-size: 15px;
        line-height: 1.55;
    }

    /* Section headings */
    .section-title {
        color: #173f67;
        font-size: 21px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    /* Uploaded image caption */
    .image-caption {
        text-align: center;
        color: #52657a;
        font-size: 14px;
        margin-top: 5px;
    }

    /* Result percentage */
    .percentage {
        text-align: center;
        color: #173f67;
        font-size: 30px;
        font-weight: 700;
        margin-top: 10px;
    }

    /* Result label */
    .percentage-label {
        text-align: center;
        color: #66788a;
        font-size: 14px;
        margin-bottom: 12px;
    }

    /* Prediction level */
    .prediction-level {
        text-align: center;
        color: #34495e;
        font-size: 17px;
        margin-bottom: 15px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #7a8795;
        font-size: 13px;
        margin-top: 35px;
        padding-top: 15px;
        border-top: 1px solid #dfe5eb;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🔍 Deepfake Detector"
    )

    st.markdown("---")

    st.markdown("**Student:**")
    st.write("Fatmata Yealie Bangura")

    st.markdown("**University:**")
    st.write("Wrexham University")

    st.markdown("**Programme:**")
    st.write("MSc Data Science and Big Data Analytics")

    st.markdown("**Models:**")
    st.write(
        """
        VGG16  
        ResNet50  
        MobileNetV2  
        EfficientNetB0  
        Xception
        """
    )

    st.markdown("**Project Topic:**")
    st.write(
        "A Comparative Study of Transfer Learning-Based "
        "Convolutional Neural Networks for Deepfake Image Detection."
    )


# ============================================================
# MAIN PAGE TITLE
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🔍 Deepfake Image Detection'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
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

        <div class="project-text">
            <b>
            A Comparative Study of Transfer Learning-Based
            Convolutional Neural Networks for Deepfake Image Detection.
            </b>
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
# SELECT CNN MODEL
# ============================================================

st.markdown(
    '<div class="section-title">Select CNN Model</div>',
    unsafe_allow_html=True
)

selected_model = st.selectbox(
    "Choose a trained model:",
    list(MODEL_PATHS.keys()),
    label_visibility="collapsed"
)


# ============================================================
# CHECK MODEL FILE
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

st.markdown(
    '<div class="section-title">Upload Image</div>',
    unsafe_allow_html=True
)

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
# DISPLAY UPLOADED IMAGE
# CENTRED AND MEDIUM SIZE
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # Three columns are ONLY used to centre the image.
    # The image itself is fixed at 450 pixels wide.

    left_col, centre_col, right_col = st.columns(
        [1, 2, 1]
    )

    with centre_col:

        st.image(
            image,
            width=450
        )

        st.markdown(
            '<div class="image-caption">'
            'Uploaded Image'
            '</div>',
            unsafe_allow_html=True
        )


    st.write("")


    # ========================================================
    # ANALYSE BUTTON
    # ========================================================

    analyse_button = st.button(
        "🔍 Analyse Image",
        type="primary",
        use_container_width=True
    )


    # ========================================================
    # RUN ANALYSIS
    # ========================================================

    if analyse_button:

        with st.spinner("Analysing image..."):

            # Resize image
            image_resized = image.resize(
                IMAGE_SIZE
            )

            # Convert to NumPy
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
            prediction_output = model.predict(
                image_array,
                verbose=0
            )

            probability_real = float(
                prediction_output[0][0]
            )

            probability_fake = (
                1.0 - probability_real
            )


        # ====================================================
        # CLASSIFICATION
        # ====================================================

        if probability_real >= 0.5:

            prediction = "REAL"

            prediction_probability = (
                probability_real
            )

        else:

            prediction = "DEEPFAKE"

            prediction_probability = (
                probability_fake
            )


        # ====================================================
        # PREDICTION LEVEL
        # ====================================================

        if prediction_probability >= 0.80:

            prediction_level = "HIGH"

        elif prediction_probability >= 0.60:

            prediction_level = "MODERATE"

        else:

            prediction_level = "LOW"


        # ====================================================
        # DETECTION RESULT
        # ====================================================

        st.markdown("---")

        st.markdown(
            '<div class="section-title">'
            'Detection Result'
            '</div>',
            unsafe_allow_html=True
        )


        # ====================================================
        # MAIN CLASSIFICATION
        # ====================================================

        if prediction == "REAL":

            st.success(
                "✓ This image is classified as REAL."
            )

        else:

            st.error(
                "⚠ This image is classified as DEEPFAKE."
            )


        # ====================================================
        # PREDICTION PERCENTAGE
        # ====================================================

        st.markdown(
            f"""
            <div class="percentage">
                {prediction_probability * 100:.2f}%
            </div>

            <div class="percentage-label">
                Prediction Probability
            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # PREDICTION LEVEL
        # ====================================================

        st.markdown(
            f"""
            <div class="prediction-level">
                <b>Prediction Level:</b>
                {prediction_level}
            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # PROBABILITY DETAILS
        # ====================================================

        probability_col1, probability_col2 = st.columns(2)

        with probability_col1:

            st.metric(
                "Real Probability",
                f"{probability_real * 100:.2f}%"
            )

        with probability_col2:

            st.metric(
                "Deepfake Probability",
                f"{probability_fake * 100:.2f}%"
            )


        # ====================================================
        # MODEL USED
        # ====================================================

        st.caption(
            f"Model used: {selected_model}"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Transfer Learning-Based Deepfake Image Detection
        <br>
        Wrexham University • Master's Dissertation
    </div>
    """,
    unsafe_allow_html=True
)
