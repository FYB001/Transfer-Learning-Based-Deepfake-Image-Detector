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
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* =========================
       GENERAL PAGE
       ========================= */

    .stApp {
        background-color: #f7f9fc;
    }

    .main .block-container {
        max-width: 1050px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }


    /* =========================
       SIDEBAR
       ========================= */

    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #d9e2ec !important;
    }

    section[data-testid="stSidebar"] > div {
        background-color: #ffffff !important;
    }

    section[data-testid="stSidebar"] .block-container {
        padding: 2rem 1.3rem 2rem 1.3rem !important;
    }


    /* =========================
       MAIN TITLE
       ========================= */

    .main-title {
        text-align: center;
        color: #173f67;
        font-size: 40px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .subtitle {
        text-align: center;
        color: #52657a;
        font-size: 16px;
        margin-bottom: 28px;
    }


    /* =========================
       PROJECT BOX
       ========================= */

    .project-box {
        background-color: #eaf3ff;
        border-left: 5px solid #2d8cff;
        border-radius: 10px;
        padding: 18px 22px;
        margin: 0 auto 30px auto;
        max-width: 850px;
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


    /* =========================
       SECTION HEADINGS
       ========================= */

    .section-title {
        color: #173f67;
        font-size: 21px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 12px;
    }


    /* =========================
       UPLOADED IMAGE
       ========================= */

    .uploaded-caption {
        text-align: center;
        color: #52657a;
        font-size: 14px;
        margin-top: 5px;
    }


    /* =========================
       RESULT
       ========================= */

    .result-box {
        background-color: #ffffff;
        border: 1px solid #dce3eb;
        border-radius: 12px;
        padding: 25px;
        margin-top: 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    .result-real {
        color: #16803c;
        font-size: 32px;
        font-weight: 800;
    }

    .result-fake {
        color: #c62828;
        font-size: 32px;
        font-weight: 800;
    }

    .prediction-percentage {
        color: #173f67;
        font-size: 28px;
        font-weight: 700;
        margin-top: 10px;
    }

    .prediction-label {
        color: #66788a;
        font-size: 14px;
        margin-bottom: 12px;
    }

    .prediction-level {
        color: #34495e;
        font-size: 17px;
        margin-bottom: 10px;
    }


    /* =========================
       FOOTER
       ========================= */

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
        """
        <div style="
            color:#173f67;
            font-size:22px;
            font-weight:700;
            margin-bottom:25px;
        ">
            🔍 Deepfake Detector
        </div>

        <div style="
            color:#173f67;
            font-size:14px;
            line-height:1.6;
        ">

            <div style="margin-bottom:22px;">
                <b style="color:#173f67;">Student:</b><br>
                <span style="color:#34495e;">
                    Fatmata Yealie Bangura
                </span>
            </div>

            <div style="margin-bottom:22px;">
                <b style="color:#173f67;">University:</b><br>
                <span style="color:#34495e;">
                    Wrexham University
                </span>
            </div>

            <div style="margin-bottom:22px;">
                <b style="color:#173f67;">Programme:</b><br>
                <span style="color:#34495e;">
                    MSc Data Science and Big Data Analytics
                </span>
            </div>

            <div style="margin-bottom:22px;">
                <b style="color:#173f67;">Models:</b><br>
                <span style="color:#34495e;">
                    VGG16<br>
                    ResNet50<br>
                    MobileNetV2<br>
                    EfficientNetB0<br>
                    Xception
                </span>
            </div>

            <div style="margin-bottom:10px;">
                <b style="color:#173f67;">Project Topic:</b>
            </div>

            <div style="
                color:#34495e;
                font-size:13px;
                line-height:1.55;
            ">
                A Comparative Study of Transfer Learning-Based
                Convolutional Neural Networks for Deepfake Image
                Detection.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🔍 Deepfake Image Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        A transfer learning-based system for classifying images as
        Real or Deepfake.
    </div>
    """,
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
# SELECT MODEL
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
# CENTRED + MEDIUM SIZE
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # Three columns are used ONLY to centre the image.
    # The rest of the page is unchanged.

    left_space, image_space, right_space = st.columns(
        [1, 2, 1]
    )

    with image_space:

        st.image(
            image,
            width=450
        )

        st.markdown(
            '<div class="uploaded-caption">'
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
    # ANALYSIS
    # ========================================================

    if analyse_button:

        with st.spinner("Analysing image..."):

            # Resize
            image_resized = image.resize(
                IMAGE_SIZE
            )

            # Convert to NumPy
            image_array = np.asarray(
                image_resized,
                dtype=np.float32
            )

            # Normalisation used during training
            image_array = image_array / 255.0

            # Add batch dimension
            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # Prediction
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

        st.markdown(
            '<div class="section-title">Detection Result</div>',
            unsafe_allow_html=True
        )


        # Result heading
        if prediction == "REAL":

            result_class = "result-real"

            result_icon = "✓"

        else:

            result_class = "result-fake"

            result_icon = "⚠"


        st.markdown(
            f"""
            <div class="result-box">

                <div class="{result_class}">
                    {result_icon} {prediction}
                </div>

                <div class="prediction-percentage">
                    {prediction_probability * 100:.2f}%
                </div>

                <div class="prediction-label">
                    Prediction Probability
                </div>

                <div class="prediction-level">
                    <b>Prediction Level:</b>
                    {prediction_level}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # PROBABILITIES
        # ====================================================

        st.write("")

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
