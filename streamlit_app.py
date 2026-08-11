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

    /* Main page */
    .stApp {
        background-color: #f7f9fc;
    }

    /* Main content width */
    .main .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Header */
    .main-title {
        text-align: center;
        color: #12355b;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .subtitle {
        text-align: center;
        color: #52606d;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    /* Information box */
    .project-box {
        background-color: #eaf3ff;
        border-left: 5px solid #2f80ed;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    }

    .project-box-title {
        color: #12355b;
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 0.3rem;
    }

    .project-box-text {
        color: #374151;
        font-size: 0.92rem;
        line-height: 1.5;
    }

    /* Upload section */
    .section-title {
        color: #12355b;
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.6rem;
    }

    /* Result card */
    .result-card {
        background-color: white;
        border-radius: 12px;
        padding: 1.4rem;
        margin-top: 1.5rem;
        border: 1px solid #d9e2ec;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }

    .result-title {
        color: #12355b;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    .prediction {
        font-size: 2rem;
        font-weight: 800;
        text-align: center;
        margin: 0.5rem 0;
    }

    .prediction-real {
        color: #16803c;
    }

    .prediction-fake {
        color: #c62828;
    }

    .prediction-level {
        text-align: center;
        font-size: 1.1rem;
        font-weight: 700;
        color: #374151;
        margin-bottom: 1rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.8rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #d9e2ec;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    .sidebar-title {
        color: #12355b;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .sidebar-text {
        color: #52606d;
        font-size: 0.88rem;
        line-height: 1.5;
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
        '<div class="sidebar-title">🔍 Deepfake Detector</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-text">

        <b>Student:</b><br>
        Fatmata Yealie Bangura

        <br><br>

        <b>University:</b><br>
        Wrexham University

        <br><br>

        <b>Programme:</b><br>
        MSc Data Science and Big Data Analytics

        <br><br>

        <b>Models:</b><br>
        VGG16<br>
        ResNet50<br>
        MobileNetV2<br>
        EfficientNetB0<br>
        Xception

        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.caption(
        "Transfer Learning-Based Deepfake Image Detection"
    )


# ============================================================
# MAIN TITLE
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

        <div class="project-box-title">
        Master's Dissertation Project
        </div>

        <div class="project-box-text">
        <b>
        A Comparative Study of Transfer Learning-Based
        Convolutional Neural Networks for Deepfake Image Detection
        </b>
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL DIRECTORY
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
# IMAGE SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)


# ============================================================
# MODEL SELECTION
# ============================================================

st.markdown(
    '<div class="section-title">Select CNN Model</div>',
    unsafe_allow_html=True
)

selected_model = st.selectbox(
    "Choose a trained model for image analysis:",
    list(MODEL_PATHS.keys())
)


# ============================================================
# MODEL PATH CHECK
# ============================================================

model_path = MODEL_PATHS[selected_model]

if not os.path.exists(model_path):

    st.error(
        f"The selected model could not be found: {model_path}"
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


with st.spinner(
    f"Loading {selected_model} model..."
):

    model = load_model(model_path)


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">Upload Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload a facial image for classification",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# ============================================================
# IMAGE DISPLAY AND ANALYSIS
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # --------------------------------------------------------
    # DISPLAY IMAGE AT NORMAL SIZE
    # --------------------------------------------------------

    image_col1, image_col2, image_col3 = st.columns(
        [1, 2, 1]
    )

    with image_col2:

        st.image(
            image,
            caption="Uploaded Image",
            width=500
        )


    # --------------------------------------------------------
    # ANALYSE BUTTON
    # --------------------------------------------------------

    st.write("")

    analyse = st.button(
        "🔍 Analyse Image",
        type="primary",
        use_container_width=True
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    if analyse:

        with st.spinner(
            "Analysing image..."
        ):

            # ------------------------------------------------
            # RESIZE IMAGE
            # ------------------------------------------------

            image_resized = image.resize(
                IMAGE_SIZE
            )


            # ------------------------------------------------
            # CONVERT TO NUMPY ARRAY
            # ------------------------------------------------

            image_array = np.asarray(
                image_resized,
                dtype=np.float32
            )


            # ------------------------------------------------
            # NORMALISATION
            # ------------------------------------------------

            image_array = image_array / 255.0


            # ------------------------------------------------
            # ADD BATCH DIMENSION
            # ------------------------------------------------

            image_array = np.expand_dims(
                image_array,
                axis=0
            )


            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            prediction_output = model.predict(
                image_array,
                verbose=0
            )


            probability_real = float(
                prediction_output[0][0]
            )


            # ------------------------------------------------
            # CALCULATE DEEPFAKE PROBABILITY
            # ------------------------------------------------

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
        # RESULT CARD
        # ====================================================

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="result-title">Detection Result</div>',
            unsafe_allow_html=True
        )


        # ====================================================
        # CLASSIFICATION
        # ====================================================

        if prediction == "REAL":

            st.markdown(
                '<div class="prediction prediction-real">REAL</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div style="
                    text-align:center;
                    font-size:1.05rem;
                    color:#374151;
                    margin-bottom:1rem;
                ">
                This image is classified as
                <b>REAL</b> with a probability of
                <b>{probability_real * 100:.2f}%</b>.
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="prediction prediction-fake">DEEPFAKE</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div style="
                    text-align:center;
                    font-size:1.05rem;
                    color:#374151;
                    margin-bottom:1rem;
                ">
                This image is classified as
                <b>DEEPFAKE</b> with a probability of
                <b>{probability_fake * 100:.2f}%</b>.
                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # PREDICTION LEVEL
        # ====================================================

        st.markdown(
            f"""
            <div style="
                text-align:center;
                font-size:1rem;
                font-weight:600;
                color:#374151;
                margin-bottom:1rem;
            ">
            Prediction Level: {prediction_level}
            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # PROBABILITY PERCENTAGES
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
        # PREDICTION CONFIDENCE
        # ====================================================

        st.markdown(
            "<div style='margin-top:1rem;'>"
            "<b>Prediction Confidence</b>"
            "</div>",
            unsafe_allow_html=True
        )

        st.progress(confidence)

        st.markdown(
            f"""
            <div style="
                text-align:center;
                font-size:1.15rem;
                font-weight:700;
                color:#12355b;
                margin-top:0.4rem;
            ">
            {confidence * 100:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )


        # Close result card
        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # ====================================================
        # ANALYSIS INFORMATION
        # ====================================================

        st.divider()

        info_col1, info_col2, info_col3 = st.columns(3)

        with info_col1:

            st.caption("MODEL")

            st.write(
                selected_model
            )

        with info_col2:

            st.caption("INPUT SIZE")

            st.write(
                "224 × 224"
            )

        with info_col3:

            st.caption("NORMALISATION")

            st.write(
                "1 / 255"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    <b>Transfer Learning-Based Deepfake Image Detection</b>
    <br>
    Master's Dissertation Project
    <br>
    Wrexham University

    </div>
    """,
    unsafe_allow_html=True
)
