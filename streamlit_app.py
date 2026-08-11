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
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM DESIGN
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #f7f9fc;
    }

    /* Reduce top spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Main title */
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #17365D;
        margin-bottom: 0.2rem;
    }

    /* Subtitle */
    .subtitle {
        font-size: 1.05rem;
        color: #5b6573;
        margin-bottom: 0.4rem;
    }

    /* Researcher */
    .researcher {
        font-size: 0.9rem;
        color: #7a8491;
        margin-bottom: 2rem;
    }

    /* Upload card */
    .upload-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    }

    /* Result card */
    .result-card {
        background: white;
        border-radius: 14px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        text-align: center;
    }

    /* Real */
    .real-result {
        color: #16803c;
        font-size: 2rem;
        font-weight: 700;
    }

    /* Fake */
    .fake-result {
        color: #c62828;
        font-size: 2rem;
        font-weight: 700;
    }

    /* Small label */
    .small-label {
        color: #7a8491;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #8a94a3;
        font-size: 0.8rem;
        padding-top: 2rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #17365D;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    </style>
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
# IMAGE SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🔍 Deepfake Detector"
    )

    st.markdown("---")

    st.markdown(
        "**CNN Model**"
    )

    selected_model = st.selectbox(
        "Select model",
        list(MODEL_PATHS.keys())
    )

    st.markdown("---")

    st.markdown(
        "**Researcher**"
    )

    st.markdown(
        "Fatmata Yealie Bangura"
    )

    st.markdown(
        "Wrexham University"
    )

    st.markdown("---")

    st.caption(
        "Master's Dissertation Research Prototype"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🔍 Deepfake Image Detection'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'A Comparative Study of Transfer Learning-Based '
    'Convolutional Neural Networks for Deepfake Image Detection'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="researcher">'
    'Fatmata Yealie Bangura&nbsp;&nbsp;|&nbsp;&nbsp;Wrexham University'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MODEL FILE CHECK
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


with st.spinner(
    f"Loading {selected_model}..."
):

    model = load_model(model_path)


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    "### Upload an image"
)

uploaded_file = st.file_uploader(
    "Choose a facial image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ],
    label_visibility="collapsed"
)


# ============================================================
# IMAGE DISPLAY
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    image_col, info_col = st.columns(
        [2.4, 1],
        gap="large"
    )

    with image_col:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with info_col:

        st.markdown(
            """
            <div class="upload-card">

            <div class="small-label">
            Selected Model
            </div>

            <h3>
            """
            + selected_model
            + """
            </h3>

            <div class="small-label">
            Input
            </div>

            <p>
            224 × 224 pixels
            </p>

            <div class="small-label">
            Normalisation
            </div>

            <p>
            1 / 255
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # ANALYSE BUTTON
    # ========================================================

    st.write("")

    analyse = st.button(
        "🔍 Analyse Image",
        type="primary",
        use_container_width=True
    )


    if analyse:

        with st.spinner(
            "Analysing image..."
        ):

            # Resize
            image_resized = image.resize(
                IMAGE_SIZE
            )

            # Convert to array
            image_array = np.asarray(
                image_resized,
                dtype=np.float32
            )

            # Normalisation
            image_array = (
                image_array / 255.0
            )

            # Batch dimension
            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # Prediction
            probability_real = float(
                model.predict(
                    image_array,
                    verbose=0
                )[0][0]
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

            result_class = "real-result"

        else:

            prediction = "DEEPFAKE"

            confidence = probability_fake

            result_class = "fake-result"


        # ====================================================
        # RESULT
        # ====================================================

        st.markdown(
            '<div class="result-card">'
            '<div class="small-label">'
            'Detection Result'
            '</div>'
            f'<div class="{result_class}">'
            f'{prediction}'
            '</div>'
            f'<p><b>Confidence:</b> '
            f'{confidence * 100:.2f}%</p>'
            '</div>',
            unsafe_allow_html=True
        )


        # ====================================================
        # PROBABILITIES
        # ====================================================

        real_col, fake_col = st.columns(
            2,
            gap="large"
        )

        with real_col:

            st.metric(
                "Real Probability",
                f"{probability_real * 100:.2f}%"
            )

            st.progress(
                probability_real
            )

        with fake_col:

            st.metric(
                "Deepfake Probability",
                f"{probability_fake * 100:.2f}%"
            )

            st.progress(
                probability_fake
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    Transfer Learning-Based Deepfake Image Detection
    <br>
    Fatmata Yealie Bangura • Wrexham University • 2026

    </div>
    """,
    unsafe_allow_html=True
)
