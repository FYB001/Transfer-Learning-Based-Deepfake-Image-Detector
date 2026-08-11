import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Deepfake Image Detection | Wrexham University",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .main {
        padding-top: 1rem;
    }

    /* Header */
    .academic-header {
        padding: 1.5rem 2rem;
        border-radius: 12px;
        background: linear-gradient(
            135deg,
            #111827 0%,
            #1e293b 100%
        );
        color: white;
        margin-bottom: 1.5rem;
        border-left: 5px solid #3b82f6;
    }

    .academic-header h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 700;
    }

    .academic-header p {
        margin: 0.4rem 0 0 0;
        color: #cbd5e1;
        font-size: 1rem;
    }

    /* Project title */
    .project-title {
        padding: 1rem 1.2rem;
        border-radius: 10px;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }

    .project-title h3 {
        margin: 0;
        color: #0f172a;
        font-size: 1.15rem;
    }

    /* Cards */
    .info-card {
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        background-color: #ffffff;
        margin-bottom: 1rem;
    }

    .info-card h4 {
        margin-top: 0;
        color: #0f172a;
    }

    /* Result cards */
    .real-result {
        padding: 1.5rem;
        border-radius: 12px;
        background-color: #ecfdf5;
        border: 1px solid #86efac;
        margin: 1rem 0;
    }

    .fake-result {
        padding: 1.5rem;
        border-radius: 12px;
        background-color: #fef2f2;
        border: 1px solid #fca5a5;
        margin: 1rem 0;
    }

    .result-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    /* Small academic label */
    .academic-label {
        font-size: 0.82rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }

    /* Footer */
    .footer {
        margin-top: 3rem;
        padding: 1.5rem;
        text-align: center;
        border-top: 1px solid #e2e8f0;
        color: #64748b;
        font-size: 0.85rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        border-right: 1px solid #e2e8f0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="academic-header">


        <h1>🔍 Deepfake Image Detection System</h1>

        <p>
            Topic: A Comparative Study of Transfer Learning-Based 
Convolutional Neural Networks for Deepfake Image 
Detection.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.markdown(
    """
    <div class="project-title">

        <div class="academic-label">
            Dissertation Project
        </div>

        <h3>
           Topic: A Comparative Study of Transfer Learning-Based 
Convolutional Neural Networks for Deepfake Image 
Detection.
        </h3>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## Researcher")

    st.markdown(
        """
        **Fatmata Yealie Bangura**

        MSc Data Science and Big Data Analytics 

        **Wrexham University**
        """
    )

    st.divider()

    st.markdown("## System Information")

    st.write("**Task:** Binary image classification")

    st.write("**Classes:** Real / Deepfake")

    st.write("**Input:** 224 × 224 pixels")

    st.write("**Normalisation:** Pixel values / 255")

    st.write("**Models:** 5 transfer-learning CNNs")

    st.divider()

    st.markdown("## Models Evaluated")

    st.write("• VGG16")

    st.write("• ResNet50")

    st.write("• MobileNetV2")

    st.write("• EfficientNetB0")

    st.write("• Xception")

    st.divider()

    st.caption(
        "This application is an academic research prototype "
        "developed as part of the dissertation project."
    )


# ============================================================
# INTRODUCTION
# ============================================================

col1, col2 = st.columns([2, 1])

with col1:

    st.markdown("### About the System")

    st.write(
        """
        This application demonstrates the practical deployment of
        five transfer learning-based convolutional neural network
        models for deepfake image detection.

        Users can upload a facial image and select one of the trained
        CNN architectures. The selected model processes the image
        and produces a classification together with the estimated
        probability of the image being real or deepfake.
        """
    )

with col2:

    st.markdown(
        """
        <div class="info-card">

        <h4>Research Objective</h4>

        To compare the effectiveness of transfer learning-based
        CNN architectures for detecting deepfake images.

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
# MODEL DESCRIPTIONS
# ============================================================

MODEL_DESCRIPTIONS = {

    "VGG16":
        "A deep CNN architecture based on stacked 3 × 3 convolutional layers.",

    "ResNet50":
        "A residual network that uses shortcut connections to support deeper learning.",

    "MobileNetV2":
        "A lightweight CNN architecture designed for computational efficiency.",

    "EfficientNetB0":
        "A compact architecture using compound scaling to balance model depth, width and resolution.",

    "Xception":
        "A CNN architecture based on depthwise separable convolutions."
}


# ============================================================
# IMAGE SIZE
# ============================================================

IMAGE_SIZE = (224, 224)


# ============================================================
# MODEL SELECTION
# ============================================================

st.markdown("### 1. Select Model")

selected_model = st.selectbox(
    "Choose the CNN architecture to use for prediction:",
    list(MODEL_PATHS.keys())
)

st.caption(
    MODEL_DESCRIPTIONS[selected_model]
)


# ============================================================
# CHECK MODEL
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
# UPLOAD SECTION
# ============================================================

st.markdown("### 2. Upload Image")

uploaded_file = st.file_uploader(
    "Upload a facial image for analysis",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ],
    help="Supported formats: JPG, JPEG, PNG and WEBP."
)


# ============================================================
# IMAGE DISPLAY
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    image_col, details_col = st.columns(
        [2, 1]
    )

    with image_col:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with details_col:

        st.markdown(
            """
            <div class="info-card">

            <h4>Image Information</h4>

            """
            + f"""
            <p><b>File:</b> {uploaded_file.name}</p>
            <p><b>Original size:</b> {image.size[0]} × {image.size[1]}</p>
            <p><b>Model input:</b> 224 × 224</p>
            <p><b>Colour mode:</b> RGB</p>
            """
            + """
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # ANALYSE BUTTON
    # ========================================================

    analyse = st.button(
        "🔍 Analyse Image",
        type="primary",
        use_container_width=True
    )


    if analyse:

        with st.spinner(
            "Processing image and generating prediction..."
        ):

            # Resize
            image_resized = image.resize(
                IMAGE_SIZE
            )

            # Convert to NumPy
            image_array = np.asarray(
                image_resized,
                dtype=np.float32
            )

            # Normalisation
            image_array = (
                image_array / 255.0
            )

            # Add batch dimension
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

            # Calculate fake probability
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
        # RESULTS
        # ====================================================

        st.markdown("### 3. Detection Result")

        st.markdown(
            f"""
            <div class="{result_class}">

                <div class="academic-label">
                    Classification Result
                </div>

                <div class="result-title">
                    {prediction}
                </div>

                <div>
                    Confidence: <b>{confidence * 100:.2f}%</b>
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # PROBABILITY METRICS
        # ====================================================

        metric1, metric2 = st.columns(2)

        with metric1:

            st.metric(
                "Real Probability",
                f"{probability_real * 100:.2f}%"
            )

            st.progress(
                min(
                    max(
                        probability_real,
                        0.0
                    ),
                    1.0
                )
            )

        with metric2:

            st.metric(
                "Deepfake Probability",
                f"{probability_fake * 100:.2f}%"
            )

            st.progress(
                min(
                    max(
                        probability_fake,
                        0.0
                    ),
                    1.0
                )
            )


        # ====================================================
        # MODEL DETAILS
        # ====================================================

        st.markdown("### Model Information")

        info1, info2, info3 = st.columns(3)

        with info1:

            st.info(
                f"**Model**\n\n{selected_model}"
            )

        with info2:

            st.info(
                "**Input Resolution**\n\n224 × 224 pixels"
            )

        with info3:

            st.info(
                "**Normalisation**\n\n1 / 255"
            )


# ============================================================
# METHODOLOGY
# ============================================================

st.divider()

st.markdown("### Research Methodology")

tab1, tab2, tab3 = st.tabs(
    [
        "Transfer Learning",
        "Classification",
        "Evaluation"
    ]
)


with tab1:

    st.write(
        """
        The system uses transfer learning-based convolutional
        neural networks. Pre-trained CNN architectures are adapted
        for binary deepfake image classification.
        """
    )


with tab2:

    st.write(
        """
        The classification task contains two classes:

        **Real (1)** and **Deepfake (0)**.

        The model produces a probability value which is used to
        determine the predicted class using a 0.5 decision threshold.
        """
    )


with tab3:

    st.write(
        """
        Model performance is evaluated using classification metrics
        including accuracy, precision, recall, F1-score and
        ROC-AUC. These metrics support comparative analysis of the
        five CNN architectures.
        """
    )


# ============================================================
# IMPORTANT DISCLAIMER
# ============================================================

st.divider()

st.warning(
    """
    **Research Prototype Disclaimer**

    This application is developed for academic research and
    demonstration purposes. Predictions should not be interpreted
    as definitive evidence that an image is authentic or manipulated.
    Detection performance may vary for images originating from
    datasets, sources or manipulation techniques that differ from
    those represented in the training data.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        <b>Transfer Learning-Based Deepfake Image Detection</b>
        <br><br>

        Fatmata Yealie Bangura
        <br>

        MSc International Health Services Management
        <br>

        Wrexham University
        <br><br>

        Master's Dissertation Research Prototype
        <br>

        © 2026

    </div>
    """,
    unsafe_allow_html=True
)
