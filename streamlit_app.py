import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Deepfake Image Detection",
    page_icon="🔍",
    layout="centered"
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🔍 Deepfake Image Detection")

st.write(
    "Upload a facial image to classify it as either "
    "Real or Deepfake using one of the five trained "
    "transfer learning CNN models."
)

st.info(
    "Models: VGG16, ResNet50, MobileNetV2, "
    "EfficientNetB0 and Xception"
)


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_DIR = "models"

MODEL_PATHS = {
    "VGG16": os.path.join(
        MODEL_DIR, "VGG16_best.keras"
    ),

    "ResNet50": os.path.join(
        MODEL_DIR, "ResNet50_best.keras"
    ),

    "MobileNetV2": os.path.join(
        MODEL_DIR, "MobileNetV2_best.keras"
    ),

    "EfficientNetB0": os.path.join(
        MODEL_DIR, "EfficientNetB0_best.keras"
    ),

    "Xception": os.path.join(
        MODEL_DIR, "Xception_best.keras"
    )
}


# ============================================================
# IMAGE SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)


# ============================================================
# MODEL SELECTION
# ============================================================

selected_model = st.selectbox(
    "Select a CNN model:",
    list(MODEL_PATHS.keys())
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
# IMAGE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)


# ============================================================
# IMAGE ANALYSIS
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button(
        "Analyse Image",
        type="primary"
    ):

        with st.spinner(
            "Analysing image..."
        ):

            # Resize to the same input size
            # used during training
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

            # Prediction
            probability_real = float(
                model.predict(
                    image_array,
                    verbose=0
                )[0][0]
            )

            # Binary classification:
            # fake = 0
            # real = 1
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
        # DISPLAY RESULT
        # ====================================================

        st.subheader(
            f"Prediction: {prediction}"
        )

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )


        # ====================================================
        # PROBABILITIES
        # ====================================================

        st.write(
            f"**Real probability:** "
            f"{probability_real * 100:.2f}%"
        )

        st.write(
            f"**Deepfake probability:** "
            f"{probability_fake * 100:.2f}%"
        )


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        st.divider()

        st.write(
            f"**Model used:** {selected_model}"
        )

        st.write(
            "**Input size:** 224 × 224 pixels"
        )

        st.write(
            "**Pixel normalisation:** 1/255"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Transfer Learning-Based Deepfake Image Detection"
)

st.caption(
    "VGG16 • ResNet50 • MobileNetV2 • "
    "EfficientNetB0 • Xception"
)
