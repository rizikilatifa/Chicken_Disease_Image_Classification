import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import io
import os
import lime
from lime import lime_image
from utils import preprocess_image, predict_disease, get_lime_explanation, get_sample_images, CLASS_NAMES, IMAGE_SIZE

# Page configuration
st.set_page_config(
    page_title="Chicken Disease Classifier",
    page_icon=":chicken:",
    layout="wide"
)

# Constants
MODEL_PATH = "models/best_chicken_model.h5"


@st.cache_resource
def load_prediction_model():
    """Load the trained model"""
    if not os.path.exists(MODEL_PATH):
        return None
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


@st.cache_data
def get_sample_images_cached():
    """Get sample images for reference"""
    data_path = "data /Train"
    if not os.path.exists(data_path):
        # Try alternative path
        data_path = "data/Train"
        if not os.path.exists(data_path):
            return None
    samples = get_sample_images(data_path, CLASS_NAMES, num_samples=1)
    return samples


def create_dummy_model():
    """Create a simple model for demonstration"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Sidebar
with st.sidebar:
    st.title("🐔 Chicken Disease Classifier")
    st.markdown("""
    This application uses deep learning to classify chicken diseases from images.

    **Supported Diseases:**
    - 🦠 **Coccidiosis** - Parasitic intestinal disease
    - 🦠 **Salmonella** - Bacterial infection
    - 🦠 **New Castle Disease** - Viral disease
    - ✅ **Healthy** - No disease detected

    Upload an image to get a prediction!
    """)

    st.markdown("---")
    st.markdown("**Model Interpretability**")
    st.markdown("""
    This app uses LIME (Local Interpretable Model-agnostic Explanations) to highlight which regions
    of the image are most important for the prediction.

    This helps veterinarians and farmers understand the model's decision-making process.
    """)

    st.markdown("---")
    st.markdown("**How to Use:**")
    st.markdown("""
    1. Upload a chicken image (JPG, JPEG, PNG)
    2. Click 'Predict' to classify the disease
    3. View LIME explanations to understand model decision-making
    """)


# Main app
def main():
    st.title("🐔 Chicken Disease Classification")
    st.subheader("Upload an image to get a disease prediction")

    # Check if model exists
    model = load_prediction_model()

    if model is None:
        st.warning("⚠️ Model not found! Please train the model first.")
        st.info("Run the notebook training cell to save the model.")

        if st.button("Train Dummy Model (for demo)", type="primary"):
            with st.spinner("Creating and training dummy model..."):
                model = create_dummy_model()
                # Create dummy data for training
                X_train = np.random.random((100, IMAGE_SIZE, IMAGE_SIZE, 3))
                y_train = np.random.randint(0, 4, (100,))
                y_train = tf.keras.utils.to_categorical(y_train, num_classes=4)
                model.fit(X_train, y_train, epochs=5, batch_size=10, verbose=0)
                os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                model.save(MODEL_PATH)
                st.success(f"✅ Model saved to {MODEL_PATH}")
        st.rerun()

    # File uploader section
    st.header("Upload Image", divider=True)
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Drag and drop an image here or click to browse"
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=400)

        # Preprocess image
        img_array = preprocess_image(uploaded_file)
        st.image(img_array[0], caption="Preprocessed Image (224x224)", use_column_width=400)

        # Predict button
        if st.button("Predict Disease", type="primary"):
            with st.spinner("Analyzing image..."):
                # Get prediction
                predicted_class, confidence, all_probs = predict_disease(model, img_array)

                # Display results
                st.success("✅ Prediction Complete!")

                # Main result
                st.markdown(f"### Predicted Disease: **{predicted_class}**")
                st.markdown(f"**Confidence:** {confidence:.2%}")

                # All class probabilities
                st.markdown("#### Class Probabilities:")
                cols = st.columns(4)
                for i, (cls, prob) in enumerate(zip(CLASS_NAMES, all_probs)):
                    cols[i].metric(f"{cls}", f"{prob:.4f}", delta=f"{prob:.4f}")
                    # Progress bar
                    cols[i].progress(prob)

        # LIME Explanation section (outside prediction button)
        st.markdown("---")
        st.subheader("🔍 Model Interpretability (LIME)")
        st.markdown("""
        LIME (Local Interpretable Model-agnostic Explanations) highlights which regions
        of the image contributed most to the prediction.
        """)

        with st.spinner("Generating LIME explanation... (this may take a moment)"):
            try:
                fig, explanation = get_lime_explanation(model, img_array, CLASS_NAMES)
                st.pyplot(fig)

                # Show explanation details
                st.markdown("#### Explanation Details:")
                st.markdown("""
                - **Green/Red regions** highlight important areas for the prediction
                - The intensity indicates the strength of contribution
                """)
            except Exception as e:
                st.error(f"Error generating LIME explanation: {e}")
                with st.expander("Show error details"):
                    st.code(str(e))

        # Sample images section
        st.markdown("---")
        st.subheader("📷 Sample Images by Class")
        samples = get_sample_images_cached()
        if samples:
            for class_name, sample_paths in samples.items():
                st.markdown(f"**{class_name}**")
                if sample_paths:
                    cols = st.columns(min(len(sample_paths), 1))
                    for idx, img_path in enumerate(sample_paths):
                        try:
                            img = Image.open(img_path)
                            img = img.resize((150, 150))
                            cols[idx].image(img)
                        except Exception as e:
                            continue
        else:
            st.info("No sample images available. Please ensure the 'data/Train' folder exists.")


if __name__ == "__main__":
    main()
