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

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Chicken Disease Classifier | AI-Powered Diagnostics",
    page_icon="🐔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #2E7D32;
        --secondary-color: #4CAF50;
        --accent-color: #FF6F00;
        --background-color: #FAFAFA;
        --card-bg: #FFFFFF;
        --text-primary: #212121;
        --text-secondary: #757575;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Hero section styling */
    .hero-container {
        background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 50%, #81C784 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(46, 125, 50, 0.3);
    }

    .hero-title {
        color: white;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    .hero-subtitle {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.3rem;
        font-weight: 400;
    }

    /* Card styling */
    .info-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    }

    /* Disease cards */
    .disease-card {
        background: linear-gradient(145deg, #ffffff 0%, #f5f5f5 100%);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border-left: 4px solid;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }

    .disease-card.coccidiosis { border-left-color: #E53935; }
    .disease-card.salmonella { border-left-color: #FB8C00; }
    .disease-card.newcastle { border-left-color: #8E24AA; }
    .disease-card.healthy { border-left-color: #43A047; }

    /* Result card styling */
    .result-card {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 8px 30px rgba(76, 175, 80, 0.2);
        border: 2px solid #4CAF50;
    }

    .result-card.disease-detected {
        background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
        border-color: #E53935;
        box-shadow: 0 8px 30px rgba(229, 57, 53, 0.2);
    }

    .result-title {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .result-disease {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .result-confidence {
        font-size: 1.5rem;
        font-weight: 600;
    }

    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #4CAF50, #81C784);
        border-radius: 10px;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2E7D32;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Upload area styling */
    .upload-area {
        border: 2px dashed #4CAF50;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        background: #F1F8E9;
        transition: all 0.3s ease;
    }

    /* Feature cards */
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1B5E20 0%, #2E7D32 100%);
    }

    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stTitle {
        color: white !important;
    }

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: white !important;
    }

    /* Button styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%);
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.3);
        transition: all 0.3s ease;
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(46, 125, 50, 0.4);
    }

    /* LIME section */
    .lime-container {
        background: #FAFAFA;
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid #E0E0E0;
        color: #666;
    }

    /* Animations */
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }

    .animate-pulse {
        animation: pulse 2s infinite;
    }

    /* Probability bar chart */
    .prob-bar {
        height: 24px;
        border-radius: 12px;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        padding: 0 1rem;
        color: white;
        font-weight: 600;
        transition: width 0.5s ease;
    }

    /* Status indicators */
    .status-healthy { color: #43A047; }
    .status-warning { color: #FB8C00; }
    .status-danger { color: #E53935; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS
# =============================================================================
MODEL_PATH = "models/best_chicken_model.h5"
MODEL_URL = os.environ.get("MODEL_URL", "https://huggingface.co/tiff12riziki/chicken-disease-classifier/resolve/main/best_chicken_model.h5")

DISEASE_INFO = {
    'Coccidiosis': {
        'icon': '🦠',
        'color': '#E53935',
        'description': 'A parasitic disease affecting the intestinal tract, causing diarrhea and reduced growth.',
        'symptoms': 'Bloody droppings, weight loss, dehydration'
    },
    'Salmonella': {
        'icon': '🔬',
        'color': '#FB8C00',
        'description': 'A bacterial infection that can spread to humans through contaminated eggs or meat.',
        'symptoms': 'Diarrhea, lethargy, reduced egg production'
    },
    'New Castle Disease': {
        'icon': '🦠',
        'color': '#8E24AA',
        'description': 'A highly contagious viral disease affecting respiratory, nervous, and digestive systems.',
        'symptoms': 'Respiratory distress, neurological signs, mortality'
    },
    'Healthy': {
        'icon': '✅',
        'color': '#43A047',
        'description': 'No disease detected. The chicken appears to be in good health.',
        'symptoms': 'Normal behavior, healthy appearance'
    }
}

# =============================================================================
# CACHED FUNCTIONS
# =============================================================================
def download_model():
    """Download model from external URL if not present"""
    if not MODEL_URL:
        return False

    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        import urllib.request
        st.info("📥 Downloading model... This may take a few minutes.")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        return True
    except Exception as e:
        st.error(f"Error downloading model: {e}")
        return False


@st.cache_resource
def load_prediction_model():
    """Load the trained model"""
    if not os.path.exists(MODEL_PATH):
        # Try to download if URL is provided
        if not download_model():
            return None

    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data
def get_sample_images_cached():
    """Get sample images for reference"""
    data_path = "data/Train"
    if not os.path.exists(data_path):
        return None
    samples = get_sample_images("data", CLASS_NAMES, num_samples=2)
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


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h1 style="color: white; font-size: 1.5rem;">🐔 PoultryDx</h1>
        <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">AI-Powered Diagnostics</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📋 About This App")
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
    <p style="color: rgba(255,255,255,0.9); font-size: 0.9rem; line-height: 1.6;">
    This application uses a <strong>Convolutional Neural Network (CNN)</strong> trained on thousands of chicken images to accurately classify poultry diseases.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎯 Supported Diseases")
    for disease, info in DISEASE_INFO.items():
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.1); padding: 0.5rem; border-radius: 8px; margin-bottom: 0.5rem;">
            <span style="font-size: 1.2rem;">{info['icon']}</span>
            <span style="color: white; font-weight: 500;">{disease}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🔬 Model Info")
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;">
        <p style="color: rgba(255,255,255,0.9); font-size: 0.85rem; margin: 0.3rem 0;">
            <strong>Architecture:</strong> CNN with BatchNorm
        </p>
        <p style="color: rgba(255,255,255,0.9); font-size: 0.85rem; margin: 0.3rem 0;">
            <strong>Input Size:</strong> 224 × 224 px
        </p>
        <p style="color: rgba(255,255,255,0.9); font-size: 0.85rem; margin: 0.3rem 0;">
            <strong>Classes:</strong> 4 disease categories
        </p>
        <p style="color: rgba(255,255,255,0.9); font-size: 0.85rem; margin: 0.3rem 0;">
            <strong>Explainability:</strong> LIME
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <p style="color: rgba(255,255,255,0.7); font-size: 0.8rem;">
            Developed with TensorFlow & Streamlit
        </p>
        <p style="color: rgba(255,255,255,0.5); font-size: 0.75rem;">
            © 2024 Chicken Disease Classification
        </p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    # Hero Section
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">🐔 Chicken Disease Classification</h1>
        <p class="hero-subtitle">AI-Powered Poultry Health Diagnostics for Early Disease Detection</p>
    </div>
    """, unsafe_allow_html=True)

    # Check if model exists
    model = load_prediction_model()

    if model is None:
        st.markdown("""
        <div class="info-card" style="text-align: center; margin-bottom: 2rem;">
            <h2 style="color: #FF6F00;">⚠️ Model Not Found</h2>
            <p style="color: #666;">Please train the model by running the notebook training cells.</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Train Demo Model", type="primary", use_container_width=True):
                with st.spinner("Creating demonstration model..."):
                    model = create_dummy_model()
                    X_train = np.random.random((100, IMAGE_SIZE, IMAGE_SIZE, 3))
                    y_train = np.random.randint(0, 4, (100,))
                    y_train = tf.keras.utils.to_categorical(y_train, num_classes=4)
                    model.fit(X_train, y_train, epochs=5, batch_size=10, verbose=0)
                    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                    model.save(MODEL_PATH)
                    st.success("✅ Model created successfully!")
                st.rerun()
        return

    # Main content area - Two column layout
    col_left, col_right = st.columns([2, 1])

    with col_left:
        # Upload Section
        st.markdown("""
        <div class="info-card" style="margin-bottom: 1.5rem;">
            <h3 style="color: #2E7D32; margin-bottom: 1rem;">📤 Upload Chicken Image</h3>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Drag and drop an image or click to browse",
            type=['jpg', 'jpeg', 'png'],
            help="Supported formats: JPG, JPEG, PNG"
        )

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)

            # Image display columns
            img_col1, img_col2 = st.columns(2)

            with img_col1:
                st.markdown("**Original Image**")
                st.image(image, use_container_width=True)

            with img_col2:
                st.markdown("**Preprocessed (224×224)**")
                img_array = preprocess_image(uploaded_file)
                st.image(img_array[0], use_container_width=True)

            # Predict button
            st.markdown("")
            predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
            with predict_col2:
                predict_btn = st.button("🔍 Analyze Image", type="primary", use_container_width=True)

            if predict_btn:
                with st.spinner("🔬 Analyzing image with deep learning model..."):
                    # Get prediction
                    predicted_class, confidence, all_probs = predict_disease(model, img_array)

                    # Determine result styling
                    is_healthy = predicted_class == 'Healthy'
                    result_class = "" if is_healthy else "disease-detected"
                    status_emoji = "✅" if is_healthy else "⚠️"

                    # Display main result
                    st.markdown(f"""
                    <div class="result-card {result_class}" style="margin-top: 1.5rem;">
                        <div class="result-title">{status_emoji} Analysis Complete</div>
                        <div class="result-disease" style="color: {DISEASE_INFO[predicted_class]['color']};">
                            {predicted_class}
                        </div>
                        <div class="result-confidence">
                            Confidence: {confidence:.1%}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Disease information
                    if not is_healthy:
                        st.markdown(f"""
                        <div class="info-card" style="margin-top: 1rem; border-left: 4px solid {DISEASE_INFO[predicted_class]['color']};">
                            <h4 style="color: {DISEASE_INFO[predicted_class]['color']}; margin-bottom: 0.5rem;">
                                {DISEASE_INFO[predicted_class]['icon']} About {predicted_class}
                            </h4>
                            <p style="color: #666; margin-bottom: 0.5rem;">{DISEASE_INFO[predicted_class]['description']}</p>
                            <p style="color: #888; font-size: 0.9rem;"><strong>Common Symptoms:</strong> {DISEASE_INFO[predicted_class]['symptoms']}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # All class probabilities
                    st.markdown("""
                    <div class="info-card" style="margin-top: 1.5rem;">
                        <h4 style="color: #2E7D32; margin-bottom: 1rem;">📊 Classification Probabilities</h4>
                    </div>
                    """, unsafe_allow_html=True)

                    prob_cols = st.columns(4)
                    for i, (cls, prob) in enumerate(zip(CLASS_NAMES, all_probs)):
                        with prob_cols[i]:
                            # Custom styled probability display
                            st.markdown(f"""
                            <div style="text-align: center; padding: 1rem; background: linear-gradient(145deg, #f5f5f5, #ffffff); border-radius: 12px; margin-bottom: 0.5rem;">
                                <div style="font-size: 1.5rem;">{DISEASE_INFO[cls]['icon']}</div>
                                <div style="font-size: 0.85rem; font-weight: 600; color: {DISEASE_INFO[cls]['color']}; margin: 0.3rem 0;">{cls}</div>
                                <div style="font-size: 1.3rem; font-weight: 700; color: #333;">{prob:.1%}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            prob_cols[i].progress(float(prob))

    with col_right:
        # Quick Reference Section
        st.markdown("""
        <div class="info-card" style="margin-bottom: 1rem;">
            <h4 style="color: #2E7D32; margin-bottom: 1rem;">📚 Disease Reference</h4>
        </div>
        """, unsafe_allow_html=True)

        for disease, info in DISEASE_INFO.items():
            st.markdown(f"""
            <div class="disease-card {disease.lower().replace(' ', '-')}" style="border-left-color: {info['color']};">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.3rem;">
                    <span style="font-size: 1.3rem;">{info['icon']}</span>
                    <span style="font-weight: 600; color: {info['color']};">{disease}</span>
                </div>
                <p style="font-size: 0.8rem; color: #666; margin: 0;">{info['description'][:60]}...</p>
            </div>
            """, unsafe_allow_html=True)

        # Model Performance Stats
        st.markdown("""
        <div class="info-card" style="margin-top: 1.5rem;">
            <h4 style="color: #2E7D32; margin-bottom: 1rem;">📈 Model Performance</h4>
        </div>
        """, unsafe_allow_html=True)

        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Classes", "4")
        with metric_col2:
            st.metric("Image Size", "224px")

    # LIME Explanation Section (Full width)
    if uploaded_file is not None:
        st.markdown("""
        <div class="lime-container" style="margin-top: 2rem;">
            <h3 style="color: #2E7D32;">🔍 AI Explainability (LIME)</h3>
            <p style="color: #666; margin-bottom: 1rem;">
                LIME highlights which regions of the image influenced the model's decision.
                <strong>Green areas</strong> supported the prediction, helping veterinarians understand AI reasoning.
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("🔬 Generating explainability visualization..."):
            try:
                fig, explanation = get_lime_explanation(model, img_array, CLASS_NAMES)
                st.pyplot(fig)

                st.info("💡 **Tip:** The highlighted regions show which parts of the image the AI focused on to make its prediction.")
            except Exception as e:
                st.warning(f"Could not generate explanation: {str(e)[:100]}")

    # Footer
    st.markdown("""
    <div class="footer">
        <p style="font-size: 0.9rem; color: #888;">
            🐔 <strong>Chicken Disease Classification System</strong> | Powered by TensorFlow & Streamlit
        </p>
        <p style="font-size: 0.8rem; color: #AAA;">
            Built for poultry health monitoring and early disease detection
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
