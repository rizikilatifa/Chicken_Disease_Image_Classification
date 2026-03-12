import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import lime
from lime import lime_image
import io
import os

# Constants
IMAGE_SIZE = 224
CLASS_NAMES = ['Coccidiosis', 'Salmonella', 'New Castle Disease', 'Healthy']


def preprocess_image(img):
    """
    Preprocess image for model input.

    Args:
        img: PIL Image or BytesIO object

    Returns:
        Preprocessed image array ready for model prediction
    """
    # Handle BytesIO object from Streamlit
    if hasattr(img, 'read'):
        img = Image.open(img)

    if isinstance(img, Image.Image):
        img = img.convert('RGB')
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

    img_array = np.array(img)
    img_array = img_array.astype('float32')
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def load_model(model_path):
    """
    Load trained model from file.

    Args:
        model_path: Path to the saved model file

    Returns:
        Loaded Keras model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = tf.keras.models.load_model(model_path)
    return model


def predict_disease(model, img_array):
    """
    Predict disease from image.

    Args:
        model: Trained Keras model
        img_array: Preprocessed image array

    Returns:
        tuple of (predicted_class, confidence, all_probabilities)
    """
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions, axis=1)[0]
    return CLASS_NAMES[predicted_class_idx], confidence, predictions[0]


def get_lime_explanation(model, img_array, class_names=None, top_labels=4):
    """
    Generate LIME explanation for the model prediction.

    Args:
        model: Trained Keras model
        img_array: Preprocessed image array
        class_names: List of class names (optional, defaults to CLASS_NAMES)
        top_labels: Number of top labels to show

    Returns:
        matplotlib figure with LIME explanation
    """
    if class_names is None:
        class_names = CLASS_NAMES

    # Create LIME explainer
    explainer = lime_image.LimeImageExplainer()

    # Generate explanation
    explanation = explainer.explain_instance(
        img_array[0].astype('uint8'),
        classifier_fn=model.predict,
        top_labels=top_labels,
        hide_color=0,
        num_samples=1000
    )

    # Get the mask and explanation
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=10,
        hide_rest=False
    )

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Original image
    axes[0].imshow(img_array[0])
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # LIME heatmap overlay
    lime_heatmap = temp.copy()
    lime_heatmap = lime_heatmap.astype('float32')
    axes[1].imshow(lime_heatmap)
    axes[1].imshow(mask, alpha=0.5, cmap='jet')
    axes[1].set_title('LIME Explanation')
    axes[1].axis('off')

    plt.tight_layout()

    return fig, explanation


def get_sample_images(data_path, class_names, num_samples=1):
    """
    Get sample images for each class for reference.

    Args:
        data_path: Path to training data directory
        class_names: List of class names
        num_samples: Number of samples per class

    Returns:
        Dictionary of class names to list of sample image paths
    """
    samples = {}

    train_dir = os.path.join(data_path, 'Train')

    if not os.path.exists(train_dir):
        return samples

    for class_name in class_names:
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.exists(class_dir):
            continue

        # Get all image files
        all_files = os.listdir(class_dir)
        images = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if len(images) > 0:
            # Get full paths for sample images
            sample_paths = [os.path.join(class_dir, img) for img in images[:num_samples]]
            samples[class_name] = sample_paths

    return samples
