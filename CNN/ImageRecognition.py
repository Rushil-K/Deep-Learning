import streamlit as st
import numpy as np
import tensorflow as tf
import gdown  # To download from Google Drive
from PIL import Image
import os

# Google Drive File ID for the model
FILE_ID = "1uAoz8Rp7vnlZcdOUJakAvS7cOQTcQWA-"
MODEL_PATH = "models/mnist_cnn_t4_optimized.h5"

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Function to download and load the model
@st.cache_resource
def download_and_load_model():
    """Downloads the model if not available locally and loads it."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)

    return tf.keras.models.load_model(MODEL_PATH)

# Load the model once using caching
model = download_and_load_model()

# Function to intelligently preprocess the image
def preprocess_image(image):
    """Processes the image by detecting and correcting the background color."""
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28), Image.Resampling.LANCZOS)  # High-quality resize
    np_image = np.array(image)  # Convert to NumPy array

    # Detect background brightness
    mean_pixel_value = np.mean(np_image)
    
    # If background is bright (closer to white), invert the image
    if mean_pixel_value > 127:
        np_image = 255 - np_image  # Invert colors (black digit on white → white digit on black)

    # Normalize pixel values
    np_image = np_image / 255.0  

    # Reshape for model input
    np_image = np_image.reshape(1, 28, 28, 1)

    return np_image

# Streamlit UI
st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✏️", layout="centered")

st.title("🖊 Handwritten Digit Recognition")
st.write("Upload or capture a digit image, and the model will predict the number.")

# File uploader or camera input
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("📂 Upload an image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

with col2:
    captured_file = st.camera_input("📸 Capture an image using your camera")

# Process the selected image
image = None
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📂 Uploaded Image", use_column_width=True)

elif captured_file:
    image = Image.open(captured_file)
    st.image(image, caption="📸 Captured Image", use_column_width=True)

# Perform prediction if an image is selected
if image:
    processed_image = preprocess_image(image)

    # Model prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)

    st.success(f"### ✨ Predicted Digit: {predicted_class}")

    st.write("### 🔢 Prediction Probabilities:")
    st.bar_chart(prediction.flatten())  # Display probabilities as a bar chart

# Footer with GitHub Follow Button
st.markdown(
    """
    ---
    **👨‍💻 Developed by [Rushil-K](https://github.com/Rushil-K)**
    
    [![Follow on GitHub](https://img.shields.io/github/followers/Rushil-K?label=Follow&style=social)](https://github.com/Rushil-K)
    """,
    unsafe_allow_html=True
)
