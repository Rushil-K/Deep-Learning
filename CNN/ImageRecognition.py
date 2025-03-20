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

    # Load the model
    return tf.keras.models.load_model(MODEL_PATH)

# Load the model once using caching
model = download_and_load_model()

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image) / 255.0  # Normalize to range [0,1]
    image = image.reshape(1, 28, 28, 1)  # Reshape for model input
    return image

# Custom CSS for Styling
st.markdown(
    """
    <style>
    body {
        background-color: #f4f4f4;
        font-family: 'Arial', sans-serif;
    }
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #4A90E2;
    }
    .upload-box {
        border: 2px dashed #4A90E2;
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        text-align: center;
    }
    .prediction-box {
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #2D3748;
        margin-top: 20px;
    }
    .bar-chart {
        margin-top: 10px;
    }
    .github-btn {
        display: flex;
        justify-content: center;
        margin-top: 30px;
    }
    .github-btn a {
        text-decoration: none;
        background-color: #24292E;
        color: white;
        padding: 10px 15px;
        border-radius: 5px;
        font-weight: bold;
    }
    .github-btn a:hover {
        background-color: #0366d6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.markdown('<h1 class="main-title">🖊 Handwritten Digit Recognition</h1>', unsafe_allow_html=True)
st.write("### Upload a handwritten digit image, and the model will predict the number.")

st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload an image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

    processed_image = preprocess_image(image)

    # Model prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)

    st.markdown(f'<div class="prediction-box">✨ Predicted Digit: {predicted_class}</div>', unsafe_allow_html=True)
    st.write("### 🔢 Prediction Probabilities:")
    st.bar_chart(prediction.flatten())  # Display probabilities as a bar chart

# GitHub Follow Button
st.markdown(
    """
    <div class="github-btn">
        <a href="https://github.com/Rushil-K" target="_blank">🚀 Follow Me on GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
