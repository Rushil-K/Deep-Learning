import streamlit as st
import numpy as np
import tensorflow as tf
import gdown  # To download from Google Drive
from PIL import Image
import os
import base64

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

# Function to create a GitHub follow button
def github_button():
    github_url = "https://github.com/Rushil-K"
    button_html = f"""
    <div style="text-align: center;">
        <a href="{github_url}" target="_blank" style="
            display: inline-block;
            background-color: #24292e;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            font-weight: bold;
            border-radius: 5px;">
            ⭐ Follow Me on GitHub
        </a>
    </div>
    """
    st.markdown(button_html, unsafe_allow_html=True)

# Streamlit UI
st.markdown(
    "<h1 style='text-align: center; color: #FF4B4B;'>🖊 Handwritten Digit Recognition</h1>",
    unsafe_allow_html=True
)

st.write("### Upload an image or capture from your camera 📷 to predict the digit.")

# File uploader or camera input
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload an image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

with col2:
    captured_file = st.camera_input("Capture an image using your camera")

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

# Display GitHub button at the bottom
st.markdown("---")
github_button()
