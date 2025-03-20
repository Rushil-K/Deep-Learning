import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import requests

# Load the trained CNN model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_cnn_t4_optimized.h5")

model = load_model()

# Preprocess the uploaded image
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to match model input
    image = np.array(image) / 255.0  # Normalize pixel values
    image = image.reshape(1, 28, 28, 1)  # Reshape for model input
    return image

# Streamlit UI
st.set_page_config(page_title="Handwritten Digit Recognition", layout="centered")

st.title("🖊️ Handwritten Digit Recognition")
st.write("Upload an image of a handwritten number (0-9) and let the AI predict it.")

# File uploader
uploaded_file = st.file_uploader("Upload a handwritten digit image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)

    # Display result
    st.subheader(f"Predicted Number: {predicted_digit}")

    # Show confidence scores
    st.bar_chart(prediction[0])

# GitHub Follow Button
st.markdown("""
    <div style="text-align: center; padding-top: 20px;">
        <a href="https://github.com/YOUR_GITHUB_USERNAME" target="_blank">
            <button style="
                background-color: black;
                color: white;
                padding: 10px 20px;
                font-size: 16px;
                border: none;
                cursor: pointer;
                border-radius: 5px;">
                ⭐ Follow me on GitHub
            </button>
        </a>
    </div>
""", unsafe_allow_html=True)

