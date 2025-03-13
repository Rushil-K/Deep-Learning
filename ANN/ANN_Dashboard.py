import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import gdown
import zipfile
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import random

# 🎯 Google Drive File ID (ZIP File Containing Dataset)
file_id = "18_IlD33FyWSy1kSSEaCBfmAeyQCXqaV1"
zip_output = "dataset.zip"

# 📥 Download ZIP dataset from Google Drive
st.info("📥 Downloading dataset ZIP file...")
gdown.download(f"https://drive.google.com/uc?id={file_id}", zip_output, quiet=False)

# 🔓 Extract ZIP file
st.info("🔓 Extracting dataset...")
with zipfile.ZipFile(zip_output, "r") as zip_ref:
    zip_ref.extractall("extracted_data")  # Extracts files into this folder

# 🔍 Find the CSV file in extracted folder
csv_file = None
for file in os.listdir("extracted_data"):
    if file.endswith(".csv"):
        csv_file = os.path.join("extracted_data", file)
        break

# ✅ Load dataset
if csv_file:
    st.success(f"✅ Dataset extracted successfully! Loading {csv_file}...")
    full_df = pd.read_csv(csv_file, encoding="ISO-8859-1")
else:
    st.error("❌ No CSV file found in ZIP! Please check the ZIP contents.")
    st.stop()

# 🧐 Show a sample of the dataset
st.write("📂 **Dataset Preview:**")
st.dataframe(full_df.head())

# 🎯 Feature & Target Selection
features = ["Age", "nmrk2627_encoded_Gender", "Income", "Purchases", "Clicks", "Spent"]
target = "Converted"

# Randomly select 50,000 records for training
def get_sample_data(df, sample_size=50000):
    return df.sample(sample_size, random_state=random.randint(0, 9999))

data = get_sample_data(full_df)

X = data[features]
y = data[target]

# 🏷️ One-hot encoding for 'Gender' column
if "Gender" in X.columns:
    X = pd.get_dummies(X, columns=["Gender"], drop_first=True)  # Converts 'Male/Female' → Binary

# 🔬 Standardizing numerical features
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X)

# 📊 Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 📥 Load the trained ANN model
st.info("📥 Loading trained ANN model...")
model = load_model("trained_model.h5")
st.success("✅ Model loaded successfully!")

# 🎛️ Sidebar - Hyperparameters
epochs = st.sidebar.slider("Epochs", min_value=10, max_value=100, step=10, value=50)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.01, 0.001, 0.0001], index=1)
activation_function = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid", "tanh", "softmax"])
optimizer_choice = st.sidebar.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])
dense_layers = st.sidebar.selectbox("Dense Layers", [2, 3, 4, 5])
neurons_per_layer = st.sidebar.selectbox("Neurons per Dense Layer", [32, 64, 128, 256, 512, 1024])
dropout_rate = st.sidebar.slider("Dropout Rate", 0.1, 0.5, step=0.1, value=0.3)

# 🚀 Select optimizer
optimizers = {"adam": Adam(learning_rate), "sgd": SGD(learning_rate), "rmsprop": RMSprop(learning_rate)}
optimizer = optimizers[optimizer_choice]

# 🔄 Retrain model with new hyperparameters
with st.spinner("🚀 Training model... Please wait!"):
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=128, validation_split=0.2, verbose=0)

st.success("🎉 Model training complete!")

# 📈 Model Evaluation
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
st.subheader("📊 Model Performance")
st.write(f"**Test Loss:** {loss:.4f}")
st.write(f"**Test Accuracy:** {accuracy:.4f}")

# 📊 Accuracy & Loss Plot
st.subheader("📈 Training Performance")
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy Plot
ax[0].plot(history.history['accuracy'], label="Train Accuracy", color="blue")
ax[0].plot(history.history['val_accuracy'], label="Validation Accuracy", color="red")
ax[0].set_title("Accuracy over Epochs")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")
ax[0].legend()

# Loss Plot
ax[1].plot(history.history['loss'], label="Train Loss", color="blue")
ax[1].plot(history.history['val_loss'], label="Validation Loss", color="red")
ax[1].set_title("Loss over Epochs")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
ax[1].legend()

st.pyplot(fig)

# 🔍 Confusion Matrix
st.subheader("📊 Confusion Matrix")
y_pred = (model.predict(X_test) > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Converted", "Converted"], yticklabels=["Not Converted", "Converted"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# 🏆 Classification Report
st.subheader("📜 Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

# 🔍 Feature Importance using SHAP
st.subheader("🔍 Feature Importance using SHAP")
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

fig, ax = plt.subplots(figsize=(8, 5))
shap.summary_plot(shap_values, X_test, show=False)
st.pyplot(fig)

st.markdown("✅ **Key Insights from Feature Importance:**")
st.write("- If 'Spent' and 'Clicks' have the highest importance, it suggests user engagement is key for conversion.")
st.write("- If 'Gender' is too high, the model might have bias.")
st.write("- If 'Age' has low importance, conversion isn’t strongly related to age.")

st.markdown("📌 **Conclusion:**")
st.write("This ANN model helps predict conversions with key insights into user behavior. Keep optimizing hyperparameters for better results!")
