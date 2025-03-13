import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import gdown
import os
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import shap

# 🎨 Custom Styling
st.set_page_config(page_title="ANN Dashboard", layout="wide")
st.markdown("""
    <style>
        .main {background-color: #f4f4f4;}
        h1, h2, h3 {color: #333366; text-align: center;}
        .stButton>button {width: 100%; background-color: #333366; color: white; font-size: 16px; border-radius: 5px;}
    </style>
""", unsafe_allow_html=True)

# 🔽 Google Drive File ID for Model and Dataset
MODEL_FILE_ID = "1NNxt6hnkAxUO8aI2sNCzPut0Nbmp8H_T"
DATASET_FILE_ID = "1OPmMFUQmeZuaiYb0FQhwOMZfEbVrWKEK"

# 📥 Download trained model if not present
if not os.path.exists("trained_model.h5"):
    st.sidebar.write("📥 Downloading trained model...")
    gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", "trained_model.h5", quiet=False)

# Load Model
model = load_model("trained_model.h5")

# 📥 Download dataset if not present
if not os.path.exists("dataset.csv"):
    st.sidebar.write("📥 Downloading dataset...")
    gdown.download(f"https://drive.google.com/uc?id={DATASET_FILE_ID}", "dataset.csv", quiet=False)

# 🔽 Load CSV Data
df = pd.read_csv("dataset.csv")

# 🎯 Feature Selection
features = ['Age', 'Gender', 'Income', 'Purchases', 'Clicks', 'Spent']
target = 'Converted'

# 🔽 Ordinal Encoding for 'Gender' (Female=0, Male=1)
encoder = OrdinalEncoder()
df[['Gender']] = encoder.fit_transform(df[['Gender']])

# 🏷️ Random Sampling (50,000 Records)
random_state = random.randint(0, 552627)
df_sample = df.sample(50000, random_state=random_state)

X = df_sample[features]
y = df_sample[target]

# 🔽 Standardization
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X)

# 🔽 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)

# 🏗️ Streamlit UI
st.title("📊 ANN Model Dashboard - Conversion Prediction")
st.sidebar.header("🔧 Model Hyperparameters")

# 🔽 Sidebar Controls
epochs = st.sidebar.slider("Epochs", 10, 100, 50, 10)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.01, 0.001, 0.0001], index=1)
activation_function = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid", "tanh", "softmax"])
optimizer_choice = st.sidebar.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])
dense_layers = st.sidebar.selectbox("Dense Layers", [2, 3, 4, 5])
neurons_per_layer = st.sidebar.selectbox("Neurons per Dense Layer", [32, 64, 128, 256, 512, 1024])
dropout_rate = st.sidebar.slider("Dropout Rate", 0.1, 0.5, 0.1, 0.3)

# 🔽 Select Optimizer
optimizers = {"adam": Adam(learning_rate), "sgd": SGD(learning_rate), "rmsprop": RMSprop(learning_rate)}
optimizer = optimizers[optimizer_choice]

# 🎯 Train Model Button
if st.sidebar.button("🚀 Train Model"):
    with st.spinner("Training model... ⏳"):
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=128, validation_split=0.2, verbose=0)
    st.success("🎉 Model training complete!")

    # 🔍 Model Evaluation
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    st.subheader("📊 Model Performance")
    st.write(f"**Test Loss:** {loss:.4f}")
    st.write(f"**Test Accuracy:** {accuracy:.4f}")

    # 📈 Training Performance Plots
    st.subheader("📈 Training Performance")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    ax[0].plot(history.history['accuracy'], label="Train Accuracy", color='blue')
    ax[0].plot(history.history['val_accuracy'], label="Validation Accuracy", color='green')
    ax[0].set_title("Accuracy over Epochs")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()

    # Loss Plot
    ax[1].plot(history.history['loss'], label="Train Loss", color='blue')
    ax[1].plot(history.history['val_loss'], label="Validation Loss", color='red')
    ax[1].set_title("Loss over Epochs")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    ax[1].legend()

    st.pyplot(fig)

    # 🔄 Confusion Matrix
    st.subheader("📊 Confusion Matrix")
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Converted", "Converted"], yticklabels=["Not Converted", "Converted"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # 📝 Classification Report
    st.subheader("📜 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    # 🔍 Feature Importance using SHAP
    st.subheader("🔍 Feature Importance")
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(fig)

    st.markdown("✅ **Key Insights from Feature Importance:**")
    st.write("- If 'Spent' and 'Clicks' have the highest importance, user engagement is key for conversion.")
    st.write("- If 'Gender' is high, the model might have bias.")
    st.write("- If 'Age' has low importance, conversion isn’t strongly related to age.")

    st.markdown("📌 **Conclusion:**")
    st.write("This ANN model helps predict conversions with key insights into user behavior. Keep optimizing hyperparameters for better results!")
