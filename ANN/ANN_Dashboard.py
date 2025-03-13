import os
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import gdown
import random
import shap
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# Set Streamlit Page Configuration
st.set_page_config(page_title="ANN Conversion Prediction", layout="wide")

# 📥 Load Dataset
DATASET_FILE_ID = "1OPmMFUQmeZuaiYb0FQhwOMZfEbVrWKEK"
if not os.path.exists("data.csv"):
    gdown.download(f"https://drive.google.com/uc?id={DATASET_FILE_ID}", "data.csv", quiet=False)

df = pd.read_csv("data.csv")

# 🎯 Feature Selection
features = ['Age', 'Gender', 'Income', 'Purchases', 'Clicks', 'Spent']
target = 'Converted'

# 🔄 Encode Categorical Features
encoder = OrdinalEncoder()
df[['Gender']] = encoder.fit_transform(df[['Gender']])

# Handle Class Imbalance with SMOTE
X = df[features]
y = df[target]

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Standardize Data
scaler = StandardScaler()
X_resampled[X.columns] = scaler.fit_transform(X_resampled)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

# Compute Class Weights
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# 🏗️ Streamlit UI
st.title("📊 ANN Model Dashboard - Conversion Prediction")
st.sidebar.header("🔧 Model Hyperparameters")

# Hyperparameters
epochs = st.sidebar.slider("Epochs", 10, 100, 50, 10)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.01, 0.001, 0.0001], index=1)
activation_function = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid", "tanh", "softmax"])
optimizer_choice = st.sidebar.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])
dense_layers = st.sidebar.selectbox("Dense Layers", [2, 3, 4, 5])
neurons_per_layer = st.sidebar.selectbox("Neurons per Layer", [32, 64, 128, 256, 512, 1024])
dropout_rate = st.sidebar.slider("Dropout Rate", 0.1, 0.5, 0.1, 0.3)

# Select Optimizer
optimizers = {"adam": Adam(learning_rate), "sgd": SGD(learning_rate), "rmsprop": RMSprop(learning_rate)}
optimizer = optimizers[optimizer_choice]

# 🎛️ Train Model Button
if st.button("🚀 Train Model"):
    with st.spinner("Training model... ⏳"):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)))
        
        for _ in range(dense_layers):
            model.add(tf.keras.layers.Dense(neurons_per_layer, activation=activation_function))
            model.add(tf.keras.layers.Dropout(dropout_rate))
        
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=128, validation_split=0.2, class_weight=class_weight_dict, verbose=0)
    
    st.success("🎉 Model training complete!")

    # Model Performance
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    st.subheader("📊 Model Performance")
    st.metric(label="Test Accuracy", value=f"{accuracy:.4f}")
    st.metric(label="Test Loss", value=f"{loss:.4f}")

    # 📈 Training Performance Plots
    st.subheader("📈 Training Performance")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    ax[0].plot(history.history['accuracy'], label="Train Accuracy", color="blue")
    ax[0].plot(history.history['val_accuracy'], label="Validation Accuracy", color="orange")
    ax[0].set_title("Accuracy over Epochs")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()
    ax[0].grid()

    # Loss Plot
    ax[1].plot(history.history['loss'], label="Train Loss", color="blue")
    ax[1].plot(history.history['val_loss'], label="Validation Loss", color="orange")
    ax[1].set_title("Loss over Epochs")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    ax[1].grid()

    st.pyplot(fig)

    # 🔄 Confusion Matrix
    st.subheader("📊 Confusion Matrix")
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", xticklabels=["Not Converted", "Converted"], yticklabels=["Not Converted", "Converted"])
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
    explainer = shap.Explainer(model, X_train[:100])
    shap_values = explainer(X_test[:100])

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test[:100], show=False)
    st.pyplot(fig)

    st.markdown("✅ **Key Insights from Feature Importance:**")
    st.write("- If 'Spent' and 'Clicks' are dominant, engagement is key for conversion.")
    st.write("- If 'Gender' is important, there may be model bias.")
    st.write("- If 'Age' has low impact, conversion isn’t age-dependent.")

st.markdown("📌 **Conclusion:**")
st.write("Follow Me on GitHub while the visuals appear...")
