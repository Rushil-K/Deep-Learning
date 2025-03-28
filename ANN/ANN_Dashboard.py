import os
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import gdown
import shap
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE
from PIL import Image

# Set Streamlit Page Configuration
st.set_page_config(page_title="ANN Conversion Prediction", layout="wide")

# --- 1. Data Loading and Preprocessing ---

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

smote = SMOTE(random_state=552627)
X_resampled, y_resampled = smote.fit_resample(X, y)
X_resampled = pd.DataFrame(X_resampled, columns=X.columns)  # Ensure DataFrame structure

# Standardize Data
scaler = StandardScaler()
X_resampled[X.columns] = scaler.fit_transform(X_resampled)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=552627)

# Compute Class Weights
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}


# --- 2. Model Building ---
def build_model(input_shape, optimizer, activation, dense_layers, neurons, dropout_rate):
    """
    Builds a TensorFlow Keras Sequential model.

    Args:
        input_shape (tuple): Shape of the input data.
        optimizer (tf.keras.optimizers.Optimizer): The optimizer to use.
        activation (str): Activation function for the dense layers.
        dense_layers (int): Number of dense layers.
        neurons (int): Number of neurons per dense layer.
        dropout_rate (float): Dropout rate.

    Returns:
        tf.keras.Model: A compiled Keras model.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(shape=input_shape))  # Corrected input_shape
    for _ in range(dense_layers):
        model.add(tf.keras.layers.Dense(neurons, activation=activation))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model



# --- 3. Streamlit UI ---

# 🎨 Custom CSS for Enhanced UI
st.markdown(
    """
    <style>
        /* Updated color scheme */
        .title {
            color: #26ae60; /* Emerald green */
            text-align: center;
            margin-bottom: 2rem;
        }
        .sidebar-header {
            color: #3498db; /* Blue */
            font-size: 1.5em;
            margin-bottom: 1rem;
        }
        .metric-label {
            font-size: 1.2em;
            color: #34495e; /* Dark gray */
        }
        .metric-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #e67e22; /* Orange */
        }
        .stButton > button {
            background-color: #26ae60; /* Emerald green */
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s ease;
            width: 100%;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); /* Add shadow */
        }
        .stButton > button:hover {
            background-color: #217f4d; /* Darker emerald */
            box-shadow: 0 3px 7px rgba(0,0,0,0.3); /* Increase shadow on hover*/
        }
        .stSlider > div > div > div > div {
            background-color: #26ae60; /* Emerald green */
        }
        .stSelectbox > div > div {
            border-color: #26ae60; /* Emerald green */
        }
        .stSelectbox > div > div:focus-within {
            border-color: #217f4d; /* Darker emerald green on focus */
            box-shadow: 0 0 0 3px rgba(38, 174, 96, 0.3); /* Green focus shadow */
        }
        .dataframe {
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            padding: 10px;
            background-color: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            overflow-x: auto;
            margin-bottom: 20px;
        }
        .stPlotlyChart {
            margin-bottom: 20px;
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            padding: 10px;
            background-color: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# 🖼️ App Title and Logo
st.title("📊 ANN Model Dashboard - Conversion Prediction")


# Sidebar for Hyperparameter Tuning
st.sidebar.header("🔧 Model Hyperparameters")
epochs = st.sidebar.slider("Epochs", 5, 12, 10, 1)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.01, 0.001, 0.0001], index=1)
activation_function = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid", "tanh", "elu"], index=0)
optimizer_choice = st.sidebar.selectbox("Optimizer", ["adam", "sgd", "rmsprop", "adamax"], index=0)
dense_layers = st.sidebar.selectbox("Dense Layers", [2, 3, 4, 5], index=1)
neurons_per_layer = st.sidebar.selectbox("Neurons per Layer", [32, 64, 128, 256], index=2)
dropout_rate = st.sidebar.slider("Dropout Rate", 0.1, 0.5, 0.2, 0.05)

# Select Optimizer
optimizers = {
    "adam": Adam(learning_rate=learning_rate),
    "sgd": SGD(learning_rate=learning_rate),
    "rmsprop": RMSprop(learning_rate=learning_rate),
    "adamax": tf.keras.optimizers.Adamax(learning_rate=learning_rate)
}
optimizer = optimizers[optimizer_choice]

# --- 4. Model Training and Evaluation ---
if st.button("🚀 Train Model"):
    with st.spinner("Training model... ⏳"):
        model = build_model(X_train.shape[1], optimizer, activation_function, dense_layers, neurons_per_layer, dropout_rate)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=128, validation_split=0.2, class_weight=class_weight_dict, verbose=0)

    st.success("🎉 Model training complete!")

    # Model Performance Evaluation
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Display Metrics with Custom Styling
    st.subheader("📊 Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<p class='metric-label'>Test Accuracy</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{accuracy:.4f}</p>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<p class='metric-label'>Test Loss</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{loss:.4f}</p>", unsafe_allow_html=True)

    # Training Performance Plots
    st.subheader("📈 Training Performance")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    ax[0].plot(history.history['accuracy'], label="Train Accuracy", color="#2ecc71")
    ax[0].plot(history.history['val_accuracy'], label="Validation Accuracy", color="#e74c3c")
    ax[0].set_title("Accuracy over Epochs")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()
    ax[0].grid(True)

    # Loss Plot
    ax[1].plot(history.history['loss'], label="Train Loss", color="#2ecc71")
    ax[1].plot(history.history['val_loss'], label="Validation Loss", color="#e74c3c")
    ax[1].set_title("Loss over Epochs")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    ax[1].grid(True)

    st.pyplot(fig)

    # Confusion Matrix
    st.subheader("📊 Confusion Matrix")
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=["Not Converted", "Converted"], yticklabels=["Not Converted", "Converted"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Classification Report
    st.subheader("📜 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.2f}"))

    # ROC Curve and AUC
    st.subheader("📈 ROC Curve and AUC")
    y_pred_proba = model.predict(X_test).ravel()
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 4))
    plt.plot(fpr, tpr, color="#3498db", lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(True)
    st.pyplot(fig)

    # Feature Importance using SHAP
    st.subheader("🔍 Feature Importance")
    explainer = shap.Explainer(model, X_train[:100])
    shap_values = explainer(X_test[:100])

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test[:100], show=False)
    st.pyplot(fig)

    # Feature Importance Stats
    st.subheader("📌 Feature Importance Stats")
    mean_abs_shap_values = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': mean_abs_shap_values})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    st.dataframe(importance_df)

# GitHub Follow Button
st.markdown(
    """
    <div style="text-align: center; margin-top: 2rem;">
        <a href="https://github.com/Rushil-K" target="_blank" rel="noopener noreferrer">
            <button style="background-color: #3498db; color: white; padding: 12px 24px; font-size: 18px; border: none; border-radius: 6px; cursor: pointer; transition: background-color 0.3s ease;">
                ⭐ Follow Me on GitHub
            </button>
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)
