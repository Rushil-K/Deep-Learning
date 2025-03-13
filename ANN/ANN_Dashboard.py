import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import random
import os

# 📌 Google Drive File ID
file_id = "18_IlD33FyWSy1kSSEaCBfmAeyQCXqaV1"  # Replace with your actual file ID
dataset_path = "dataset.csv"

# Check if dataset is already downloaded
if not os.path.exists(dataset_path):
    with st.spinner("Downloading dataset from Google Drive... ⏳"):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", dataset_path, quiet=False)
    st.success("Dataset downloaded successfully! 🎉")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(dataset_path)

full_df = load_data()

# Streamlit UI
st.title("📊 ANN Model Dashboard - Conversion Prediction")
st.sidebar.header("🔧 Model Hyperparameters")

# Randomly select 50,000 records for training
def get_sample_data(df, sample_size=50000):
    return df.sample(sample_size, random_state=random.randint(0, 9999))

data = get_sample_data(full_df)

# Preprocessing
features = ['Age', 'Gender', 'Income', 'Purchases', 'Clicks', 'Spent']
target = 'Converted'

X = data[features]
y = data[target]

# One-hot encoding for categorical features
X = pd.get_dummies(X, columns=['Gender'], drop_first=True)

# Standardizing numerical features
scaler = StandardScaler()
num_cols = ['Age', 'Income', 'Purchases', 'Clicks', 'Spent']
X[num_cols] = scaler.fit_transform(X[num_cols])

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Load the trained model
model = load_model("trained_model.h5")

# Sidebar - Hyperparameters
epochs = st.sidebar.slider("Epochs", min_value=10, max_value=100, step=10, value=50)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.01, 0.001, 0.0001], index=1)
activation_function = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid", "tanh", "softmax"])
optimizer_choice = st.sidebar.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])
dense_layers = st.sidebar.selectbox("Dense Layers", [2, 3, 4, 5])
neurons_per_layer = st.sidebar.selectbox("Neurons per Dense Layer", [32, 64, 128, 256, 512, 1024])
dropout_rate = st.sidebar.slider("Dropout Rate", 0.1, 0.5, step=0.1, value=0.3)

# Select optimizer
optimizers = {"adam": Adam(learning_rate), "sgd": SGD(learning_rate), "rmsprop": RMSprop(learning_rate)}
optimizer = optimizers[optimizer_choice]

# Fine-tune the model instead of full retraining
with st.spinner("Fine-tuning the model... ⏳"):
    for layer in model.layers[:-1]:  # Freeze all layers except the last one
        layer.trainable = False
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=128, validation_split=0.2, verbose=0)

st.success("🎉 Model fine-tuning complete!")

# Model Evaluation
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

st.subheader("📊 Model Performance")
st.write(f"**Test Loss:** {loss:.4f}")
st.write(f"**Test Accuracy:** {accuracy:.4f}")

# Plot Accuracy & Loss
st.subheader("📈 Training Performance")
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy Plot
ax[0].plot(history.history['accuracy'], label="Train Accuracy")
ax[0].plot(history.history['val_accuracy'], label="Validation Accuracy")
ax[0].set_title("Accuracy over Epochs")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")
ax[0].legend()

# Loss Plot
ax[1].plot(history.history['loss'], label="Train Loss")
ax[1].plot(history.history['val_loss'], label="Validation Loss")
ax[1].set_title("Loss over Epochs")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
ax[1].legend()

st.pyplot(fig)

# Confusion Matrix
st.subheader("📊 Confusion Matrix")
y_pred = (model.predict(X_test) > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Converted", "Converted"], yticklabels=["Not Converted", "Converted"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# Classification Report
st.subheader("📜 Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

# Feature Importance using Permutation Importance
st.subheader("🔍 Feature Importance")
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.linear_model import LogisticRegression

# Using Logistic Regression as an alternative feature importance estimator
perm_model = LogisticRegression(max_iter=500)
perm_model.fit(X_train, y_train)
perm = PermutationImportance(perm_model, random_state=552627).fit(X_test, y_test)

feature_importance = pd.DataFrame({"Feature": X.columns, "Importance": perm.feature_importances_})
feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=feature_importance, palette="coolwarm")
ax.set_title("Feature Importance")
st.pyplot(fig)

st.markdown("✅ **Key Insights from Feature Importance:**")
st.write("- If 'Spent' and 'Clicks' have the highest importance, it suggests user engagement is key for conversion.")
st.write("- If 'Gender' is too high, the model might have bias.")
st.write("- If 'Age' has low importance, conversion isn’t strongly related to age.")

st.markdown("📌 **Conclusion:**")
st.write("This ANN model helps predict conversions with key insights into user behavior. Keep optimizing hyperparameters for better results!")
