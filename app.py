import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf

from src.config import MODEL_PATH, CLASS_NAMES_PATH
from src.model_utils import load_class_names, predict_emotion

st.set_page_config(page_title="Emotion Detection Demo", page_icon="😊", layout="centered")

st.title("😊 Facial Emotion Detection")
st.markdown("### Final Project Demo")

with st.container():
    st.markdown(
        """
        **Project Summary**  
        This demo uses a Convolutional Neural Network (CNN) trained on facial expression images
        to classify emotions from a face photo. The model predicts one of these emotions:
        **angry, disgust, fear, happy, neutral, sad,** or **surprise**.
        """
    )

with st.expander("Project Information"):
    st.write("- Student Name: Frida Villa Hernandez")
    st.write("- Dataset: FER-2013")
    st.write("- Model: CNN image classifier")

st.divider()

# ---------- Model loading ----------
if not MODEL_PATH.exists():
    st.error("Model file not found. Please run `python train.py` first.")
    st.stop()

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = load_class_names(CLASS_NAMES_PATH)
except Exception as exc:
    st.error(f"Could not load model or labels: {exc}")
    st.stop()

# ---------- Upload ----------
uploaded_file = st.file_uploader(
    "Upload a face image for prediction",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Choose one clear face image to start the demo.")
    st.stop()

image = Image.open(uploaded_file)
st.image(image, caption="Uploaded Image", use_container_width=True)

# ---------- Prediction ----------
with st.spinner("Analyzing facial expression..."):
    label, probabilities = predict_emotion(model, class_names, image)

confidence = float(max(probabilities) * 100)

st.success(f"Predicted Emotion: **{label.capitalize()}**")
st.metric("Confidence", f"{confidence:.2f}%")

# ---------- Cleaner probability output ----------
results = pd.DataFrame({
    "Emotion": [name.capitalize() for name in class_names],
    "Probability": probabilities,
})
results = results.sort_values(by="Probability", ascending=False).reset_index(drop=True)
results["Probability"] = results["Probability"] * 100

st.markdown("### Top Results")
top_results = results.head(3).copy()
top_results["Probability"] = top_results["Probability"].map(lambda x: f"{x:.2f}%")
st.table(top_results)

st.markdown("### Probability Distribution")
chart_data = results.set_index("Emotion")
st.bar_chart(chart_data)
