import streamlit as st
import numpy as np
import joblib
import os
from PIL import Image
from keras.preprocessing.image import img_to_array

# Load the model
try:
    model = joblib.load('best_model.pkl')  
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()

def preprocess_image(image, target_size=(64, 64)):
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0
    return image

def predict_image(model, image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Streamlit UI
st.title("🌲 EcoFlame Shield: Protecting Nature with AI 🔥")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=400)  # Set appropriate width
    
    if model:
        prediction = predict_image(model, image)
        class_label = "🔥 Wildfire Detected" if prediction < 0.5 else "✅ No Wildfire"
        
        # Center-align prediction output
        st.markdown(f"<h2 style='text-align: center; color: Light Pink;'>{class_label}</h2>", unsafe_allow_html=True)
    else:
        st.error("⚠️ Model not loaded correctly!")
