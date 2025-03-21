import streamlit as st
import numpy as np
import joblib
import os
from PIL import Image
from keras.preprocessing.image import img_to_array

try:
    model = joblib.load('best_model.pkl')  
    
except Exception as e:
    st.error(f"⚠️ Error loading model or scaler: {e}")
    st.stop()

def preprocess_image(image, target_size=(64,64)):
    image = image.resize(target_size)  
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
  
    image /= 255.0
    return image

def predict_image(model, image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

st.title("EcoFlame Shield: Protecting Nature with AI")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")
    
    if model:
        prediction = predict_image(model, image)
        class_label = "Wildfire Detected" if prediction < 0.5 else "No Wildfire"
        st.write(f"Prediction: {class_label}")
    else:
        st.error("⚠️ Model not loaded correctly!")
