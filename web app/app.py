import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- Function to Load the Model (with caching) ---
# Caching the model load function prevents it from reloading every time the user interacts with the app.
@st.cache_resource
def load_model():
    """Loads and returns the trained Keras model."""
    # We'll use '3-conv-CNN.h5'. You can change this to your other model if you wish.
    model_path = 'models/3-conv-CNN.h5'
    model = tf.keras.models.load_model(model_path)
    return model

# --- Function to Preprocess Image and Predict ---
def predict_image(image_data, model):
    """
    Takes an image, preprocesses it to fit the model's input requirements,
    and returns the model's prediction.
    """
    # IMPORTANT: The target size must be the same as the input size used for training your model.
    # For example, if your model was trained on 150x150 images, use (150, 150).
    size = (150, 150)
    
    # Resize and fit the image to the target size
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    
    # Convert image to a numpy array
    img_array = np.asarray(image)
    
    # If the image is grayscale, convert it to 3 channels (RGB)
    # as most CNNs expect 3-channel images.
    if img_array.ndim == 2: # Check if it's grayscale
        img_array = np.stack((img_array,)*3, axis=-1)

    # Normalize the image (scale pixel values to between 0 and 1)
    img_array = img_array / 255.0
    
    # Create a batch of 1 to feed into the model
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Make the prediction
    prediction = model.predict(img_batch)
    
    return prediction

# --- Streamlit Web App Interface ---

# Set the title and a small description
st.title("🩺 Tuberculosis Detection from Chest X-Ray")
st.write("Upload a chest X-ray image, and the model will predict whether it shows signs of tuberculosis.")

# Load the trained model
model = load_model()

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # When a file is uploaded, open it as an image
    image = Image.open(uploaded_file).convert('RGB')
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Chest X-ray', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Get the model's prediction
    prediction = predict_image(image, model)
    score = prediction[0][0] # The output is likely a single value in a nested list
    
    # Display the result
    st.write(f"**Prediction Score:** `{score:.4f}`")
    
    # A threshold of 0.5 is common for binary classification.
    # If the score is > 0.5, it's class 1 (Tuberculosis), otherwise class 0 (Normal).
    if score > 0.5:
        st.error(f"**Result:** Tuberculosis Detected (Confidence: {score*100:.2f}%)")
    else:
        st.success(f"**Result:** Normal (Confidence: {(1-score)*100:.2f}%)")