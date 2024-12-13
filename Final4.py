import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import gdown

# Page configuration
st.set_page_config(
    page_title="Card Classification",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Card Classification System - Powered by TensorFlow"
    }
)

# Custom CSS (previous CSS remains the same)
st.markdown("""
    <style>
        /* Previous CSS styles */
    </style>
""", unsafe_allow_html=True)

# Enhanced model loading function with comprehensive error handling
def load_model_safely():
    # Potential model file locations to check
    model_paths = [
        'best_model.keras',  # Current directory
        'models/best_model.keras',  # models subdirectory
        '/content/best_model.keras',  # Google Colab typical path
        os.path.join(os.getcwd(), 'best_model.keras'),  # Full path in current working directory
    ]

    # Try loading from different potential paths
    for path in model_paths:
        try:
            st.info(f"Attempting to load model from: {path}")
            
            # Check if file exists
            if not os.path.exists(path):
                st.warning(f"File not found: {path}")
                continue
            
            # Attempt to load the model
            model = load_model(path)
            st.success(f"Model successfully loaded from {path}")
            return model
        
        except Exception as e:
            st.error(f"Error loading model from {path}: {str(e)}")
    
    # If all attempts fail
    st.error("Could not load the model from any of the expected locations.")
    return None

# Dynamically get class names 
def get_class_names(model):
    try:
        # Get output layer units (number of classes)
        num_classes = model.layers[-1].output_shape[-1]
        
        # Generate default class names if not predefined
        return [f'Class_{i}' for i in range(num_classes)]
    except Exception as e:
        st.error(f"Error extracting class names: {str(e)}")
        return []

# Rest of the previous code remains the same...
def preprocess_image(img):
    img = img.convert("RGB")
    img = np.array(img)
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def predict(model, img, class_names):
    prediction = model.predict(img, verbose=0)
    predicted_class_idx = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class_idx]
    return class_names[predicted_class_idx], confidence, prediction[0]

def main():
    st.markdown("""
        <h1 style='text-align: center; color: #2E7D32;'>🃏 Card Classification</h1>
        <p style='text-align: center; font-size: 1.2em;'>Upload an image or use the live feed to identify playing cards</p>
        <hr>
    """, unsafe_allow_html=True)

    # Load the model
    model = load_model_safely()

    if model is None:
        st.error("❌ Failed to load model. Please check the following:")
        st.info("""
        1. Ensure 'best_model.keras' is in the same directory
        2. Verify the model file is not corrupted
        3. Check TensorFlow and Keras versions are compatible
        4. Upload the model file if missing
        """)
        return

    # Get class names
    class_names = get_class_names(model)

    # Sidebar and rest of the UI remains the same as in previous version
    with st.sidebar:
        st.markdown("### 🃏 Input Options")
        option = st.radio("Select Input Method:", ["Upload Image 📁", "Live Feed 📸"])
        
        st.markdown("### ℹ️ About")
        st.info(f"""
            This application uses machine learning to identify playing cards.
            Current model can classify {len(class_names)} different card types.
        """)

    # Image upload and prediction logic remains the same
    if option == "Upload Image 📁":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with st.spinner("🔍 Analyzing image..."):
                processed_img = preprocess_image(image)
                class_name, confidence, all_predictions = predict(model, processed_img, class_names)
                
            # Display results (previous implementation)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                    <div class="prediction-box">
                        <h3>Primary Prediction</h3>
                        <h2 style='color: #4CAF50;'>{class_name}</h2>
                        <h4>Confidence: {confidence:.1%}</h4>
                    </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown("### Top Predictions")
                predictions = list(zip(class_names, all_predictions))
                sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]
                
                for name, prob in sorted_predictions:
                    st.markdown(f"{name}: {prob:.1%}")

if __name__ == "__main__":
    main()
