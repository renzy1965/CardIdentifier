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

# Custom CSS
st.markdown("""
    <style>
        .main > div {
            padding: 2rem;
            border-radius: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

def load_model_multiple_sources():
    # Potential model file locations
    model_paths = [
        # Local C drive paths (adjust as needed)
        r'C:\best_model.keras',
        r'C:\Users\YourUsername\best_model.keras',
        
        # Current directory
        'best_model.keras',
        os.path.join(os.getcwd(), 'best_model.keras'),
        
        # Google Drive download option (replace with your actual Google Drive link)
        'models/best_model.keras'
    ]

    # Additional Google Drive download option
    gdrive_model_url = st.text_input(
        "Optional: Paste Google Drive model sharing link",
        placeholder="https://drive.google.com/drive/folders/1uSiQPjl_vrpzPXFkB8XcfkoBpwhhrktN?usp=drive_link",
    )

    # If a Google Drive link is provided, add it to potential paths
    if gdrive_model_url and gdrive_model_url.startswith('https://'):
        try:
            # Ensure models directory exists
            os.makedirs('models', exist_ok=True)
            output_path = 'models/best_model.keras'
            
            # Download from Google Drive
            with st.spinner('🔄 Downloading model from Google Drive...'):
                gdown.download(gdrive_model_url, output=output_path, quiet=False)
            
            # Add the downloaded path to model paths
            model_paths.append(output_path)
        except Exception as e:
            st.error(f"Error downloading from Google Drive: {e}")

    # Try loading from different paths
    for path in model_paths:
        try:
            # Expand user directory if needed
            full_path = os.path.expanduser(path)
            
            # Check if file exists
            if not os.path.exists(full_path):
                continue
            
            # Attempt to load the model
            st.info(f"Attempting to load model from: {full_path}")
            model = load_model(full_path)
            
            st.success(f"Model successfully loaded from {full_path}")
            return model
        
        except Exception as e:
            st.warning(f"Could not load model from {path}: {str(e)}")
    
    # If all attempts fail
    st.error("Could not load the model from any location.")
    return None

def get_class_names(model):
    try:
        # Get output layer units (number of classes)
        num_classes = model.layers[-1].output_shape[-1]
        
        # Generate default class names
        return [f'Card_{i+1}' for i in range(num_classes)]
    except Exception as e:
        st.error(f"Error extracting class names: {str(e)}")
        return []

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
    model = load_model_multiple_sources()

    if model is None:
        st.error("❌ Failed to load model. Please check the following:")
        st.info("""
        Troubleshooting steps:
        1. Ensure the model file exists
        2. Check file path and permissions
        3. Verify model file is not corrupted
        4. Use the Google Drive link option if needed
        """)
        return

    # Get class names
    class_names = get_class_names(model)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🃏 Input Options")
        option = st.radio("Select Input Method:", ["Upload Image 📁", "Live Feed 📸"])
        
        st.markdown("### ℹ️ About")
        st.info(f"""
            This application uses machine learning to identify playing cards.
            Current model can classify {len(class_names)} different card types.
        """)

    # Image upload and prediction
    if option == "Upload Image 📁":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with st.spinner("🔍 Analyzing image..."):
                processed_img = preprocess_image(image)
                class_name, confidence, all_predictions = predict(model, processed_img, class_names)
                
            # Results display
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                    <div style='background-color: #f0f0f0; padding: 20px; border-radius: 10px;'>
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

    else:  # Live Feed option
        st.markdown("### 📸 Live Camera Feed")
        camera_input = st.camera_input("Take a picture")
        
        if camera_input:
            image = Image.open(camera_input)
            
            with st.spinner("🔍 Analyzing image..."):
                processed_img = preprocess_image(image)
                class_name, confidence, all_predictions = predict(model, processed_img, class_names)
                
            # Results display (same as upload method)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                    <div style='background-color: #f0f0f0; padding: 20px; border-radius: 10px;'>
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
