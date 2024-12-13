import streamlit as st
import tensorflow as tf
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
        .prediction-box {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border: 1px solid #dee2e6;
            margin: 1rem 0;
        }
        .prediction-list {
            list-style-type: none;
            padding: 0;
        }
        .prediction-list li {
            padding: 0.5rem 0;
            border-bottom: 1px solid #eee;
            font-size: 1.1rem;
        }
        .prediction-list li:last-child {
            border-bottom: none;
        }
        .confidence {
            float: right;
            color: #4CAF50;
            font-weight: bold;
        }
        .stButton>button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
        }
        .css-1v0mbdj.etr89bj1 {
            margin-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Model loading with download functionality
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)
        model_path = 'models/best_model.keras'
        
        # Check if model exists, if not, download it
        if not os.path.exists(model_path):
            with st.spinner('🔄 Downloading model... Please wait.'):
                # Replace with the actual Google Drive link to your model
                model_url = "https://drive.google.com/drive/folders/1fVN_B96_B2KrO1_pah39-ablO4lpXblp?usp=drive_link"
                gdown.download(model_url, output=model_path, quiet=False)
        
        # Load and return the model
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return None

# Rest of the code remains the same as in the previous version...
# (You can copy the entire previous implementation after this function)

# Dynamically get class names from the dataset
def get_class_names(model):
    try:
        # Try to get class names from the model's output layer
        return list(model.get_config()['layers'][-1]['config']['units'])
    except:
        # Fallback to manual list if automatic extraction fails
        # Replace with actual class names from your dataset
        return [
            'Ace of Clubs', 'Ace of Diamonds', 'Ace of Hearts', 'Ace of Spades', 
            # Add all 53 card names here
        ]


def preprocess_image(img):
    img = img.convert("RGB")
    img = np.array(img)
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def predict(model, img):
    prediction = model.predict(img, verbose=0)
    predicted_class_idx = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class_idx]
    return class_names[predicted_class_idx], confidence, prediction[0]

def display_results(class_name, confidence, all_predictions):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="prediction-box">
                <h3>Primary Prediction</h3>
                <h2 style='color: #4CAF50;'>{}</h2>
                <h4>Confidence: {:.1%}</h4>
            </div>
        """.format(class_name, confidence), unsafe_allow_html=True)
        
        st.markdown("### Card Information")
        st.info(f"Classification result for {class_name}")
    
    with col2:
        st.markdown("### Top Predictions")
        predictions_with_names = list(zip(class_names, all_predictions))
        # Filter predictions with confidence > 0 and sort by confidence
        valid_predictions = [(name, prob) for name, prob in predictions_with_names if prob > 0]
        sorted_predictions = sorted(valid_predictions, key=lambda x: x[1], reverse=True)[:5]
        
        # Display predictions as a clean list
        st.markdown('<ul class="prediction-list">', unsafe_allow_html=True)
        for name, prob in sorted_predictions:
            st.markdown(
                f'<li>{name}<span class="confidence">{prob:.1%}</span></li>',
                unsafe_allow_html=True
            )
        st.markdown('</ul>', unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
        <h1 style='text-align: center; color: #2E7D32;'>🃏 Card Classification</h1>
        <p style='text-align: center; font-size: 1.2em;'>Upload an image or use the live feed to identify playing cards</p>
        <hr>
    """, unsafe_allow_html=True)

    model = load_model()

    if model is None:
        st.error("❌ Failed to load model. Please check the model file.")
        return

    # Dynamically get class names
    global class_names
    class_names = get_class_names(model)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🃏 Input Options")
        option = st.radio("Select Input Method:", ["Upload Image 📁", "Live Feed 📸"])
        
        st.markdown("### ℹ️ About")
        st.info("""
            This application uses machine learning to identify playing cards.
            It can recognize cards from a standard deck with high accuracy.
        """)

    if option == "Upload Image 📁":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with st.spinner("🔍 Analyzing image..."):
                processed_img = preprocess_image(image)
                class_name, confidence, all_predictions = predict(model, processed_img)
                
            display_results(class_name, confidence, all_predictions)

    else:  # Live Feed option
        st.markdown("### 📸 Live Camera Feed")
        camera_input = st.camera_input("Take a picture")
        
        if camera_input:
            image = Image.open(camera_input)
            
            with st.spinner("🔍 Analyzing image..."):
                processed_img = preprocess_image(image)
                class_name, confidence, all_predictions = predict(model, processed_img)
                
            display_results(class_name, confidence, all_predictions)

if __name__ == "__main__":
    main()
