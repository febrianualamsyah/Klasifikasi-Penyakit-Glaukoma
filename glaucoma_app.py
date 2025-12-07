import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io

# --- Function Definitions (re-included for self-contained app) ---

def preprocess_image(pil_image):
    # Ensure the image is in RGB format if it's not (e.g., RGBA or grayscale)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    img = pil_image.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
    return img_array

def predict_image(model, preprocessed_image):
    predictions = model.predict(preprocessed_image)
    # Assuming 2 classes: 0 for Normal, 1 for Glaucoma
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    class_labels = ['Normal', 'Glaucoma']
    predicted_label = class_labels[predicted_class_index]
    
    # Get the confidence for the predicted class
    confidence = predictions[0][predicted_class_index]

    return predicted_label, confidence

# --- Streamlit Application ---

st.title('Glaucoma Detection using VGG19 Model')
st.write('Upload an eye image to classify it as Glaucoma or Normal.')

# Cache the model loading for efficiency
@st.cache_resource
def load_vgg19_model():
    model = load_model('VGG19_FOLD_1.h5')
    return model

model = load_vgg19_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    try:
        # Convert the uploaded file to a PIL Image object
        pil_image = Image.open(io.BytesIO(uploaded_file.read()))

        # Preprocess the image
        preprocessed_img = preprocess_image(pil_image)

        # Make prediction
        predicted_label, confidence = predict_image(model, preprocessed_img)

        # Display results
        st.success(f"Prediction: {predicted_label}")
        st.write(f"Confidence: {confidence:.2f}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.write("Please ensure the uploaded file is a valid image.")

st.markdown("--- Streamlit App End --- ")

# Instructions on how to run the app
st.subheader("How to run this application:")
st.markdown("1. Save this code as a Python file (e.g., `glaucoma_app.py`).")
st.markdown("2. Make sure you have the `VGG19_FOLD_1.h5` model file in the same directory.")
st.markdown("3. Open your terminal or command prompt.")
st.markdown("4. Navigate to the directory where you saved the file.")
st.markdown("5. Run the application using the command: `streamlit run glaucoma_app.py`")
st.markdown("6. Streamlit will provide a local URL (e.g., `http://localhost:8501`) and possibly a network URL if you're on a remote server/Colab.")
st.markdown("In Google Colab, a public URL will appear above this output, which you can click to access the app.")

