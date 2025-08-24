import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Set page configuration
st.set_page_config(page_title="Car Damage Assessment", layout="centered")

# Title and description
st.title("🚗 Car Damage Assessment")
st.markdown("""
Upload an image of your car, and our AI model will determine whether it's **Damaged** or **Undamaged**.
If damaged, it will also assess the **Severity** of the damage.
""")

# Load the trained model
@st.cache_resource
def load_trained_model():
    model_path = 'car_damage_mobilenetv2.keras'  # Ensure this file is in the same directory
    return load_model(model_path)

model = load_trained_model()
class_labels = ['Damaged', 'Undamaged']

# File uploader
uploaded_file = st.file_uploader("Choose a car image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Car Image', use_column_width=True)

    # Preprocess the image
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_index]
    confidence = prediction[0][predicted_index]

    # Display prediction
    st.subheader("Prediction:")
    st.write(f"**{predicted_class}** with a confidence of **{confidence * 100:.2f}%**.")

    # Assess severity if damaged
    if predicted_class == 'Damaged':
        if confidence >= 0.75:
            severity = "Major Damage - Consider filing an insurance claim."
        else:
            severity = "Minor Damage - May not require an insurance claim."
        st.subheader("Damage Severity Assessment:")
        st.write(severity)
