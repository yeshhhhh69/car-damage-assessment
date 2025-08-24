# Car Damage Assessment using Deep Learning

This project is a **Car Damage Assessment System** that uses a deep learning model to detect and classify **damages on cars** from images.  
It is built with **TensorFlow/Keras** for training and **Streamlit** for deployment as an interactive web app.  

---

## Features
- Upload a car image and detect **damaged** vs **undamaged** areas.  
- Trained deep learning model (`.keras` file) for object detection.  
- Streamlit app for **real-time prediction** and visualization.  
- Transfer learning + fine-tuning on pre-trained models for better accuracy.  
- 🛠Easy-to-run web interface, no ML expertise required.  

---

## Tech Stack
- **Python**  
- **TensorFlow / Keras** (model training)  
- **OpenCV** (image preprocessing)  
- **Streamlit** (web app)  
- **Matplotlib / Seaborn** (visualization)  

---

## Project Structure
├── app.py # Streamlit app
├── model/
│ ├── car_damage_model.keras # Trained model
├── requirements.txt # Dependencies
├── README.md # Project Documentation
└── dataset/ # (optional) dataset structure used

---

## ⚙️ Installation & Usage
1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/car-damage-assessment.git
   cd car-damage-assessment
2. Create virtual environment (optional but recommended)
   python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

3. install dependencies
  pip install -r requirements.txt

4. Run the Streamlit app
  streamlit run app.py

5. Upload a car image and see the damage detection results!
