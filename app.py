import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction Portal",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling for an Attractive Layout
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained SVM model safely
@st.cache_resource
def load_model():
    with open('svm.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# App Header
st.markdown("<h1 style='text-align: center; color: #333333;'>🩺 Health Risk Prediction Portal</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666666;'>Fill in your health metrics below to get an instant evaluation from the SVM model.</p>", unsafe_allow_html=True)
st.markdown("---")

# Input Form Layout
with st.form("prediction_form"):
    st.markdown("### Patient Information & Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        city = st.text_input("City", value="New York")
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.5)
        family_history_diabetes = st.selectbox("Family History of Diabetes", ["No", "Yes"])
        physical_activity_level = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
        diet_type = st.selectbox("Diet Type", ["Standard", "Vegetarian", "Vegan", "Keto", "Other"])
        smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        alcohol_consumption = st.selectbox("Alcohol Consumption", ["None", "Low", "Moderate", "High"])
        
    with col2:
        hours_sleep_per_night = st.slider("Hours of Sleep per Night", min_value=1.0, max_value=15.0, value=7.0)
        stress_level = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=5)
        fasting_blood_sugar = st.number_input("Fasting Blood Sugar", min_value=50.0, max_value=300.0, value=95.0)
        hba1c_level = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)
        blood_pressure_systolic = st.number_input("Blood Pressure Systolic", min_value=70, max_value=200, value=120)
        blood_pressure_diastolic = st.number_input("Blood Pressure Diastolic", min_value=40, max_value=130, value=80)
        waist_circumference_cm = st.number_input("Waist Circumference (cm)", min_value=40.0, max_value=200.0, value=85.0)
        income_bracket = st.selectbox("Income Bracket", ["Low", "Middle", "High"])

    st.markdown("")
    submit_button = st.form_submit_button(label="Run Prediction")

# Process input data and run inference
if submit_button:
    # Note: Ensure the encoding of categorical columns matches what your model expects during training (e.g., One-Hot Encoding or Label Encoding). 
    # Below is a sample dataframe creation matching your model's feature signature.
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [1 if gender == "Female" else (0 if gender == "Male" else 2)],
        'city': [city], # If city was label encoded or dropped, handle accordingly
        'bmi': [bmi],
        'family_history_diabetes': [1 if family_history_diabetes == "Yes" else 0],
        'physical_activity_level': [0 if physical_activity_level == "Low" else (1 if physical_activity_level == "Moderate" else 2)],
        'diet_type': [diet_type],
        'smoking_status': [smoking_status],
        'alcohol_consumption': [alcohol_consumption],
        'hours_sleep_per_night': [hours_sleep_per_night],
        'stress_level': [stress_level],
        'fasting_blood_sugar': [fasting_blood_sugar],
        'hba1c_level': [hba1c_level],
        'blood_pressure_systolic': [blood_pressure_systolic],
        'blood_pressure_diastolic': [blood_pressure_diastolic],
        'waist_circumference_cm': [waist_circumference_cm],
        'income_bracket': [income_bracket]
    })
    
    try:
        # If your pipeline includes preprocessors, pass raw input_data directly. 
        # Otherwise, ensure data types align with model expectations.
        prediction = model.predict(input_data)
        
        st.markdown("---")
        st.markdown("### 📊 Prediction Result")
        if prediction[0] == 1:
            st.error(f"**Result:** High Risk Detected (Class: {prediction[0]}). Please consult a medical professional.")
        else:
            st.success(f"**Result:** Low Risk (Class: {prediction[0]}). Keep up the healthy lifestyle!")
            
    except Exception as e:
        st.warning("Note: If your model expects numerical categorical features or specific one-hot encoding columns matching training time, make sure data transformations align.")
        st.info(f"Technical details: {e}")
