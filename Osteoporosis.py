"""
Osteoporosis Risk Assessment Application
A Streamlit-based web application for predicting osteoporosis risk using machine learning.

This script consolidates all necessary components (UI, Logic, Model Loading) into a single file 
for simplified deployment and execution.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import glob
from typing import Tuple, Dict, Any

# ==============================================================================
# 0. INPUT MAPPING CONFIGURATION
# ==============================================================================
VALUE_MAPPING = {
    'Hormonal Changes': {
        'Normal': 'Normal',
        'Postmenopausal': 'Postmenopausal',
        'Perimenopausal': 'Normal',
        'Low Testosterone': 'Normal'
    },
    'Race/Ethnicity': {
        'Caucasian': 'Caucasian',
        'Asian': 'Asian',
        'African American': 'African American',
        'Hispanic': 'Caucasian',
        'Other': 'Caucasian'
    },
    'Body Weight': {
        'Normal': 'Normal',
        'Underweight': 'Underweight',
        'Overweight': 'Normal'
    },
    'Calcium Intake': {
        'Adequate': 'Adequate',
        'Low': 'Low',
        'High': 'Adequate'
    },
    'Physical Activity': {
        'Sedentary': 'Sedentary',
        'Active': 'Active',
        'Moderate': 'Active'
    },
    'Alcohol Consumption': {
        'None': 'Unknown',
        'Moderate': 'Moderate',
        'Heavy': 'Moderate'
    },
    'Medical Conditions': {
        'None': 'Unknown',
        'Rheumatoid Arthritis': 'Rheumatoid Arthritis',
        'Thyroid Disorders': 'Hyperthyroidism',
        'Celiac Disease': 'Unknown',
        'Kidney Disease': 'Unknown',
        'Other': 'Unknown'
    },
    'Medications': {
        'None': 'Unknown',
        'Corticosteroids': 'Corticosteroids',
        'Anticonvulsants': 'Unknown',
        'Thyroid Medication': 'Unknown',
        'Other': 'Unknown'
    }
}

# ==============================================================================
# 1. MODEL LOADING & UTILS
# ==============================================================================

def get_model_paths() -> Dict[str, str]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')

    male_model_path = os.path.join(models_dir, 'osteoporosis_male_random_forest_model.pkl')
    female_model_path = os.path.join(models_dir, 'osteoporosis_female_random_forest_model.pkl')

    return {
        'male_model': male_model_path,
        'female_model': female_model_path,
        'encoders': os.path.join(models_dir, 'label_encoders.pkl'),
        'scaler': os.path.join(models_dir, 'scaler.pkl')
    }

@st.cache_resource
def load_model_assets() -> Tuple[Any, Any, Dict, Any]:
    paths = get_model_paths()
    try:
        male_model = joblib.load(paths['male_model'])
        female_model = joblib.load(paths['female_model'])
        label_encoders = joblib.load(paths['encoders'])
        scaler = joblib.load(paths['scaler'])
        return male_model, female_model, label_encoders, scaler
    except Exception as e:
        st.error(f"Error loading model assets: {str(e)}")
        st.stop()

def get_feature_names() -> list:
    return [
        'Age', 'Gender', 'Hormonal Changes', 'Family History', 'Race/Ethnicity',
        'Body Weight', 'Calcium Intake', 'Vitamin D Intake', 'Physical Activity',
        'Smoking', 'Alcohol Consumption', 'Medical Conditions', 'Medications',
        'Prior Fractures'
    ]

# ==============================================================================
# 2. UI & APPLICATION LOGIC
# ==============================================================================

def apply_custom_css():
    st.markdown("""
        <style>
        .stApp { background-color: #121212; color: #FFFFFF; }
        h1, h2, h3 { color: #FF4B4B !important; }
        label { color: #E0E0E0 !important; }
        div.stButton > button:first-child {
            background-color: #FF4B4B; color: white; border: none; width: 100%; font-weight: bold;
        }
        .result-container {
            padding: 20px; border-radius: 10px; background-color: #1E1E1E; 
            border: 1px solid #FF4B4B; text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)

def get_user_inputs():
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=55)
        gender = st.selectbox("Gender", options=["Male", "Female"])
        hormonal_changes = st.selectbox("Hormonal Status", options=["Normal", "Postmenopausal", "Perimenopausal", "Low Testosterone"])
        family_history = st.selectbox("Family History of Osteoporosis", options=["No", "Yes"])
        race = st.selectbox("Race/Ethnicity", options=["Caucasian", "Asian", "African American", "Hispanic", "Other"])
        body_weight = st.selectbox("Body Weight Category", options=["Normal", "Underweight", "Overweight"])
        calcium_intake = st.selectbox("Daily Calcium Intake", options=["Adequate", "Low", "High"])

    with col2:
        vitamin_d = st.selectbox("Vitamin D Intake", options=["Sufficient", "Insufficient"])
        physical_activity = st.selectbox("Physical Activity Level", options=["Moderate", "Active", "Sedentary"])
        smoking = st.selectbox("Smoking Status", options=["No", "Yes"])
        alcohol = st.selectbox("Alcohol Consumption", options=["None", "Moderate", "Heavy"])
        medical_conditions = st.selectbox("Medical Conditions Affecting Bone Health", options=["None", "Rheumatoid Arthritis", "Thyroid Disorders", "Celiac Disease", "Kidney Disease", "Other"])
        medications = st.selectbox("Medications Affecting Bone Density", options=["None", "Corticosteroids", "Anticonvulsants", "Thyroid Medication", "Other"])
        prior_fractures = st.selectbox("History of Fractures (after age 50)", options=["No", "Yes"])

    return {
        'Age': age, 'Gender': gender, 'Hormonal Changes': hormonal_changes,
        'Family History': family_history, 'Race/Ethnicity': race, 'Body Weight': body_weight,
        'Calcium Intake': calcium_intake, 'Vitamin D Intake': vitamin_d,
        'Physical Activity': physical_activity, 'Smoking': smoking, 'Alcohol Consumption': alcohol,
        'Medical Conditions': medical_conditions, 'Medications': medications, 'Prior Fractures': prior_fractures
    }

def generate_recommendations(user_inputs, risk_score, prediction):
    recommendations = []
    if prediction == 1:
        recommendations.append("### 🚨 **Immediate Actions Required:**")
        recommendations.append("• **Schedule a bone density test (DXA scan)**")
        recommendations.append("• **Consult an endocrinologist or rheumatologist**")
    else:
        recommendations.append("### ✅ **Continue Good Bone Health Practices:**")
    return recommendations

def make_prediction(user_inputs, male_model, female_model, label_encoders, scaler):
    model_input = {}
    for col, value in user_inputs.items():
        if col in VALUE_MAPPING:
            model_input[col] = VALUE_MAPPING[col].get(value, value)
        else:
            model_input[col] = value

    df_input = pd.DataFrame([model_input])
    for col in label_encoders.keys():
        if col in df_input.columns:
            le = label_encoders[col]
            val = df_input[col].iloc[0]
            try:
                df_input[col] = le.transform([val])
            except ValueError:
                df_input[col] = 0

    expected_features = get_feature_names()
    df_input = df_input[expected_features]
    scaled_array = scaler.transform(df_input)
    df_scaled = pd.DataFrame(scaled_array, columns=expected_features)

    model = male_model if user_inputs['Gender'] == "Male" else female_model
    prediction = model.predict(df_scaled)[0]

    if hasattr(model, 'predict_proba'):
        prediction_proba = model.predict_proba(df_scaled)[0]
        risk_score = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
    else:
        risk_score = float(prediction)

    return prediction, risk_score

# ==============================================================================
# 3. WRAPPED EXECUTION METHOD
# ==============================================================================

def run_osteoporosis_analysis():
    """Method to be called from main.py"""
    apply_custom_css()
    male_model, female_model, label_encoders, scaler = load_model_assets()

    st.title("🦴 Osteoporosis Risk Assessment")
    st.write("Enter the following details to estimate your bone health risk.")

    with st.container():
        user_inputs = get_user_inputs()

    if st.button("Calculate Osteoporosis Risk"):
        try:
            prediction, risk_score = make_prediction(user_inputs, male_model, female_model, label_encoders, scaler)
            prediction_label = "Osteoporosis" if prediction == 1 else "No Osteoporosis"
            prediction_color = "#FF4B4B" if prediction == 1 else "#00D9A3"

            st.markdown(f"""
            <div class="result-container">
            <h3>Assessment Results</h3>
            <p style="font-size: 18px; color: #E0E0E0;">Risk Score: <strong style="color: {prediction_color};">{risk_score*100:.1f}%</strong></p>
            <p style="font-size: 18px; color: #E0E0E0;">Prediction: <strong style="color: {prediction_color};">{prediction_label}</strong></p>
            </div>
            """, unsafe_allow_html=True)

            for rec in generate_recommendations(user_inputs, risk_score, prediction):
                st.markdown(rec)
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

    st.info("**Note:** This tool is for informational purposes only.")

if __name__ == "__main__":
    run_osteoporosis_analysis()
