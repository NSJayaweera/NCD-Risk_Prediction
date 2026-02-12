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
from typing import Tuple, Dict, Any

# ==============================================================================
# 0. INPUT MAPPING CONFIGURATION
# ==============================================================================
# Maps user-friendly UI options to the exact labels expected by the trained models.
# Values not in this map will be passed as-is.
VALUE_MAPPING = {
    'Hormonal Changes': {
        'Normal': 'Normal', 
        'Postmenopausal': 'Postmenopausal', 
        'Perimenopausal': 'Normal',     # Mapped to baseline
        'Low Testosterone': 'Normal'    # Mapped to baseline
    },
    'Race/Ethnicity': {
        'Caucasian': 'Caucasian', 
        'Asian': 'Asian', 
        'African American': 'African American', 
        'Hispanic': 'Caucasian',        # Mapped to majority group if not distinct in model
        'Other': 'Caucasian'            # Mapped to baseline
    },
    'Body Weight': {
        'Normal': 'Normal', 
        'Underweight': 'Underweight', 
        'Overweight': 'Normal'          # Model likely only flags Underweight as risk
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
        'None': 'Unknown',              # 'Unknown' appears to be the baseline/non-risk class in training data
        'Moderate': 'Moderate', 
        'Heavy': 'Unknown'              # Mapped to Unknown as model lacks 'Heavy' class
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
    """
    Get the absolute paths to model files.
    The models are expected to be in a 'models' directory alongside this script (or in the project root).
    """
    # Get the directory where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    
    return {
        'male_model': os.path.join(models_dir, 'osteoporosis_male_model.pkl'),
        'female_model': os.path.join(models_dir, 'osteoporosis_female_model.pkl'),
        'encoders': os.path.join(models_dir, 'label_encoders.pkl'),
        'scaler': os.path.join(models_dir, 'scaler.pkl')
    }

@st.cache_resource
def load_model_assets() -> Tuple[Any, Any, Dict, Any]:
    """
    Load the trained models, label encoders, and scaler.
    Uses Streamlit's cache_resource to load models only once.
    """
    paths = get_model_paths()
    
    try:
        male_model = joblib.load(paths['male_model'])
        female_model = joblib.load(paths['female_model'])
        label_encoders = joblib.load(paths['encoders'])
        scaler = joblib.load(paths['scaler'])
        
        return male_model, female_model, label_encoders, scaler
    
    except FileNotFoundError as e:
        # Construct helpful error message if files are missing
        st.error(f"Error: Model file not found at {e.filename}")
        st.info("Ensure that the 'models' folder exists in the same directory as this script and contains the .pkl files.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model assets: {str(e)}")
        st.stop()

def get_feature_names() -> list:
    """Get the list of expected feature names for the model."""
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
    """Apply custom dark theme CSS styling"""
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
    """Render input form and collect user data"""
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=55, help="Your current age in years")
        
        gender = st.selectbox("Gender", options=["Male", "Female"], 
                             help="Biological sex - women have higher risk due to lower bone density")
        
        hormonal_changes = st.selectbox("Hormonal Status",
                                       options=["Normal", "Postmenopausal", "Perimenopausal", "Low Testosterone"],
                                       help="Hormonal status affects bone density.")
        
        family_history = st.selectbox("Family History of Osteoporosis", options=["No", "Yes"],
                                     help="Parent or sibling with osteoporosis or hip fracture")
        
        race = st.selectbox("Race/Ethnicity", 
                           options=["Caucasian", "Asian", "African American", "Hispanic", "Other"],
                           help="Caucasian and Asian individuals typically have higher risk")
        
        body_weight = st.selectbox("Body Weight Category", options=["Normal", "Underweight", "Overweight"],
                                  help="Low body weight (BMI < 18.5) or small frame increases risk")
        
        calcium_intake = st.selectbox("Daily Calcium Intake", options=["Adequate", "Low", "High"],
                                     help="Adequate = 1000-1200mg/day. Low calcium intake increases risk.")

    with col2:
        vitamin_d = st.selectbox("Vitamin D Intake", options=["Sufficient", "Insufficient"],
                                help="Vitamin D is essential for calcium absorption. Deficiency increases risk.")
        
        physical_activity = st.selectbox("Physical Activity Level", options=["Moderate", "Active", "Sedentary"],
                                        help="Weight-bearing exercise strengthens bones. Sedentary lifestyle increases risk.")
        
        smoking = st.selectbox("Smoking Status", options=["No", "Yes"],
                              help="Smoking interferes with calcium absorption and reduces bone density")
        
        alcohol = st.selectbox("Alcohol Consumption", options=["None", "Moderate", "Heavy"],
                              help="Heavy drinking (>2 drinks/day) interferes with bone formation")
        
        medical_conditions = st.selectbox("Medical Conditions Affecting Bone Health",
                                         options=["None", "Rheumatoid Arthritis", "Thyroid Disorders", 
                                                 "Celiac Disease", "Kidney Disease", "Other"],
                                         help="Certain conditions increase risk")
        
        medications = st.selectbox("Medications Affecting Bone Density",
                                  options=["None", "Corticosteroids", "Anticonvulsants", 
                                          "Thyroid Medication", "Other"],
                                  help="Long-term use of certain medications can cause bone loss")
        
        prior_fractures = st.selectbox("History of Fractures (after age 50)", options=["No", "Yes"],
                                      help="Previous fractures from minor falls indicate weakened bones")
    
    return {
        'Age': age, 'Gender': gender, 'Hormonal Changes': hormonal_changes,
        'Family History': family_history, 'Race/Ethnicity': race, 'Body Weight': body_weight,
        'Calcium Intake': calcium_intake, 'Vitamin D Intake': vitamin_d,
        'Physical Activity': physical_activity, 'Smoking': smoking, 'Alcohol Consumption': alcohol,
        'Medical Conditions': medical_conditions, 'Medications': medications, 'Prior Fractures': prior_fractures
    }

def generate_recommendations(user_inputs, risk_score):
    """Generate personalized health recommendations based on risk factors"""
    recommendations = []
    
    if user_inputs['Calcium Intake'] == "Low":
        recommendations.append("• **Increase calcium intake** to 1000-1200mg daily (dairy, leafy greens, fortified foods)")
    
    if user_inputs['Vitamin D Intake'] == "Insufficient":
        recommendations.append("• **Boost Vitamin D** through sunlight exposure (10-15 min daily) or supplements (800-1000 IU)")
    
    if user_inputs['Physical Activity'] == "Sedentary":
        recommendations.append("• **Start weight-bearing exercise** like walking, jogging, or resistance training (30 min, 4-5x/week)")
    
    if user_inputs['Smoking'] == "Yes":
        recommendations.append("• **Quit smoking** - it significantly accelerates bone loss")
    
    if user_inputs['Alcohol Consumption'] == "Heavy":
        recommendations.append("• **Reduce alcohol consumption** to ≤1 drink/day for women, ≤2 for men")
    
    if user_inputs['Body Weight'] == "Underweight":
        recommendations.append("• **Maintain healthy body weight** - consult a nutritionist if BMI < 18.5")
    
    if risk_score > 0.5:
        recommendations.append("• **Schedule a bone density test (DXA scan)** with your healthcare provider")
        recommendations.append("• **Consult an endocrinologist or rheumatologist** for comprehensive evaluation")
    
    return recommendations

def make_prediction(user_inputs, male_model, female_model, label_encoders, scaler):
    """Make osteoporosis risk prediction using gender-specific models"""
    # 1. Create a dictionary for model input, applying mapping
    model_input = {}
    
    for col, value in user_inputs.items():
        # Apply mapping if exists for this column
        if col in VALUE_MAPPING:
            mapped_value = VALUE_MAPPING[col].get(value, value)
            model_input[col] = mapped_value
        else:
            model_input[col] = value
            
    # 2. Convert to DataFrame
    df_input = pd.DataFrame([model_input])
    
    # 3. Apply label encoding
    for col in label_encoders.keys():
        if col in df_input.columns:
            le = label_encoders[col]
            val = df_input[col].iloc[0]
            try:
                # Handle unseen labels carefully
                df_input[col] = le.transform([val])
            except ValueError:
                # Fallback to the most common class or 0 if truly unknown
                st.warning(f"Value '{val}' not found in trained model features for '{col}'. Using default.")
                df_input[col] = 0 # Default fallback
    
    # 4. Apply scaling
    try:
        df_scaled = scaler.transform(df_input)
    except ValueError as e:
        # Handle feature name mismatch if any
        st.error(f"Scaling error: {e}")
        st.stop()
    
    # 5. Select appropriate model based on gender input (using raw input string from user_inputs)
    gender_input = user_inputs['Gender']
    if gender_input == "Male":
        model = male_model
    else:
        model = female_model
        
    # 6. Make prediction
    prediction = model.predict(df_scaled)[0]
    
    # 7. Handle proba calculation for different model types
    if hasattr(model, 'predict_proba'):
        prediction_proba = model.predict_proba(df_scaled)[0]
        # Get probability of positive class (Index 1)
        risk_score = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
    else:
        # Fallback for models without predict_proba
        raw_pred = model.predict(df_scaled)
        if hasattr(raw_pred, 'shape') and len(raw_pred.shape) > 1 and raw_pred.shape[-1] == 1:
             risk_score = float(raw_pred[0][0])
        else:
             risk_score = float(prediction)

    return prediction, risk_score

# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================

def main():
    """Main application function"""
    # Apply custom styling
    apply_custom_css()
    
    # Load model assets
    male_model, female_model, label_encoders, scaler = load_model_assets()
    
    # Page title and description
    st.title("🦴 Osteoporosis Risk Assessment")
    st.write("Enter the following details to estimate your bone health risk.")
    
    # Get user inputs
    with st.container():
        user_inputs = get_user_inputs()
    
    # Prediction button
    if st.button("Calculate Osteoporosis Risk"):
        try:
            # Pass all models to prediction function
            prediction, risk_score = make_prediction(user_inputs, male_model, female_model, label_encoders, scaler)
            
            # Display results
            st.markdown(f"""
            <div class="result-container">
            <h3>Estimated Osteoporosis Risk</h3>
            <h1 style="color: #FF4B4B; font-size: 54px;">{risk_score:.1%}</h1>
            <p style="color: #E0E0E0; font-style: italic;">
            {"⚠️ High risk detected. Bone density screening (DXA scan) strongly recommended." if risk_score > 0.5 
             else "Lower relative risk based on provided factors. Continue healthy bone habits."}
            </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Personalized recommendations (Use original user inputs, not mapped ones)
            recommendations = generate_recommendations(user_inputs, risk_score)
            
            if recommendations:
                st.markdown("### 🦴 **Bone Health Recommendations:**")
                for rec in recommendations:
                    st.markdown(rec)
            else:
                st.success("✅ Continue your current healthy bone practices!")
        
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.error("Please ensure all inputs are valid.")
        
        # Medical disclaimer
        st.info("**Note:** This tool is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.")

if __name__ == "__main__":
    main()
