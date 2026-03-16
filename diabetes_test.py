import pytest
import sys
import os
import joblib
import pandas as pd
import numpy as np
import warnings

# Suppress the InconsistentVersionWarning from scikit-learn
warnings.filterwarnings("ignore", category=UserWarning)
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

# Add the parent directory to sys.path so we can import the models
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Load the actual pre-trained model and scaler
# Using the SVM best model logic from the previous application
MODELS_DIR = os.path.join(parent_dir, "Models")

@pytest.fixture(scope="module")
def ml_pipeline():
    """Loads the real scaler and best model to perform proper tests"""
    model_path = os.path.join(MODELS_DIR, "best_diabetes_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler_pickle.pkl")
    
    assert os.path.exists(model_path), "Model file not found"
    assert os.path.exists(scaler_path), "Scaler file not found"
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler

def predict_patient(model, scaler, patient_data):
    """Helper method to scale and predict identically to diabetes.py"""
    feature_order = ['blood_glucose', 'physical_activity', 'diet', 'medication_adherence', 
                     'stress_level', 'sleep_hours', 'hydration_level', 'bmi']
    
    df = pd.DataFrame([patient_data])[feature_order]
    scaled_features = scaler.transform(df)
    
    # 1 indicates High Risk, 0 indicates Low Risk
    prediction = int(model.predict(scaled_features)[0])
    return prediction

# ==========================================================
# 5 TEST CASES SOURCED FROM REAL DATA (balanced_diabetes_data.csv)
# ==========================================================

# Patient 275 - High risk in CSV
def test_real_patient_275_high_risk(ml_pipeline):
    model, scaler = ml_pipeline
    patient = {
        'blood_glucose': 96.51, 'physical_activity': 11.67, 'diet': 0, 
        'medication_adherence': 0, 'stress_level': 2, 'sleep_hours': 10.36, 
        'hydration_level': 0, 'bmi': 16.7
    }
    assert predict_patient(model, scaler, patient) == 1

# Patient 854 - High risk in CSV
def test_real_patient_854_high_risk(ml_pipeline):
    model, scaler = ml_pipeline
    patient = {
        'blood_glucose': 148.82, 'physical_activity': 7.85, 'diet': 1, 
        'medication_adherence': 0, 'stress_level': 2, 'sleep_hours': 5.02, 
        'hydration_level': 1, 'bmi': 27.4
    }
    assert predict_patient(model, scaler, patient) == 1

# Patient 931 - Low risk in CSV
def test_real_patient_931_low_risk(ml_pipeline):
    model, scaler = ml_pipeline
    patient = {
        'blood_glucose': 174.76, 'physical_activity': 43.43, 'diet': 1, 
        'medication_adherence': 1, 'stress_level': 2, 'sleep_hours': 5.59, 
        'hydration_level': 1, 'bmi': 24.8
    }
    assert predict_patient(model, scaler, patient) == 0

# Patient 546 - Low risk in CSV
def test_real_patient_546_low_risk(ml_pipeline):
    model, scaler = ml_pipeline
    patient = {
        'blood_glucose': 157.31, 'physical_activity': 36.26, 'diet': 0, 
        'medication_adherence': 1, 'stress_level': 1, 'sleep_hours': 4.0, 
        'hydration_level': 1, 'bmi': 19.0
    }
    assert predict_patient(model, scaler, patient) == 0

# Patient 399 - Low risk in CSV
def test_real_patient_399_low_risk(ml_pipeline):
    model, scaler = ml_pipeline
    patient = {
        'blood_glucose': 148.29, 'physical_activity': 75.42, 'diet': 1, 
        'medication_adherence': 1, 'stress_level': 0, 'sleep_hours': 10.81, 
        'hydration_level': 1, 'bmi': 20.6
    }
    assert predict_patient(model, scaler, patient) == 0

# ==========================================================
# 5 TEST CASES FOR HYPOTHETICAL/NEW DATA SCENARIOS
# ==========================================================

# Hypothetical Case 1: Ideal Healthy Individual (Low Risk)
def test_hypothetical_ideal_patient_low_risk(ml_pipeline):
    model, scaler = ml_pipeline
    patient = {
        'blood_glucose': 90.0, 'physical_activity': 150.0, 'diet': 1, 
        'medication_adherence': 1, 'stress_level': 0, 'sleep_hours': 8.0, 
        'hydration_level': 1, 'bmi': 22.0
    }
    assert predict_patient(model, scaler, patient) == 0

# Hypothetical Case 2: Extreme Unhealthy Lifestyle (High Risk)
def test_hypothetical_sedentary_unhealthy_high_risk(ml_pipeline):
    model, scaler = ml_pipeline
    patient = {
        'blood_glucose': 200.0, 'physical_activity': 0.0, 'diet': 0, 
        'medication_adherence': 0, 'stress_level': 2, 'sleep_hours': 3.0, 
        'hydration_level': 0, 'bmi': 35.0
    }
    assert predict_patient(model, scaler, patient) == 1

# Hypothetical Case 3: Borderline Patient (Moderate habits, low BMI)
def test_hypothetical_moderate_habits_low_bmi(ml_pipeline):
    model, scaler = ml_pipeline
    patient = {
        'blood_glucose': 110.0, 'physical_activity': 30.0, 'diet': 1, 
        'medication_adherence': 1, 'stress_level': 1, 'sleep_hours': 6.0, 
        'hydration_level': 1, 'bmi': 18.5
    }
    # Prediction logic can vary, testing if it resolves 0 (Low Risk) given diet/medication
    assert predict_patient(model, scaler, patient) == 0

# Hypothetical Case 4: High Stress, Poor Sleep, Good Diet (Mixed)
def test_hypothetical_stressed_sleepless_low_risk(ml_pipeline):
    model, scaler = ml_pipeline
    patient = {
        'blood_glucose': 120.0, 'physical_activity': 100.0, 'diet': 1, 
        'medication_adherence': 1, 'stress_level': 2, 'sleep_hours': 4.0, 
        'hydration_level': 1, 'bmi': 24.0
    }
    assert predict_patient(model, scaler, patient) == 0

# Hypothetical Case 5: Aging Patient with High BMI and Glucose
def test_hypothetical_high_glucose_high_bmi_high_risk(ml_pipeline):
    model, scaler = ml_pipeline
    patient = {
        'blood_glucose': 250.0, 'physical_activity': 10.0, 'diet': 0, 
        'medication_adherence': 1, 'stress_level': 1, 'sleep_hours': 7.0, 
        'hydration_level': 1, 'bmi': 42.0
    }
    assert predict_patient(model, scaler, patient) == 1

if __name__ == "__main__":
    pytest.main(["-v", __file__])
