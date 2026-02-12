import pandas as pd
import joblib
import os
import sys

# Add current directory to path so we can import from Osteoporosis.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from Osteoporosis import load_model_assets, make_prediction, VALUE_MAPPING
    print("SUCCESS: Successfully imported modules from Osteoporosis.py")
except ImportError as e:
    print(f"ERROR: Error importing from Osteoporosis.py: {e}")
    sys.exit(1)

def run_tests():
    print("\nLOADING MODELS...")
    try:
        male_model, female_model, label_encoders, scaler = load_model_assets()
        print("SUCCESS: Models loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load models: {e}")
        return

    # Define Test Cases
    test_cases = [
        {
            "name": "High Risk Female (Postmenopausal, Smoker, Low Calcium)",
            "inputs": {
                'Age': 65, 
                'Gender': 'Female', 
                'Hormonal Changes': 'Postmenopausal',
                'Family History': 'Yes', 
                'Race/Ethnicity': 'Caucasian', 
                'Body Weight': 'Underweight',
                'Calcium Intake': 'Low', 
                'Vitamin D Intake': 'Insufficient',
                'Physical Activity': 'Sedentary', 
                'Smoking': 'Yes', 
                'Alcohol Consumption': 'Heavy', 
                'Medical Conditions': 'Rheumatoid Arthritis', 
                'Medications': 'Corticosteroids', 
                'Prior Fractures': 'Yes'
            }
        },
        {
            "name": "Low Risk Male (Active, Young, Healthy)",
            "inputs": {
                'Age': 30, 
                'Gender': 'Male', 
                'Hormonal Changes': 'Normal',
                'Family History': 'No', 
                'Race/Ethnicity': 'African American', 
                'Body Weight': 'Normal',
                'Calcium Intake': 'Adequate', 
                'Vitamin D Intake': 'Sufficient',
                'Physical Activity': 'Active', 
                'Smoking': 'No', 
                'Alcohol Consumption': 'None', 
                'Medical Conditions': 'None', 
                'Medications': 'None', 
                'Prior Fractures': 'No'
            }
        },
         {
            "name": "Moderate Risk Female (Elderly but Healthy Habits)",
            "inputs": {
                'Age': 75, 
                'Gender': 'Female', 
                'Hormonal Changes': 'Postmenopausal',
                'Family History': 'No', 
                'Race/Ethnicity': 'Asian', 
                'Body Weight': 'Normal',
                'Calcium Intake': 'Adequate', 
                'Vitamin D Intake': 'Sufficient',
                'Physical Activity': 'Moderate', 
                'Smoking': 'No', 
                'Alcohol Consumption': 'Moderate',
                'Medical Conditions': 'None', 
                'Medications': 'None', 
                'Prior Fractures': 'No'
            }
        },
        {
            "name": "High Risk Male (Smoker, Heavy Drinker, Medical Issues)",
            "inputs": {
                'Age': 60, 
                'Gender': 'Male', 
                'Hormonal Changes': 'Normal', 
                'Family History': 'Yes', 
                'Race/Ethnicity': 'Caucasian', 
                'Body Weight': 'Underweight',
                'Calcium Intake': 'Low', 
                'Vitamin D Intake': 'Insufficient',
                'Physical Activity': 'Sedentary', 
                'Smoking': 'Yes', 
                'Alcohol Consumption': 'Heavy',
                'Medical Conditions': 'Thyroid Disorders', 
                'Medications': 'Corticosteroids', 
                'Prior Fractures': 'Yes'
            }
        }
    ]

    print("\nSTARTING PREDICTION TESTS")
    print("=" * 60)

    for case in test_cases:
        print(f"\nTEST CASE: {case['name']}")
        inputs = case['inputs']
        
        try:
            # Make Prediction
            prediction, risk_score = make_prediction(inputs, male_model, female_model, label_encoders, scaler)
            
            print(f"   Gender: {inputs['Gender']}")
            print(f"   Risk Score: {risk_score:.4f} ({risk_score*100:.1f}%)")
            print(f"   Prediction: {'Osteoporosis' if prediction == 1 else 'Normal'}")
            
            if risk_score > 0.5:
                print("   Result: HIGH RISK")
            else:
                print("   Result: LOW RISK")
                
        except Exception as e:
            print(f"   ERROR: {e}")

    print("\n" + "=" * 60)
    print("TESTS COMPLETED.")

if __name__ == "__main__":
    run_tests()
