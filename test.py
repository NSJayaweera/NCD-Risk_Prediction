"""
Test script for Osteoporosis Risk Prediction
Tests both manually created cases AND real samples from osteoporosis_data.csv
"""
import sys
sys.path.insert(0, '.')

from Osteoporosis import load_model_assets, make_prediction
import pandas as pd
import random

print("=" * 70)
print("OSTEOPOROSIS RISK PREDICTION - COMPREHENSIVE TEST")
print("=" * 70)

# Load models
print("\nLOADING MODELS...")
male_model, female_model, label_encoders, scaler = load_model_assets()
print("SUCCESS: Models loaded successfully.")

# ============================================================================
# PART 1: MANUAL TEST CASES (Edge cases and specific scenarios)
# ============================================================================
print("\n" + "=" * 70)
print("PART 1: MANUAL TEST CASES")
print("=" * 70)

manual_test_cases = [
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
            'Age': 70, 
            'Gender': 'Female', 
            'Hormonal Changes': 'Postmenopausal',
            'Family History': 'No', 
            'Race/Ethnicity': 'Asian', 
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
        "name": "High Risk Male (Smoker, Heavy Drinker, Medical Issues)",
        "inputs": {
            'Age': 60, 
            'Gender': 'Male', 
            'Hormonal Changes': 'Low Testosterone',
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
    },
    # NEW TEST CASES - Covering more diverse scenarios
    {
        "name": "Low Risk Young Female Athlete",
        "inputs": {
            'Age': 25, 
            'Gender': 'Female', 
            'Hormonal Changes': 'Normal',
            'Family History': 'No', 
            'Race/Ethnicity': 'Caucasian', 
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
        "name": "Moderate Risk Underweight Young Female",
        "inputs": {
            'Age': 28, 
            'Gender': 'Female', 
            'Hormonal Changes': 'Normal',
            'Family History': 'Yes', 
            'Race/Ethnicity': 'Asian', 
            'Body Weight': 'Underweight',
            'Calcium Intake': 'Low', 
            'Vitamin D Intake': 'Insufficient',
            'Physical Activity': 'Active', 
            'Smoking': 'No', 
            'Alcohol Consumption': 'None',
            'Medical Conditions': 'None', 
            'Medications': 'None', 
            'Prior Fractures': 'No'
        }
    },
    {
        "name": "Low Risk Middle-aged Male (Good Habits)",
        "inputs": {
            'Age': 45, 
            'Gender': 'Male', 
            'Hormonal Changes': 'Normal',
            'Family History': 'No', 
            'Race/Ethnicity': 'African American', 
            'Body Weight': 'Normal',
            'Calcium Intake': 'Adequate', 
            'Vitamin D Intake': 'Sufficient',
            'Physical Activity': 'Active', 
            'Smoking': 'No', 
            'Alcohol Consumption': 'Moderate',
            'Medical Conditions': 'None', 
            'Medications': 'None', 
            'Prior Fractures': 'No'
        }
    },
    {
        "name": "High Risk Elderly Female (Multiple Risk Factors)",
        "inputs": {
            'Age': 75, 
            'Gender': 'Female', 
            'Hormonal Changes': 'Postmenopausal',
            'Family History': 'Yes', 
            'Race/Ethnicity': 'Caucasian', 
            'Body Weight': 'Underweight',
            'Calcium Intake': 'Low', 
            'Vitamin D Intake': 'Insufficient',
            'Physical Activity': 'Sedentary', 
            'Smoking': 'Yes', 
            'Alcohol Consumption': 'Moderate',
            'Medical Conditions': 'Rheumatoid Arthritis', 
            'Medications': 'Corticosteroids', 
            'Prior Fractures': 'Yes'
        }
    },
    {
        "name": "Moderate Risk Male (Family History, Sedentary)",
        "inputs": {
            'Age': 55, 
            'Gender': 'Male', 
            'Hormonal Changes': 'Normal',
            'Family History': 'Yes', 
            'Race/Ethnicity': 'Caucasian', 
            'Body Weight': 'Normal',
            'Calcium Intake': 'Adequate', 
            'Vitamin D Intake': 'Sufficient',
            'Physical Activity': 'Sedentary', 
            'Smoking': 'No', 
            'Alcohol Consumption': 'Moderate',
            'Medical Conditions': 'None', 
            'Medications': 'None', 
            'Prior Fractures': 'No'
        }
    },
    {
        "name": "High Risk Postmenopausal Female (Prior Fractures)",
        "inputs": {
            'Age': 62, 
            'Gender': 'Female', 
            'Hormonal Changes': 'Postmenopausal',
            'Family History': 'Yes', 
            'Race/Ethnicity': 'Asian', 
            'Body Weight': 'Normal',
            'Calcium Intake': 'Low', 
            'Vitamin D Intake': 'Insufficient',
            'Physical Activity': 'Sedentary', 
            'Smoking': 'No', 
            'Alcohol Consumption': 'None',
            'Medical Conditions': 'None', 
            'Medications': 'None', 
            'Prior Fractures': 'Yes'
        }
    },
    {
        "name": "Low Risk Young Male (Good Nutrition)",
        "inputs": {
            'Age': 22, 
            'Gender': 'Male', 
            'Hormonal Changes': 'Normal',
            'Family History': 'No', 
            'Race/Ethnicity': 'Asian', 
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
        "name": "Moderate Risk Female (Thyroid Disorder, Good Habits)",
        "inputs": {
            'Age': 50, 
            'Gender': 'Female', 
            'Hormonal Changes': 'Postmenopausal',
            'Family History': 'No', 
            'Race/Ethnicity': 'Caucasian', 
            'Body Weight': 'Normal',
            'Calcium Intake': 'Adequate', 
            'Vitamin D Intake': 'Sufficient',
            'Physical Activity': 'Active', 
            'Smoking': 'No', 
            'Alcohol Consumption': 'None',
            'Medical Conditions': 'Thyroid Disorders', 
            'Medications': 'None', 
            'Prior Fractures': 'No'
        }
    },
    {
        "name": "High Risk Male (Corticosteroid Use)",
        "inputs": {
            'Age': 58, 
            'Gender': 'Male', 
            'Hormonal Changes': 'Normal',
            'Family History': 'No', 
            'Race/Ethnicity': 'Caucasian', 
            'Body Weight': 'Normal',
            'Calcium Intake': 'Adequate', 
            'Vitamin D Intake': 'Sufficient',
            'Physical Activity': 'Sedentary', 
            'Smoking': 'Yes', 
            'Alcohol Consumption': 'Heavy',
            'Medical Conditions': 'Rheumatoid Arthritis', 
            'Medications': 'Corticosteroids', 
            'Prior Fractures': 'No'
        }
    },
    {
        "name": "Low Risk Middle-aged Female (Perimenopausal, Active)",
        "inputs": {
            'Age': 48, 
            'Gender': 'Female', 
            'Hormonal Changes': 'Perimenopausal',
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
        "name": "Moderate Risk Sedentary Young Male",
        "inputs": {
            'Age': 35, 
            'Gender': 'Male', 
            'Hormonal Changes': 'Normal',
            'Family History': 'Yes', 
            'Race/Ethnicity': 'Asian', 
            'Body Weight': 'Underweight',
            'Calcium Intake': 'Low', 
            'Vitamin D Intake': 'Insufficient',
            'Physical Activity': 'Sedentary', 
            'Smoking': 'No', 
            'Alcohol Consumption': 'None',
            'Medical Conditions': 'None', 
            'Medications': 'None', 
            'Prior Fractures': 'No'
        }
    },
    {
        "name": "High Risk Female Smoker (Poor Nutrition)",
        "inputs": {
            'Age': 55, 
            'Gender': 'Female', 
            'Hormonal Changes': 'Postmenopausal',
            'Family History': 'No', 
            'Race/Ethnicity': 'Caucasian', 
            'Body Weight': 'Underweight',
            'Calcium Intake': 'Low', 
            'Vitamin D Intake': 'Insufficient',
            'Physical Activity': 'Sedentary', 
            'Smoking': 'Yes', 
            'Alcohol Consumption': 'Heavy',
            'Medical Conditions': 'None', 
            'Medications': 'None', 
            'Prior Fractures': 'No'
        }
    },
    {
        "name": "Low Risk Elderly Male (Excellent Habits)",
        "inputs": {
            'Age': 68, 
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
    }
]

for test_case in manual_test_cases:
    print(f"\nTEST CASE: {test_case['name']}")
    try:
        prediction, risk_score = make_prediction(
            test_case['inputs'], 
            male_model, 
            female_model, 
            label_encoders, 
            scaler
        )
        
        print(f"   Gender: {test_case['inputs']['Gender']}")
        print(f"   Risk Score: {risk_score:.4f} ({risk_score*100:.1f}%)")
        print(f"   Prediction: {'Osteoporosis' if prediction == 1 else 'No Osteoporosis'}")
            
    except Exception as e:
        print(f"   ERROR: {e}")

# ============================================================================
# PART 2: REAL DATA SAMPLES FROM CSV
# ============================================================================
print("\n" + "=" * 70)
print("PART 2: REAL DATA SAMPLES (from osteoporosis_data.csv)")
print("=" * 70)

# Load the CSV
df = pd.read_csv('data/osteoporosis_data.csv')

# Get samples: 3 with Osteoporosis=0, 3 with Osteoporosis=1
osteo_negative = df[df['Osteoporosis'] == 0].sample(n=3, random_state=42)
osteo_positive = df[df['Osteoporosis'] == 1].sample(n=3, random_state=42)

real_samples = pd.concat([osteo_negative, osteo_positive])

print(f"\nTesting {len(real_samples)} real samples from training data...")

for idx, row in real_samples.iterrows():
    # Convert row to input format
    inputs = {
        'Age': int(row['Age']),
        'Gender': row['Gender'],
        'Hormonal Changes': row['Hormonal Changes'],
        'Family History': row['Family History'],
        'Race/Ethnicity': row['Race/Ethnicity'],
        'Body Weight': row['Body Weight'],
        'Calcium Intake': row['Calcium Intake'],
        'Vitamin D Intake': row['Vitamin D Intake'],
        'Physical Activity': row['Physical Activity'],
        'Smoking': row['Smoking'],
        'Alcohol Consumption': row['Alcohol Consumption'] if pd.notna(row['Alcohol Consumption']) else 'None',
        'Medical Conditions': row['Medical Conditions'] if pd.notna(row['Medical Conditions']) else 'None',
        'Medications': row['Medications'] if pd.notna(row['Medications']) else 'None',
        'Prior Fractures': row['Prior Fractures']
    }
    
    actual_label = int(row['Osteoporosis'])
    
    print(f"\n[Sample ID: {int(row['Id'])}] {row['Gender']}, Age {int(row['Age'])} | Actual: {'OSTEO' if actual_label == 1 else 'HEALTHY'}")
    
    try:
        prediction, risk_score = make_prediction(inputs, male_model, female_model, label_encoders, scaler)
        
        print(f"   Risk Score: {risk_score:.4f} ({risk_score*100:.1f}%)")
        print(f"   Prediction: {'Osteoporosis' if prediction == 1 else 'No Osteoporosis'}")
        print(f"   Actual: {'Osteoporosis' if actual_label == 1 else 'No Osteoporosis'}")
        
        # Check if prediction matches
        match = "[CORRECT]" if prediction == actual_label else "[WRONG]"
        print(f"   Result: {match}")
        
    except Exception as e:
        print(f"   ERROR: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("TESTS COMPLETED")
print("=" * 70)
print("\nThis test covers:")
print("  + Manual edge cases (high/low/moderate risk)")
print("  + Real training data samples (both classes)")
print("  + Both male and female predictions)")
print("\nFor production use, the model should:")
print("  - Score high-risk cases > 70%")
print("  - Score low-risk cases < 30%")
print("  - Correctly classify most training samples")
print("=" * 70)
