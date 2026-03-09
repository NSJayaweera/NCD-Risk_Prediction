import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Wrap everything in a function
def run_diabetes_analysis():
    # CSS remains inside the function so it applies when called
    st.markdown("""
        <style>
        .stApp { background-color: #121212; color: #FFFFFF; }
        h1, h2, h3 { color: #4BFF4B !important; }
        label { color: #E0E0E0 !important; }
        div.stButton > button:first-child {
            background-color: #4BFF4B; color: black; border: none; width: 100%; font-weight: bold;
        }
        .result-container {
            padding: 20px; border-radius: 10px; background-color: #1E1E1E; border: 1px solid #4BFF4B; text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)

    @st.cache_resource
    def load_assets():
        best_m_name = 'Models/best_diabetes_model.pkl'
        second_m_name = 'Models/second_best_diabetes_model.pkl'
        s_name = 'Models/scaler_pickle.pkl'
        
        models = {
            "Best Model (SVM)": joblib.load(best_m_name),
            "Second Model (Gradient Boosting)": joblib.load(second_m_name)
        }
        scaler = joblib.load(s_name)
        return models, scaler

    try:
        models, scaler = load_assets()
    except Exception as e:
        st.error(f"Model files not found. Ensure .pkl files are in the 'Models' directory. Error: {e}")
        return

    st.title("Diabetes Health Risk Assessment")
    
    # Model Selection in Main Area
    selected_model_name = st.radio(
        "Select Prediction Model",
        options=list(models.keys()),
        horizontal=True,
        help="Choose between the top two performing models."
    )
    model = models[selected_model_name]

    st.write("Enter the following details to estimate diabetes risk.")

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            blood_glucose = st.number_input("Blood Glucose Level (mg/dL)", min_value=50.0, max_value=400.0, value=100.0, step=1.0)
            
            physical_activity = st.number_input("Physical Activity (hours/week)", min_value=0.0, max_value=168.0, value=5.0, step=0.1)
            
            diet = st.selectbox("Diet Quality", 
                                options=[(1, "Healthy"), (0, "Unhealthy")], 
                                format_func=lambda x: x[1],
                                help="Select 'Healthy' if you possess a balanced diet.")[0]
            
            medication_adherence = st.selectbox("Medication Adherence", 
                                                options=[(1, "Yes"), (0, "No")], 
                                                format_func=lambda x: x[1],
                                                help="Do you strictly adhere to prescribed medications?")[0]

        with col2:
            stress_level = st.selectbox("Stress Level", 
                                        options=[(0, "Low"), (1, "Medium"), (2, "High")], 
                                        format_func=lambda x: x[1])[0]
            
            sleep_hours = st.number_input("Sleep Hours per Night", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
            
            hydration_level = st.selectbox("Hydration Level", 
                                           options=[(1, "High (Good)"), (0, "Low (Poor)")], 
                                           format_func=lambda x: x[1])[0]
            
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

    if st.button("Calculate Diabetes Risk"):
        # Features: ['blood_glucose', 'physical_activity', 'diet', 'medication_adherence', 'stress_level', 'sleep_hours', 'hydration_level', 'bmi']
        input_dict = {
            'blood_glucose': blood_glucose,
            'physical_activity': physical_activity,
            'diet': diet,
            'medication_adherence': medication_adherence,
            'stress_level': stress_level,
            'sleep_hours': sleep_hours,
            'hydration_level': hydration_level,
            'bmi': bmi
        }

        df = pd.DataFrame([input_dict])
        
        # Scale the features
        try:
            # Reorder columns to match scaler's expected input
            feature_order = ['blood_glucose', 'physical_activity', 'diet', 'medication_adherence', 
                             'stress_level', 'sleep_hours', 'hydration_level', 'bmi']
            df = df[feature_order]
            
            scaled_features = scaler.transform(df)
            
            # Predict
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(scaled_features)[0][1] # Probability of class 1 (Diabetes/High Risk)
                risk_score = prob
            else:
                # If model doesn't support predict_proba, use predict
                risk_score = float(model.predict(scaled_features)[0])
            
            st.markdown(f"""
            <div class="result-container">
            <h3>Estimated Diabetes Risk Score ({selected_model_name})</h3>
            <h1 style="color: #4BFF4B; font-size: 54px;">{risk_score:.2f}</h1>
            <p style="color: #E0E0E0; font-style: italic;">
            {"High risk detected. Consult a specialist." if risk_score > 0.5 else "Low risk detected."}
            </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("**Note:** This tool is for informational purposes only and is not a substitute for professional medical advice.")
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    run_diabetes_analysis()
