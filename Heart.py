import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Wrap everything in a function
def run_heart_analysis():
    # CSS remains inside the function so it applies when called
    st.markdown("""
        <style>
        .stApp { background-color: #121212; color: #FFFFFF; }
        h1, h2, h3 { color: #FF4B4B !important; }
        label { color: #E0E0E0 !important; }
        div.stButton > button:first-child {
            background-color: #FF4B4B; color: white; border: none; width: 100%; font-weight: bold;
        }
        .result-container {
            padding: 20px; border-radius: 10px; background-color: #1E1E1E; border: 1px solid #FF4B4B; text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)

    @st.cache_resource
    def load_assets():
        m_name = 'heart_disease_gbr_model.pkl'
        c_name = 'model_columns.pkl'
        return joblib.load(m_name), joblib.load(c_name)

    try:
        model, model_columns = load_assets()
    except:
        st.error("Model files not found. Ensure .pkl files are in the root directory.")
        return

    st.title("Heart Health Risk Assessment")
    st.write("Enter the following details to estimate cardiovascular risk.")

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=52)
            sex = st.selectbox("Biological Sex", options=[(1, "Male"), (0, "Female")], format_func=lambda x: x[1])[0]

            cp = st.selectbox("Chest Pain Type",
                              options=[
                                  (0, "Typical Heart-Related Pain"),
                                  (1, "Atypical (Unusual) Chest Pain"),
                                  (2, "Non-Heart Related Pain"),
                                  (3, "No Symptoms (Asymptomatic)")
                              ],
                              format_func=lambda x: x[1],
                              help="Select 'Typical' if the pain feels like pressure or squeezing in the chest.")[0]

            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=125,
                                       help="Your blood pressure while sitting still.")
            chol = st.number_input("Total Cholesterol (mg/dL)", value=212)

            fbs = st.selectbox("Is Fasting Blood Sugar > 120 mg/dL?",
                               options=[(1, "Yes (High)"), (0, "No (Normal)")],
                               format_func=lambda x: x[1],
                               help="Select 'Yes' if a recent blood test showed high sugar after fasting.")[0]

        with col2:
            restecg = st.selectbox("Resting Heart Rhythm (ECG)",
                                   options=[
                                       (0, "Normal"),
                                       (1, "Minor Irregularity (ST-T Abnormality)"),
                                       (2, "Thickened Heart Muscle (Hypertrophy)")
                                   ],
                                   format_func=lambda x: x[1],
                                   help="Results from a resting electrocardiogram (ECG).")[0]

            thalach = st.number_input("Maximum Heart Rate Achieved", value=168,
                                      help="The highest heart rate reached during intense physical activity.")
            exang = st.selectbox("Chest Pain During Exercise?", options=[(1, "Yes"), (0, "No")], format_func=lambda x: x[1])[0]

            oldpeak = st.number_input("Heart Stress Level (ST Depression)", value=1.0, step=0.1,
                                      help="A measure of how much the heart is stressed during exercise vs rest.")

            slope = st.selectbox("Heart Recovery Pattern (ST Slope)",
                                 options=[
                                     (0, "Steady Rise (Upsloping)"),
                                     (1, "Flat"),
                                     (2, "Downward (Downsloping)")
                                 ],
                                 format_func=lambda x: x[1],
                                 help="How the heart's electrical activity reacts to peak exercise.")[0]

            ca = st.selectbox("Major Vessels Visible (0-3)", options=[0, 1, 2, 3],
                              help="Number of major blood vessels seen clearly on an X-ray (fluoroscopy). Higher is generally better.")

            thal = st.selectbox("Blood Flow Status",
                                options=[
                                    (0, "Normal Flow"),
                                    (1, "Permanent Blockage (Fixed)"),
                                    (2, "Partial Blockage (Reversible)")
                                ],
                                format_func=lambda x: x[1],
                                help="Results from a Thalassemia stress test measuring blood flow to the heart.")[0]

    if st.button("Calculate Heart Risk"):
        input_dict = {
            'age': age, 'sex': sex, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs,
            'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak,
            'cp': cp, 'restecg': restecg, 'slope': slope, 'ca': ca, 'thal': thal
        }

        df = pd.DataFrame([input_dict])
        df['log_chol'] = np.log1p(df['chol'])
        df['log_oldpeak'] = np.log1p(df['oldpeak'])
        df['hr_reserve'] = (220 - df['age']) - df['thalach']

        input_final = pd.get_dummies(df, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])
        input_final = input_final.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(input_final)[0]
        risk_score = 1 - prediction

        st.markdown(f"""
        <div class="result-container">
        <h3>Estimated Cardiovascular Risk</h3>
        <h1 style="color: #FF4B4B; font-size: 54px;">{max(0, min(1, risk_score)):.1%}</h1>
        <p style="color: #E0E0E0; font-style: italic;">
        {"High probability of health issues detected. Consult a specialist." if risk_score > 0.5 else "Lower relative risk detected based on provided markers."}
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.info("**Note:** This tool is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.")

if __name__ == "__main__":
    run_heart_analysis()