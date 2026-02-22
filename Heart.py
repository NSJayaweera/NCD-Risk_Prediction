import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor


def run_heart_analysis():
    # 1. CSS Styling
    st.markdown("""
            <style>
            .stApp { background-color: #121212; color: #FFFFFF; }
            h1, h2, h3 { color: #FF4B4B !important; }
            label { color: #E0E0E0 !important; }
            div.stButton > button:first-child {
                background-color: #FF4B4B; color: white; border: none; width: 100%; font-weight: bold;
            }
            .result-container {
                padding: 20px; border-radius: 10px; background-color: #1E1E1E; border: 1px solid #333; text-align: center;
            }
            </style>
            """, unsafe_allow_html=True)

    # 2. Asset Loading
    @st.cache_resource
    def load_assets():
        # Adjust paths if your folder structure is different
        gbr_model = joblib.load('heartModel/heart_disease_gbr_model.pkl')
        xgb_model = joblib.load('heartModel/best_xgb_model_enhanced.joblib')

        # CatBoost uses a specific loading method
        cat_model = CatBoostRegressor()
        cat_model.load_model('heartModel/best_catboost_model_enhanced.cbm')

        # Load the feature columns used during training for alignment
        model_columns = joblib.load('heartModel/feature_columns.joblib')

        return {
            "Gradient Boosting": gbr_model,
            "XGBoost": xgb_model,
            "CatBoost": cat_model
        }, model_columns

    try:
        models_dict, model_columns = load_assets()
    except Exception as e:
        st.error(f"Error loading models: {e}. Ensure the 'models/' folder contains the .joblib and .cbm files.")
        return

    # 3. UI Header & Model Selection
    st.title("Heart Health Risk Assessment")

    # Model Selection Sidebar
    selected_model_name = st.selectbox(
        "Choose Analysis Algorithm",
        options=["Gradient Boosting", "XGBoost", "CatBoost"],
        help="Select the machine learning model you wish to use for this specific assessment."
    )

    st.markdown("---")

    # 4. Input Features
    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=30)
            sex = st.selectbox("Biological Sex", options=[(1, "Male"), (0, "Female")], format_func=lambda x: x[1])[0]

            cp = st.selectbox("Chest Pain Type",
                              options=[
                                  (0, "Typical Heart-Related Pain"),
                                  (1, "Atypical (Unusual) Chest Pain"),
                                  (2, "Non-Heart Related Pain"),
                                  (3, "No Symptoms (Asymptomatic)")
                              ],
                              index=3,
                              format_func=lambda x: x[1],
                              help="Select 'Typical' if the pain feels like pressure or squeezing in the chest.")[0]

            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=120,
                                       help="Your blood pressure while sitting still.")
            chol = st.number_input("Total Cholesterol (mg/dL)", value=190)

            fbs = st.selectbox("Is Fasting Blood Sugar > 120 mg/dL?",
                               options=[(1, "Yes (High)"), (0, "No (Normal)")],
                               index=1,
                               format_func=lambda x: x[1],
                               help="Select 'Yes' if a recent blood test showed high sugar after fasting.")[0]

        with col2:
            restecg = st.selectbox("Resting Heart Rhythm (ECG)",
                                   options=[
                                       (0, "Normal"),
                                       (1, "Minor Irregularity (ST-T Abnormality)"),
                                       (2, "Thickened Heart Muscle (Hypertrophy)")
                                   ],
                                   index=0,
                                   format_func=lambda x: x[1],
                                   help="Results from a resting electrocardiogram (ECG).")[0]

            thalach = st.number_input("Maximum Heart Rate Achieved", value=170,
                                      help="The highest heart rate reached during intense physical activity.")
            exang = st.selectbox("Chest Pain During Exercise?", options=[(1, "Yes"), (0, "No")], index=0,
                                 format_func=lambda x: x[1])[0]

            oldpeak = st.number_input("Heart Stress Level (ST Depression)", value=0.0, step=0.1,
                                      help="A measure of how much the heart is stressed during exercise vs rest.")

            slope = st.selectbox("Heart Recovery Pattern (ST Slope)",
                                 options=[
                                     (0, "Steady Rise (Upsloping)"),
                                     (1, "Flat"),
                                     (2, "Downward (Downsloping)")
                                 ],
                                 index=0,
                                 format_func=lambda x: x[1],
                                 help="How the heart's electrical activity reacts to peak exercise.")[0]

            ca = st.selectbox("Major Vessels Visible (0-3)", options=[0, 1, 2, 3],
                              index=3,
                              help="Number of major blood vessels seen clearly on an X-ray (fluoroscopy). Higher is generally better.")

            thal = st.selectbox("Blood Flow Status",
                                options=[
                                    (0, "Normal Flow"),
                                    (1, "Permanent Blockage (Fixed)"),
                                    (2, "Partial Blockage (Reversible)")
                                ],
                                format_func=lambda x: x[1],
                                help="Results from a Thalassemia stress test measuring blood flow to the heart.")[0]

    # 5. Prediction Logic
    if st.button("Calculate Heart Risk"):
        # Create Dataframe
        input_dict = {
            'age': age, 'sex': sex, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs,
            'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak,
            'cp': cp, 'restecg': restecg, 'slope': slope, 'ca': ca, 'thal': thal
        }

        df = pd.DataFrame([input_dict])

        # Apply Enhanced Feature Engineering
        df['log_chol'] = np.log1p(df['chol'])
        df['log_oldpeak'] = np.log1p(df['oldpeak'])
        df['hr_reserve'] = (220 - df['age']) - df['thalach']

        # Categorical Encoding
        input_final = pd.get_dummies(df, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])

        # Align with the columns the models expect
        input_final = input_final.reindex(columns=model_columns, fill_value=0)

        # Get the selected model and predict
        model = models_dict[selected_model_name]
        prediction = model.predict(input_final)[0]

        # Score interpretation
        risk_score = 1 - prediction

        # Display Result
        score_color = "#28a745" if risk_score <= 0.5 else "#FF4B4B"
        border_color = score_color
        st.markdown(f"""
                <div class="result-container" style="border: 2px solid {border_color};">
                    <h3 style="color: #E0E0E0;">Estimated Cardiovascular Risk</h3>
                    <p>Analysis by: <strong>{selected_model_name}</strong></p>
                    <h1 style="color: {score_color}; font-size: 64px; margin: 10px 0;">{max(0, min(1, risk_score)):.1%}</h1>
                    <p style="color: #E0E0E0; font-style: italic;">
                        {"High probability of health issues detected. Consult a specialist." if risk_score > 0.5
        else "Lower relative risk detected based on provided markers."}
                    </p>
                </div>
                """, unsafe_allow_html=True)

        st.info(
            "**Note:** This tool is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.")


if __name__ == "__main__":
    run_heart_analysis()