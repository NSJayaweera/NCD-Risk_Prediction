import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor


def run_heart_analysis():
    # 1. CSS Styling
    st.markdown("""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

            /* ── Base ── */
            .stApp {
                background-color: #0A0A0F;
                color: #E8E8F0;
                font-family: 'DM Sans', sans-serif;
            }

            /* Subtle grid texture overlay */
            .stApp::before {
                content: '';
                position: fixed;
                inset: 0;
                background-image:
                    linear-gradient(rgba(255,75,75,0.03) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,75,75,0.03) 1px, transparent 1px);
                background-size: 40px 40px;
                pointer-events: none;
                z-index: 0;
            }

            /* ── Typography ── */
            h1, h2, h3 {
                font-family: 'Space Mono', monospace !important;
                color: #FF4B4B !important;
                letter-spacing: -0.02em;
            }

            h1 {
                font-size: 2rem !important;
                text-transform: uppercase;
                border-bottom: 2px solid #FF4B4B;
                padding-bottom: 0.5rem;
                margin-bottom: 1.5rem !important;
            }

            label, .stMarkdown p {
                color: #C0C0D0 !important;
                font-size: 0.875rem;
                font-weight: 500;
                letter-spacing: 0.01em;
            }

            /* ── Selectbox & Number Inputs ── */
            div[data-baseweb="select"] > div,
            div[data-baseweb="input"] > div {
                background-color: #13131A !important;
                border: 1px solid #2A2A3A !important;
                border-radius: 8px !important;
                color: #E8E8F0 !important;
                transition: border-color 0.2s ease;
            }

            div[data-baseweb="select"] > div:hover,
            div[data-baseweb="input"] > div:hover {
                border-color: #FF4B4B !important;
            }

            div[data-baseweb="select"] svg { fill: #FF4B4B !important; }

            /* Dropdown menu */
            div[data-baseweb="popover"] {
                background-color: #13131A !important;
                border: 1px solid #2A2A3A !important;
                border-radius: 8px !important;
            }

            li[role="option"]:hover {
                background-color: #1E1E2E !important;
                color: #FF4B4B !important;
            }

            /* ── Button ── */
            .stButton button {
                background-color: #0D0D14 !important;
                color: #E8E8F0 !important;
                border-width: 2px !important;
                border-style: solid !important;
                border-color: #888899 !important;
                width: 100% !important;
                font-family: 'DM Sans', sans-serif !important;
                font-weight: 600 !important;
                font-size: 0.9rem !important;
                padding: 0.85rem 1.5rem !important;
                border-radius: 8px !important;
                cursor: pointer !important;
                transition: all 0.25s ease !important;
                box-shadow: none !important;
            }

            .stButton button:hover {
                background-color: #13131A !important;
                border-color: #FF4B4B !important;
                color: #FF4B4B !important;
                transform: translateY(-2px) !important;
                box-shadow: 0 8px 24px rgba(255, 75, 75, 0.2) !important;
            }

            .stButton button:active {
                transform: translateY(0) !important;
            }

            /* ── Result Card ── */
            .result-container {
                padding: 2.5rem 2rem;
                border-radius: 16px;
                background: linear-gradient(135deg, #13131A 0%, #0F0F18 100%);
                border: 1px solid #2A2A3A;
                text-align: center;
                position: relative;
                overflow: hidden;
            }

            .result-container::before {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: radial-gradient(circle at center, rgba(255,75,75,0.05) 0%, transparent 60%);
                pointer-events: none;
            }

            /* ── Section Divider ── */
            hr {
                border: none !important;
                border-top: 1px solid #1E1E2E !important;
                margin: 1.5rem 0 !important;
            }

            /* ── Info Box ── */
            div[data-testid="stInfo"] {
                background-color: #13131A !important;
                border: 1px solid #2A2A3A !important;
                border-left: 3px solid #FF4B4B !important;
                border-radius: 8px !important;
                color: #C0C0D0 !important;
            }

            /* ── Column Sections ── */
            div[data-testid="column"] {
                background-color: #0D0D14;
                border: 1px solid #1A1A26;
                border-radius: 12px;
                padding: 1.25rem !important;
            }

            /* ── Selectbox Tooltip ── */
            .stTooltipIcon { color: #FF4B4B !important; }

            /* ── Scrollbar ── */
            ::-webkit-scrollbar { width: 6px; }
            ::-webkit-scrollbar-track { background: #0A0A0F; }
            ::-webkit-scrollbar-thumb { background: #2A2A3A; border-radius: 3px; }
            ::-webkit-scrollbar-thumb:hover { background: #FF4B4B; }

            /* ── Header accent bar ── */
            .header-accent {
                display: inline-block;
                width: 40px;
                height: 3px;
                background: #FF4B4B;
                border-radius: 2px;
                margin-bottom: 0.75rem;
            }

            /* ── Risk badge ── */
            .risk-badge {
                display: inline-block;
                padding: 4px 14px;
                border-radius: 20px;
                font-family: 'Space Mono', monospace;
                font-size: 0.7rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 1rem;
            }

            /* ── Radio Buttons ── */
            div[data-testid="stRadio"] label {
                color: #C0C0D0 !important;
                font-family: 'DM Sans', sans-serif !important;
                font-weight: 500 !important;
            }

            div[data-testid="stRadio"] > div {
                gap: 1rem;
            }

            div[data-testid="stRadio"] input[type="radio"] + div svg {
                fill: #FF4B4B !important;
                stroke: #FF4B4B !important;
            }
            </style>
            """, unsafe_allow_html=True)

    # 2. Asset Loading
    @st.cache_resource
    def load_assets():
        # Adjust paths if your folder structure is different
        gbr_model = joblib.load('heartModel/heart_disease_gbr_model.pkl')

        # CatBoost uses a specific loading method
        cat_model = CatBoostRegressor()
        cat_model.load_model('heartModel/best_catboost_model_enhanced.cbm')

        # Load the feature columns used during training for alignment
        model_columns = joblib.load('heartModel/feature_columns.joblib')

        return {
            "Gradient Boosting": gbr_model,
            "CatBoost": cat_model
        }, model_columns

    try:
        models_dict, model_columns = load_assets()
    except Exception as e:
        st.error(f"Error loading models: {e}. Ensure the 'models/' folder contains the .joblib and .cbm files.")
        return

    # 3. UI Header & Model Selection
    st.title("Heart Health Risk Assessment")

    # Model Selection Radio
    selected_model_name = st.radio(
        "Choose Analysis Algorithm",
        options=["Gradient Boosting", "CatBoost"],
        horizontal=True,
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

    st.markdown("---")

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
        badge_bg = "rgba(40,167,69,0.15)" if risk_score <= 0.5 else "rgba(255,75,75,0.15)"
        badge_label = "LOW RISK" if risk_score <= 0.5 else "HIGH RISK"

        st.markdown(f"""
                <div class="result-container" style="border: 1px solid {border_color}33; box-shadow: 0 0 40px {border_color}22;">
                    <p style="color: #808090; font-family: 'Space Mono', monospace; font-size: 0.7rem; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.5rem;">
                        Analysis by {selected_model_name}
                    </p>
                    <span class="risk-badge" style="background:{badge_bg}; color:{score_color}; border: 1px solid {score_color}44;">
                        {badge_label}
                    </span>
                    <h3 style="color: #808090; font-family: 'DM Sans', sans-serif; font-size: 0.8rem; font-weight: 500; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.25rem;">
                        Estimated Cardiovascular Risk
                    </h3>
                    <div style="font-family: 'Space Mono', monospace; color: {score_color}; font-size: 5rem; font-weight: 700; line-height: 1; margin: 0.5rem 0 1rem;">
                        {max(0, min(1, risk_score)):.1%}
                    </div>
                    <p style="color: #A0A0B0; font-size: 0.875rem; max-width: 420px; margin: 0 auto; line-height: 1.6;">
                        {"High probability of health issues detected. Please consult a specialist promptly." if risk_score > 0.5
        else "Lower relative risk detected based on the provided clinical markers."}
                    </p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Tiered guidance based on risk score
        clamped = max(0, min(1, risk_score))

        if clamped < 0.30:
            tier_icon = "✅"
            tier_title = "Your heart health indicators look good"
            tier_color = "#28a745"
            tier_bg = "rgba(40,167,69,0.07)"
            tier_border = "rgba(40,167,69,0.3)"
            steps = [
                ("Keep it up", "Your current lifestyle habits appear to be supporting good heart health. Stay consistent with regular physical activity — even a 30-minute walk most days makes a difference."),
                ("Eat well", "A balanced diet rich in fruits, vegetables, whole grains, and healthy fats helps maintain healthy cholesterol and blood pressure over the long term."),
                ("Stay on top of check-ups", "Even with a low risk score, a routine check-up with your doctor once a year is a great habit to catch any early changes."),
                ("Know your numbers", "Keep an eye on your blood pressure and cholesterol periodically. Knowing your baseline makes it easier to spot any shifts early."),
            ]
        elif clamped < 0.60:
            tier_icon = "🟡"
            tier_title = "A few areas worth paying attention to"
            tier_color = "#f0a500"
            tier_bg = "rgba(240,165,0,0.07)"
            tier_border = "rgba(240,165,0,0.3)"
            steps = [
                ("Talk to your doctor", "This score suggests it would be worth discussing your heart health at your next appointment. There's no need to worry — a doctor can give you a clearer picture and personalised advice."),
                ("Move a little more", "Gradually increasing physical activity — even light walking, swimming, or cycling — can meaningfully improve cardiovascular health over time."),
                ("Review your diet", "Reducing processed foods, excess salt, and saturated fats while increasing fibre and vegetables is one of the most effective lifestyle changes for heart health."),
                ("Manage stress", "Chronic stress can quietly affect blood pressure and heart health. Simple habits like better sleep, breathing exercises, or time outdoors can help."),
                ("Monitor key metrics", "Try to track your blood pressure and cholesterol more regularly. Many pharmacies offer free checks, and your GP can arrange bloodwork."),
            ]
        else:
            tier_icon = "🔴"
            tier_title = "We recommend speaking with a healthcare professional"
            tier_color = "#FF4B4B"
            tier_bg = "rgba(255,75,75,0.07)"
            tier_border = "rgba(255,75,75,0.3)"
            steps = [
                ("Schedule a check-up soon", "Based on these markers, we'd encourage you to book an appointment with your GP or a cardiologist in the near future. This is a proactive step, not a cause for alarm."),
                ("Be open with your doctor", "Share these results and any symptoms you may have noticed — even mild ones like occasional breathlessness or fatigue. Context helps your doctor help you."),
                ("Medication may be an option", "In some cases, your doctor might suggest medication to manage blood pressure or cholesterol. These are well-established, effective tools that many people benefit from."),
                ("Focus on small, sustainable changes", "You don't need to overhaul everything at once. Small consistent steps — quitting smoking, reducing alcohol, gentle exercise, and better sleep — add up significantly."),
                ("Don't go it alone", "Consider involving a family member or friend in your health journey. Support makes a real difference, and many clinics offer heart health programmes with structured guidance."),
            ]

        steps_html = ""
        for title, desc in steps:
            steps_html += f"""
                <div style="display:flex; gap:1rem; margin-bottom:1rem; align-items:flex-start;">
                    <div style="min-width:6px; height:6px; border-radius:50%; background:{tier_color}; margin-top:0.45rem; flex-shrink:0;"></div>
                    <div>
                        <p style="color:#E8E8F0; font-weight:600; font-size:0.875rem; margin:0 0 0.2rem;">{title}</p>
                        <p style="color:#A0A0B0; font-size:0.85rem; line-height:1.6; margin:0;">{desc}</p>
                    </div>
                </div>"""

        guidance_html = f"""
            <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
            <div style="background:{tier_bg}; border:1px solid {tier_border}; border-radius:14px; padding:1.75rem 2rem; margin-bottom:1rem;">
                <p style="font-family:'Space Mono',monospace; color:{tier_color}; font-size:0.7rem; letter-spacing:0.12em; text-transform:uppercase; margin:0 0 0.4rem;">
                    {tier_icon} Personalised Guidance
                </p>
                <p style="color:#E8E8F0; font-family:'DM Sans',sans-serif; font-size:1.1rem; font-weight:600; margin:0 0 1.25rem;">
                    {tier_title}
                </p>
                {steps_html}
            </div>
            <div style="background:#13131A; border:1px solid #2A2A3A; border-left:3px solid #FF4B4B;
                        border-radius:8px; padding:0.85rem 1rem; margin-top:0.5rem;">
                <p style="color:#C0C0D0; font-family:'DM Sans',sans-serif; font-size:0.82rem;
                            line-height:1.5; margin:0;">
                    <strong>Note:</strong> This tool is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.
                </p>
            </div>"""

        components.html(guidance_html, height=len(steps) * 110 + 100, scrolling=False)

if __name__ == "__main__":
    run_heart_analysis()
