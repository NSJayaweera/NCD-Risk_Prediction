import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Wrap everything in a function
def run_diabetes_analysis():
    # CSS remains inside the function so it applies when called
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

        /* ── Base ── */
        .stApp {
            background-color: #0A0A0F;
            color: #E8E8F0;
            font-family: 'DM Sans', sans-serif;
        }

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

        /* ── Tooltip ── */
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

        div[data-testid="stRadio"] > div { gap: 1rem; }

        div[data-testid="stRadio"] input[type="radio"] + div svg {
            fill: #FF4B4B !important;
            stroke: #FF4B4B !important;
        }
        </style>
        """, unsafe_allow_html=True)

    @st.cache_resource
    def load_assets():
        best_m_name = 'models/best_diabetes_model.pkl'
        second_m_name = 'models/second_best_diabetes_model.pkl'
        s_name = 'models/scaler_pickle.pkl'

        models = {
            "SVM": joblib.load(best_m_name),
            "Gradient Boosting": joblib.load(second_m_name)
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

    st.markdown("---")

    st.write("Enter the following details to estimate diabetes risk.")

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            blood_glucose = st.number_input("Blood Glucose Level (mg/dL)", min_value=50.0, max_value=400.0, value=100.0,
                                            step=1.0)

            physical_activity = st.number_input("Physical Activity (hours/week)", min_value=0.0, max_value=168.0,
                                                value=5.0, step=0.1)

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

    st.markdown("---")

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
                prob = model.predict_proba(scaled_features)[0][1]  # Probability of class 1 (Diabetes/High Risk)
                risk_score = prob
            else:
                # If model doesn't support predict_proba, use predict
                risk_score = float(model.predict(scaled_features)[0])

            score_color = "#28a745" if risk_score <= 0.5 else "#FF4B4B"
            badge_bg = "rgba(40,167,69,0.15)" if risk_score <= 0.5 else "rgba(255,75,75,0.15)"
            badge_label = "LOW RISK" if risk_score <= 0.5 else "HIGH RISK"

            st.markdown(f"""
                <div class="result-container" style="border: 1px solid {score_color}33; box-shadow: 0 0 40px {score_color}22;">
                    <p style="color: #808090; font-family: 'Space Mono', monospace; font-size: 0.7rem; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.5rem;">
                        Analysis by {selected_model_name}
                    </p>
                    <span class="risk-badge" style="background:{badge_bg}; color:{score_color}; border: 1px solid {score_color}44;">
                        {badge_label}
                    </span>
                    <h3 style="color: #808090; font-family: 'DM Sans', sans-serif; font-size: 0.8rem; font-weight: 500; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.25rem; border: none !important; padding: 0 !important;">
                        Estimated Diabetes Risk
                    </h3>
                    <div style="font-family: 'Space Mono', monospace; color: {score_color}; font-size: 5rem; font-weight: 700; line-height: 1; margin: 0.5rem 0 1rem;">
                        {risk_score:.1%}
                    </div>
                    <p style="color: #A0A0B0; font-size: 0.875rem; max-width: 420px; margin: 0 auto; line-height: 1.6;">
                        {"High risk detected. Please consult a specialist promptly." if risk_score > 0.5 else "Lower relative risk detected based on the provided markers."}
                    </p>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # Tiered guidance
            clamped = max(0.0, min(1.0, float(risk_score)))

            if clamped < 0.30:
                tier_icon = "✅"
                tier_title = "Your diabetes risk indicators look reassuring"
                tier_color = "#28a745"
                tier_bg = "rgba(40,167,69,0.07)"
                tier_border = "rgba(40,167,69,0.3)"
                steps = [
                    ("Keep up your healthy habits",
                     "Your current lifestyle appears to be supporting good metabolic health. Staying consistent with physical activity and a balanced diet makes a real long-term difference."),
                    ("Monitor blood glucose periodically",
                     "Even at low risk, knowing your baseline blood glucose through an occasional check-up helps catch any early shifts before they become concerns."),
                    ("Maintain a healthy weight",
                     "Keeping your BMI in a healthy range is one of the most effective ways to stay at low risk for Type 2 diabetes over time."),
                    ("Stay hydrated and sleep well",
                     "Good hydration and 7–9 hours of sleep per night support healthy insulin sensitivity and overall metabolic function."),
                ]
            elif clamped < 0.60:
                tier_icon = "🟡"
                tier_title = "A few lifestyle areas worth addressing"
                tier_color = "#f0a500"
                tier_bg = "rgba(240,165,0,0.07)"
                tier_border = "rgba(240,165,0,0.3)"
                steps = [
                    ("Speak with your doctor",
                     "It would be worth mentioning your metabolic health at your next appointment. A simple fasting blood glucose or HbA1c test can give you a clearer picture — it's quick and routine."),
                    ("Review your diet",
                     "Reducing refined sugars and processed carbohydrates while increasing fibre, vegetables, and whole grains can meaningfully improve blood sugar regulation."),
                    ("Increase physical activity",
                     "Aim for at least 150 minutes of moderate activity per week. Even brisk walking after meals helps manage blood glucose levels effectively."),
                    ("Manage stress",
                     "Chronic stress raises cortisol, which can push blood sugar higher. Simple habits like better sleep, short breaks, and time outdoors can help."),
                    ("Track your BMI",
                     "If your BMI is above the healthy range, even a modest reduction of 5–10% body weight can significantly lower diabetes risk."),
                ]
            else:
                tier_icon = "🔴"
                tier_title = "We recommend speaking with a healthcare professional"
                tier_color = "#FF4B4B"
                tier_bg = "rgba(255,75,75,0.07)"
                tier_border = "rgba(255,75,75,0.3)"
                steps = [
                    ("Book a check-up soon",
                     "Based on these markers, we'd encourage you to book an appointment with your GP in the near future. A blood test can confirm your current status and guide the right next steps — this is a proactive, not alarming, move."),
                    ("Ask about diabetes screening",
                     "A fasting glucose test or HbA1c check is simple, fast, and gives your doctor the information needed to recommend a personalised plan."),
                    ("Focus on diet changes",
                     "Cutting back on sugary drinks, white bread, and processed foods while adding more fibre, lean protein, and vegetables is one of the most impactful changes you can make."),
                    ("Move more, consistently",
                     "Daily physical activity — even 20–30 minutes of walking — significantly improves insulin sensitivity. You don't need to do intense exercise to see real benefits."),
                    ("Medication is an option",
                     "If lifestyle changes aren't enough, effective medications exist. Many people manage diabetes well with the right support — your doctor can help you understand your options without pressure."),
                ]

            # Render guidance as a single block to avoid Streamlit gaps
            steps_html = ""
            for title, desc in steps:
                steps_html += f"""
                    <div style="display:flex; gap:1rem; align-items:flex-start; padding:0.6rem 0; border-top:1px solid {tier_border};">
                        <div style="min-width:6px; height:6px; border-radius:50%; background:{tier_color};
                                    margin-top:0.45rem; flex-shrink:0;"></div>
                        <div>
                            <p style="color:#E8E8F0; font-weight:600; font-size:0.875rem; margin:0 0 0.2rem;">{title}</p>
                            <p style="color:#A0A0B0; font-size:0.85rem; line-height:1.6; margin:0;">{desc}</p>
                        </div>
                    </div>"""

            guidance_top = f"""
                <div style="background:{tier_bg}; border:1px solid {tier_border}; border-radius:14px;
                            padding:1.75rem 2rem; margin-bottom:0.5rem;">
                    <p style="font-family:'Space Mono',monospace; color:{tier_color}; font-size:0.7rem;
                              letter-spacing:0.12em; text-transform:uppercase; margin:0 0 0.4rem;">
                        {tier_icon} Personalised Guidance
                    </p>
                    <p style="color:#E8E8F0; font-family:'DM Sans',sans-serif; font-size:1.1rem;
                              font-weight:600; margin:0 0 1rem;">
                        {tier_title}
                    </p>"""

            guidance_bottom = "</div>"

            st.markdown(guidance_top + steps_html + guidance_bottom, unsafe_allow_html=True)

            st.markdown("""
                <div style="background:#13131A; border:1px solid #2A2A3A; border-left:3px solid #FF4B4B;
                            border-radius:8px; padding:0.85rem 1rem; margin-top:0.5rem;">
                    <p style="color:#C0C0D0; font-family:'DM Sans',sans-serif; font-size:0.82rem;
                                line-height:1.5; margin:0;">
                        <strong>Note:</strong> This tool is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.
                    </p>
                </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error during prediction: {e}")


if __name__ == "__main__":
    run_diabetes_analysis()
