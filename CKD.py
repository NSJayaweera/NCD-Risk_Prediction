import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import math


def run_ckd_analysis():

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

        h3 {
            font-size: 1rem !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
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

        /* ── Metric ── */
        div[data-testid="stMetric"] {
            background-color: #13131A;
            border: 1px solid #2A2A3A;
            border-radius: 10px;
            padding: 1rem 1.25rem;
        }

        div[data-testid="stMetricLabel"] p {
            color: #808090 !important;
            font-size: 0.75rem !important;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        div[data-testid="stMetricValue"] {
            color: #E8E8F0 !important;
            font-family: 'Space Mono', monospace !important;
            font-size: 1.5rem !important;
        }

        /* ── Progress bar ── */
        div[data-testid="stProgressBar"] > div > div {
            background-color: #FF4B4B !important;
        }

        div[data-testid="stProgressBar"] > div {
            background-color: #1E1E2E !important;
            border-radius: 4px !important;
        }

        /* ── Info / Success / Error / Warning boxes ── */
        div[data-testid="stInfo"] {
            background-color: #13131A !important;
            border: 1px solid #2A2A3A !important;
            border-left: 3px solid #FF4B4B !important;
            border-radius: 8px !important;
            color: #C0C0D0 !important;
        }

        div[data-testid="stSuccess"] {
            background-color: rgba(40,167,69,0.08) !important;
            border: 1px solid rgba(40,167,69,0.3) !important;
            border-left: 3px solid #28a745 !important;
            border-radius: 8px !important;
        }

        div[data-testid="stError"] {
            background-color: rgba(255,75,75,0.08) !important;
            border: 1px solid rgba(255,75,75,0.3) !important;
            border-left: 3px solid #FF4B4B !important;
            border-radius: 8px !important;
        }

        div[data-testid="stWarning"] {
            background-color: rgba(240,165,0,0.08) !important;
            border: 1px solid rgba(240,165,0,0.3) !important;
            border-left: 3px solid #f0a500 !important;
            border-radius: 8px !important;
        }

        /* ── Section Divider ── */
        hr {
            border: none !important;
            border-top: 1px solid #1E1E2E !important;
            margin: 1.5rem 0 !important;
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

        /* ── Caption ── */
        .stCaption, small { color: #505060 !important; font-size: 0.75rem !important; }

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

    # ── Load models ──
    @st.cache_resource
    def load_models():
        SAVE_DIR = "ckdModel/"
        stacking = joblib.load(SAVE_DIR + "stacking_model.pkl")
        bagging  = joblib.load(SAVE_DIR + "bagging_model.pkl")
        with open(SAVE_DIR + "feature_order.json", "r") as f:
            feature_order = json.load(f)
        return stacking, bagging, feature_order

    try:
        stacking_model, bagging_model, feature_order = load_models()
    except Exception as e:
        st.error(f"Error loading models: {e}. Ensure the 'ckdModel/' folder contains the .pkl and .json files.")
        return

    # ── eGFR Calculator (CKD-EPI 2021) ──
    def calculate_egfr(creatinine, age, gender):
        if creatinine <= 0 or age <= 0:
            return None
        kappa      = 0.7   if gender == 0 else 0.9
        alpha      = -0.241 if gender == 0 else -0.302
        sex_factor = 1.012  if gender == 0 else 1.0
        ratio = creatinine / kappa
        if ratio < 1:
            egfr = 142 * (ratio ** alpha) * (0.9938 ** age) * sex_factor
        else:
            egfr = 142 * (ratio ** -1.200) * (0.9938 ** age) * sex_factor
        return round(egfr, 1)

    # ── Header ──
    st.title("Chronic Kidney Disease Risk Prediction")
    st.write("Enter patient details below to predict CKD risk using AI models.")

    st.markdown("---")

    # ── Model selection ──
    selected_model_name = st.radio(
        "Choose Prediction Model",
        options=["Stacking Model", "Bagging Model", "Both"],
        horizontal=True,
        help="Select which trained model(s) to use for prediction."
    )

    st.markdown("---")

    # ── Input Form ──
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Basic Information")
        age    = st.number_input("Age (years)", min_value=18, max_value=120, value=50)
        gender = st.radio("Gender", options=["Female", "Male"], horizontal=True)
        gender_val = 1 if gender == "Male" else 0

        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0)
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
        bmi    = round(weight / ((height / 100) ** 2), 1)
        st.markdown(f"""
            <div style="background:#13131A; border:1px solid #2A2A3A; border-left:3px solid #FF4B4B;
                        border-radius:8px; padding:0.6rem 1rem; margin:0.5rem 0;">
                <p style="color:#808090; font-size:0.7rem; text-transform:uppercase;
                          letter-spacing:0.08em; margin:0 0 0.2rem;">Calculated BMI</p>
                <p style="color:#E8E8F0; font-family:'Space Mono',monospace;
                          font-size:1.25rem; font-weight:700; margin:0;">{bmi}</p>
            </div>
        """, unsafe_allow_html=True)

        bp_systolic  = st.number_input("Systolic BP (mmHg)", min_value=60, max_value=260, value=120)
        bp_diastolic = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=160, value=80)
        diabetes     = st.radio("Diabetes Diagnosed", options=["No", "Yes"], horizontal=True)
        diabetes_val = 1 if diabetes == "Yes" else 0

    with col2:
        st.subheader("Laboratory Values")
        blood_urea_nitrogen = st.number_input("Blood Urea Nitrogen (mg/dL)", min_value=1.0, max_value=200.0, value=15.0, step=0.5)
        urine_albumin       = st.number_input("Urine Albumin (mg/L)", min_value=0.0, max_value=9000.0, value=3.0, step=1.0)
        urine_creatinine    = st.number_input("Urine Creatinine (mg/dL)", min_value=1.0, max_value=1200.0, value=150.0, step=1.0)
        albumin_serum       = st.number_input("Serum Albumin (g/dL)", min_value=0.5, max_value=8.0, value=4.0, step=0.1)
        serum_creatinine    = st.number_input("Serum Creatinine (mg/dL)", min_value=0.1, max_value=20.0, value=1.0, step=0.1)
        uric_acid           = st.number_input("Uric Acid (mg/dL)", min_value=0.5, max_value=25.0, value=5.0, step=0.1)

    # ── Auto-calculated values ──
    st.markdown("---")
    st.subheader("Auto-Calculated Values")

    acr          = round(urine_albumin / urine_creatinine * 1000, 2) if urine_creatinine > 0 else 0
    bun_cr_ratio = round(blood_urea_nitrogen / serum_creatinine, 2)  if serum_creatinine > 0 else 0
    egfr_val     = calculate_egfr(serum_creatinine, age, gender_val)

    c1, c2, c3 = st.columns(3)
    c1.metric("Albumin-Creatinine Ratio (mg/g)", f"{acr}")
    c2.metric("BUN / Creatinine Ratio",           f"{bun_cr_ratio}")
    c3.metric("eGFR (CKD-EPI) — display only",    f"{egfr_val}")

    st.markdown("---")

    # ── Predict ──
    if st.button("Predict CKD Risk"):

        patient_data = {
            "age"                      : age,
            "gender"                   : gender_val,
            "bmi"                      : bmi,
            "bp_systolic"              : bp_systolic,
            "bp_diastolic"             : bp_diastolic,
            "serum_creatinine"         : serum_creatinine,
            "blood_urea_nitrogen"      : blood_urea_nitrogen,
            "urine_albumin"            : urine_albumin,
            "urine_creatinine"         : urine_creatinine,
            "albumin_creatinine_ratio" : acr,
            "albumin_serum"            : albumin_serum,
            "uric_acid"                : uric_acid,
            "diabetes_diagnosed"       : diabetes_val,
            "bun_creatinine_ratio"     : bun_cr_ratio,
        }

        patient_df = pd.DataFrame([patient_data])
        for col in feature_order:
            if col not in patient_df.columns:
                patient_df[col] = np.nan
        patient_df = patient_df[feature_order]

        stack_prob = stacking_model.predict_proba(patient_df)[0][1]
        stack_pred = stacking_model.predict(patient_df)[0]
        bag_prob   = bagging_model.predict_proba(patient_df)[0][1]
        bag_pred   = bagging_model.predict(patient_df)[0]

        # ── Determine what to show based on model selection ──
        if selected_model_name == "Stacking Model":
            results = [("Stacking Model", stack_prob, stack_pred)]
        elif selected_model_name == "Bagging Model":
            results = [("Bagging Model", bag_prob, bag_pred)]
        else:
            results = [
                ("Stacking Model", stack_prob, stack_pred),
                ("Bagging Model",  bag_prob,   bag_pred),
            ]

        # ── Result cards ──
        if len(results) == 1:
            name, prob, pred = results[0]
            score_color = "#FF4B4B" if pred == 1 else "#28a745"
            badge_bg    = "rgba(255,75,75,0.15)" if pred == 1 else "rgba(40,167,69,0.15)"
            badge_label = "CKD DETECTED" if pred == 1 else "NO CKD DETECTED"

            st.markdown(f"""
                <div style="padding:2.5rem 2rem; border-radius:16px;
                            background:linear-gradient(135deg,#13131A 0%,#0F0F18 100%);
                            border:1px solid {score_color}33; box-shadow:0 0 40px {score_color}22;
                            text-align:center; position:relative; overflow:hidden; margin-bottom:1rem;">
                    <p style="font-family:'Space Mono',monospace; color:#808090; font-size:0.7rem;
                              letter-spacing:0.15em; text-transform:uppercase; margin:0 0 0.5rem;">
                        Analysis by {name}
                    </p>
                    <span class="risk-badge" style="background:{badge_bg}; color:{score_color}; border:1px solid {score_color}44;">
                        {badge_label}
                    </span>
                    <h3 style="color:#808090 !important; font-family:'DM Sans',sans-serif !important;
                               font-size:0.8rem !important; font-weight:500; letter-spacing:0.12em;
                               text-transform:uppercase; margin-bottom:0.25rem; border:none !important; padding:0 !important;">
                        Estimated CKD Risk
                    </h3>
                    <div style="font-family:'Space Mono',monospace; color:{score_color};
                                font-size:5rem; font-weight:700; line-height:1; margin:0.5rem 0 1rem;">
                        {prob:.1%}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.progress(float(prob))

        else:
            res_cols = st.columns(2)
            for i, (name, prob, pred) in enumerate(results):
                score_color = "#FF4B4B" if pred == 1 else "#28a745"
                badge_bg    = "rgba(255,75,75,0.15)" if pred == 1 else "rgba(40,167,69,0.15)"
                badge_label = "CKD DETECTED" if pred == 1 else "NO CKD DETECTED"
                with res_cols[i]:
                    st.markdown(f"""
                        <div style="padding:1.75rem 1.5rem; border-radius:14px;
                                    background:linear-gradient(135deg,#13131A 0%,#0F0F18 100%);
                                    border:1px solid {score_color}33; box-shadow:0 0 30px {score_color}18;
                                    text-align:center; margin-bottom:0.75rem;">
                            <p style="font-family:'Space Mono',monospace; color:#808090; font-size:0.65rem;
                                      letter-spacing:0.15em; text-transform:uppercase; margin:0 0 0.4rem;">
                                {name}
                            </p>
                            <span class="risk-badge" style="background:{badge_bg}; color:{score_color}; border:1px solid {score_color}44;">
                                {badge_label}
                            </span>
                            <div style="font-family:'Space Mono',monospace; color:{score_color};
                                        font-size:3.5rem; font-weight:700; line-height:1; margin:0.4rem 0 0.75rem;">
                                {prob:.1%}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.progress(float(prob))

            # ── Agreement banner ──
            st.markdown("---")
            if stack_pred == bag_pred:
                if stack_pred == 1:
                    st.error("Both models agree: **CKD Detected** — Clinical evaluation is recommended.")
                else:
                    st.success("Both models agree: **No CKD Detected**")
            else:
                st.warning("Models disagree — Further clinical evaluation is recommended.")

        # ── Tiered guidance ──
        st.markdown("---")

        avg_prob = sum(p for _, p, _ in results) / len(results)
        clamped  = max(0.0, min(1.0, float(avg_prob)))

        if clamped < 0.30:
            tier_icon   = "✅"
            tier_title  = "Your kidney health indicators look reassuring"
            tier_color  = "#28a745"
            tier_bg     = "rgba(40,167,69,0.07)"
            tier_border = "rgba(40,167,69,0.3)"
            steps = [
                ("Stay hydrated", "Drinking adequate water daily (around 2 litres for most adults) supports kidney filtration and helps prevent stone formation."),
                ("Keep blood pressure in check", "High blood pressure is the leading cause of kidney damage. Regular monitoring and a low-salt diet go a long way."),
                ("Maintain a healthy weight", "Excess weight puts additional strain on the kidneys over time. Staying active and eating a balanced diet supports long-term kidney health."),
                ("Routine check-ups", "Even at low risk, an annual kidney function test (creatinine, eGFR) is a simple and worthwhile habit, especially if you have diabetes or high blood pressure."),
            ]
        elif clamped < 0.60:
            tier_icon   = "🟡"
            tier_title  = "A few areas worth keeping an eye on"
            tier_color  = "#f0a500"
            tier_bg     = "rgba(240,165,0,0.07)"
            tier_border = "rgba(240,165,0,0.3)"
            steps = [
                ("Speak with your doctor", "These results suggest it's worth discussing your kidney health at your next appointment. A simple blood test (eGFR, creatinine) can give a clearer picture."),
                ("Monitor blood pressure closely", "Target BP below 130/80 mmHg if you have kidney concerns. Uncontrolled hypertension is one of the fastest ways kidney function declines."),
                ("Watch your diet", "Reducing sodium, processed foods, and excess protein can meaningfully reduce kidney workload. A dietitian can help you build a kidney-friendly meal plan."),
                ("Manage diabetes carefully", "If you have diabetes, tight blood sugar control is one of the most protective things you can do for your kidneys. Regular HbA1c checks are important."),
                ("Avoid nephrotoxic medications", "NSAIDs like ibuprofen and certain antibiotics can stress the kidneys. Always check with your doctor before taking these regularly."),
            ]
        else:
            tier_icon   = "🔴"
            tier_title  = "We recommend speaking with a healthcare professional promptly"
            tier_color  = "#FF4B4B"
            tier_bg     = "rgba(255,75,75,0.07)"
            tier_border = "rgba(255,75,75,0.3)"
            steps = [
                ("Book an appointment soon", "Based on these markers, we'd strongly encourage you to see a GP or nephrologist in the near future. Early-stage CKD is very manageable with the right care."),
                ("Request a full kidney panel", "Ask your doctor for a comprehensive panel — eGFR, urine ACR, creatinine, BUN, and electrolytes. This gives the clearest picture of kidney function."),
                ("Control blood pressure and blood sugar", "These are the two biggest modifiable risk factors for CKD progression. Even small improvements make a measurable difference in slowing decline."),
                ("Review all medications with your doctor", "Some medications — including common pain relievers — can worsen kidney function. Your doctor can review your current medications and suggest safer alternatives."),
                ("Dietary changes are important", "A kidney-friendly diet (lower potassium, phosphorus, and sodium) can slow progression significantly. A renal dietitian referral is worth asking about."),
            ]

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

        st.markdown(guidance_top + steps_html + "</div>", unsafe_allow_html=True)

        st.markdown("""
            <div style="background:#13131A; border:1px solid #2A2A3A; border-left:3px solid #FF4B4B;
                        border-radius:8px; padding:0.85rem 1rem; margin-top:0.5rem;">
                <p style="color:#C0C0D0; font-family:'DM Sans',sans-serif; font-size:0.82rem;
                            line-height:1.5; margin:0;">
                    <strong>Note:</strong> This tool is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.
                </p>
            </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    run_ckd_analysis()
