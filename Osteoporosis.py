import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import os
from typing import Tuple, Dict, Any

# INPUT MAPPING CONFIGURATION
VALUE_MAPPING = {
    'Hormonal Changes': {
        'Normal': 'Normal',
        'Postmenopausal': 'Postmenopausal',
        'Perimenopausal': 'Normal',
        'Low Testosterone': 'Normal'
    },
    'Race/Ethnicity': {
        'Caucasian': 'Caucasian',
        'Asian': 'Asian',
        'African American': 'African American',
        'Hispanic': 'Caucasian',
        'Other': 'Caucasian'
    },
    'Body Weight': {
        'Normal': 'Normal',
        'Underweight': 'Underweight',
        'Overweight': 'Normal'
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
        'None': 'Unknown',
        'Moderate': 'Moderate',
        'Heavy': 'Moderate'
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

# MODEL LOADING & UTILS
def get_model_paths() -> Dict[str, str]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    return {
        'male_rf_model':   os.path.join(models_dir, 'osteoporosis_male_random_forest_model.pkl'),
        'male_ada_model':  os.path.join(models_dir, 'osteoporosis_male_adaboost_model_2nd.pkl'),
        'female_rf_model': os.path.join(models_dir, 'osteoporosis_female_random_forest_model.pkl'),
        'female_et_model': os.path.join(models_dir, 'osteoporosis_female_extra_trees_model_2nd.pkl'),
        'encoders':        os.path.join(models_dir, 'label_encoders.pkl'),
        'scaler':          os.path.join(models_dir, 'scaler.pkl')
    }

@st.cache_resource
def load_model_assets() -> Tuple[Dict[str, Any], Dict, Any]:
    paths = get_model_paths()
    try:
        models = {
            'male_rf':   joblib.load(paths['male_rf_model']),
            'male_ada':  joblib.load(paths['male_ada_model']),
            'female_rf': joblib.load(paths['female_rf_model']),
            'female_et': joblib.load(paths['female_et_model']),
        }
        label_encoders = joblib.load(paths['encoders'])
        scaler = joblib.load(paths['scaler'])
        return models, label_encoders, scaler
    except FileNotFoundError as e:
        st.error(f"Error: Model file not found at {e.filename}")
        st.info("Ensure that the 'models' folder exists in the same directory as this script and contains the .pkl files.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model assets: {str(e)}")
        st.stop()

def get_feature_names() -> list:
    return [
        'Age', 'Gender', 'Hormonal Changes', 'Family History', 'Race/Ethnicity',
        'Body Weight', 'Calcium Intake', 'Vitamin D Intake', 'Physical Activity',
        'Smoking', 'Alcohol Consumption', 'Medical Conditions', 'Medications',
        'Prior Fractures'
    ]

# UI & APPLICATION LOGIC
MALE_MODELS = {
    "Random Forest": "male_rf",
    "AdaBoost": "male_ada",
}
FEMALE_MODELS = {
    "Random Forest": "female_rf",
    "Extra Trees": "female_et",
}


def get_user_inputs():
    # Top section: gender + model selection
    gender = st.radio("Gender", options=["Male", "Female"],
                      horizontal=True,
                      help="Biological sex — women have higher risk due to lower bone density")

    if gender == "Male":
        model_display_name = st.radio(
            "Select Prediction Model", options=list(MALE_MODELS.keys()) + ["Both"],
            horizontal=True,
            help="Choose which trained male model to use for prediction."
        )
        if model_display_name == "Both":
            selected_model_keys = list(MALE_MODELS.values())
        else:
            selected_model_keys = [MALE_MODELS[model_display_name]]
    else:
        model_display_name = st.radio(
            "Select Prediction Model", options=list(FEMALE_MODELS.keys()) + ["Both"],
            horizontal=True,
            help="Choose which trained female model to use for prediction."
        )
        if model_display_name == "Both":
            selected_model_keys = list(FEMALE_MODELS.values())
        else:
            selected_model_keys = [FEMALE_MODELS[model_display_name]]

    st.markdown("---")

    # Main inputs
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=55,
                              help="Your current age in years")

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
        'Medical Conditions': medical_conditions, 'Medications': medications,
        'Prior Fractures': prior_fractures,
        '_selected_model_keys': selected_model_keys,
        '_model_display_name': model_display_name,
    }


def make_prediction(user_inputs, model_key, all_models, label_encoders, scaler):
    """Run prediction for a single model key using a copy of user_inputs."""
    # Work on a clean copy — strip internal keys
    raw = {k: v for k, v in user_inputs.items() if not k.startswith('_')}

    model_input = {}
    for col, value in raw.items():
        if col in VALUE_MAPPING:
            model_input[col] = VALUE_MAPPING[col].get(value, value)
        else:
            model_input[col] = value

    df_input = pd.DataFrame([model_input])

    for col in label_encoders.keys():
        if col in df_input.columns:
            le = label_encoders[col]
            val = df_input[col].iloc[0]
            try:
                df_input[col] = le.transform([val])
            except ValueError:
                st.warning(f"Value '{val}' not found in trained model features for '{col}'. Using default.")
                df_input[col] = 0

    expected_features = get_feature_names()
    df_input = df_input[expected_features]

    scaled_array = scaler.transform(df_input)
    df_scaled = pd.DataFrame(scaled_array, columns=expected_features)

    model = all_models[model_key]
    prediction = model.predict(df_scaled)[0]

    if hasattr(model, 'predict_proba'):
        prediction_proba = model.predict_proba(df_scaled)[0]
        risk_score = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
    else:
        raw_pred = model.predict(df_scaled)
        if hasattr(raw_pred, 'shape') and len(raw_pred.shape) > 1 and raw_pred.shape[-1] == 1:
            risk_score = float(raw_pred[0][0])
        else:
            risk_score = float(prediction)

    return prediction, risk_score

# MAIN ENTRY POINT

def run_osteoporosis_analysis():
    """Main entry point called from app.py"""

    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

        .stApp {
            background-color: #0A0A0F;
            color: #E8E8F0;
            font-family: 'DM Sans', sans-serif;
        }

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

        label, .stMarkdown p { color: #C0C0D0 !important; font-size: 0.875rem; font-weight: 500; }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            background-color: #13131A !important;
            border: 1px solid #2A2A3A !important;
            border-radius: 8px !important;
            color: #E8E8F0 !important;
            transition: border-color 0.2s ease;
        }

        div[data-baseweb="select"] > div:hover,
        div[data-baseweb="input"] > div:hover { border-color: #FF4B4B !important; }

        div[data-baseweb="select"] svg { fill: #FF4B4B !important; }

        div[data-baseweb="popover"] {
            background-color: #13131A !important;
            border: 1px solid #2A2A3A !important;
            border-radius: 8px !important;
        }

        li[role="option"]:hover { background-color: #1E1E2E !important; color: #FF4B4B !important; }

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
            transition: all 0.25s ease !important;
            box-shadow: none !important;
        }

        .stButton button:hover {
            background-color: #13131A !important;
            border-color: #FF4B4B !important;
            color: #FF4B4B !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 24px rgba(255,75,75,0.2) !important;
        }

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

        div[data-testid="stProgressBar"] > div > div {
            background-color: #FF4B4B !important;
        }

        div[data-testid="stProgressBar"] > div {
            background-color: #1E1E2E !important;
            border-radius: 4px !important;
        }

        hr { border: none !important; border-top: 1px solid #1E1E2E !important; margin: 1.5rem 0 !important; }

        div[data-testid="column"] {
            background-color: #0D0D14;
            border: 1px solid #1A1A26;
            border-radius: 12px;
            padding: 1.25rem !important;
        }

        .stTooltipIcon { color: #FF4B4B !important; }
        .stCaption, small { color: #505060 !important; font-size: 0.75rem !important; }

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

        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0A0A0F; }
        ::-webkit-scrollbar-thumb { background: #2A2A3A; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #FF4B4B; }

        .header-accent {
            display: inline-block; width: 40px; height: 3px;
            background: #FF4B4B; border-radius: 2px; margin-bottom: 0.75rem;
        }
        </style>
    """, unsafe_allow_html=True)

    all_models, label_encoders, scaler = load_model_assets()
    st.title("Osteoporosis Risk Assessment")
    st.write("Enter the following details to estimate your bone health risk.")

    with st.container():
        user_inputs = get_user_inputs()

    model_name      = user_inputs.get('_model_display_name', '')
    gender          = user_inputs.get('Gender', '')
    selected_keys   = user_inputs.get('_selected_model_keys', [])
    show_both       = model_name == "Both"

    # Resolve display names for each key
    all_display_names = {**MALE_MODELS, **FEMALE_MODELS}
    key_to_display = {v: k for k, v in all_display_names.items()}

    st.caption(f"Using **{model_name}** for {gender} prediction")

    st.markdown("---")

    if st.button("Calculate Osteoporosis Risk"):
        try:
            raw_inputs_copy = {k: v for k, v in user_inputs.items() if not k.startswith('_')}

            # Run all selected models
            results = []
            for key in selected_keys:
                pred, score = make_prediction(user_inputs, key, all_models, label_encoders, scaler)
                results.append((key_to_display.get(key, key), pred, score))

            # ── Result cards ──
            if len(results) == 1:
                name, prediction, risk_score = results[0]
                is_positive  = prediction == 1
                score_color  = "#FF4B4B" if is_positive else "#28a745"
                badge_bg     = "rgba(255,75,75,0.15)" if is_positive else "rgba(40,167,69,0.15)"
                badge_label  = "OSTEOPOROSIS INDICATED" if is_positive else "LOW RISK"

                st.markdown(f"""
                    <div style="padding:2.5rem 2rem; border-radius:16px;
                         background:linear-gradient(135deg,#13131A 0%,#0F0F18 100%);
                         border:1px solid {score_color}33;
                         box-shadow:0 0 40px {score_color}22;
                         text-align:center; position:relative; overflow:hidden; margin-bottom:1rem;">
                        <p style="font-family:'Space Mono',monospace; color:#808090; font-size:0.7rem;
                                  letter-spacing:0.15em; text-transform:uppercase; margin:0 0 0.5rem;">
                            Analysis by {name}
                        </p>
                        <span style="display:inline-block; padding:4px 14px; border-radius:20px;
                                     font-family:'Space Mono',monospace; font-size:0.7rem;
                                     letter-spacing:0.08em; text-transform:uppercase; margin-bottom:1rem;
                                     background:{badge_bg}; color:{score_color}; border:1px solid {score_color}44;">
                            {badge_label}
                        </span>
                        <h3 style="color:#808090 !important; font-family:'DM Sans',sans-serif !important;
                                   font-size:0.8rem !important; font-weight:500; letter-spacing:0.12em;
                                   text-transform:uppercase; margin-bottom:0.25rem; border:none !important; padding:0 !important;">
                            Estimated Osteoporosis Risk
                        </h3>
                        <div style="font-family:'Space Mono',monospace; color:{score_color};
                                    font-size:5rem; font-weight:700; line-height:1; margin:0.5rem 0 1rem;">
                            {risk_score*100:.1f}%
                        </div>
                        <p style="color:#A0A0B0; font-size:0.875rem; max-width:420px; margin:0 auto; line-height:1.6;">
                            {f"These results suggest signs worth discussing with your doctor. Early action makes a big difference."
                              if is_positive else
                              "Your bone health indicators look reassuring based on the provided information."}
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                st.progress(float(risk_score))

            else:
                res_cols = st.columns(2)
                for i, (name, prediction, risk_score) in enumerate(results):
                    is_positive = prediction == 1
                    score_color = "#FF4B4B" if is_positive else "#28a745"
                    badge_bg    = "rgba(255,75,75,0.15)" if is_positive else "rgba(40,167,69,0.15)"
                    badge_label = "OSTEOPOROSIS INDICATED" if is_positive else "LOW RISK"
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
                                <span style="display:inline-block; padding:4px 14px; border-radius:20px;
                                             font-family:'Space Mono',monospace; font-size:0.7rem;
                                             letter-spacing:0.08em; text-transform:uppercase; margin-bottom:0.75rem;
                                             background:{badge_bg}; color:{score_color}; border:1px solid {score_color}44;">
                                    {badge_label}
                                </span>
                                <div style="font-family:'Space Mono',monospace; color:{score_color};
                                            font-size:3.5rem; font-weight:700; line-height:1; margin:0.4rem 0 0.75rem;">
                                    {risk_score*100:.1f}%
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        st.progress(float(risk_score))

                # ── Agreement banner ──
                st.markdown("---")
                pred_a = results[0][1]
                pred_b = results[1][1]
                if pred_a == pred_b:
                    if pred_a == 1:
                        st.error("Both models agree: **Osteoporosis Indicated** — Clinical evaluation is recommended.")
                    else:
                        st.success("Both models agree: **Low Risk Detected**")
                else:
                    st.warning("Models disagree — Further clinical evaluation is recommended.")

            st.markdown("---")

            # Tiered guidance — average risk score across all selected models
            avg_score   = sum(s for _, _, s in results) / len(results)
            avg_pred    = results[0][1] if len(results) == 1 else (1 if sum(p for _, p, _ in results) > len(results) / 2 else 0)
            is_positive = avg_pred == 1
            clamped     = max(0.0, min(1.0, float(avg_score)))

            if not is_positive and clamped < 0.40:
                tier_icon   = "✅"
                tier_title  = "Your bone health indicators look good"
                tier_color  = "#28a745"
                tier_bg     = "rgba(40,167,69,0.07)"
                tier_border = "rgba(40,167,69,0.3)"
                steps = [
                    ("Keep up your calcium intake",
                     "Aim for 1000–1200mg of calcium daily through dairy, leafy greens, or fortified foods. Your current intake appears adequate — keep it up."),
                    ("Stay active",
                     "Weight-bearing activities like walking, dancing, or light resistance training are excellent for maintaining bone density over time."),
                    ("Get regular sunlight",
                     "10–15 minutes of sunlight daily supports healthy Vitamin D levels, which is essential for calcium absorption."),
                    ("Schedule routine check-ups",
                     "Even with a low risk score, a bone density check (DXA scan) is recommended for women over 50 and men over 65 as part of routine care."),
                ]
            elif not is_positive and clamped < 0.65:
                tier_icon   = "🟡"
                tier_title  = "A few areas worth keeping an eye on"
                tier_color  = "#f0a500"
                tier_bg     = "rgba(240,165,0,0.07)"
                tier_border = "rgba(240,165,0,0.3)"
                steps = [
                    ("Talk to your doctor at your next visit",
                     "Mention your bone health — your GP can assess whether a baseline DXA scan would be useful given your profile."),
                    ("Review your calcium and Vitamin D",
                     f"{'Your calcium intake appears low — try to increase it through diet or supplements.' if raw_inputs_copy.get('Calcium Intake') == 'Low' else 'Maintain your current calcium intake.'} "
                     f"{'Vitamin D deficiency is common — a simple supplement (800–1000 IU/day) can help.' if raw_inputs_copy.get('Vitamin D Intake') == 'Insufficient' else ''}"),
                    ("Add weight-bearing activity",
                     "Even 20–30 minutes of walking most days can meaningfully support bone strength. Resistance training is especially beneficial."),
                    ("Limit bone-affecting habits",
                     f"{'Smoking accelerates bone loss — quitting is one of the most impactful steps you can take.' if raw_inputs_copy.get('Smoking') == 'Yes' else ''}"
                     f"{'Excess alcohol can interfere with bone formation — moderating intake helps.' if raw_inputs_copy.get('Alcohol Consumption') == 'Heavy' else ''}"
                     "Small consistent changes add up significantly over time."),
                ]
            else:
                tier_icon   = "🔴"
                tier_title  = "We recommend speaking with a healthcare professional"
                tier_color  = "#FF4B4B"
                tier_bg     = "rgba(255,75,75,0.07)"
                tier_border = "rgba(255,75,75,0.3)"
                steps = [
                    ("Book an appointment with your doctor",
                     "Based on these markers, we'd encourage you to speak with your GP or an endocrinologist. They can arrange a DXA scan to get a clear picture of your bone density — this is a routine, painless test."),
                    ("Ask about a bone density scan (DXA)",
                     "A DXA scan is the gold standard for diagnosing osteoporosis. It's quick, non-invasive, and gives your doctor the information needed to recommend the right next steps."),
                    ("Nutrition is a priority",
                     f"{'Increasing your calcium intake to 1000–1200mg/day is important. ' if raw_inputs_copy.get('Calcium Intake') == 'Low' else ''}"
                     f"{'Vitamin D supplements (800–1000 IU/day) are often recommended and safe. ' if raw_inputs_copy.get('Vitamin D Intake') == 'Insufficient' else ''}"
                     "Your doctor can advise on whether prescription-strength supplements are needed."),
                    ("Medication options exist",
                     "Effective treatments are available — bisphosphonates and other medications can significantly slow bone loss and reduce fracture risk. There's no need to worry; these are well-established and widely used."),
                    ("Make your home safer",
                     "Simple changes like removing loose rugs, improving lighting, and using non-slip mats can meaningfully reduce fall risk while you work on your bone health."),
                ]

            steps_html = ""
            for title, desc in steps:
                if desc.strip():
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

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.error("Please ensure all inputs are valid.")

def main():
    run_osteoporosis_analysis()

if __name__ == "__main__":
    main()
