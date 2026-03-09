import streamlit as st
import Heart
import Osteoporosis
import diabetes
import CKD

st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

        /* ── Base ── */
        .stApp {
            background-color: #0A0A0F;
            color: #E8E8F0;
            font-family: 'DM Sans', sans-serif;
        }

        /* Subtle grid texture */
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
            margin-bottom: 1.25rem !important;
        }

        label, .stMarkdown p {
            color: #C0C0D0 !important;
            font-size: 0.875rem;
            font-weight: 500;
        }

        /* ── Model Card Buttons ── */
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
            padding: 1.1rem 1.5rem !important;
            border-radius: 10px !important;
            cursor: pointer !important;
            transition: all 0.25s ease !important;
            text-align: left !important;
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

        /* ── Info Box ── */
        div[data-testid="stInfo"] {
            background-color: #13131A !important;
            border: 1px solid #2A2A3A !important;
            border-left: 3px solid #FF4B4B !important;
            border-radius: 8px !important;
            color: #C0C0D0 !important;
        }

        /* ── Divider ── */
        hr {
            border: none !important;
            border-top: 1px solid #1E1E2E !important;
            margin: 1.5rem 0 !important;
        }

        /* ── Caption ── */
        .stCaption, small {
            color: #505060 !important;
            font-size: 0.75rem !important;
        }

        /* ── Write / paragraph text ── */
        .stMarkdown {
            color: #A0A0B0;
            line-height: 1.7;
        }

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

        /* ── Result container (for sub-modules) ── */
        .result-container {
            padding: 2rem;
            border-radius: 14px;
            background: linear-gradient(135deg, #13131A 0%, #0F0F18 100%);
            border: 1px solid #FF4B4B33;
            text-align: center;
        }

        /* ── Column card panels ── */
        div[data-testid="column"] {
            background-color: #0D0D14;
            border: 1px solid #1A1A26;
            border-radius: 12px;
            padding: 1rem !important;
        }

        /* Selectbox & inputs */
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

        .stTooltipIcon { color: #FF4B4B !important; }
        </style>
        """, unsafe_allow_html=True)

# Set page configuration
st.set_page_config(page_title="NCD Risk Prediction System", layout="wide")

# --- STATE MANAGEMENT ---
# Initialize session state for navigation if it doesn't exist
if 'page' not in st.session_state:
    st.session_state.page = 'Home'


# Callback function to change pages immediately
def nav_to(page_name):
    st.session_state.page = page_name


# HOME PAGE
if st.session_state.page == 'Home':
    st.title("Non-Communicable Disease Prediction Platform")
    st.write("""
    This project addresses the growing global burden of Non-Communicable Diseases (NCDs) by 
    developing an integrated machine learning-based web application for early risk prediction. 
    NCDs like heart disease and diabetes are responsible for over 75% of deaths globally, 
    often developing "silently" over years without apparent symptoms.
    """)

    st.info("""
    **Our Mission:** To shift NCD screening from a reactive, hospital-centric process to a 
    proactive, citizen-level assessment that empowers you with personalized health insights.
    """)

    st.write("Click on one of the models below to assess your specific risk profile:")

    # Creating a 2x2 Grid for Model Selection Buttons
    col1, col2 = st.columns(2)

    with col1:
        st.button("Heart Disease Risk Prediction",
                  on_click=nav_to, args=('Heart',), use_container_width=True)

        st.button("Diabetes (Type 2) Risk Prediction",
                  on_click=nav_to, args=('Diabetes',), use_container_width=True)

    with col2:
        st.button("Chronic Kidney Disease Prediction",
                  on_click=nav_to, args=('CKD',), use_container_width=True)

        st.button("Osteoporosis Risk Prediction",
                  on_click=nav_to, args=('Osteoporosis',), use_container_width=True)

    st.divider()
    st.caption("Note: This tool is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.")

# NAVIGATION LOGIC FOR MODELS
elif st.session_state.page == 'Heart':
    st.button("← Back to Home", on_click=nav_to, args=('Home',))
    # Check if Heart module is loaded and has the function
    if 'Heart' in globals() and hasattr(Heart, 'run_heart_analysis'):
        Heart.run_heart_analysis()
    else:
        st.error("Heart Disease module not found or function missing.")

elif st.session_state.page == 'Diabetes':
    st.button("← Back to Home", on_click=nav_to, args=('Home',))
    diabetes.run_diabetes_analysis()

elif st.session_state.page == 'CKD':
    st.button("← Back to Home", on_click=nav_to, args=('Home',))
    CKD.run_ckd_analysis()

elif st.session_state.page == 'Osteoporosis':
    st.button("← Back to Home", on_click=nav_to, args=('Home',))
    Osteoporosis.run_osteoporosis_analysis()
