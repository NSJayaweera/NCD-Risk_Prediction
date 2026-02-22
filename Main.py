import streamlit as st
import Heart
import Osteoporosis

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
    st.header("Diabetes (Type 2) Risk Prediction")
    st.write("This model uses parameters like blood glucose and BMI to identify high-risk individuals.")

elif st.session_state.page == 'CKD':
    st.button("← Back to Home", on_click=nav_to, args=('Home',))
    st.header("Chronic Kidney Disease (CKD) Prediction")
    st.write("This model predicts the likelihood of CKD and assesses the potential need for dialysis.")

elif st.session_state.page == 'Osteoporosis':
    st.button("← Back to Home", on_click=nav_to, args=('Home',))
    Osteoporosis.run_osteoporosis_analysis()