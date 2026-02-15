<<<<<<< HEAD
**Project Overview**
- *Project Title:* Non-Communicable Disease Prediction using Machine Learning
- *Group:* 40
- *University:* Informatics Institute of Technology in collaboration with Robert Gordon University Aberdeen
- *Degree:* BSc (Hons) in Artificial Intelligence and Data Science
- *Supervisor:* Mr. Prashan Rathnayaka

**Team Members**
- Neelesh Jayaweera – 20241799
- Sangavitha Chandramowleeswaran – 20241812
- Anuda Hettiarachchi – 20242097
- Isum Gamage – 20242052

**Project Description**<br>
This project aims to develop an integrated web application that leverages machine learning models to provide early, multi-disease risk assessment for Non-Communicable Diseases (NCDs). The system focuses on four major NCDs:
- Heart Disease
- Chronic Kidney Disease
- Diabetes
- Osteoporosis
The application is designed to be user-friendly, providing individuals with accessible and actionable health insights to encourage proactive health management.

**Key Features**
- Multi-Disease Prediction: Four independent ML models for different NCDs
- Web-Based Interface: Intuitive dashboard for input and visualization
- Real-Time Inference: Immediate risk assessment based on user inputs
- Personalized Recommendations: Tailored health advice based on risk levels
- Explainable Results: Clear, interpretable risk presentations
=======
# Osteoporosis Risk Assessment Application 🦴

A machine learning-powered web application for predicting osteoporosis risk based on personal health data and lifestyle factors. This tool uses gender-specific Random Forest models to provide personalized risk assessments and actionable health recommendations.

---

## 🚀 Features

- **Personalized Risk Assessment**: Input age, gender, lifestyle habits, and medical history to get an instant risk prediction.
- **Gender-Specific Models**:  Utilizes separate, optimized Random Forest models for males and females to ensure high accuracy.
- **Actionable Recommendations**: Provides tailored advice on calcium intake, Vitamin D, exercise, and lifestyle changes based on your specific risk profile.
- **Interactive UI**: Clean, dark-themed interface built with Streamlit for a premium user experience.

## 🛠️ Installation & Usage

1.  **Clone the Repository** (if applicable) or download the project files.
2.  **Install Dependencies**:
    Ensure you have Python installed. Install the required libraries using pip:
    ```bash
    pip install streamlit pandas scikit-learn joblib
    ```
3.  **Run the Application**:
    Navigate to the project directory in your terminal and run:
    ```bash
    streamlit run Osteoporosis.py
    ```
4.  **Access the App**:
    The application will open automatically in your default web browser (usually at `http://localhost:8501`).

## 🧠 Model Information

The application uses pre-trained **Random Forest Classifiers** located in the `models/` directory:
-   `osteoporosis_male_random_forest_model.pkl`
-   `osteoporosis_female_random_forest_model.pkl`

> **Note on Age Sensitivity**: The trained models are highly sensitive to **Age**. Individuals over **45 years old** may frequently receive a "High Risk" prediction. This reflects the patterns learned from the training data and is intended to be conservative for older demographics.

## 📂 Project Structure

-   `Osteoporosis.py`: Main application script containing UI layout, logic, and prediction pipeline.
-   `models/`: Directory containing the trained model files (`.pkl`), label encoders, and scaler.
-   `data/`: Contains the dataset used (e.g., `osteoporosis_data.csv`).
-   `notebooks/`: Contains the Jupyter notebooks used for data analysis and model training (`MASTER_Complete_Pipeline.ipynb`).

---
**DSGP Group 40** | 2026
>>>>>>> Osteoporosis
