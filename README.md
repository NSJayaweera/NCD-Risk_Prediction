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
