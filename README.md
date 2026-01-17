# Osteoporosis Risk Prediction Model (DSGP Group 40)

This repository contains a clean, step-by-step implementation of the **Osteoporosis Risk Prediction Model** using Google Colab notebooks only. The structure is designed so that anyone can follow the pipeline easily without worrying about local Python setup.

## 📁 Repository Structure

```text
notebooks/
  01_Environment_Setup.ipynb      # Install libraries + configure Colab
  02_Data_Preparation.ipynb       # Load dataset + EDA
  03_Data_Preprocessing.ipynb     # Missing values, encoding, scaling, feature engineering
  04_Model_Training.ipynb         # Gender-specific XGBoost training (Male/Female models)
  05_Model_Evaluation.ipynb       # Metrics, confusion matrix, ROC, feature importance
  06_SHAP_Explainability.ipynb    # SHAP analysis + interpretability plots

data/
  dataset_loaded.csv              # Saved after EDA (created by notebook 02)
  preprocessed_data.csv           # Saved after preprocessing (created by notebook 03)

models/
  osteoporosis_male_model.pkl     # Trained male model (created by notebook 04)
  osteoporosis_female_model.pkl   # Trained female model (created by notebook 04)
  age_scaler.pkl                  # StandardScaler for Age feature
  training_data.pkl               # Helper object for evaluation + SHAP
  shap_explainers.pkl             # SHAP explainers and values

figures/
  confusion_matrices.png
  roc_curves.png
  performance_comparison.png
  feature_importance.png
  shap_summary_male.png
  shap_summary_female.png
  shap_force_male.png
  shap_force_female.png

outputs/
  model_performance_comparison.csv
  male_feature_importance.csv
  female_feature_importance.csv
```

> Note: Most files under `data/`, `models/`, `figures/`, and `outputs/` are **generated inside Colab** when you run the notebooks.

## 🚀 How to Use (Google Colab Only)

1. **Open the repository in GitHub**  
   https://github.com/isumenuka/osteoporosis-risk-prediction

2. **Open each notebook in Google Colab**
   - Click on a notebook (e.g., `01_Environment_Setup.ipynb`)
   - Click the **"Open in Colab"** button (or copy the GitHub URL into Colab > File > Open Notebook > GitHub)

3. **Run notebooks in order**

   1. `01_Environment_Setup.ipynb`  
      Installs all required libraries (`xgboost`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `shap`, `joblib`) and creates folders.

   2. `02_Data_Preparation.ipynb`  
      - Load `osteoporosis_cleaned_reorganized.csv` from Google Drive or local upload
      - Perform basic EDA, demographic analysis, and save `data/dataset_loaded.csv`

   3. `03_Data_Preprocessing.ipynb`  
      - Handle missing values
      - Encode binary and multi-category features
      - Scale **Age** using `StandardScaler`
      - Create interaction features (e.g., `Age_x_Hormonal`)
      - Save `data/preprocessed_data.csv` and `models/age_scaler.pkl`

   4. `04_Model_Training.ipynb`  
      - Split data into male and female cohorts
      - Train two XGBoost models (Male/Female)
      - Save models to `models/osteoporosis_male_model.pkl` and `models/osteoporosis_female_model.pkl`

   5. `05_Model_Evaluation.ipynb`  
      - Evaluate accuracy, precision, recall, F1, AUC-ROC
      - Generate confusion matrices and ROC curves
      - Save performance tables and plots

   6. `06_SHAP_Explainability.ipynb`  
      - Run SHAP summary plots, bar plots, and force plots
      - Save explainers to `models/shap_explainers.pkl`

## ✅ Design Goals

- **Google Colab–First**: No local Python or virtualenv setup is required.
- **Separation of Concerns**: Each notebook handles one clear stage: setup → EDA → preprocessing → training → evaluation → explainability.
- **Reproducible Pipeline**: All intermediate artefacts are saved (`data/`, `models/`, `outputs/`, `figures/`).
- **Explainable ML**: SHAP analysis integrated as a first-class step for clinical transparency.

## 📦 Dataset

- File: `osteoporosis_cleaned_reorganized.csv`
- Location: Keep it in your **Google Drive** or upload directly inside Colab.
- Characteristics: 1,958 records, 15 clinically meaningful risk indicators, perfectly balanced classes (0/1).

## 🔗 Useful References

The logic of these notebooks follows your detailed documentation:
- *Osteoporosis_Component_Documentation.docx*
- *Osteoporosis_Model_Complete_Guide.pdf*

If you update your methodology there, we can easily sync the notebooks.

---

If you want, I can later add:
- A separate notebook for **inference + simple web form demo**
- A notebook for **GitHub → Colab workflow** (clone repo, run directly from GitHub)
