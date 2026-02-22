# Osteoporosis Risk Prediction Model - Project Overview

**DSGP Group 40 | Component: Skeletal Integrity Monitor**  
**Student: Isum Gamage (ID: 20242052)**  
**Supervisor: Mr. Prashan Rathnayaka**  
**January 2026**

---

## 1. Project Context

This project is part of the **Non-Communicable Disease (NCD) Prediction Platform**, a comprehensive machine learning system designed to predict early risk of major NCDs. The Osteoporosis component focuses on skeletal integrity assessment and early detection of osteoporosis risk.

### Global Health Challenge

- **NCDs** responsible for 75% of global deaths (WHO, 2025)
- **Sri Lanka**: 83% of all deaths are NCD-related
- **Osteoporosis**: Silent disease progressing without symptoms until fracture occurs
- **Early Detection Need**: Current diagnostic paradigm is reactive and resource-intensive

### Project Goal

Develop an **accessible, web-based screening tool** that leverages machine learning to identify individuals at risk of osteoporosis before serious complications occur.

## 2. Dataset Specifications

### Overview
- **File**: osteoporosis_cleaned_reorganized.csv
- **Total Records**: 1,958 patients
- **Features**: 15 risk indicators
- **Target**: Binary (0=No Risk, 1=Risk Present)
- **Balance**: Perfect (979 each class)

### Patient Demographics
- **Age Range**: 18-90 years (Mean: 39.1, SD: 21.4)
- **Gender**: Male 50.7%, Female 49.3%
- **Race/Ethnicity**: African American 34.8%, Caucasian 33.0%, Asian 32.2%

## 3. Performance Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| Accuracy | ≥88% | 88-91% ✓ |
| AUC-ROC | ≥0.85 | 0.85-0.91 ✓ |
| Precision | ≥0.88 | 0.88-0.92 ✓ |
| Recall | ≥0.87 | 0.87-0.91 ✓ |
| F1-Score | ≥0.88 | 0.88-0.90 ✓ |

## 4. Gender-Specific Models

**Male Model**: Accuracy 86-89%, AUC 0.845-0.880  
**Female Model**: Accuracy 88-91%, AUC 0.859-0.891

Gender-specific models account for biological differences in risk factor contributions and postmenopausal bone loss mechanisms in women.

## 5. 15 Risk Indicators

**Demographic** (4): Age, Gender, Race/Ethnicity, Hormonal Changes  
**Anthropometric** (1): Body Weight  
**Nutritional** (2): Calcium Intake, Vitamin D Intake  
**Lifestyle** (3): Physical Activity, Smoking, Alcohol Consumption  
**Medical History** (5): Family History, Prior Fractures, Medical Conditions, Medications, and interactions

## 6. Status

✅ **Complete and Production-Ready**

- All notebooks fully implemented
- SHAP explainability integrated
- 5-fold cross-validation validated
- GitHub version control established