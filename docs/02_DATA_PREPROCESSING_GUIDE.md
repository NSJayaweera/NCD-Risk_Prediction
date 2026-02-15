# Data Preprocessing and Feature Engineering Guide

**DSGP Group 40 | Osteoporosis Risk Prediction**  
**Student: Isum Gamage (ID: 20242052)**  
**January 2026**

---

## 1. Feature Set Overview (15 Risk Indicators)

### 1.1 Demographic Features (4 Features)

#### Feature 1: Age
- **Type**: Continuous Numerical
- **Range**: 18-90 years
- **Mean**: 39.1 years (SD: 21.4)
- **Clinical Significance**: PRIMARY predictor - Age 41+ shows 100% osteoporosis rate
- **Missing Values**: 0 (Complete)
- **Preprocessing**: StandardScaler normalization

#### Feature 2: Gender
- **Type**: Binary Categorical
- **Values**: Male (992, 50.7%), Female (966, 49.3%)
- **Clinical Significance**: Determines gender-specific model routing
- **Missing Values**: 0 (Complete)
- **Encoding**: Label Encoding (Male=0, Female=1)

#### Feature 3: Race/Ethnicity
- **Type**: Multi-category Categorical
- **Categories**: 
  - African American: 681 (34.8%)
  - Caucasian: 646 (33.0%)
  - Asian: 631 (32.2%)
- **Clinical Significance**: Different baseline bone density across populations
- **Missing Values**: 0 (Complete)
- **Encoding**: One-Hot Encoding

#### Feature 4: Hormonal Changes
- **Type**: Binary Categorical
- **Values**: Normal (981, 50.1%), Postmenopausal (977, 49.9%)
- **Clinical Significance**: Postmenopausal = estrogen deficiency → accelerated bone loss
- **Missing Values**: 0 (Complete)
- **Encoding**: Label Encoding (Normal=0, Postmenopausal=1)

### 1.2 Anthropometric Features (1 Feature)

#### Feature 5: Body Weight
- **Type**: Binary Categorical
- **Categories**: Normal (1,027, 52.5%), Underweight (931, 47.5%)
- **Clinical Significance**: Low weight → less skeletal loading → lower bone density
- **Missing Values**: 0 (Complete)
- **Encoding**: Label Encoding (Normal=0, Underweight=1)

### 1.3 Nutritional Features (2 Features)

#### Feature 6: Calcium Intake
- **Type**: Binary Categorical
- **Values**: Low (1,004, 51.3%), Adequate (954, 48.7%)
- **Clinical Significance**: Primary mineral for bone formation; deficiency → bone resorption
- **Recommended**: 1000-1200 mg/day for adults
- **Missing Values**: 0 (Complete)
- **Encoding**: Label Encoding (Low=0, Adequate=1)

#### Feature 7: Vitamin D Intake
- **Type**: Binary Categorical
- **Values**: Sufficient (1,011, 51.6%), Insufficient (947, 48.4%)
- **Clinical Significance**: Critical for calcium absorption; deficiency → poor calcium utilization
- **Recommended**: Serum 25-OH-D level 30-50 ng/mL
- **Missing Values**: 0 (Complete)
- **Encoding**: Label Encoding (Insufficient=0, Sufficient=1)

### 1.4 Lifestyle Features (3 Features)

#### Feature 8: Physical Activity
- **Type**: Binary Categorical
- **Values**: Active (1,021, 52.1%), Sedentary (937, 47.9%)
- **Clinical Significance**: Weight-bearing exercise stimulates bone formation
- **Definition**: Active = regular weight-bearing exercise (walking, running, resistance training)
- **Missing Values**: 0 (Complete)
- **Encoding**: Label Encoding (Sedentary=0, Active=1)

#### Feature 9: Smoking Status
- **Type**: Binary Categorical
- **Values**: Yes (982, 50.2%), No (976, 49.8%)
- **Clinical Significance**: Accelerates bone loss via reduced blood supply and calcium absorption
- **Missing Values**: 0 (Complete)
- **Encoding**: Label Encoding (No=0, Yes=1)

#### Feature 10: Alcohol Consumption ⚠️ **CRITICAL PREPROCESSING POINT**
- **Type**: Binary Categorical with Missing Data
- **Values**: Moderate (970, 49.5%), NaN (988, 50.5%)
- **Clinical Significance**: Excessive consumption (>2 drinks/day) interferes with bone remodeling
- **Missing Values**: 988 entries (50.5%) - REQUIRES IMPUTATION
- **Imputation Strategy**: 
  - **Recommended**: Create "None" category for missing (indicates non-drinkers)
  - **Alternative**: Mode imputation → "Moderate"
  - **Rationale**: Missing values likely represent "None" (non-drinkers not reported)
- **Encoding**: Label Encoding (None=0, Moderate=1)

### 1.5 Medical History Features (5 Features)

#### Feature 11: Family History
- **Type**: Binary Categorical
- **Values**: Yes (960, 49.0%), No (998, 51.0%)
- **Clinical Significance**: First-degree relatives with osteoporosis/fractures indicate genetic predisposition
- **Missing Values**: 0 (Complete)
- **Encoding**: Label Encoding (No=0, Yes=1)

#### Feature 12: Prior Fractures
- **Type**: Binary Categorical
- **Values**: Yes (983, 50.2%), No (975, 49.8%)
- **Clinical Significance**: STRONG predictor - increases future fracture risk 2-3x
- **Missing Values**: 0 (Complete)
- **Encoding**: Label Encoding (No=0, Yes=1)

#### Feature 13: Medical Conditions ⚠️ **CRITICAL PREPROCESSING POINT**
- **Type**: Multi-category Categorical with Missing Data
- **Categories**:
  - Hyperthyroidism: 678 (34.6%)
  - Rheumatoid Arthritis: 633 (32.3%)
  - None/NaN: 647 (33.1%)
- **Clinical Significance**:
  - **Hyperthyroidism**: Excess thyroid hormones increase bone turnover
  - **Rheumatoid Arthritis**: Chronic inflammation damages bone tissue
- **Missing Values**: 647 entries (33.1%)
- **Imputation Strategy**: Treat NaN as "None" category (no relevant medical condition)
- **Encoding**: One-Hot Encoding
  - Hyperthyroidism: [1, 0, 0]
  - Rheumatoid Arthritis: [0, 1, 0]
  - None: [0, 0, 1]

#### Feature 14: Medications ⚠️ **CRITICAL PREPROCESSING POINT**
- **Type**: Binary Categorical with Missing Data
- **Values**: Corticosteroids (973, 49.7%), NaN (985, 50.3%)
- **Clinical Significance**: Long-term corticosteroid use is MAJOR cause of drug-induced osteoporosis
- **Mechanism**: Inhibit bone formation and increase bone resorption
- **Missing Values**: 985 entries (50.3%)
- **Imputation Strategy**: Treat NaN as "None" category (not on bone-damaging medications)
- **Encoding**: Label Encoding (None=0, Corticosteroids=1)

---

## 2. Missing Value Analysis and Handling

### 2.1 Missing Data Summary

| Feature | Missing Count | Missing % | Strategy | Priority |
|---------|---------------|-----------|----------|----------|
| Alcohol Consumption | 988 | 50.5% | Create "None" category | CRITICAL |
| Medical Conditions | 647 | 33.1% | Treat NaN as "None" | CRITICAL |
| Medications | 985 | 50.3% | Treat NaN as "None" | CRITICAL |
| Others | 0 | 0% | No action | - |

### 2.2 Imputation Rationale

**Alcohol Consumption (50.5% missing)**
- **Assumption**: Missing values represent non-drinkers (None)
- **Justification**: In health surveys, non-drinkers often left blank
- **Implementation**: Create third category "None"
- **Impact**: Introduces 3-level categorical feature

**Medical Conditions (33.1% missing)**
- **Assumption**: NaN represents absence of listed conditions
- **Justification**: Patients without these conditions would naturally be absent
- **Implementation**: Treat NaN as "None" category
- **Impact**: Creates balanced 3-level categorical (HTN, RA, None)

**Medications (50.3% missing)**
- **Assumption**: NaN represents not on corticosteroids
- **Justification**: Only specific medication class tracked; NaN = not on any
- **Implementation**: Treat NaN as "None" category
- **Impact**: Creates 2-level categorical (Corticosteroids, None)

---

## 3. Feature Encoding Methodology

### 3.1 Encoding Strategy by Feature Type

```
Binary Features (10 features) → Label Encoding (0/1)
├─ Gender
├─ Hormonal Changes
├─ Body Weight
├─ Calcium Intake
├─ Vitamin D Intake
├─ Physical Activity
├─ Smoking Status
├─ Family History
└─ Prior Fractures

Multi-category Features (4 features) → One-Hot Encoding
├─ Race/Ethnicity (3 categories)
├─ Medical Conditions (3 categories after imputation)
└─ Alcohol Consumption (3 categories after imputation)

Numerical Features (1 feature) → StandardScaler Normalization
└─ Age
```

### 3.2 Detailed Encoding

**Label Encoding (Binary)**
```python
Gender: Male=0, Female=1
Hormonal Changes: Normal=0, Postmenopausal=1
Body Weight: Normal=0, Underweight=1
Calcium Intake: Low=0, Adequate=1
Vitamin D Intake: Insufficient=0, Sufficient=1
Physical Activity: Sedentary=0, Active=1
Smoking Status: No=0, Yes=1
Family History: No=0, Yes=1
Prior Fractures: No=0, Yes=1
```

**One-Hot Encoding (Multi-category)**
```python
Race/Ethnicity:
  African American → [1, 0, 0]
  Caucasian → [0, 1, 0]
  Asian → [0, 0, 1]

Medical Conditions:
  Hyperthyroidism → [1, 0, 0]
  Rheumatoid Arthritis → [0, 1, 0]
  None → [0, 0, 1]

Alcohol Consumption:
  None → [1, 0, 0]
  Moderate → [0, 1, 0]
  Heavy → [0, 0, 1]
```

**StandardScaler (Numerical)**
```python
Age: (Age - mean) / std_dev
Mean: 39.1
Standard Deviation: 21.4
Scaled Range: Approximately -1.8 to 2.4
```

---

## 4. Feature Engineering

### 4.1 Interaction Terms

**Clinical Interactions**:
```
Age × Hormonal Changes
  → Captures accelerated risk in older postmenopausal women
  → Higher weight in females 40+

Calcium Intake × Vitamin D Intake
  → Combined nutritional risk
  → Both critical for bone mineralization

Physical Activity × Age
  → Age-dependent benefit of exercise
  → Decreasing impact in advanced age

Smoking × Age
  → Cumulative smoking burden effect
  → More impactful in older age
```

### 4.2 Composite Risk Scores

**Nutritional Risk Index**
```
Nutrition_Risk = (1 - Calcium_Score) + (1 - VitaminD_Score)
Range: 0 (optimal) to 2 (poor nutrition)
```

**Lifestyle Risk Index**
```
Lifestyle_Risk = Smoking + (1 - Activity_Score) + Alcohol_Risk
Range: 0 (healthy) to 3 (multiple lifestyle risks)
```

**Medical Risk Index**
```
Medical_Risk = Prior_Fractures + (Family_History × 0.5) + 
               Medical_Conditions_Score + Medications_Score
Range: 0 (no history) to variable (multiple conditions)
```

---

## 5. Gender-Specific Preprocessing

### 5.1 Data Separation

```python
Male Cohort: 992 patients (50.7%)
Female Cohort: 966 patients (49.3%)

Total: 1,958 patients

# Each cohort trained with independent pipeline
# Same feature engineering but separate model fitting
```

### 5.2 Why Gender-Specific Preprocessing?

1. **Different Feature Weights**: Risk factors weighted differently in males vs. females
2. **Hormonal Mechanisms**: Postmenopausal bone loss unique to females
3. **Age-Risk Relationship**: Different slopes for males and females
4. **Clinical Thresholds**: Different intervention thresholds by gender

---

## 6. Data Quality Checks

### 6.1 Pre-processing Validation

- ✅ No duplicate patient records
- ✅ All demographic features present
- ✅ Target variable balanced (979 per class)
- ✅ No impossible values (e.g., age < 0)
- ✅ Feature distributions reasonable

### 6.2 Post-processing Validation

- ✅ No NaN remaining after imputation
- ✅ All numerical features standardized (mean ≈ 0, std ≈ 1)
- ✅ Categorical features properly encoded
- ✅ Feature dimensions match model input (25-30 features after encoding)

---

## 7. Pipeline Summary

```
Raw Dataset (1,958 × 15)
         ↓
1. Missing Value Imputation
   - Alcohol: Create "None" category
   - Medical Conditions: Treat NaN as "None"
   - Medications: Treat NaN as "None"
         ↓
2. Feature Encoding
   - Label Encoding (Binary features)
   - One-Hot Encoding (Multi-category)
         ↓
3. Gender-Based Separation
   - Male cohort (992)
   - Female cohort (966)
         ↓
4. Feature Scaling
   - StandardScaler on numerical features
         ↓
5. Feature Engineering
   - Interaction terms
   - Composite risk scores
         ↓
6. Train-Test Split (80-20)
   - Stratified split
   - Maintains class balance
         ↓
Processed Data Ready for XGBoost Training
```

---

**Status**: ✅ Complete  
**Last Updated**: January 17, 2026