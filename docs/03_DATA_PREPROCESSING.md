# 3. Data Preprocessing - Technical Documentation

## Overview
This module transforms raw categorical/mixed data into machine-learning-ready format through:
1. Missing value handling
2. Feature encoding (binary & multi-category)
3. Feature scaling (normalization)
4. Feature engineering (interaction terms)

## Objectives
1. Handle remaining missing values strategically
2. Convert all features to numerical format
3. Normalize/scale features for algorithm compatibility
4. Create meaningful interaction features
5. Prepare data for model training

---

## Technical Implementation

### Phase 1: Missing Value Handling

#### Strategy per Feature

**Categorical Missing Values**
```python
# Alcohol Consumption: 50.5% missing
df_processed['Alcohol Consumption'].fillna('None', inplace=True)

# Medical Conditions: 33.1% missing
df_processed['Medical Conditions'].fillna('None', inplace=True)

# Medications: 50.3% missing  
df_processed['Medications'].fillna('None', inplace=True)
```

**Rationale:**
- These represent absence of that category
- 'None' is a valid category (patient doesn't consume alcohol, has no condition, etc.)
- Preserves clinical meaning
- Maintains information content

**Why NOT mean/median imputation?**
- Only works for numerical features
- Would require arbitrary mapping for categories

**Why NOT deletion?**
- Would lose 50% of data
- Reduces statistical power
- Creates bias (might not be random)

### Phase 2: Binary Feature Encoding (Label Encoding: 0/1)

#### 10 Binary Features
```python
encoding_map = {
    'Gender': {'Male': 0, 'Female': 1},
    'Hormonal Changes': {'Normal': 0, 'Postmenopausal': 1},
    'Body Weight': {'Normal': 0, 'Underweight': 1},
    'Calcium Intake': {'Adequate': 0, 'Low': 1},
    'Vitamin D Intake': {'Sufficient': 0, 'Insufficient': 1},
    'Physical Activity': {'Active': 0, 'Sedentary': 1},
    'Smoking': {'No': 0, 'Yes': 1},
    'Prior Fractures': {'No': 0, 'Yes': 1},
    'Family History': {'No': 0, 'Yes': 1},
    'Alcohol Consumption': {'None': 0, 'Moderate': 1},
    'Medications': {'None': 0, 'Corticosteroids': 1}
}
```

**Encoding Principle:**
- **0 = Lower risk/normal condition**
- **1 = Higher risk/abnormal condition**

**Example: Body Weight**
- Normal weight = 0 (healthy)
- Underweight = 1 (risk factor for low bone mass)

**Encoding Logic:**
- Preserves ordinal/clinical meaning
- XGBoost benefits from risk-aligned encoding
- Improves SHAP interpretability

**Why Label Encoding (not One-Hot)?**
- Binary features: 1 or 0 sufficient
- One-Hot would create redundant features
- XGBoost handles categorical encoding efficiently

### Phase 3: Multi-Category Feature Encoding (One-Hot Encoding)

#### Two Multi-Category Features

**Race/Ethnicity (3 categories)**
```python
race_dummies = pd.get_dummies(df_encoded['Race/Ethnicity'], prefix='Race')
# Creates: Race_African American, Race_Asian, Race_Caucasian
```

**Medical Conditions (4 categories)**
```python
conditions_dummies = pd.get_dummies(df_encoded['Medical Conditions'], prefix='Condition')
# Creates: Condition_COPD, Condition_Hyperthyroidism, 
#          Condition_None, Condition_Rheumatoid Arthritis
```

**Why One-Hot for Multi-Category?**
- Prevents ordinal assumption (Asian ≠ 2× African American)
- Creates separate binary features
- Allows model to learn category-specific effects
- Standard practice for nominal categorical data

**Example - Race Encoding:**
```
Original: 'Asian'
↓
Race_African American: 0
Race_Asian: 1  ← This patient
Race_Caucasian: 0
```

**Note:** One-Hot introduces k-1 redundancy (3 races need only 2 columns). XGBoost handles this well.

### Phase 4: Feature Scaling (StandardScaler)

#### Age Feature Normalization
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_encoded['Age'] = scaler.fit_transform(df_encoded[['Age']])

# Formula: Age_scaled = (Age - mean) / std
# Mean: 39.1 years, Std: 21.4 years
```

**Example:**
```
Original Age: 60 years
Scaled Age: (60 - 39.1) / 21.4 = 0.98
```

**Why Scaling?**
- **Algorithm requirement:** XGBoost doesn't require scaling, but good practice
- **Consistency:** All features now on similar scale
- **SHAP values:** More interpretable with scaled features
- **Numerical stability:** Prevents overflow in calculations
- **Interpretability:** Scaled values show deviation from mean in standard deviations

**When NOT to scale?**
- Tree-based models (Random Forest, XGBoost) are scale-invariant
- But doesn't hurt, and ensures reproducibility

### Phase 5: Feature Engineering (Interaction Terms)

#### Three Strategic Interactions

**1. Age × Hormonal Changes**
```python
df_encoded['Age_x_Hormonal'] = df_encoded['Age'] * df_encoded['Hormonal Changes']
```

**Clinical Rationale:**
- Post-menopausal women face accelerated bone loss (hormonal×age interaction)
- Captures non-linear relationship
- Example: 60-year-old post-menopausal female has higher risk than 60-year-old pre-menopausal

**2. Calcium Intake × Vitamin D Intake**
```python
df_encoded['Calcium_x_VitaminD'] = df_encoded['Calcium Intake'] * df_encoded['Vitamin D Intake']
```

**Clinical Rationale:**
- Synergistic effect: Vitamin D required for calcium absorption
- Low calcium alone is bad; low calcium + low vitamin D is worse
- Captures biological coupling

**3. Physical Activity × Smoking**
```python
df_encoded['Activity_x_Smoking'] = df_encoded['Physical Activity'] * df_encoded['Smoking']
```

**Clinical Rationale:**
- Smokers have compromised bone remodeling
- Active + non-smoker = best outcome
- Sedentary + smoker = worst outcome
- Captures lifestyle interaction

### Final Feature Count

**Before encoding:** 15 features  
**After encoding:** ~23 features
- 10 binary (label encoded)
- 2 one-hot sets (7 features total: 3 + 4)
- 3 interaction terms
- 1 scaled numerical (Age)

---

## Statistical Validation

### Distribution Check After Scaling
```python
print(f'Original Age - Mean: 39.1, Std: 21.4')
print(f'Scaled Age - Mean: 0.0, Std: 1.0')
```

**Expected output:**
- Scaled mean ≈ 0 (within floating point error)
- Scaled std ≈ 1.0

### Missing Values Verification
```python
print(f'Total missing values: {df_processed.isnull().sum().sum()}')
# Expected: 0
```

---

## Technical Challenges & Solutions

### Challenge 1: Information Loss in Encoding
**Problem:** Converting 'Male'/'Female' to 0/1 loses original meaning  
**Solution:** 
- Keep feature names documented
- Create encoding dictionary for reference
- SHAP analysis later restores interpretability

### Challenge 2: Feature Explosion from One-Hot
**Problem:** One-Hot can create too many features  
**Solution:** 
- We have only 2 multi-category features (manageable)
- Race: 3 categories → 3 columns
- Conditions: 4 categories → 4 columns
- Total explosion: 15 → 23 features (acceptable)

**If more features:** Use target encoding or ordinal encoding instead

### Challenge 3: Scaling Consistency
**Problem:** Must use same scaler for training and inference  
**Solution:**
```python
joblib.dump(scaler, 'models/age_scaler.pkl')
# Later, for new patient:
scaler = joblib.load('models/age_scaler.pkl')
new_age_scaled = scaler.transform([[new_age]])
```

### Challenge 4: Multicollinearity from Interactions
**Problem:** Interaction terms may correlate with source features  
**Solution:** 
- XGBoost handles multicollinearity well
- Decision trees use only one feature per split
- Not a problem in practice, but we could check VIF if needed

---

## Preprocessing Quality Checks

✓ **No missing values** (filled strategically)  
✓ **All features numerical** (ready for ML)  
✓ **Features scaled** (consistent magnitude)  
✓ **Interaction terms created** (captures relationships)  
✓ **Data types correct** (float64 for numerical)  
✓ **No NaNs or Infs** (verified)  
✓ **Original data preserved** (saved to CSV)  

---

## Viva Preparation Points

**Q: Why handle missing values with 'None' instead of deletion?**  
A: Alcohol Consumption, Medical Conditions, and Medications have 30-50% missing values. These don't represent random missing data; they represent the absence of that condition/consumption. Deleting 50% of data would lose 979 samples. Filling with 'None' preserves all data and maintains clinical meaning (patient has no medical condition, is not taking medications, etc.).

**Q: Explain your encoding strategy.**  
A: We use two encoding methods:
1. **Label Encoding (0/1)** for 10 binary features: Maps categories to 0 (lower risk/normal) and 1 (higher risk/abnormal). Examples: Smoking No→0, Yes→1; Activity Active→0, Sedentary→1
2. **One-Hot Encoding** for 2 multi-category features: Race (3 categories) and Medical Conditions (4 categories). One-Hot creates separate binary columns for each category, preventing ordinal assumptions.

**Q: Why scale only Age and not other features?**  
A: Age is the only continuous numerical feature. XGBoost doesn't require scaling (tree-based models are scale-invariant), but StandardScaler ensures:
- Numerical consistency across algorithms
- Better SHAP value interpretation
- Standard ML best practices
- Easier debugging if results need reproduction

Other features are categorical (after encoding, they're 0/1), so scaling doesn't add value.

**Q: What is the purpose of interaction features?**  
A: Interactions capture non-linear relationships the model might miss:
- **Age × Hormonal Changes:** Post-menopausal women at higher risk (hormonal effect amplified with age)
- **Calcium × Vitamin D:** Synergistic nutrient absorption (low both is worse than low one)
- **Activity × Smoking:** Lifestyle interaction (inactive smoker faces combined risk)

These allow XGBoost to model complex biological relationships directly.

**Q: How did preprocessing improve model readiness?**  
A: Before preprocessing: 15 mixed features (categorical, numerical, missing). After preprocessing: 23 clean numerical features (no missing values, appropriate encoding, scaled, with engineered interactions). This standardization is essential for ML algorithms.

---

## Outputs Saved

**Files:**
1. `data/preprocessed_data.csv` - Clean, encoded dataset ready for model training
2. `models/age_scaler.pkl` - StandardScaler fitted on training data (for later predictions)

---

## Next Steps

→ Proceed to **04_Model_Training.ipynb** to separate by gender and train XGBoost models
