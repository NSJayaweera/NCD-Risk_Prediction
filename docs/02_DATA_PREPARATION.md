# 2. Data Preparation & Exploratory Data Analysis (EDA) - Technical Documentation

## Overview
This module loads the osteoporosis dataset and performs comprehensive exploratory data analysis (EDA) to understand data characteristics, distributions, and patterns.

## Objectives
1. Load dataset from Google Drive or Colab upload
2. Perform basic dataset inspection
3. Analyze missing values and data quality
4. Examine target variable distribution
5. Analyze demographic characteristics
6. Understand risk distribution by subgroups
7. Prepare data for preprocessing

---

## Dataset Characteristics

### Source Information
- **Filename:** osteoporosis_cleaned_reorganized.csv
- **Total Records:** 1,958 patients
- **Total Features:** 16 (including ID and target)
- **Clinical Indicators:** 15 risk factors
- **Target Variable:** Osteoporosis (Binary: 0=No Risk, 1=Risk)
- **Class Balance:** Perfectly balanced (979 each class)

### Feature Categories

#### Demographic Factors (3 features)
- Age (years): 18-90
- Gender: Male/Female
- Race/Ethnicity: Asian, Caucasian, African American

#### Anthropometric (1 feature)
- Body Weight: Underweight/Normal/Overweight/Obese

#### Nutritional Status (2 features)
- Calcium Intake: Low/Adequate/High
- Vitamin D Intake: Insufficient/Sufficient/High

#### Lifestyle Factors (3 features)
- Physical Activity: Sedentary/Moderate/Active
- Smoking: Yes/No
- Alcohol Consumption: None/Moderate/Heavy

#### Medical History (6 features)
- Family History: Yes/No
- Hormonal Changes: Normal/Postmenopausal
- Prior Fractures: Yes/No
- Medical Conditions: None/Rheumatoid Arthritis/Hyperthyroidism/COPD
- Medications: None/Corticosteroids/Hormone Therapy
- Osteoporosis: Yes/No (TARGET)

---

## Technical Implementation

### 1. Dataset Loading Methods

#### Method 1: Google Drive Mount (Recommended)
```python
from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv('/content/drive/MyDrive/osteoporosis_cleaned_reorganized.csv')
```
**Advantages:** 
- Persistent storage (survives session restart)
- Easy to share/backup
- Works with large files

#### Method 2: Direct Upload
```python
from google.colab import files
uploaded = files.upload()
df = pd.read_csv('osteoporosis_cleaned_reorganized.csv')
```
**Advantages:** 
- Simpler for single use
- No authentication needed

**Note:** Uploads are temporary (lost when session ends)

### 2. Data Inspection

#### Dataset Info (df.info())
```
Output shows:
- Column names
- Data types (int64, object, float64)
- Non-null counts
- Memory usage
```

**What it tells us:**
- Categorical vs numerical columns
- Data type correctness
- Missing values overview

#### Statistical Summary (df.describe())
```
Output shows for numerical columns:
- Count: Non-null entries
- Mean: Average value
- Std: Standard deviation
- Min/Max: Range
- 25%, 50%, 75%: Quartiles
```

**Analysis:**
- Age: Mean 39.1 years (±21.4 std) → broad age range
- Balanced distribution across demographics

### 3. Missing Values Analysis

#### Calculation
```python
missing_values = df.isnull().sum()  # Count NaNs per column
missing_percentage = (missing_values / len(df)) * 100  # Calculate %
```

#### Expected Pattern in Our Data

| Feature | Missing % | Strategy |
|---------|-----------|----------|
| Alcohol Consumption | ~50.5% | Fill with 'None' (not consuming) |
| Medical Conditions | ~33.1% | Fill with 'None' (no condition) |
| Medications | ~50.3% | Fill with 'None' (not taking) |
| Others | 0% | Complete data |

**Why this pattern?**
- Categorical features have logical "absence" meaning
- Not truly "missing" but represents absence of condition/medication
- Missing at Random (MAR) mechanism

**Handling Strategy:**
- NOT random deletion (loses 50% of data)
- NOT mean/median imputation (inappropriate for categorical)
- YES: Create 'None' category (clinically meaningful)

### 4. Target Variable Distribution

#### Analysis
```python
df['Osteoporosis'].value_counts()
# Output: 0: 979, 1: 979 (perfectly balanced!)
```

**Significance:**
- **Class Balance = 50-50**: No need for SMOTE or class weights
- Both classes equally represented
- Rare in real-world data (usually imbalanced)
- Simplifies model training

#### Visualization Impact
- Equal-sized bars in bar plot
- No class imbalance bias in predictions

### 5. Demographic Analysis

#### Gender Distribution
```python
df['Gender'].value_counts()
# Output: Male: 992 (50.7%), Female: 966 (49.3%)
```

**Why important?**
- Nearly perfect 50-50 split
- Justifies gender-specific model creation
- Each gender has sufficient samples (n>900)

#### Age Distribution Analysis
```python
df.groupby('Gender')['Age'].describe()
```

**Findings:**
- Male: Mean age 39.2 years
- Female: Mean age 39.0 years
- Age ranges evenly from 18-90
- No age-gender bias

#### Race/Ethnicity Distribution
```python
df['Race/Ethnicity'].value_counts()
# Output: ~33% each (Asian, Caucasian, African American)
```

**Significance:**
- Good ethnic diversity
- No racial bias in sample
- Allows generalization across populations

### 6. Risk Distribution by Demographics

#### Gender-Risk Analysis
```python
pd.crosstab(df['Gender'], df['Osteoporosis'], normalize='index')
```

**Expected pattern:**
- Both genders have ~50% risk
- Justifies separate models (different biological mechanisms)

#### Age-Risk Analysis
```python
age_bins = [0, 30, 40, 50, 100]
df['Age_Group'] = pd.cut(df['Age'], bins=age_bins)
pd.crosstab(df['Age_Group'], df['Osteoporosis'])
```

**Expected pattern:**
- Risk increases with age
- <30 years: Low risk
- 30-40: Moderate risk
- 40-50: Higher risk
- >50: Highest risk

---

## Technical Challenges & Solutions

### Challenge 1: Google Drive Authentication
**Problem:** `PermissionError` when mounting Google Drive  
**Solution:**
- First time: Click authorization link, grant permissions
- Colab saves credential (won't ask again in same session)
- If stuck: Use direct upload instead

### Challenge 2: Large Dataset Memory Usage
**Problem:** Reading large CSV can cause memory issues  
**Solution (for larger datasets):**
```python
df = pd.read_csv('file.csv', chunksize=1000)
for chunk in df:
    # Process chunk
```
**Our dataset:** 1,958 rows = trivial (instant load)

### Challenge 3: Encoding Issues
**Problem:** Special characters in data cause decode errors  
**Solution:**
```python
df = pd.read_csv('file.csv', encoding='utf-8')  # Default, handles most cases
# If fails: try encoding='latin-1' or encoding='iso-8859-1'
```

### Challenge 4: Missing Value Interpretation
**Problem:** Unclear if NaN means "missing" or "absent"  
**Solution:** Domain knowledge + statistical context
- Alcohol Consumption: NaN = Not consuming (fill 'None')
- Medical Conditions: NaN = No condition (fill 'None')
- Never fill Age/Gender with NaN (these are always present)

---

## Statistical Insights

### Data Quality Assessment
✓ **Perfect class balance** (979-979)  
✓ **Good demographic diversity** (Gender, Race evenly distributed)  
✓ **Wide age range** (18-90 years, well-distributed)  
✓ **Logical missing patterns** (50% for optional features)  
✓ **No obvious errors** (Age in valid range, target is binary)  

### Potential Bias Concerns
✗ **Gender bias?** NO - nearly perfect 50-50  
✗ **Age bias?** NO - uniform distribution across age groups  
✗ **Ethnic bias?** NO - equal representation  
✗ **Class imbalance?** NO - perfectly balanced  

---

## Viva Preparation Points

**Q: How many samples did you use and why?**  
A: 1,958 patient records with 15 clinical features. Dataset is perfectly balanced (979 with osteoporosis risk, 979 without), allowing fair model training without class weighting or synthetic data generation.

**Q: What is the missing data problem and how did you handle it?**  
A: Three features have ~50% missing values:
- Alcohol Consumption: 50.5% missing → filled with 'None' (not consuming)
- Medical Conditions: 33.1% missing → filled with 'None' (no condition)
- Medications: 50.3% missing → filled with 'None' (not taking)

These are Missing At Random (MAR) representing absence of conditions, not random missing data. Filling with 'None' maintains clinical meaning.

**Q: How did you justify gender-specific models?**  
A: Gender distribution is nearly 50-50 (Male: 992, Female: 966). More importantly, bone metabolism differs between genders due to hormonal factors (estrogen in females, lower testosterone-related bone loss in males). Separate models capture these biological differences and improve prediction accuracy.

**Q: What EDA insights did you find?**  
A:
- Perfect class balance (no imbalance handling needed)
- Age uniformly distributed 18-90 years
- Equal ethnic representation (no bias)
- Risk increases with age (expected for osteoporosis)
- Gender-risk similar but with different underlying mechanisms

---

## Outputs Saved

**File:** `data/dataset_loaded.csv`
- Complete dataset after inspection
- Used as input to preprocessing notebook
- Preserves original data for reproducibility

---

## Next Steps

→ Proceed to **03_Data_Preprocessing.ipynb** to clean, encode, and engineer features
