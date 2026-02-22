# 1. Environment Setup - Technical Documentation

## Overview
This module sets up the Google Colab environment with all necessary dependencies for the osteoporosis risk prediction model.

## Objectives
1. Install all required Python libraries
2. Verify library versions and compatibility
3. Configure visualization settings
4. Create directory structure for model artifacts
5. Check GPU availability (optional acceleration)

---

## Libraries Installed & Why

### Core Machine Learning Stack

| Library | Version | Purpose | Why Chosen |
|---------|---------|---------|------------|
| **XGBoost** | Latest | Gradient boosting classifier | Superior performance (AUC 0.85-0.91) for binary classification, handles missing values well, built-in regularization |
| **scikit-learn** | Latest | ML preprocessing & metrics | Industry standard for preprocessing (StandardScaler, LabelEncoder, train_test_split), comprehensive evaluation metrics |
| **pandas** | Latest | Data manipulation | Essential for DataFrame operations, handles CSV I/O, data filtering by gender |
| **NumPy** | Latest | Numerical computing | Efficient array operations, mathematical functions for feature engineering |
| **SHAP** | Latest | Model explainability | Provides Shapley values for individual prediction explanations (critical for healthcare) |
| **joblib** | Latest | Model serialization | Save/load trained models efficiently, handle large objects |
| **Matplotlib & Seaborn** | Latest | Visualization | Create publication-quality plots, confusion matrices, ROC curves |
| **TensorFlow** | Latest | GPU detection | Check GPU availability for potential acceleration |
| **SciPy** | Latest | Statistical analysis | Additional statistical tests if needed |

---

## Technical Implementation

### 1. Library Installation (pip)
```python
# Used -q flag for quiet output (less clutter in Colab)
!pip install -q xgboost scikit-learn pandas numpy matplotlib seaborn shap joblib scipy tensorflow
```

**Why quiet mode?** Colab output can be overwhelming with verbose pip output. `-q` keeps it clean.

### 2. Import Verification
After installation, we import each library to verify:
- ✓ Successful installation
- ✓ No dependency conflicts
- ✓ Compatible versions

### 3. Configuration for Colab Environment

#### Matplotlib/Seaborn Settings
```python
sns.set_style("whitegrid")  # Professional styling with gridlines
sns.set_palette("husl")     # Perceptually uniform color palette
plt.rcParams['figure.figsize'] = (12, 6)  # Default figure size
plt.rcParams['font.size'] = 10            # Readable font
```

**Why?** Colab resets settings between cells. Pre-configuring ensures consistent, professional visualizations.

#### GPU Detection
```python
import tensorflow as tf
gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
```

**Why?** Google Colab provides free GPU. Using it accelerates XGBoost training:
- Without GPU: ~30-60 seconds per model
- With GPU: ~5-10 seconds per model (but XGBoost CPU is already very fast)

### 4. Directory Structure
```python
directories = ['models', 'outputs', 'figures', 'data']
for directory in directories:
    os.makedirs(directory, exist_ok=True)
```

**Rationale:**
- `models/`: Store trained .pkl files for reuse
- `outputs/`: CSV results, metrics, feature importance
- `figures/`: Visualizations (confusion matrices, ROC, SHAP plots)
- `data/`: Intermediate CSV files (loaded, preprocessed)

---

## Technical Challenges & Solutions

### Challenge 1: Library Version Conflicts
**Problem:** Different libraries may have incompatible version requirements  
**Solution:** Use latest stable versions (pip installs latest by default)  
**Mitigation:** If conflict occurs, use specific version pinning:
```python
!pip install xgboost==2.0.0 scikit-learn==1.3.2
```

### Challenge 2: Memory Limitations in Colab
**Problem:** Large dataset operations may cause OOM errors  
**Solution:** 
- Use Colab's GPU runtime (12GB VRAM)
- Process data in chunks if needed
- Our 1,958-row dataset is very manageable

### Challenge 3: GPU Availability Variability
**Problem:** Colab doesn't guarantee GPU availability  
**Solution:** 
- Check GPU availability in notebook
- Code works with or without GPU (XGBoost CPU is fast)
- GPU mainly benefits deep learning, not XGBoost

---

## Possibilities & Capabilities After Setup

✓ Can load CSV files from Google Drive or local upload  
✓ Can process 1000s of rows efficiently  
✓ Can generate publication-quality visualizations  
✓ Can save and load trained models  
✓ Can calculate SHAP values for 100s of samples  
✓ Can perform cross-validation and hyperparameter tuning  
✓ Can export results to CSV/PNG  

## Impossibilities & Limitations

✗ Cannot use local Python imports (must be pip packages)  
✗ Cannot use GPU-optimized libraries without installing them first  
✗ Cannot exceed Colab's runtime limit (~12 hours per session)  
✗ Cannot access local machine file system (use Google Drive instead)  
✗ Cannot install system-level packages without special commands  

---

## Viva Preparation Points

**Q: Why do you need to set up the environment in Colab?**  
A: Because Google Colab provides a free cloud computing environment with pre-installed Python, but we need specific ML libraries (XGBoost, SHAP, scikit-learn) that don't come by default. Setting up ensures reproducibility and all necessary tools are available.

**Q: What are the key libraries and their purposes?**  
A:
- **XGBoost**: Our main classifier algorithm (chosen for high AUC and interpretability)
- **scikit-learn**: Data preprocessing (scaling, encoding) and evaluation metrics
- **SHAP**: Model explainability (critical for healthcare ML)
- **pandas/NumPy**: Data manipulation and numerical operations
- **Matplotlib/Seaborn**: Visualization of results

**Q: Why use GPU?**  
A: GPU accelerates XGBoost training. While not strictly necessary for our small dataset, it demonstrates best practices for ML at scale. Modern ML requires GPU consideration.

**Q: What could go wrong during setup?**  
A: Library conflicts (rare), network issues (pip), or Colab environment resets. Solutions: use specific versions, retry pip install, restart runtime.

---

## Expected Output

When this notebook runs successfully:

```
============================================================
LIBRARY VERSIONS
============================================================
XGBoost version: 2.0.0
Pandas version: 2.0.3
NumPy version: 1.24.3
SHAP version: 0.43.0
============================================================
✓ All libraries successfully imported!
✓ Visualization settings configured!
✓ GPU is available for acceleration!
✓ Created directory: models
✓ Created directory: outputs
✓ Created directory: figures
✓ Created directory: data

✓ Environment setup complete! Ready to proceed to data preparation.
```

---

## Next Steps

→ Proceed to **02_Data_Preparation.ipynb** to load and explore the dataset
