# Osteoporosis Risk Prediction Model (Master Pipeline)

This repository features a **comprehensive, all-in-one Master Pipeline** for osteoporosis risk prediction. It consolidates the entire machine learning workflow—from data loading to advanced model explainability—into a single, easy-to-run Google Colab notebook: `notebooks/00_MASTER_Complete_Pipeline.ipynb`.

## 🧠 Models Implemented

The pipeline trains and evaluates **12 distinct machine learning algorithms** to ensure robust performance comparison:

1.  **Logistic Regression**: Baseline linear model for binary classification.
2.  **Decision Tree**: Interpretable tree-based model.
3.  **Random Forest**: Ensemble of decision trees for reduced variance.
4.  **Gradient Boosting**: Sequential ensemble method for high accuracy.
5.  **AdaBoost**: Adaptive boosting algorithm focusing on hard-to-classify samples.
6.  **XGBoost**: Extreme Gradient Boosting, optimized for speed and performance.
7.  **Bagging Classifier**: Bootstrap Aggregating to improve stability.
8.  **Stacking Classifier**: Meta-model combining Random Forest, Gradient Boosting, and XGBoost.
9.  **K-Nearest Neighbors (KNN)**: Distance-based classification.
10. **Support Vector Machine (SVM)**: Effective in high-dimensional spaces.
11. **Extra Trees Classifier**: Extremely Randomized Trees for variance reduction.
12. **Neural Network (Deep Learning)**: Custom TensorFlow/Keras architecture for capturing non-linear relationships.

## � Key Outputs & Reports

The pipeline generates detailed analytical reports and high-quality visualizations:

### 1. Model Performance Leaderboard
*   **Ranked Table**: A comparative leaderboard ranking all 12 algorithms based on **Accuracy** and **ROC-AUC** scores.
*   **Performance CSV**: Auto-generated `model_performance_comparison.csv` containing detailed metrics (Precision, Recall, F1-Score) for every model.

### 2. Confusion Matrices
*   **Visualization**: A grid of confusion matrices for **all 12 models**, allowing for visual inspection of True Positives, False Positives, True Negatives, and False Negatives.
*   **Comparison**: Side-by-side analysis to identify which models minimize False Negatives (critical for medical diagnosis).

### 3. SHAP Explainability Analysis
Advanced interpretability plots to understand *why* the model makes specific predictions:
*   **SHAP Summary Plot**: Global view of feature importance and impact direction (e.g., how Age or Calcium Intake affects risk).
*   **SHAP Bar Plot**: Ranked importance of all risk factors.
*   **SHAP Force Plots**: Local explanations for individual patient predictions.

### 4. Training Loss Curves
*   **Loss Visualization**: Training and validation loss curves for the top-performing algorithms (including the Neural Network and XGBoost) to diagnose overfitting or underfitting.

### 5. ROC Curves
*   **AUC Analysis**: Receiver Operating Characteristic (ROC) curves plotted for all models to evaluate trade-offs between sensitivity and specificity.

---
**DSGP Group 40** | January 2026
