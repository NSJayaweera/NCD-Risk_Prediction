# Quick Reference Guide

**Osteoporosis Risk Prediction Model - Fast Lookup**

---

## 📄 Documentation Quick Links

| Document | Purpose | Read Time |
|----------|---------|----------|
| [00_TABLE_OF_CONTENTS.md](00_TABLE_OF_CONTENTS.md) | Master index & navigation | 15 min |
| [01_PROJECT_OVERVIEW.md](01_PROJECT_OVERVIEW.md) | Project background | 20 min |
| [02_DATA_PREPROCESSING_GUIDE.md](02_DATA_PREPROCESSING_GUIDE.md) | Data handling | 25 min |
| [03_MODEL_TRAINING_GUIDE.md](03_MODEL_TRAINING_GUIDE.md) | Model training | 30 min |
| [04_SHAP_EXPLAINABILITY_GUIDE.md](04_SHAP_EXPLAINABILITY_GUIDE.md) | Feature importance | 25 min |
| [05_DASHBOARD_DEPLOYMENT_GUIDE.md](05_DASHBOARD_DEPLOYMENT_GUIDE.md) | Deployment | 25 min |
| [06_CLINICAL_VALIDATION_AND_RESULTS.md](06_CLINICAL_VALIDATION_AND_RESULTS.md) | Clinical validation | 30 min |

---

## 📊 15 Clinical Risk Indicators

### Demographics (4 features)
1. **Age** (18-90 years) - STRONGEST predictor
2. **Gender** (Male/Female) - Routes to specific model
3. **Race/Ethnicity** (African American, Caucasian, Asian)
4. **Hormonal Status** (Normal/Postmenopausal) - Female-specific

### Anthropometry (1 feature)
5. **Body Weight** (Normal/Underweight)

### Nutrition (2 features)
6. **Calcium Intake** (Low/Adequate)
7. **Vitamin D Intake** (Insufficient/Sufficient)

### Lifestyle (3 features)
8. **Physical Activity** (Sedentary/Active)
9. **Smoking Status** (Yes/No)
10. **Alcohol Consumption** (None/Moderate/Heavy) - 50% missing

### Medical History (5 features)
11. **Family History** (Yes/No)
12. **Prior Fractures** (Yes/No)
13. **Medical Conditions** (None/HTN/RA) - 33% missing
14. **Medications** (None/Corticosteroids) - 50% missing
15. **[Reserved for future features]**

---

## 📈 Model Performance at a Glance

### Male Model (992 patients)
```
Accuracy:    85%  | Precision: 83%  | Recall:  87%
F1-Score:    0.85 | AUC-ROC:   0.91 | Test samples: 198
```

### Female Model (966 patients)
```
Accuracy:    86%  | Precision: 84%  | Recall:  88%
F1-Score:    0.86 | AUC-ROC:   0.92 | Test samples: 193
```

### 5-Fold Cross-Validation
```
Male:   Mean AUC = 0.91 ± 0.02 (robust)
Female: Mean AUC = 0.92 ± 0.02 (robust)
```

---

## 🎨 Top 5 Feature Importance (SHAP Values)

### Male Model
```
1. Age                 (0.185)  ████████████ STRONGEST
2. Prior Fractures     (0.142)  ████████
3. Smoking Status      (0.098)  █████
4. Family History      (0.076)  ████
5. Physical Activity   (0.064)  ███
```

### Female Model
```
1. Age                 (0.198)  ████████████ STRONGEST
2. Hormonal Changes    (0.167)  █████████ GENDER-SPECIFIC
3. Prior Fractures     (0.148)  ████████
4. Smoking Status      (0.102)  █████
5. Family History      (0.082)  ████
```

---

## 📊 Risk Stratification

```
RISK LEVEL      PROBABILITY    CLINICAL ACTION
─────────────────────────────────────────────
Low Risk        < 30%          Routine monitoring
Moderate        30-60%         Annual screening
High Risk       60-80%         DEXA + specialist
Very High       > 80%          Urgent evaluation
```

---

## 👨‍💻 API Endpoints

### Main Prediction
```
POST /v1/predict
Input: Patient data (15 features)
Output: Probability, risk category, feature importance, recommendations
Auth: JWT Bearer token
```

### Health Check
```
GET /v1/health
Output: Server status, models loaded, versions
Auth: None
```

### Model Metrics
```
GET /v1/metrics
Output: Performance metrics for both models
Auth: JWT Bearer token
```

---

## 🚀 Deployment Checklist

### Pre-Deployment
- [ ] Models trained and tested
- [ ] SHAP values calculated
- [ ] Frontend built and tested
- [ ] Backend API tested locally
- [ ] Environment variables configured
- [ ] Security: HIPAA audit complete
- [ ] Database: Migrations run
- [ ] Documentation: Updated

### Docker Deployment
```bash
# Build
docker build -t osteoporosis-api:1.0 .

# Run
docker run -p 5000:5000 \
  -e FLASK_ENV=production \
  -e SECRET_KEY=your_secret \
  osteoporosis-api:1.0
```

### AWS ECS Deployment
```bash
# Push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com
docker tag osteoporosis-api:1.0 <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/osteoporosis-api:1.0
docker push <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/osteoporosis-api:1.0

# Update ECS service
aws ecs update-service --cluster osteoporosis-cluster --service osteoporosis-api --force-new-deployment
```

### Post-Deployment
- [ ] Health check: GET /v1/health returns 200
- [ ] Sample prediction: Test with known patient
- [ ] Monitoring: Prometheus metrics visible
- [ ] Logging: CloudWatch logs streaming
- [ ] Performance: API response <500ms
- [ ] Security: HTTPS/TLS enabled
- [ ] Database: Backup configured

---

## 🔍 Troubleshooting

### Model Not Loading
```
ERROR: FileNotFoundError: osteoporosis_male_model.pkl

SOLUTION:
1. Check model file exists: models/osteoporosis_male_model.pkl
2. Verify path in app configuration
3. Run 04_Model_Training.ipynb first
```

### API Returns 400 Bad Request
```
ERROR: Invalid age: must be between 18-90

SOLUTION:
1. Check all 15 features provided
2. Validate value ranges
3. Check required fields not null
```

### SHAP Values Missing
```
ERROR: shap_values not calculated

SOLUTION:
1. Ensure 06_SHAP_Explainability.ipynb ran
2. Check explainer objects saved
3. Verify SHAP version compatible with XGBoost
```

### Prediction Latency High (>1000ms)
```
ERROR: Slow API response

SOLUTION:
1. Check CPU/memory usage
2. Profile predict function
3. Consider model quantization
4. Batch predictions if possible
```

---

## 📄 Command Reference

### Training
```bash
# Run training pipeline
python notebooks/04_Model_Training.ipynb

# Generate SHAP values
python notebooks/06_SHAP_Explainability.ipynb
```

### Deployment
```bash
# Start development server
python app/backend.py

# Start with gunicorn (production)
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Docker build
docker build -t osteoporosis-api:1.0 .

# Run locally
local npm start  # Frontend
python app.py    # Backend
```

### Testing
```bash
# Validate data
python src/preprocessing.py

# Test prediction
python -c "from src.prediction import predict; predict(sample_data)"

# Run unit tests
pytest tests/
```

---

## 📚 File Locations

```
Models:
  models/osteoporosis_male_model.pkl
  models/osteoporosis_female_model.pkl
  models/scaler.pkl

Data:
  data/osteoporosis_cleaned_reorganized.csv

Notebooks:
  notebooks/01_Data_Exploration.ipynb
  notebooks/02_Data_Preprocessing.ipynb
  notebooks/03_Feature_Engineering.ipynb
  notebooks/04_Model_Training.ipynb
  notebooks/06_SHAP_Explainability.ipynb

Documentation:
  docs/00_TABLE_OF_CONTENTS.md
  docs/01_PROJECT_OVERVIEW.md
  docs/02_DATA_PREPROCESSING_GUIDE.md
  docs/03_MODEL_TRAINING_GUIDE.md
  docs/04_SHAP_EXPLAINABILITY_GUIDE.md
  docs/05_DASHBOARD_DEPLOYMENT_GUIDE.md
  docs/06_CLINICAL_VALIDATION_AND_RESULTS.md
```

---

## 📎 Key Contacts

**Project Lead**: Isum Gamage (ID: 20242052)  
**Institution**: DSGP Group 40  
**Repository**: https://github.com/isumenuka/osteoporosis-risk-prediction

---

## ⏱ Critical Timelines

```
Model Training:          ~30-45 min
Cross-Validation:        ~20 min
Hyperparameter Tuning:   ~60 min
SHAP Computation:        ~15 min
Full Pipeline:           ~2.5 hours
```

---

**Last Updated**: January 17, 2026  
**Status**: ✅ Complete