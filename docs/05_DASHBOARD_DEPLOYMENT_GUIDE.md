# Interactive Dashboard and Deployment Guide

**DSGP Group 40 | Osteoporosis Risk Prediction**  
**Student: Isum Gamage (ID: 20242052)**  
**January 2026**

---

## 1. Dashboard Architecture Overview

### 1.1 System Architecture

```
┌─────────────────────────────────┐
│                  FRONTEND LAYER (React)                  │
├─────────────────────────────────┤
│  Input Form | Risk Gauge | Recommendations Display   │
└─────────────────────────────────┘
                        ↑↓ HTTP/REST API
┌─────────────────────────────────┐
│              BACKEND LAYER (Flask/FastAPI)             │
├─────────────────────────────────┤
│  Data Validation | Model Routing | Prediction Engine  │
└─────────────────────────────────┘
                        ↑↓
┌─────────────────────────────────┐
│               ML MODEL LAYER (XGBoost)                 │
├─────────────────────────────────┤
│  Male Model (992) | Female Model (966) | SHAP Values  │
└─────────────────────────────────┘
```

### 1.2 Tech Stack

```
Frontend:       React 18 + TypeScript + Tailwind CSS
Backend:        Flask or FastAPI with Python 3.8+
ML Framework:   XGBoost, SHAP, scikit-learn
Database:       Optional (PostgreSQL for patient records)
Deployment:     Docker + AWS/GCP/Heroku
Monitoring:     Prometheus + Grafana (optional)
```

---

## 2. Frontend Components

### 2.1 Input Form

**Patient Data Collection Interface**

```jsx
Form Fields:
├─ Demographics
│  ├─ Age (slider: 18-90)
│  ├─ Gender (radio: Male/Female)
│  ├─ Race/Ethnicity (dropdown)
│  └─ Hormonal Status (dropdown: Normal/Postmenopausal)
├─ Anthropometry
│  └─ Body Weight (radio: Normal/Underweight)
├─ Nutrition
│  ├─ Calcium Intake (radio: Low/Adequate)
│  └─ Vitamin D Intake (radio: Insufficient/Sufficient)
├─ Lifestyle
│  ├─ Physical Activity (radio: Sedentary/Active)
│  ├─ Smoking (radio: Yes/No)
│  └─ Alcohol Consumption (dropdown: None/Moderate/Heavy)
└─ Medical History
   ├─ Family History (checkbox)
   ├─ Prior Fractures (checkbox)
   ├─ Medical Conditions (dropdown: None/HTN/RA)
   └─ Medications (dropdown: None/Corticosteroids)

Actions:
  - Clear Form
  - Reset to Defaults
  - Calculate Risk
```

**Form Validation**:
```javascript
Validation Rules:
  ✓ Age: 18-90 (no negatives, no >150)
  ✓ Gender: Required (routes to correct model)
  ✓ All fields: No null/undefined
  ✓ Dropdown: Only predefined values
  ✓ Real-time error messages

Example Error: "Please enter valid age between 18-90"
```

### 2.2 Risk Result Display

**Visual Risk Representation**

```
┌─────────────────────────────────┐
│              OSTEOPOROSIS RISK ASSESSMENT              │
├─────────────────────────────────┤
│                                                         │
│  Risk Score: 78%                                      │
│                                                         │
│  ██████████████████■░░░░  <- Gauge to 100%   │
│                                                         │
│  Risk Category: HIGH RISK                             │
│  Confidence Interval (95%): 73%-83%                  │
│                                                         │
└─────────────────────────────────┘
```

**Color Coding**:
```
Risk Level    Color    Range        Clinical Action
───────────────────────────────────────────────
Low Risk      Green    < 30%        Routine monitoring
Moderate      Yellow   30-60%       Lifestyle changes
High Risk     Orange   60-80%       Medical consultation
Very High     Red      > 80%        Urgent evaluation
```

### 2.3 Feature Importance Chart

**SHAP-based Feature Importance Visualization**

```javascript
Component: HorizontalBarChart

Data Structure:
[
  { feature: 'Age', importance: 0.185, color: 'red' },
  { feature: 'Prior Fractures', importance: 0.142, color: 'red' },
  { feature: 'Smoking Status', importance: 0.098, color: 'red' },
  // ...
]

Rendering:
  - Show top 8 features
  - Bar length proportional to importance
  - Red = increasing risk
  - Blue = decreasing risk (protective)
  - Interactive tooltip: "Feature X contributes ±Y% to risk"
```

### 2.4 Personalized Recommendations

**Smart Recommendation Engine**

```javascript
Recommendation Logic:

IF Risk > 70% AND Age > 50:
  └─ "Consult bone specialist immediately"
  └─ "Consider DEXA scan"
  └─ "Discuss bisphosphonate therapy"

IF Smoking = Yes:
  └─ "Smoking cessation is critical - improves outcomes"
  └─ "Refer to smoking cessation program"

IF Calcium Intake = Low OR Vitamin D = Insufficient:
  └─ "Increase dietary calcium (1000-1200 mg/day)"
  └─ "Vitamin D supplementation recommended (800-2000 IU/day)"

IF Physical Activity = Sedentary:
  └─ "Weight-bearing exercise 30 min/day recommended"
  └─ "Examples: brisk walking, dancing, jogging"

IF Female AND Postmenopausal AND Age > 60:
  └─ "Hormone replacement therapy consideration"
  └─ "Regular bone density monitoring"
```

**Output Format**:
```jsx
<div className="recommendations">
  <h3>Personalized Recommendations</h3>
  <ul>
    <li className="critical">Action 1 (Critical)</li>
    <li className="important">Action 2 (Important)</li>
    <li className="informational">Action 3 (Informational)</li>
  </ul>
</div>
```

---

## 3. Backend API Endpoints

### 3.1 API Specifications

**Base URL**: `https://api.osteoporosis-risk.com/v1`

#### Endpoint 1: Predict Risk

**POST** `/predict`

```javascript
Request Body:
{
  "age": 65,
  "gender": "Female",
  "race": "Caucasian",
  "hormonal_status": "Postmenopausal",
  "body_weight": "Normal",
  "calcium_intake": "Low",
  "vitamin_d_intake": "Insufficient",
  "physical_activity": "Sedentary",
  "smoking": "Yes",
  "alcohol_consumption": "Moderate",
  "family_history": true,
  "prior_fractures": false,
  "medical_conditions": "None",
  "medications": "Corticosteroids"
}

Response (200 OK):
{
  "patient_id": "uuid-1234",
  "prediction": {
    "probability": 0.78,
    "risk_category": "HIGH_RISK",
    "confidence_interval": {
      "lower": 0.73,
      "upper": 0.83
    }
  },
  "feature_importance": [
    { "feature": "Age", "shap_value": 0.20 },
    { "feature": "Postmenopausal", "shap_value": 0.15 },
    // ...
  ],
  "recommendations": [
    "Consult bone specialist immediately",
    "Consider DEXA scan",
    // ...
  ],
  "model_version": "1.0.0",
  "timestamp": "2026-01-17T09:30:00Z"
}
```

**Error Handling** (400 Bad Request):
```javascript
{
  "error": "Invalid age: must be between 18-90",
  "status": 400,
  "details": {
    "field": "age",
    "value": 150,
    "constraint": "18-90"
  }
}
```

#### Endpoint 2: Model Health Check

**GET** `/health`

```javascript
Response (200 OK):
{
  "status": "healthy",
  "models_loaded": {
    "male_model": true,
    "female_model": true
  },
  "last_trained": "2025-12-20T10:30:00Z",
  "model_version": "1.0.0",
  "api_version": "1.0.0"
}
```

#### Endpoint 3: Model Metrics

**GET** `/metrics`

```javascript
Response (200 OK):
{
  "male_model": {
    "accuracy": 0.85,
    "precision": 0.83,
    "recall": 0.87,
    "f1_score": 0.85,
    "auc_roc": 0.91,
    "samples_trained": 794
  },
  "female_model": {
    "accuracy": 0.86,
    "precision": 0.84,
    "recall": 0.88,
    "f1_score": 0.86,
    "auc_roc": 0.92,
    "samples_trained": 773
  }
}
```

### 3.2 Data Validation Layer

```python
# Backend validation (Python)

def validate_patient_data(data):
    errors = []
    
    # Age validation
    if not isinstance(data['age'], int) or data['age'] < 18 or data['age'] > 90:
        errors.append({"field": "age", "error": "Must be 18-90"})
    
    # Gender validation
    if data['gender'] not in ['Male', 'Female']:
        errors.append({"field": "gender", "error": "Must be Male or Female"})
    
    # All other validations...
    
    if errors:
        raise ValidationError(errors)
    
    return True
```

---

## 4. Gender-Specific Model Routing

### 4.1 Routing Logic

```python
def predict_osteoporosis_risk(patient_data):
    # Step 1: Validate input
    validate_patient_data(patient_data)
    
    # Step 2: Preprocess features
    X = preprocess_features(patient_data)
    
    # Step 3: Route based on gender
    if patient_data['gender'] == 'Male':
        model = male_model
        explainer = male_explainer
    else:
        model = female_model
        explainer = female_explainer
    
    # Step 4: Generate prediction
    probability = model.predict_proba(X)[0][1]  # Positive class
    
    # Step 5: Generate explanations
    shap_values = explainer.shap_values(X)
    feature_importance = get_feature_importance(shap_values)
    
    # Step 6: Generate recommendations
    recommendations = generate_recommendations(patient_data, probability, feature_importance)
    
    # Step 7: Return results
    return {
        'probability': probability,
        'risk_category': categorize_risk(probability),
        'feature_importance': feature_importance,
        'recommendations': recommendations
    }
```

### 4.2 Risk Categorization Function

```python
def categorize_risk(probability):
    if probability < 0.30:
        return "LOW_RISK"
    elif probability < 0.60:
        return "MODERATE_RISK"
    elif probability < 0.80:
        return "HIGH_RISK"
    else:
        return "VERY_HIGH_RISK"
```

---

## 5. Deployment Instructions

### 5.1 Docker Deployment

**Dockerfile**:
```dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY models/ ./models/

EXPOSE 5000

CMD ["python", "app.py"]
```

**requirements.txt**:
```
Flask==2.3.0
xgboost==1.7.0
shap==0.41.0
scikit-learn==1.0.0
joblib==1.2.0
numpy==1.21.0
pandas==1.3.0
pydantic==1.8.0
gunicorn==20.1.0
```

**Docker Build & Run**:
```bash
# Build image
docker build -t osteoporosis-api:1.0 .

# Run container
docker run -p 5000:5000 -e FLASK_ENV=production osteoporosis-api:1.0
```

### 5.2 AWS Deployment (ECS/ECR)

```bash
# 1. Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com
docker tag osteoporosis-api:1.0 <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/osteoporosis-api:1.0
docker push <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/osteoporosis-api:1.0

# 2. Create ECS Task Definition
# (Configure via AWS Console or CloudFormation)

# 3. Deploy to ECS Cluster
aws ecs update-service --cluster osteoporosis-cluster --service osteoporosis-api --force-new-deployment
```

### 5.3 Heroku Deployment

```bash
# 1. Create Heroku app
heroku create osteoporosis-risk-api

# 2. Set environment variables
heroku config:set FLASK_ENV=production

# 3. Deploy
git push heroku main

# 4. View logs
heroku logs --tail
```

---

## 6. Security and Privacy

### 6.1 HIPAA Compliance

```
✅ Data Encryption
   - HTTPS/TLS for all communications
   - AES-256 for data at rest
   - TLS 1.2+ minimum

✅ Access Control
   - API key authentication
   - JWT tokens with expiration (1 hour)
   - Role-based access control (Admin/Clinician/Patient)

✅ Data Minimization
   - Don't store full patient records (unless consented)
   - Store only: age, gender, medical history
   - Hash all IDs (no direct patient identifiers)

✅ Audit Logging
   - Log all predictions with timestamp
   - Track user access patterns
   - Maintain 7-year retention (HIPAA requirement)
```

### 6.2 API Authentication

```python
from functools import wraps
import jwt

SECRET_KEY = os.getenv('SECRET_KEY')

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return {'error': 'Missing authorization token'}, 401
        
        try:
            data = jwt.decode(token.split(' ')[1], SECRET_KEY, algorithms=['HS256'])
            request.user = data
        except jwt.ExpiredSignatureError:
            return {'error': 'Token expired'}, 401
        except jwt.InvalidTokenError:
            return {'error': 'Invalid token'}, 401
        
        return f(*args, **kwargs)
    return decorated

@app.route('/predict', methods=['POST'])
@require_auth
def predict():
    # Prediction logic
    pass
```

### 6.3 GDPR Compliance (if EU users)

```
✅ Right to Access
   - Users can request their prediction history

✅ Right to Deletion
   - Users can request data deletion (right to be forgotten)

✅ Data Portability
   - Users can export their data in standard format

✅ Consent Management
   - Explicit opt-in for data storage
   - Regular consent renewal
```

---

## 7. Monitoring and Maintenance

### 7.1 Performance Monitoring

```python
from prometheus_client import Counter, Histogram, Gauge

# Metrics
prediction_count = Counter('predictions_total', 'Total predictions')
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction latency')
model_accuracy = Gauge('model_accuracy', 'Model accuracy on recent data')

@app.route('/predict', methods=['POST'])
@require_auth
def predict():
    with prediction_duration.time():
        # Prediction logic
        result = generate_prediction(data)
    
    prediction_count.inc()
    return result
```

### 7.2 Model Monitoring

**Performance Tracking**:
```
Monitor:
  - Prediction distribution (should match training distribution)
  - Confidence intervals (should be stable)
  - Error rates (should not exceed baseline)
  - Response times (should be <500ms)

Alert Thresholds:
  - Error rate > 5% → Alert
  - Response time > 1000ms → Warning
  - Prediction drift > 10% → Alert
  - Model accuracy degradation > 5% → Retrain
```

### 7.3 Regular Retraining

```
Retraining Schedule:
  - Monthly: Full model retraining
  - Quarterly: Feature importance review
  - Annually: Complete model validation

Retraining Triggers:
  - Accuracy drops >5%
  - Significant data distribution shift detected
  - New evidence contradicts model behavior
  - Regulatory/clinical practice updates
```

---

## 8. Usage Example

### Frontend (React)
```javascript
const response = await fetch('https://api.osteoporosis-risk.com/v1/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  },
  body: JSON.stringify({
    age: 65,
    gender: 'Female',
    // ... other fields
  })
});

const result = await response.json();
console.log(`Risk: ${result.prediction.probability * 100}%`);
```

---

**Status**: ✅ Complete  
**Last Updated**: January 17, 2026