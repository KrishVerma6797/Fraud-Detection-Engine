## 💳 Real-Time Fraud Detection Engine

An end-to-end Machine Learning system built using **XGBoost** to detect fraudulent financial transactions in real time.

This project implements the complete ML lifecycle:

- Data preprocessing
- Model training
- Performance evaluation
- Model explainability (SHAP)
- Deployment using Streamlit

---

## 📌 Project Overview

Fraud detection is a highly imbalanced **binary classification problem** where the goal is to predict whether a transaction is:

- `1` → Fraudulent  
- `0` → Legitimate  

This system provides:

- Fraud probability score
- Adjustable decision threshold
- Business risk estimation
- Model performance dashboard
- Feature importance visualization

---

## 🏗️ Project Structure

```
ML PROJECT - FRAUD DET/
│
├── app/
│   ├── predict.py
│   └── streamlit_app.py
│
├── data/
│   └── realistic_fraud_dataset_200k.csv
│
├── models/
│   ├── scaler.pkl
│   └── xgb_model.pkl
│
├── src/
│   ├── train.py
│   ├── preprocess.py
│   ├── evaluate.py
│   └── explain.py
│
├── requirements.txt
└── README.md
```

---

## 🧠 Model Architecture

### 🔹 Algorithm
XGBoost (Gradient Boosted Decision Trees)

### 🔹 Problem Type
Binary Classification (Fraud vs Non-Fraud)

### 🔹 Train-Test Split
- 80% Training
- 20% Testing
- Stratified sampling
- `random_state = 42`

### 🔹 Hyperparameters

```python
n_estimators = 200
max_depth = 5
learning_rate = 0.1
eval_metric = "logloss"
```

---

## 📊 Features Used

The model is trained on the following transaction-level features:

- `amount`
- `hour`
- `is_international`
- `transaction_gap`
- `location_risk`
- `device_risk`
- `merchant_risk`

All numerical features are scaled using **StandardScaler**.

---

## ⚙️ Core Modules

### 🔹 Training — `src/train.py`

- Loads dataset
- Separates features and target
- Applies StandardScaler
- Trains XGBClassifier
- Evaluates performance
- Saves model and scaler to `/models`

---

### 🔹 Preprocessing — `src/preprocess.py`

- Handles feature-target separation
- Performs scaling
- Returns fitted scaler

---

### 🔹 Evaluation — `src/evaluate.py`

Model evaluation includes:

- Classification Report
- ROC-AUC Score
- PR-AUC Score
- Confusion Matrix

---

### 🔹 Explainability — `src/explain.py`

- Uses SHAP TreeExplainer
- Generates SHAP summary plot
- Interprets feature contributions

---

### 🔹 Inference — `app/predict.py`

- Loads trained model and scaler
- Accepts transaction feature input
- Returns:
  - Binary prediction
  - Fraud probability score

---

### 🔹 Deployment — `app/streamlit_app.py`

Interactive Streamlit dashboard featuring:

- Real-time transaction input
- Adjustable decision threshold
- Fraud probability visualization
- Expected business loss estimation
- ROC curve visualization
- Precision-Recall curve visualization
- Feature importance chart

---

## 💰 Business Logic Layer

The system integrates cost-sensitive fraud decision modeling.

### 🔹 Default Parameters

- Fraud Loss: ₹5000
- False Positive Cost: ₹200
- Decision Threshold: 0.4

### 🔹 Expected Loss Formula

- If fraud predicted → `fraud_loss × probability`
- If legitimate → `false_positive_cost × probability`

This enables business-oriented fraud risk estimation rather than pure ML prediction.

---

## 🛠️ Installation & Setup

### Step 1: Clone Repository

```bash
git clone <your-repository-link>
cd ML PROJECT - FRAUD DET
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Train the Model

```bash
python src/train.py
```

### Step 4: Run the Application

```bash
streamlit run app/streamlit_app.py
```

---

## 📈 Model Performance Metrics

The system evaluates performance using:

- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC

Visual performance monitoring includes:

- ROC Curve
- Precision-Recall Curve
- Feature Importance Plot

---

## 🔮 Future Improvements

- Hyperparameter tuning using GridSearchCV
- SMOTE for class imbalance handling
- FastAPI REST endpoint for production inference
- Docker containerization
- Model monitoring and drift detection

---

## 📦 Tech Stack

- Python
- XGBoost
- Scikit-Learn
- Pandas
- NumPy
- SHAP
- Streamlit
- Matplotlib

---

## 📄 License

MIT License