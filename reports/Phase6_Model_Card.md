# Phase 6: Model Card

This Model Card details the technical specifications, performance metadata, operational limitations, and ethical boundaries of the best-performing machine learning model developed for the Cardiovascular Disease Risk Classification Pipeline.

---

## 1. Model Overview
- **Model Name:** Tuned XGBoost Classifier
- **Model Type:** Extreme Gradient Boosting Ensemble (Decision Trees)
- **Version:** 1.0.0
- **Release Date:** June 18, 2026
- **Serialized Artifact Path:** `/models/best_cardio_model.joblib`
- **Developer:** Assadiki Zaid (MIG Department, S4, EHTP)
- **License:** Open Academic Use

---

## 2. Intended Clinical Use
- **Primary Use Case:** A clinical decision-support tool for medical professionals to aid in identifying patients at high risk of developing cardiovascular disease during routine physical examinations.
- **Target Audience:** General practitioners, cardiologists, and triage nurses in outpatient clinic settings.
- **Decision Support Type:** Categorical risk assessment (CVD high-risk vs. low-risk) based on biometric, laboratory, and self-reported lifestyle features.
- **Out-of-Scope Uses:** This model is **not** a diagnostic device. It should not be used as a standalone diagnostic system or to automate medical billing, insurance eligibility, or prescription decisions without direct clinician validation.

---

## 3. Preprocessing & Training Summary
- **Data Source:** Kaggle Cardiovascular Disease Dataset (Svetlana Ulianova), consisting of 70,000 records.
- **Cleaned Dataset Size:** 63,942 records (following the removal of unphysical biometric and blood pressure anomalies).
- **Train/Test Split:** 80% Training (51,153 records), 20% Testing (12,789 records), stratified by target variable to maintain class proportion.
- **Pipeline Transformations (No Data Leakage):**
  - **Numerical Scaling:** `StandardScaler` fitted exclusively on `X_train` and applied to `age`, `height`, `weight`, `ap_hi`, `ap_lo`, `bmi`, `pulse_pressure`, and `age_cholesterol_risk`.
  - **Categorical Encoding:** `OneHotEncoder(drop='first')` applied to multi-class ordinal variables `cholesterol` and `gluc`.
  - **Pass-through Features:** Binary variables `gender`, `smoke`, `alco`, and `active` are fed directly without transformations.
- **Optimization Strategy:** Hyperparameter optimization via 3-fold cross-validated Grid Search on the training split, optimizing for the $F_1$-score.
  - **Selected Hyperparameters:** `n_estimators=100`, `max_depth=3`, `learning_rate=0.05`.

---

## 4. Metrics Evaluation (Test Set)
The model was evaluated on a 20% stratified holdout test set (12,789 patients):

- **Accuracy:** 73.57%
- **Precision:** 75.39%
- **Recall (Sensitivity):** 70.21%
- **$F_1$-score:** 72.71%
- **ROC-AUC:** 80.12%

---

## 5. Operational Limitations & Edge Cases
- **Biometric Outliers:** The model was trained on cleaned clinical data. It may output erratic or unreliable predictions if presented with inputs outside realistic human physiology (e.g., systolic BP < 80 mmHg or > 220 mmHg, or weight < 40 kg).
- **Pediatric Cohorts:** The study population represents individuals aged 30 to 65. The model's predictions are invalid for pediatric patients (< 18 years old).
- **Geographic Bias:** The source data is clinical telemetry from Eastern European populations; its predictive weights may not generalize perfectly to other global cohorts with different genetic predispositions or dietary lifestyles.
- **Pregnancy:** High blood pressure readings during pregnancy (e.g., preeclampsia) represent distinct pathophysiological conditions not captured by this general classification model.

---

## 6. Ethical Considerations
- **Fairness & Subpopulation Biases:** Standard evaluations indicate slight differences in recall between gender groups, likely due to biological differences in how blood pressure correlates with cardiovascular events. Clinicians must apply caution to ensure these variances do not lead to under-diagnosis in women.
- **Data Privacy Constraints:** The dataset contains no personal identifiers (names, national IDs, precise geographic coordinates). Any future clinical deployment must comply with medical data privacy regulations (e.g., HIPAA or GDPR) to protect patient telemetry.
- **Human-in-the-Loop Oversight:** This model is designed to augment, not replace, clinical judgement. Final diagnostic decisions, medication prescriptions, and therapeutic interventions must be made exclusively by a qualified healthcare professional.
