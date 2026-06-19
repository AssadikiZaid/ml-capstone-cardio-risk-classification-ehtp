# Capstone Project Technical Report: Cardiovascular Disease Risk Classification Pipeline

**Course Module:** AI & Data Science Basics (Semester S4)  
**Student Name:** Assadiki Zaid  
**Academic Mentor:** Dr. Rym Nassih  
**Department:** MIG Department, École Hassania des Travaux Publics (EHTP)  
**Date:** June 18, 2026  
**Repository URL:** [https://github.com/AssadikiZaid/ml-capstone-cardio-risk-classification-ehtp](https://github.com/AssadikiZaid/ml-capstone-cardio-risk-classification-ehtp)

---

## Table of Contents
1. **Executive Summary** (Clinical Director Briefing)
2. **Phase 1: Problem Framing & AI Context Note**
3. **Phase 2: Data Audit, Outlier Treatment, & Encoding**
4. **Phase 3: Feature Engineering Log & Data Management (SQLite)**
5. **Phase 4 & 5: Algorithm Benchmarking & Model Performance**
6. **Phase 6: Deployment Guidelines & Operational Limits**
7. **Academic Integrity & Individual Contribution Disclosures**
8. **Oral Presentation Outline (5-Minute Slide Deck)**

---

## 1. Executive Summary

### TO: Clinical Director & Hospital Administration  
### FROM: Assadiki Zaid, Lead Data Scientist  
### SUBJECT: Deployment of Cardiovascular Risk Classification Pipeline

Cardiovascular diseases (CVDs) remain the leading cause of global mortality, accounting for an estimated 17.9 million deaths annually. For healthcare providers, early identification of high-risk patients is the key to reducing long-term mortality and optimizing clinical resource allocation. However, general practitioners face high patient volumes, and traditional risk calculators (such as Framingham or SCORE) often rely on static, linear thresholds that fail to capture complex, non-linear interactions between patient biometrics, lab values, and lifestyle factors.

This technical report presents a clinical decision-support pipeline trained on 70,000 patient records. Through rigorous data wrangling, outlier removal, and clinical feature engineering, we constructed a predictive model to identify patients at high risk of CVD. 

### Key Clinical Outcomes:
- **Surprising Hemodynamic Insight:** Rather than relying solely on raw Systolic Blood Pressure (`ap_hi`), our pipeline engineered **Pulse Pressure** (the difference between systolic and diastolic blood pressure, `ap_hi - ap_lo`). The clinical validation demonstrated that Pulse Pressure provides a significantly clearer decision boundary between healthy and diseased individuals under the age of 55, acting as a direct indicator of early-stage arterial stiffness.
- **High Diagnostic Security:** The tuned XGBoost Classifier achieved an **accuracy of 73.57%** and an **$F_1$-score of 72.71%** on an independent test set. Critically, we prioritized the $F_1$-score over raw accuracy to balance the model. This minimizes dangerous False Negatives (preventing undiagnosed high-risk patients from being discharged) while maintaining high Precision (avoiding the waste of clinical resources on false positives).
- **SQLite Database Architecture:** A local database (`data/cardio_cleaned.db`) was successfully initialized to store patient data, enabling rapid SQL queries for clinical cohorts and hospital administration.
- **Ready for Deployment:** The final pipeline is fully serialized and prepared for integration into Electronic Health Record (EHR) systems to alert general practitioners during routine checkups.

---

## 2. Phase 1: Problem Framing & AI Context Note

### Background & Motivations
Early cardiovascular risk screening is highly cost-effective compared to treating acute events like heart attacks or stroke. The objective of this project is to develop a machine learning pipeline that ingests a patient's routine biometrics (age, gender, height, weight), basic lab readings (cholesterol, glucose levels), and subjective habits (smoking, alcohol intake, physical activity) to output a binary classification: whether the patient is at high risk of cardiovascular disease (`cardio = 1`) or not (`cardio = 0`).

---

### AI Context Note: Expert Systems vs. Probabilistic Connectionism

In early AI (Course 2, Chapters 1-3), the dominant paradigm was **Symbolic AI** and **Expert Systems** (e.g., MYCIN in the 1970s). These systems operated on explicit, hand-coded rules created by human experts, such as:
$$\text{IF } \textit{systolic\_bp} > 140 \text{ AND } \textit{cholesterol} = \text{high} \text{ THEN } \textit{cardio\_risk} = \text{high}$$

While expert systems were highly transparent and explainable, they suffered from significant historical limitations:
1. **The Knowledge Acquisition Bottleneck:** Human medical knowledge is hard to translate into exhaustive, rigid rules. As complexity grows, rule conflicts occur.
2. **Fragility and Inability to Learn:** Expert systems cannot adapt to new patterns or empirical data. They are deterministic and struggle with the inherent noise and probability of biology.
3. **Handling High-Dimensional Interactions:** A rule-based system cannot easily map complex interactions (such as how a patient's weight, height, age, and activity level compound to affect vascular health).

This project represents the modern paradigm shift from symbolic rules to **Empirical Connectionism and Machine Learning**. Instead of coding rules, we feed an algorithm a dataset of 70,000 real medical observations. The algorithm optimizes weights based on empirical loss functions. Rather than applying binary thresholds, models like XGBoost and Logistic Regression map continuous, probabilistic risk gradients across a high-dimensional feature space, resolving the fragility of early AI expert systems.

---

## 3. Phase 2: Data Audit, Outlier Treatment, & Encoding

### Data Audit Summary
The raw dataset was audited for quality issues:
- **Missing Values:** 0 null values were found across all 70,000 records.
- **Duplicates:** 0 duplicate rows were identified.
- **Biomedical Outliers:** Severe unphysical anomalies were identified in blood pressure readings:
  - Diastolic pressure (`ap_lo`) had minimum values of -70 mmHg and maximum values of 11,000 mmHg.
  - Systolic pressure (`ap_hi`) had minimum values of -150 mmHg and maximum values of 16,000 mmHg.
  - These values are physiologically impossible (systolic pressure above 240 mmHg represents hypertensive crisis, while values over 10,000 are telemetry logging errors).

### Outlier Method Comparison
We compared the Interquartile Range (IQR) and Z-score methods:
- **IQR Method:** Identified **1,439 outliers** on blood pressure columns.
- **Z-score Method (Threshold $|Z| > 3$):** Identified **983 outliers**.
- **Comparison:** The Z-score method failed to detect several unphysical values. This occurred because extreme values (e.g., 16,000 mmHg) heavily distorted the mean and standard deviation, pushing the Z-score boundaries outward. The IQR method proved more robust but still flagged valid clinical readings (e.g., systolic BP of 180 mmHg) as outliers.
- **Treatment:** We applied physical, clinical thresholds to ensure data integrity:
  - Systolic BP: $[80, 220]$ mmHg
  - Diastolic BP: $[50, 140]$ mmHg
  - Constraint: $\textit{ap\_hi} > \textit{ap\_lo}$
  - Height: $[130, 220]$ cm
  - Weight: $[40, 180]$ kg
  - This resulted in a clean cohort of **63,942 records** (removing 6,058 anomalous entries).

### Categorical Encoding Justification
- **`cholesterol` and `gluc`:** One-hot encoded. Because they represent ordinal categories (1: normal, 2: above normal, 3: well above normal), using them as integers forces the model to assume a linear relationship. One-hot encoding allows the model to learn non-linear risk increments for each category.
- **`smoke`, `alco`, `active`:** Retained in binary format (0 and 1).
- **`gender`:** Recoded to binary (0 and 1) from its original (1 and 2).

---

## 4. Phase 3: Feature Engineering Log & Data Management (SQLite)

### Feature Engineering Log

| Feature Name | Mathematical Formula | Clinical Motivation | Performance Impact |
| :--- | :--- | :--- | :--- |
| **Body Mass Index (BMI)** | $$\text{BMI} = \frac{\text{weight (kg)}}{(\text{height (m)})^2}$$ | Normalizes weight relative to height. BMI is a key marker of metabolic obesity and physical strain. | Increased classification power; ranked high in Mutual Information. |
| **Pulse Pressure** | $$\text{PP} = \text{ap\_hi} - \text{ap\_lo}$$ | Represents force generated by the heart with each contraction. A high PP indicates arterial stiffness. | The most significant indicator for patients under 55. |
| **Age-Cholesterol Risk** | $$\text{ACR} = \text{Age (Years)} \times \text{cholesterol}$$ | Captures the compounding risk of vascular aging combined with high blood cholesterol. | Improved F1-score of the linear Logistic Regression baseline. |

---

### Data Management: SQL Analytical Validation
The processed dataset was stored in SQLite table `cardio`. The following four queries were executed to validate the data:

```sql
-- Query 1: Percentage of CVD patients by activity and alcohol consumption
SELECT 
    active, 
    alco, 
    AVG(cardio) * 100 AS cvd_percentage, 
    COUNT(*) AS patient_count
FROM cardio 
GROUP BY active, alco;

-- Query 2: Average Pulse Pressure and BMI by Cholesterol level for patients older than 50
SELECT 
    cholesterol, 
    AVG(pulse_pressure) AS avg_pulse_pressure, 
    AVG(bmi) AS avg_bmi, 
    COUNT(*) AS patient_count
FROM cardio 
WHERE (age / 365.25) > 50 
GROUP BY cholesterol;

-- Query 3: High-risk patients without CVD diagnosis
SELECT COUNT(*) AS high_risk_no_cvd_count 
FROM cardio 
WHERE cardio = 0 
  AND cholesterol = 3 
  AND gluc = 3 
  AND bmi > 30;

-- Query 4: Baseline target verification
SELECT 
    cardio, 
    COUNT(*) AS total_patients, 
    AVG(age / 365.25) AS avg_age_years, 
    AVG(bmi) AS avg_bmi
FROM cardio 
GROUP BY cardio;
```

---

## 5. Phase 4 & 5: Algorithm Benchmarking & Model Performance

The pipelines were evaluated on a 20% stratified test set:

| Model | Accuracy | Precision | Recall | $F_1$-score | ROC-AUC | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tuned XGBoost** | **73.57%** | **75.39%** | **70.21%** | **72.71%** | **80.12%** | **~1.20s** |
| Logistic Regression | 72.84% | 74.92% | 68.61% | 71.63% | 79.28% | ~0.15s |
| Random Forest | 71.21% | 72.10% | 69.15% | 70.59% | 77.40% | ~3.80s |
| K-Nearest Neighbors | 69.80% | 70.35% | 68.10% | 69.21% | 74.50% | ~0.45s |

### Hyperparameter Tuning Rationale:
We selected XGBoost for hyperparameter tuning. Through 3-fold cross-validation on the training set, we optimized parameters to maximize the $F_1$-score. The best parameters found were `learning_rate=0.05`, `max_depth=3`, and `n_estimators=100`. This configuration minimized overfitting, leading to the best generalization metrics on the holdout test set.

---

## 6. Phase 6: Deployment Guidelines & Operational Limits

### Deployment Guidelines
- **Integration:** The serialized file `best_cardio_model.joblib` should be wrapped in an API and integrated into the clinic's Electronic Health Record (EHR) system.
- **Workflow:** When a clinician inputs a patient's vitals (blood pressure, height, weight) and lab results (cholesterol, glucose), the model calculates the risk score. If the score exceeds a threshold (e.g., 0.50), an alert flags the patient for further evaluation.

### Operational Limits & Edge Cases
1. **Vitals Outside Bounds:** The model may output unreliable predictions for inputs outside realistic physiological limits (e.g., systolic BP < 80 mmHg or > 220 mmHg).
2. **Pediatric Generalization:** The model is not validated for patients under 18.
3. **Data Bias:** Telemetry is sourced from specific demographic regions and may have geographic biases.

---

## 7. Academic Integrity & Individual Contribution Disclosures

### AI-Assisted Content Appendix
- **Generative AI Use Disclosure:** In accordance with course guidelines, Generative AI (Gemini) was utilized to support the writing of this technical report and code structures.
- **Verification Steps:**
  1. **Code Execution:** All code cells and scripts were executed locally to verify that pipelines run without errors.
  2. **Mathematical Verification:** The Gini Impurity, Logistic Sigmoid, and XGBoost objective equations were verified against academic textbooks.
  3. **SQL Validation:** The SQLite schema and queries were executed locally to confirm the output records matched our expectations.

---

### Individual Contribution Statement
As the sole author of this project, I, **Assadiki Zaid**, executed all phases of the capstone project:
- **Phase 1:** Formulated the clinical classification problem and drafted the AI historical context.
- **Phase 2:** Wrote the wrangling scripts and compared the IQR and Z-score outlier detection methods.
- **Phase 3:** Engineered the clinical features (BMI, Pulse Pressure, and Risk Interaction) and set up the local SQLite database.
- **Phase 4 & 5:** Built the scikit-learn preprocessing pipelines, performed model benchmarking, optimized the XGBoost model, and serialized the final weights.
- **Phase 6 & 7:** Authored the Model Card, final technical report, and presentation slides.

---

## 8. Oral Presentation Outline (5-Minute Slide Deck)

### Slide 1: Title & Overview (0:30 min)
- **Title:** Cardiovascular Disease Risk Classification Pipeline
- **Presenter:** Assadiki Zaid (Academic Mentor: Dr. Rym Nassih)
- **Problem Statement:** Early detection of CVD to prevent mortality and optimize clinical resources using raw clinical measurements.

### Slide 2: Pipeline Design & Repository Structure (0:45 min)
- **Visual:** Diagram of the data flow: Raw CSV $\rightarrow$ Ingest script $\rightarrow$ EDA and physical cleaning $\rightarrow$ Feature Engineering $\rightarrow$ SQLite Storage $\rightarrow$ Preprocessing Pipelines $\rightarrow$ Tuned XGBoost Model.
- **Key point:** Emphasize the "No Data Leakage" policy using scikit-learn Pipelines.

### Slide 3: EDA & The Surprising Insight (1:00 min)
- **Key finding:** Instead of raw Systolic Blood Pressure (`ap_hi`) being the sole predictor, the engineered feature **Pulse Pressure** (`ap_hi - ap_lo`) provides a significantly cleaner separation between healthy and high-risk patients, especially for patients under the age of 55.

### Slide 4: Model Benchmarking & Performance (1:30 min)
- **Visual:** Benchmarking table comparing Logistic Regression, KNN, Random Forest, and XGBoost.
- **Metrics discussion:** Explain why the $F_1$-score is prioritized over raw accuracy (balancing dangerous false negatives and costly false positives).
- **Result:** Tuned XGBoost achieved the best performance ($F_1$-score: 72.71%, Accuracy: 73.57%).

### Slide 5: Clinical Deployment & Summary (0:45 min)
- **Deployment:** Integrate `best_cardio_model.joblib` into EHR systems to trigger clinician alerts during routine checkups.
- **Operational limits:** Underline the need for human-in-the-loop oversight and limitations with demographic biases and pediatric patients.
- **Closing:** Thank you. Questions?
