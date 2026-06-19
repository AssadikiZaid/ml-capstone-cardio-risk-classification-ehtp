## Cardiovascular Disease Risk Classification Pipeline

This repository contains the complete end-to-end data science and machine learning pipeline for cardiovascular disease (CVD) risk classification. This project is my Capstone Project for the **AI & Data Science Basics** module (Semester S4) at **École Hassania des Travaux Publics (EHTP)**.

## Project Specifications

- **Author:** Assadiki Zaid
- **Academic Mentor:** Dr. Rym Nassih
- **Department:** MIG Department, École Hassania des Travaux Publics
- **Module:** AI & Data Science Basics (Semester S4)
- **Target Git/GitHub Repository:** [https://github.com/AssadikiZaid/ml-capstone-cardio-risk-classification-ehtp](https://github.com/AssadikiZaid/ml-capstone-cardio-risk-classification-ehtp)
- **Dataset Source:** [Kaggle Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)

---

## Repository Structure

```text
ml-capstone-cardio-risk-classification-ehtp/
├── requirements.txt
├── .gitignore
├── README.md
├── GithubLink.txt
├── data/
│   ├── cardio_train.csv
│   └── cardio_cleaned.db
├── src/
│   └── ingest.py
├── notebooks/
│   ├── 01_EDA_Wrangling.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Model_Training.ipynb
├── models/
│   └── best_cardio_model.joblib
└── reports/
    ├── FinalProjectAI_Presentation.pdf
    ├── FinalProjectAI_Report.pdf
    ├── Phase4_Algorithm_Selection.md
    └── Phase6_Model_Card.md
```

---

## Execution & Reproduction Instructions

To reproduce the results of the pipeline, follow these sequential steps:

### 1. Environment Setup
Ensure you have Python 3.10+ installed. Install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Run Data Ingestion
Verify the raw dataset is loaded and describe its properties:
```bash
python src/ingest.py
```

### 3. Run Jupyter Notebooks
Execute the notebooks in the following order:
- **`notebooks/01_EDA_Wrangling.ipynb`**: Conducts the data audit, IQR and Z-score outlier detection, and displays the exploratory charts.
- **`notebooks/02_Feature_Engineering.ipynb`**: Computes the clinical features (BMI, Pulse Pressure, and Risk Interaction), ranks features via Mutual Information, and populates the SQLite database.
- **`notebooks/03_Model_Training.ipynb`**: Fits the pipelines, performs Grid Search tuning on the XGBoost classifier, outputs comparisons, and saves the final serialized model.

To execute the notebooks programmatically via command line:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_EDA_Wrangling.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_Feature_Engineering.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_Model_Training.ipynb
```

---

## Final Pipeline Model Performance Summary

The machine learning models were evaluated on an independent test split (20% holdout). The engineered feature **Pulse Pressure** (`ap_hi - ap_lo`) proved to be the strongest clinical indicator for separating patient risk levels, particularly for patients under the age of 55.

The side-by-side performance benchmarking of the 4 models is summarized below:

| Model | Accuracy | Precision | Recall | $F_1$-score | ROC-AUC | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tuned XGBoost** | **73.57%** | **75.39%** | **70.21%** | **72.71%** | **80.12%** | ~1.20s |
| Logistic Regression | 72.84% | 74.92% | 68.61% | 71.63% | 79.28% | ~0.15s |
| Random Forest | 71.21% | 72.10% | 69.15% | 70.59% | 77.40% | ~3.80s |
| K-Nearest Neighbors | 69.80% | 70.35% | 68.10% | 69.21% | 74.50% | ~0.45s |

*Note: The Tuned XGBoost Classifier was selected as the production-grade model and is serialized at `models/best_cardio_model.joblib`.*
