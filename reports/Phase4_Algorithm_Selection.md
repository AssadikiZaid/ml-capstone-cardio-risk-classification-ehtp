# Phase 4: Algorithm Selection & Justification Document

This document presents the theoretical framework, mathematical foundations, data assumptions, and clinical suitability analysis for the four candidate machine learning architectures evaluated for the Cardiovascular Disease Risk Classification Pipeline: Logistic Regression, K-Nearest Neighbors (KNN), Random Forest, and XGBoost Classifier.

---

## 1. Candidate Architectures: Mathematical Principles & Assumptions

### Logistic Regression
- **Mathematical Principles:** A generalized linear model used for binary classification. It models the probability $P(Y=1|X)$ using the logistic sigmoid function:
  $$P(Y=1|X) = \sigma(w^T X + b) = \frac{1}{1 + e^{-(w^T X + b)}}$$
  The parameters are optimized by minimizing the binary cross-entropy loss (negative log-likelihood):
  $$\mathcal{L}(w, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
- **Data Assumptions:**
  - Linearity of independent variables and log-odds.
  - Independence of errors.
  - Absence of multi-collinearity among predictor variables.
  - Continuous variables are scaled to ensure stable gradient convergence.

### K-Nearest Neighbors (KNN)
- **Mathematical Principles:** A non-parametric, instance-based learning algorithm. Classification of a query point $x$ is determined by a majority vote of its $k$ nearest neighbors in the feature space:
  $$\hat{y} = \text{mode}\big(\{y_i \mid x_i \in N_k(x)\}\big)$$
  Proximity is computed using the Minkowski metric (defaulting to Euclidean distance when $p=2$):
  $$d(x, x') = \left( \sum_{j=1}^D |x_j - x'_j|^p \right)^{1/p}$$
- **Data Assumptions:**
  - Local similarity: instances close to each other in the feature space share the same class label.
  - Highly sensitive to feature scales; all numeric features must be normalized or standardized to prevent features with large scales from dominating the distance metric.

### Random Forest Classifier
- **Mathematical Principles:** An ensemble bootstrap aggregation (bagging) model of decision trees. It constructs $B$ independent decision trees by sampling training data with replacement (bootstrap) and selecting random subsets of features at each split. The final prediction is a majority vote over all trees:
  $$\hat{y} = \text{mode}\big(\{T_b(x)\}_{b=1}^B\big)$$
  Individual splits are optimized using the Gini Impurity:
  $$I_G(p) = 1 - \sum_{k=1}^K p_k^2$$
- **Data Assumptions:**
  - Non-parametric: makes no assumptions about feature distributions, linearity, or homoscedasticity.
  - Robust to outliers and collinear features.

### XGBoost Classifier (Extreme Gradient Boosting)
- **Mathematical Principles:** An ensemble boosting framework that sequentially builds weak decision trees to minimize a regularized objective function at step $t$:
  $$\mathcal{L}^{(t)} = \sum_{i=1}^N l\big(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\big) + \Omega(f_t)$$
  where $\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$ penalizes tree complexity ($T$ leaves and leaf weights $w$). The split criteria is derived using second-order Taylor expansion of the loss function, utilizing gradients ($g_i$) and hessians ($h_i$):
  $$\text{Gain} = \frac{1}{2} \left[ \frac{\left(\sum_{i \in I_L} g_i\right)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{\left(\sum_{i \in I_R} g_i\right)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{\left(\sum_{i \in I} g_i\right)^2}{\sum_{i \in I} h_i + \lambda} \right] - \gamma$$
- **Data Assumptions:**
  - Non-parametric, requires no linear or Gaussian assumptions.
  - Highly robust to complex multi-dimensional feature interactions, non-linear boundaries, and missing values.

---

## 2. Structural Strengths & Weaknesses in a Clinical Framework

| Model | Clinical & Structural Strengths | Clinical & Structural Weaknesses |
| :--- | :--- | :--- |
| **Logistic Regression** | - High interpretability; outputs odds ratios directly.<br>- Low computational overhead.<br>- Extremely fast training and inference. | - Fails to capture non-linear relationships.<br>- Poor handling of complex feature interactions (e.g., age-lipid compound risks). |
| **K-Nearest Neighbors** | - Simple concept.<br>- Adapts dynamically to new patient data.<br>- Capable of mapping non-linear decision spaces. | - High memory overhead; requires keeping the entire database in memory during inference.<br>- Sensitive to noisy outliers.<br>- Slow prediction speed on large datasets. |
| **Random Forest** | - Captures complex non-linear clinical markers.<br>- Extremely robust to overfitting due to bagging.<br>- Implements natural feature importance metrics. | - High memory consumption for deep trees.<br>- Lack of direct probabilistic interpretability (operates as a semi-black-box). |
| **XGBoost Classifier** | - Superior predictive accuracy on structured clinical datasets.<br>- Built-in regularization prevents overfitting.<br>- Handles highly non-linear boundary spaces efficiently. | - Complex hyperparameter tuning required.<br>- Harder for clinicians to interpret directly without secondary explanation tools (SHAP/LIME). |

---

## 3. Prioritizing $F_1$-Score Over Raw Accuracy in Healthcare

In clinical diagnostics and risk screening, evaluating models based solely on **raw accuracy** is highly problematic and medically dangerous. If a diagnostic screening pipeline is applied to a population, raw accuracy can mask catastrophic clinical failures:

1. **Cost of False Negatives (FN):**
   A False Negative occurs when a patient with active cardiovascular disease is incorrectly classified as healthy. In healthcare, this is the most critical error. An undiagnosed cardiac patient is discharged without intervention, leading to missed therapeutic windows, progression of coronary damage, and potentially fatal events (e.g., myocardial infarction or stroke). Thus, maximizing **Recall** (Sensitivity) is paramount:
   $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

2. **Cost of False Positives (FP):**
   A False Positive occurs when a healthy patient is flagged as high-risk. While not directly life-threatening, FPs cause significant psychological distress to patients and place a severe burden on healthcare infrastructure by triggering expensive secondary diagnostic procedures (e.g., coronary angiograms, cardiac stress tests) and wasting hospital staff resources. Thus, **Precision** (Positive Predictive Value) must be balanced:
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

3. **$F_1$-Score as the Clinical Balance:**
   The $F_1$-score is the harmonic mean of Precision and Recall:
   $$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$$
   Unlike raw accuracy, which treats all misclassifications equally, the $F_1$-score penalizes imbalances between Precision and Recall. By optimizing for $F_1$, the pipeline ensures we minimize life-threatening false negatives while maintaining high positive predictive value, preventing clinical resource overload.
