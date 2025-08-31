# Credit Risk Prediction Using a Hybrid Attention-Based Deep Learning Model

## Project Overview

This project develops a hybrid attention-based deep learning model for predicting credit risk from structured financial data. The model integrates feature embeddings, multi-layer perceptrons (MLPs), and attention mechanisms to capture complex relationships in tabular loan applicant data. Its primary goal is to predict the probability of borrower default, improving accuracy and interpretability relative to traditional methods.

The model was trained and evaluated on the Lending Club dataset (855,969 records, 81 features) and tested on additional holdout and unseen datasets to validate generalization.

---

## Objectives

* Develop a model capable of accurately predicting borrower default.
* Improve performance over baseline logistic regression, particularly in imbalanced and feature-limited scenarios.
* Increase interpretability through attention mechanisms highlighting key feature interactions.
* Assess model generalization on new, previously unseen data.

---

## Tools and Technologies

* **Programming Language:** Python
* **Libraries:** scikit-learn, PyTorch, NumPy, Pandas, Matplotlib, Seaborn
* **Machine Learning Techniques:** Logistic Regression (baseline), Multi-Layer Perceptrons, Attention Networks, Ensemble Methods
* **Evaluation Metrics:** ROC-AUC, Accuracy, F1-score, Precision, Recall

---

## Data

### Primary Dataset

* **Source:** Lending Club (Mehta, 2020)
* **Size:** 855,969 records, 73 original features
* **Preprocessing:**

  * Removed features with excessive missing values or high cardinality
  * Addressed missing values using imputation or missing indicators
  * Encoded categorical variables (ordinal and one-hot encoding)
  * Normalized numerical features with PowerTransformer and StandardScaler
  * Applied random undersampling to address class imbalance

### Additional Test Datasets

* **Subset Dataset:** Removed highly predictive loan grade features to evaluate model robustness under missing information.
* **New Dataset:** Independent Kaggle dataset (32,581 records) with limited features to test generalization to unseen data.

---

## Model Architecture

The **Hybrid Attention Network** consists of two parallel components:

1. **Attention Branch:** Uses feature embeddings and multi-head attention to capture interactions between categorical and numerical features.
2. **MLP Branch:** Processes numerical features through configurable hidden layers with ReLU activations and dropout.

Outputs from both branches are fused and passed through a sigmoid layer to produce a probability of default. Ensembles of multiple model instances, weighted by validation performance, further enhance stability and accuracy.

---

## Baseline Model

* **Model:** Logistic Regression
* **Configuration:** L2 regularization, class\_weight='balanced', solver='liblinear', max\_iter=1000
* **Purpose:** Provides a transparent, interpretable benchmark for evaluating the performance of the hybrid deep learning model.

---

## Evaluation

Models were assessed on multiple datasets using ROC-AUC, accuracy, F1-score, precision, and recall:

* The hybrid attention ensemble consistently outperformed logistic regression across all metrics.
* Gains were largest for recall on the minority class, highlighting improved handling of imbalanced data.
* Attention heatmaps revealed meaningful feature interactions, such as the influence of debt consolidation on default probability.
* Performance decreased slightly on the new dataset, reflecting distribution shifts and underscoring the importance of continuous monitoring and retraining.

**Key Results (Ensemble vs Logistic Regression):**

| Dataset     | ROC-AUC | Accuracy | F1 (Macro) |
| ----------- | ------- | -------- | ---------- |
| Train       | 0.829   | 0.759    | 0.778      |
| Validation  | 0.830   | 0.759    | 0.778      |
| Test        | 0.822   | 0.754    | 0.774      |
| Test Subset | 0.821   | 0.755    | 0.774      |
| New Dataset | 0.748   | 0.647    | 0.689      |

---

## Interpretability

Attention mechanisms highlighted key feature relationships consistent with domain knowledge. For example, debt consolidation interacted with other loan purposes and credit indicators, providing transparency and justifying model predictions.

---

## Limitations and Ethical Considerations

* **Deployment Bias:** Model is intended solely for credit risk assessment and should not be applied to unrelated decisions such as hiring or insurance.
* **Data Bias:** Historical Lending Club data contains socioeconomic and geographic biases, which could propagate unfair outcomes.
* **Generalization:** Performance decreases when features are missing or distributions shift, emphasizing the need for monitoring and retraining.
* **False Positives/Negatives:** Incorrect predictions can result in financial loss for lenders or unfair denial for applicants; explainability methods are recommended to mitigate risks.

---

## Future Work

* Integrate SHAP or other explainability techniques for actionable insights.
* Extend model for real-time credit scoring deployment.
* Implement adaptive retraining pipelines to address evolving borrower profiles and economic conditions.
