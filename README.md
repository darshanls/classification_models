# ML Classification Model Comparison Dashboard

A machine learning project that implements **6 classification models** on the **Breast Cancer Wisconsin (Diagnostic)** dataset and provides an interactive **Streamlit web application** for model comparison, evaluation, and prediction.

**Live App:** [Streamlit App Link](#) *(update after deployment)*

---

## a. Problem Statement

Breast cancer is one of the most common cancers affecting women worldwide. Early and accurate diagnosis is critical for improving patient outcomes and survival rates. This project applies six different machine learning classification algorithms to the Breast Cancer Wisconsin (Diagnostic) dataset to classify tumours as **Malignant** or **Benign** based on 30 numeric features computed from digitized images of fine needle aspirate (FNA) of breast masses.

The goal is to compare model performance across multiple evaluation metrics and identify the best-suited algorithm for this binary classification task.

---

## b. Dataset Description

| Property | Details |
|----------|---------|
| **Name** | Breast Cancer Wisconsin (Diagnostic) |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) / sklearn |
| **Instances** | 569 (426 train / 143 test) |
| **Features** | 30 (all numeric, real-valued) |
| **Classes** | 2 — Malignant (212 samples, 37.3%) and Benign (357 samples, 62.7%) |
| **Task** | Binary Classification |
| **Missing Values** | None |

**Features** are computed from digitized images of FNA of breast masses and describe characteristics of cell nuclei:
- **Mean features (10):** radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension
- **SE features (10):** Standard error of the above 10 measurements
- **Worst features (10):** Largest/worst values of the above 10 measurements

**Preprocessing:**
- 75/25 stratified train-test split (426 training, 143 testing samples)
- StandardScaler applied for feature normalization

---

## c. Models Used

### Model Comparison Table (Evaluation Metrics)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9930 | 0.9973 | 0.9931 | 0.9930 | 0.9930 | 0.9851 |
| Decision Tree | 0.9510 | 0.9534 | 0.9524 | 0.9510 | 0.9513 | 0.8972 |
| K-Nearest Neighbors (kNN) | 0.9860 | 0.9964 | 0.9863 | 0.9860 | 0.9860 | 0.9702 |
| Naive Bayes (Gaussian) | 0.9301 | 0.9904 | 0.9299 | 0.9301 | 0.9298 | 0.8493 |
| Random Forest (Ensemble) | 0.9441 | 0.9948 | 0.9449 | 0.9441 | 0.9443 | 0.8814 |
| XGBoost (Ensemble) | 0.9720 | 0.9971 | 0.9722 | 0.9720 | 0.9719 | 0.9400 |

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Achieved the highest accuracy (99.30%) and the best MCC (0.9851) among all models. Its strong performance indicates that the dataset features are largely linearly separable after standardization. The near-perfect AUC (0.9973) confirms excellent discriminative ability between malignant and benign classes. This is the best overall model for this dataset. |
| **Decision Tree** | Achieved 95.10% accuracy with the lowest AUC (0.9534) among all models. Single decision trees tend to overfit on training data by creating overly specific splits. The MCC of 0.8972 is decent but noticeably behind kNN and Logistic Regression. Its recall of 0.96 for malignant class is good, but precision lags at 0.91, indicating some false positives. |
| **K-Nearest Neighbors (kNN)** | Achieved an impressive 98.60% accuracy — the second-best performer — with an AUC of 0.9964. kNN benefits greatly from StandardScaler normalization applied during preprocessing. Its perfect 1.00 precision for malignant class means zero false positives for cancer detection. The k=5 setting provides a strong balance between bias and variance on this feature space. |
| **Naive Bayes (Gaussian)** | Achieved 93.01% accuracy with a high AUC of 0.9904. Despite the naive independence assumption between features (violated here since features like radius and perimeter are correlated), Gaussian NB performs reasonably well. The gap between its AUC (high) and accuracy (lower) suggests well-calibrated probabilities but a slightly suboptimal hard decision boundary. |
| **Random Forest (Ensemble)** | Achieved 94.41% accuracy with an excellent AUC of 0.9948. As an ensemble of 150 decision trees with bagging and random feature subsets, it reduces single-tree overfitting. However, it slightly underperforms the standalone Decision Tree on accuracy in this split, possibly due to majority-vote smoothing diluting strong individual tree predictions on this particular test partition. |
| **XGBoost (Ensemble)** | Achieved 97.20% accuracy with AUC of 0.9971 — the third-best model overall. As a gradient boosting ensemble, XGBoost builds 150 trees sequentially, each correcting prior errors. Its high precision (0.98 for malignant) and strong MCC (0.9400) demonstrate robust discriminative power. It outperforms Random Forest here, showing that sequential error-correction (boosting) can beat parallel averaging (bagging) on this data. |

---

## Project Structure

```
classification_models/
│── app.py                    # Streamlit web application (main entry point)
│── requirements.txt          # Python dependencies
│── README.md                 # Project documentation
│── model/
│   ├── __init__.py           # Package initializer
│   ├── ml_models.py          # Core ML functions (training, evaluation)
│   └── train_models.py       # Standalone training script
│── data/
│   ├── model_comparison.csv  # Model comparison metrics
│   ├── sample_dataset.csv    # Sample test dataset (114 instances)
│   └── dataset_for_app.csv   # Small dataset for app upload testing (50 instances)
```

---

## Streamlit App Features

1. **Dataset Upload (CSV):** Upload test data in CSV format to evaluate models on custom data
2. **Model Selection Dropdown:** Choose from 6 trained classification models
3. **Evaluation Metrics Display:** View Accuracy, AUC, Precision, Recall, F1 Score, and MCC for each model
4. **Confusion Matrix & Classification Report:** Visual confusion matrix heatmap and detailed classification report
5. **ROC Curves:** Compare ROC curves of all models on a single plot
6. **Visual Comparison Charts:** Bar charts comparing all metrics across all models
7. **Prediction Download:** Download predictions as CSV after running inference on uploaded data

---

## How to Run Locally

```bash
# Clone the repository
git clone https://github.com/<your-username>/classification_models.git
cd classification_models

# Install dependencies
pip install -r requirements.txt

# (Optional) Run training script to regenerate data files
python -m model.train_models

# Launch Streamlit app
streamlit run app.py
```

---

## Deployment

Deployed on **Streamlit Community Cloud**:
1. Push code to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in with GitHub → New App → Select repository → Choose `app.py` → Deploy

---

## Technologies Used

- **Python 3.11**
- **Streamlit** — Web application framework
- **scikit-learn** — ML models and evaluation metrics
- **XGBoost** — Gradient boosting classifier
- **Pandas / NumPy** — Data manipulation
- **Matplotlib / Seaborn** — Visualization
