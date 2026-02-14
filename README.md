# ML Classification Model Comparison Dashboard

A machine learning project that implements **6 classification models** on the **Breast Cancer Wisconsin (Diagnostic)** dataset and provides an interactive **Streamlit web application** for model comparison, evaluation, and prediction.

**Live App:** [Streamlit App Link](https://ml-assignment-bits.streamlit.app/)

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
| **Instances** | 569 (483 train / 86 test) |
| **Features** | 30 (all numeric, real-valued) |
| **Classes** | 2 — Malignant (212 samples, 37.3%) and Benign (357 samples, 62.7%) |
| **Task** | Binary Classification |
| **Missing Values** | None |

**Features** are computed from digitized images of FNA of breast masses and describe characteristics of cell nuclei:
- **Mean features (10):** radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension
- **SE features (10):** Standard error of the above 10 measurements
- **Worst features (10):** Largest/worst values of the above 10 measurements

**Preprocessing:**
- 85/15 stratified train-test split (483 training, 86 testing samples)
- StandardScaler applied for feature normalization

---

## c. Models Used

### Model Comparison Table (Evaluation Metrics)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9884 | 0.9942 | 0.9886 | 0.9884 | 0.9883 | 0.9753 |
| Decision Tree | 0.9535 | 0.9502 | 0.9535 | 0.9535 | 0.9535 | 0.9005 |
| K-Nearest Neighbors (kNN) | 0.9884 | 0.9977 | 0.9886 | 0.9884 | 0.9883 | 0.9753 |
| Naive Bayes (Gaussian) | 0.9186 | 0.9850 | 0.9183 | 0.9186 | 0.9183 | 0.8250 |
| Random Forest (Ensemble) | 0.9186 | 0.9928 | 0.9222 | 0.9186 | 0.9193 | 0.8313 |
| XGBoost (Ensemble) | 0.9651 | 0.9965 | 0.9652 | 0.9651 | 0.9650 | 0.9252 |

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Tied for the highest accuracy (98.84%) along with kNN and achieved an excellent MCC of 0.9753. Its strong performance indicates that the dataset features are largely linearly separable after standardization. The AUC of 0.9942 confirms excellent discriminative ability between malignant and benign classes. One of the best overall models for this dataset. |
| **Decision Tree** | Achieved 95.35% accuracy with the lowest AUC (0.9502) among all models. Single decision trees tend to overfit on training data by creating overly specific splits. The MCC of 0.9005 is decent but noticeably behind kNN and Logistic Regression. Its balanced precision and recall (both 0.9535) show consistent performance, though it remains the weakest on AUC. |
| **K-Nearest Neighbors (kNN)** | Tied for the highest accuracy (98.84%) with Logistic Regression and achieved the best AUC (0.9977) among all models. kNN benefits greatly from StandardScaler normalization applied during preprocessing. Its perfect 1.00 precision for the malignant class means zero false positives for cancer detection. The k=5 setting provides a strong balance between bias and variance on this feature space. |
| **Naive Bayes (Gaussian)** | Achieved 91.86% accuracy with a high AUC of 0.9850. Despite the naive independence assumption between features (violated here since features like radius and perimeter are correlated), Gaussian NB performs reasonably well. The gap between its AUC (high) and accuracy (lower) suggests well-calibrated probabilities but a slightly suboptimal hard decision boundary. |
| **Random Forest (Ensemble)** | Achieved 91.86% accuracy with an excellent AUC of 0.9928. As an ensemble of 150 decision trees with bagging and random feature subsets, it reduces single-tree overfitting. However, it underperforms the standalone Decision Tree on accuracy in this split, possibly due to majority-vote smoothing diluting strong individual tree predictions on this particular test partition. Its precision (0.9222) exceeds its recall (0.9186), indicating slightly conservative predictions. |
| **XGBoost (Ensemble)** | Achieved 96.51% accuracy with AUC of 0.9965 — the third-best model overall. As a gradient boosting ensemble, XGBoost builds 150 trees sequentially, each correcting prior errors. Its high precision (0.9652) and strong MCC (0.9252) demonstrate robust discriminative power. It outperforms Random Forest here, showing that sequential error-correction (boosting) can beat parallel averaging (bagging) on this data. |

---

## Project Structure

```
classification_models/
│── app.py                    # Streamlit web application (main entry point)
│── requirements.txt          # Python dependencies
│── README.md                 # Project documentation
│── model/
│   ├── __init__.py           # Package initializer
│   ├── ml_models.py          # Core ML functions (training, evaluation, save/load)
│   ├── model_evaluation.py   # Model evaluation script (metrics, plots, reports)
│   ├── cached.py             # Cached functions for Streamlit (auto-loads pkl)
│   └── train_models.py       # Standalone training script (generates pkl files)
│── saved_models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── k-nearest_neighbors_knn.pkl
│   ├── naive_bayes_gaussian.pkl
│   ├── random_forest_ensemble.pkl
│   ├── xgboost_ensemble.pkl
│   ├── scaler.pkl            # Fitted StandardScaler
│   └── artifacts.pkl         # Test data, metrics, confusion matrices, reports
│── data/
│   ├── model_comparison.csv  # Model comparison metrics
│   ├── sample_dataset.csv    # Sample test dataset (86 instances)
│   └── dataset_for_app.csv   # Small dataset for app upload testing (50 instances)
│── evaluation_output/        # Generated by model_evaluation.py
│   ├── metrics_comparison.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   └── model_comparison.csv
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
git clone https://github.com/darshanls/classification_models.git
cd classification_models

# Install dependencies
pip install -r requirements.txt

# (Optional) Run training script to regenerate pkl models and data files
python -m model.train_models

# Launch Streamlit app
streamlit run app.py
```

---

## Technologies Used

- **Python 3.11**
- **Streamlit** — Web application framework
- **scikit-learn** — ML models and evaluation metrics
- **XGBoost** — Gradient boosting classifier
- **Pandas / NumPy** — Data manipulation
- **Matplotlib / Seaborn** — Visualization
