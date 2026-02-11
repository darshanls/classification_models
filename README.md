# Machine Learning Classification Project

## 🎯 Problem Statement

The objective of this project is to build and evaluate multiple classification models on the Wine Quality dataset from Kaggle. We aim to predict wine quality ratings based on physicochemical properties using various machine learning classification algorithms. This multi-class classification problem demonstrates the comparative performance of different ML algorithms on real-world wine quality assessment.

## 📊 Dataset Description

**Dataset:** Wine Quality Dataset (Red and White Wine)

**Source:** Kaggle - https://www.kaggle.com/datasets/rajyellow46/wine-quality

| Property | Value |
|----------|-------|
| **Total Instances** | 6,463 (after cleaning) |
| **Number of Features** | 12 physicochemical properties |
| **Target Variable** | Quality (Multi-class: 3, 4, 5, 6, 7, 8, 9) |
| **Task Type** | Multi-class Classification |

### Features Description

| Feature | Description | Type |
|---------|-------------|------|
| fixed acidity | Tartaric acid content (g/dm³) | Numeric |
| volatile acidity | Acetic acid content (g/dm³) | Numeric |
| citric acid | Citric acid content (g/dm³) | Numeric |
| residual sugar | Residual sugar content (g/dm³) | Numeric |
| chlorides | Chloride content (g/dm³) | Numeric |
| free sulfur dioxide | Free SO₂ (mg/dm³) | Numeric |
| total sulfur dioxide | Total SO₂ (mg/dm³) | Numeric |
| density | Wine density (g/cm³) | Numeric |
| pH | pH value | Numeric |
| sulphates | Potassium sulphate (g/dm³) | Numeric |
| alcohol | Alcohol content (% by volume) | Numeric |
| type | Wine type (red/white) - encoded | Categorical |

### Target Variable
- **quality**: Wine quality rating
  - 3 = Low quality
  - 4 = Below average  
  - 5 = Average
  - 6 = Good
  - 7 = Very good
  - 8 = Excellent
  - 9 = Outstanding

## 🤖 Models Used

Six different machine learning classification models were implemented and evaluated:

1. **Logistic Regression** - Linear classification model with multi-class support
2. **Decision Tree Classifier** - Tree-based non-linear model
3. **K-Nearest Neighbors (KNN)** - Instance-based learning algorithm
4. **Naive Bayes (Gaussian)** - Probabilistic classifier assuming normal distribution
5. **Random Forest (Ensemble)** - Bagging ensemble method with 100 trees
6. **XGBoost (Ensemble)** - Gradient boosting ensemble method

## 📈 Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.5367 | 0.7552 | 0.5130 | 0.5367 | 0.5073 | 0.2636 |
| Decision Tree | 0.6102 | 0.6339 | 0.6088 | 0.6102 | 0.6092 | 0.4209 |
| KNN | 0.5522 | 0.6752 | 0.5342 | 0.5522 | 0.5412 | 0.3146 |
| Naive Bayes | 0.3929 | 0.6383 | 0.4239 | 0.3929 | 0.4042 | 0.1360 |
| Random Forest (Ensemble) | 0.6999 | 0.7967 | 0.7050 | 0.6999 | 0.6877 | 0.5373 |
| XGBoost (Ensemble) | 0.6713 | 0.7840 | 0.6630 | 0.6713 | 0.6629 | 0.4973 |

*Note: Metrics calculated on test set (20% of data) with stratified sampling. Dataset contains 6,463 wine samples with quality ratings 3-9.*

## 📝 Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Provides baseline performance (53.67% accuracy). The linear decision boundaries struggle with the complex non-linear relationships in wine quality assessment. However, it achieves decent AUC (0.7551) indicating reasonable class separation capability. Fast training time and highly interpretable coefficients. |
| Decision Tree | Shows moderate performance (61.02% accuracy) with good AUC (0.6339). The tree captures non-linear relationships between wine properties and quality but tends to overfit specific chemical patterns. Provides interpretable decision rules for wine quality assessment. |
| KNN | Achieves moderate performance (55.22% accuracy) with decent AUC (0.6752). The instance-based approach works reasonably well when similar wines have similar quality ratings. Performance is sensitive to the choice of k=5 and feature scaling of chemical properties. |
| Naive Bayes | Shows lowest performance (39.29% accuracy). The Gaussian assumption is violated for many wine chemical properties which have skewed distributions. The independence assumption between wine features is unrealistic (e.g., acidity and pH are correlated). Very fast training but poor predictive power. |
| Random Forest (Ensemble) | Best overall performance (69.99% accuracy) with high AUC (0.7967). The ensemble of decision trees effectively captures complex interactions between wine chemical properties and quality. Feature randomness provides robustness against outliers in wine measurements. Good balance between performance and interpretability through feature importance. |
| XGBoost (Ensemble) | Strong performance (67.13% accuracy) with high AUC (0.7840). Gradient boosting with regularization effectively models the complex relationships between wine physicochemical properties and quality ratings. Handles the dataset's feature correlations well and provides excellent generalization. |

## 🏗️ Project Structure

```
MLTest/
│-- streamlit_app.py          # Streamlit web application
│-- requirements.txt          # Python dependencies
│-- README.md                 # Project documentation
│-- model/
│   │-- ml_models.py          # ML classifier class implementation
│   │-- train_models.py       # Model training script
│   │-- *.pkl files           # Saved trained models
│-- model/
│   │-- dataset_for_app.csv       # Dataset for Streamlit app
│   │-- model_comparison.csv      # Model performance comparison
```

## 🚀 Streamlit App Features

The deployed Streamlit application includes:

- ✅ **Dataset Upload Option (CSV)** - Upload custom test data for predictions
- ✅ **Model Selection Dropdown** - Choose from 6 different ML models
- ✅ **Display of Evaluation Metrics** - View Accuracy, AUC, Precision, Recall, F1, MCC
- ✅ **Confusion Matrix** - Visual representation of model predictions
- ✅ **Classification Report** - Detailed per-class metrics
- ✅ **Model Comparison Table** - Side-by-side comparison of all models
- ✅ **Interactive Visualizations** - Bar charts, pie charts, and correlation matrices
- ✅ **Prediction Interface** - Generate predictions on uploaded data
- ✅ **Download Results** - Export predictions as CSV

## 🔗 Links

- **GitHub Repository:** [Add your GitHub repo link here]
- **Live Streamlit App:** [Add your deployed Streamlit app link here]

## 💻 Local Setup Instructions

1. Clone the repository:
```bash
git clone [your-repo-url]
cd MLTest
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the models:
```bash
python model/train_models.py
```

4. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

5. Open your browser and navigate to `http://localhost:8501`

## 📦 Requirements

```
streamlit>=1.28.1
scikit-learn>=1.3.2
numpy>=1.26.0
pandas>=2.0.3
matplotlib>=3.7.2
seaborn>=0.12.2
xgboost>=1.7.6
plotly>=5.17.0
```

## 🎓 Assignment Information

- **Course:** Machine Learning
- **Assignment:** Classification Models with Streamlit Deployment
- **Evaluation Metrics:** Accuracy, AUC, Precision, Recall, F1 Score, MCC
- **Dataset Requirements:** ✅ 12+ features, ✅ 500+ instances
- **Models Implemented:** ✅ All 6 required classification models

## 📊 Key Insights

1. **Ensemble Methods Dominate**: Random Forest achieves best performance (69.99% accuracy) for wine quality prediction
2. **Random Forest is Best**: Achieves highest scores across most metrics (69.99% accuracy, 0.7967 AUC, 0.5373 MCC)
3. **Wine Quality is Complex**: Multi-class nature (7 quality levels) makes prediction challenging compared to binary classification
4. **Linear Models Limited**: Logistic Regression struggles with complex non-linear relationships between chemical properties
5. **Naive Bayes Poor Fit**: Gaussian assumptions violated by wine chemical property distributions
6. **Feature Interactions Matter**: Tree-based models capture complex interactions between acidity, alcohol, and other compounds

## 📜 License

This project is created for educational purposes as part of a Machine Learning course assignment.

---
*Built with ❤️ using Python, Scikit-learn, XGBoost, and Streamlit*
