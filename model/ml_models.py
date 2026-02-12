"""
Core ML functions for training and evaluating classification models.
Dataset: Breast Cancer Wisconsin (Diagnostic) - UCI/sklearn
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)


def load_dataset():
    """Load and return the Breast Cancer Wisconsin dataset."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    return X, y, data.feature_names, data.target_names


def preprocess_data(X, y, test_size=0.15, random_state=85):
    """Split and scale the data."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def get_models():
    """Return a dictionary of all 6 classification models."""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=10000, random_state=85),
        'Decision Tree': DecisionTreeClassifier(random_state=85),
        'K-Nearest Neighbors (kNN)': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes (Gaussian)': GaussianNB(),
        'Random Forest (Ensemble)': RandomForestClassifier(n_estimators=150, random_state=85),
        'XGBoost (Ensemble)': XGBClassifier(
            n_estimators=150,
            random_state=85,
            eval_metric='logloss'
        )
    }
    return models


def evaluate_model(model, X_test, y_test):
    """Calculate all 6 evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)

    # Get prediction probabilities for AUC
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = None

    metrics = {
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'AUC': round(auc, 4) if auc is not None else 'N/A',
        'Precision': round(precision_score(y_test, y_pred, average='weighted'), 4),
        'Recall': round(recall_score(y_test, y_pred, average='weighted'), 4),
        'F1 Score': round(f1_score(y_test, y_pred, average='weighted'), 4),
        'MCC': round(matthews_corrcoef(y_test, y_pred), 4)
    }

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Malignant', 'Benign'])

    return metrics, cm, report, y_pred


def train_and_evaluate_all(X_train, X_test, y_train, y_test):
    """Train all 6 models and return comprehensive results."""
    models = get_models()
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics, cm, report, y_pred = evaluate_model(model, X_test, y_test)
        results[name] = {
            'model': model,
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_pred
        }

    return results


# ─── Model Persistence (pkl) ───

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'saved_models')


def save_trained_artifacts(results, X_test, y_test, scaler):
    """Save trained models, scaler, and test data to pkl files."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save each model individually
    for name, res in results.items():
        safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        filepath = os.path.join(MODELS_DIR, f'{safe_name}.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(res['model'], f)

    # Save scaler
    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # Save test data and full results metadata (metrics, cm, report, predictions)
    artifacts = {
        'X_test': X_test,
        'y_test': y_test,
        'results_meta': {
            name: {
                'metrics': res['metrics'],
                'confusion_matrix': res['confusion_matrix'],
                'classification_report': res['classification_report'],
                'predictions': res['predictions']
            } for name, res in results.items()
        }
    }
    with open(os.path.join(MODELS_DIR, 'artifacts.pkl'), 'wb') as f:
        pickle.dump(artifacts, f)


def load_trained_artifacts():
    """Load trained models and artifacts from pkl files. Returns None if files missing."""
    artifacts_path = os.path.join(MODELS_DIR, 'artifacts.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')

    if not os.path.exists(artifacts_path) or not os.path.exists(scaler_path):
        return None

    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Load artifacts (test data + metrics)
    with open(artifacts_path, 'rb') as f:
        artifacts = pickle.load(f)

    # Load each model and reconstruct full results dict
    results = {}
    for name, meta in artifacts['results_meta'].items():
        safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        model_path = os.path.join(MODELS_DIR, f'{safe_name}.pkl')
        if not os.path.exists(model_path):
            return None
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        results[name] = {
            'model': model,
            'metrics': meta['metrics'],
            'confusion_matrix': meta['confusion_matrix'],
            'classification_report': meta['classification_report'],
            'predictions': meta['predictions']
        }

    return results, artifacts['X_test'], artifacts['y_test'], scaler
