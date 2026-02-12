"""
Model Evaluation Script
Dataset: Breast Cancer Wisconsin (Diagnostic) - UCI/Kaggle
Evaluates 6 classification models with comprehensive metrics and visualizations.

Run: python -m model.model_evaluation
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
    roc_curve, auc
)
from model.ml_models import (
    load_dataset, preprocess_data, get_models, evaluate_model,
    train_and_evaluate_all, load_trained_artifacts, save_trained_artifacts
)


def print_header(title):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def evaluate_all_models():
    """Main evaluation pipeline."""

    # ─── 1. Load Dataset ───
    print_header("1. DATASET LOADING")
    X, y, feature_names, target_names = load_dataset()
    print(f"Dataset Shape       : {X.shape}")
    print(f"Number of Features  : {len(feature_names)}")
    print(f"Target Classes      : {list(target_names)}")
    print(f"Class Distribution  :")
    print(f"  - Malignant (0)   : {(y == 0).sum()}")
    print(f"  - Benign    (1)   : {(y == 1).sum()}")

    # ─── 2. Preprocessing ───
    print_header("2. PREPROCESSING (85/15 SPLIT)")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    print(f"Training samples    : {len(y_train)}")
    print(f"Testing samples     : {len(y_test)}")
    print(f"Scaling             : StandardScaler (fit on train, transform both)")

    # ─── 3. Try loading pre-trained models, else train fresh ───
    print_header("3. MODEL TRAINING")
    loaded = load_trained_artifacts()
    if loaded is not None:
        results, X_test_loaded, y_test_loaded, scaler_loaded = loaded
        print("Loaded pre-trained models from saved_models/*.pkl")
        # Use the saved test data for consistency
        X_test = X_test_loaded
        y_test = y_test_loaded
        scaler = scaler_loaded
    else:
        print("Training all 6 models from scratch...")
        results = train_and_evaluate_all(X_train, X_test, y_train, y_test)
        save_trained_artifacts(results, X_test, y_test, scaler)
        print("Models trained and saved to saved_models/*.pkl")

    # ─── 4. Metrics Comparison Table ───
    print_header("4. METRICS COMPARISON TABLE")
    comparison_data = []
    for name, res in results.items():
        comparison_data.append({'Model': name, **res['metrics']})
    comp_df = pd.DataFrame(comparison_data)
    print(comp_df.to_string(index=False))

    # ─── 5. Individual Model Evaluation ───
    print_header("5. INDIVIDUAL MODEL EVALUATION")
    for name, res in results.items():
        print(f"\n{'─' * 60}")
        print(f"  Model: {name}")
        print(f"{'─' * 60}")

        # Metrics
        print("\n  Metrics:")
        for metric, value in res['metrics'].items():
            if isinstance(value, float):
                print(f"    {metric:<12}: {value:.4f}")
            else:
                print(f"    {metric:<12}: {value}")

        # Confusion Matrix
        cm = res['confusion_matrix']
        print(f"\n  Confusion Matrix:")
        print(f"                  Predicted")
        print(f"                  Malignant  Benign")
        print(f"    Actual Malig.   {cm[0][0]:>5}     {cm[0][1]:>5}")
        print(f"    Actual Benign   {cm[1][0]:>5}     {cm[1][1]:>5}")

        # Classification Report
        print(f"\n  Classification Report:")
        for line in res['classification_report'].split('\n'):
            print(f"    {line}")

    # ─── 6. Best Model Summary ───
    print_header("6. BEST MODEL SUMMARY")
    metric_names = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']
    for metric in metric_names:
        best_model = None
        best_val = -1
        for name, res in results.items():
            val = res['metrics'].get(metric)
            if isinstance(val, float) and val > best_val:
                best_val = val
                best_model = name
        if best_model:
            print(f"  Best {metric:<12}: {best_model} ({best_val:.4f})")

    # ─── 7. Generate Evaluation Plots ───
    print_header("7. GENERATING EVALUATION PLOTS")
    os.makedirs('evaluation_output', exist_ok=True)

    # 7a. Metrics Comparison Bar Chart
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Model Comparison - All Metrics', fontsize=16, fontweight='bold')
    colors = ['#667eea', '#764ba2', '#4facfe', '#43e97b', '#fa709a', '#f6d365']

    for idx, metric in enumerate(metric_names):
        ax = axes[idx // 3][idx % 3]
        values = []
        names = []
        for name, res in results.items():
            val = res['metrics'].get(metric)
            if isinstance(val, float):
                values.append(val)
                names.append(name.split('(')[0].strip())
            else:
                values.append(0)
                names.append(name.split('(')[0].strip())
        bars = ax.barh(names, values, color=colors)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1.05)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('evaluation_output/metrics_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: evaluation_output/metrics_comparison.png")

    # 7b. Confusion Matrices
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold')

    for idx, (name, res) in enumerate(results.items()):
        ax = axes[idx // 3][idx % 3]
        sns.heatmap(res['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Malignant', 'Benign'],
                    yticklabels=['Malignant', 'Benign'], ax=ax)
        ax.set_title(name.split('(')[0].strip(), fontsize=11, fontweight='bold')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')

    plt.tight_layout()
    plt.savefig('evaluation_output/confusion_matrices.png', dpi=150, bbox_inches='tight')
    print("  Saved: evaluation_output/confusion_matrices.png")

    # 7c. ROC Curves
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title('ROC Curves - All Models', fontsize=14, fontweight='bold')

    for idx, (name, res) in enumerate(results.items()):
        model = res['model']
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[idx], lw=2,
                    label=f'{name.split("(")[0].strip()} (AUC={roc_auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('evaluation_output/roc_curves.png', dpi=150, bbox_inches='tight')
    print("  Saved: evaluation_output/roc_curves.png")

    # ─── 8. Save comparison CSV ───
    comp_df.to_csv('evaluation_output/model_comparison.csv', index=False)
    print(f"\n  Saved: evaluation_output/model_comparison.csv")

    print_header("EVALUATION COMPLETE")
    print(f"  All outputs saved to: evaluation_output/")
    print(f"  Files generated:")
    print(f"    - metrics_comparison.png")
    print(f"    - confusion_matrices.png")
    print(f"    - roc_curves.png")
    print(f"    - model_comparison.csv")


if __name__ == '__main__':
    evaluate_all_models()
