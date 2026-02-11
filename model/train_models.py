"""
Script to train all classification models and generate comparison metrics.
Dataset: Breast Cancer Wisconsin (Diagnostic) - UCI/sklearn

Run: python -m model.train_models
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from model.ml_models import load_dataset, preprocess_data, train_and_evaluate_all


def main():
    print("=" * 60)
    print("Classification Model Training & Evaluation")
    print("Dataset: Breast Cancer Wisconsin (Diagnostic)")
    print("=" * 60)

    # Load dataset
    X, y, feature_names, target_names = load_dataset()
    print(f"\nDataset Shape: {X.shape}")
    print(f"Number of Features: {len(feature_names)}")
    print(f"Classes: {list(target_names)}")
    print(f"Class Distribution:\n{y.value_counts().to_string()}")

    # Preprocess
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Train all models
    print("\n" + "=" * 60)
    print("Training Models...")
    print("=" * 60)
    results = train_and_evaluate_all(X_train, X_test, y_train, y_test)

    # Build comparison table
    print("\n" + "=" * 80)
    print("MODEL COMPARISON TABLE")
    print("=" * 80)

    comparison_data = []
    for name, result in results.items():
        metrics = result['metrics']
        comparison_data.append({'Model': name, **metrics})

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))

    # Save comparison to CSV
    os.makedirs('data', exist_ok=True)
    comparison_df.to_csv('data/model_comparison.csv', index=False)
    print("\nComparison saved to data/model_comparison.csv")

    # Save sample test dataset for app upload testing
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['target'] = y_test.values
    test_df.to_csv('data/sample_dataset.csv', index=False)
    print("Sample test dataset saved to data/sample_dataset.csv")

    # Save a smaller subset for quick app upload testing
    app_df = pd.DataFrame(X.values[:50], columns=feature_names)
    app_df['target'] = y.values[:50]
    app_df.to_csv('data/dataset_for_app.csv', index=False)
    print("App test dataset saved to data/dataset_for_app.csv")

    # Print detailed classification reports
    for name, result in results.items():
        print(f"\n{'=' * 60}")
        print(f"Classification Report: {name}")
        print("=" * 60)
        print(result['classification_report'])


if __name__ == '__main__':
    main()
