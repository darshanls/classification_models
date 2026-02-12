"""
Cached functions for the Streamlit app.
Separated from app.py so inspect.getsource() tokenizes this clean file
instead of app.py which contains complex CSS animation strings.
"""

import streamlit as st
import pandas as pd
from model.ml_models import (
    load_dataset, preprocess_data, train_and_evaluate_all,
    save_trained_artifacts, load_trained_artifacts
)


@st.cache_data
def get_trained_results():
    """Load models from pkl if available, otherwise train + save."""
    X, y, feature_names, target_names = load_dataset()

    # Try loading pre-trained models from pkl files
    loaded = load_trained_artifacts()
    if loaded is not None:
        results, X_test, y_test, scaler = loaded
        return results, X_test, y_test, feature_names, target_names, scaler

    # Fallback: train from scratch and save pkl files
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    results = train_and_evaluate_all(X_train, X_test, y_train, y_test)
    save_trained_artifacts(results, X_test, y_test, scaler)
    return results, X_test, y_test, feature_names, target_names, scaler


@st.cache_data
def get_comparison_df(_results):
    """Build a comparison DataFrame from results."""
    rows = []
    for name, res in _results.items():
        rows.append({'Model': name, **res['metrics']})
    return pd.DataFrame(rows)
