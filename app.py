"""
ML Classification Model Comparison Dashboard
Dataset: Breast Cancer Wisconsin (Diagnostic) - UCI/sklearn
Built with Streamlit | 6 Models x 6 Metrics
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from model.ml_models import (
    load_dataset, preprocess_data, get_models,
    evaluate_model, train_and_evaluate_all
)
from model.cached import get_trained_results, get_comparison_df

# ─── Page Configuration ───
st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    /* ─── Keyframe Animations ─── */
    @keyframes shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInLeft {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    @keyframes scaleIn {
        from { opacity: 0; transform: scale(0.85); }
        to { opacity: 1; transform: scale(1); }
    }
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(40px); }
        to { opacity: 1; transform: translateX(0); }
    }
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3); }
        50% { box-shadow: 0 4px 25px rgba(102, 126, 234, 0.5); }
    }

    .main-title {
        font-size: 2.4rem !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #667eea 100%) !important;
        background-size: 200% auto !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        text-align: center !important;
        margin-bottom: 0.2rem !important;
        font-family: 'Inter', sans-serif !important;
        line-height: 1.2 !important;
        animation: shimmer 3s ease-in-out infinite !important;
    }
    .sub-title {
        font-size: 1.05rem !important;
        color: #6b7280 !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
        font-family: 'Inter', sans-serif !important;
        line-height: 1.4 !important;
        animation: fadeInUp 0.8s ease-out 0.3s both !important;
    }
    .metric-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: scaleIn 0.5s ease-out both;
    }
    .metric-container:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e3a5f;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 0.3rem;
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1e3a5f;
        border-left: 4px solid #667eea;
        padding-left: 12px;
        margin: 1.5rem 0 1rem 0;
        animation: fadeInLeft 0.6s ease-out both;
    }
    .info-box {
        background-color: #f0f4ff;
        border: 1px solid #d0d9f5;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
    }
    
    /* Model control panel title */
    .panel-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #667eea;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 0.5rem;
    }
    .model-card {
        background: white;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        text-align: center;
        border: 2px solid transparent;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: all 0.3s ease;
        cursor: default;
        animation: scaleIn 0.5s ease-out both;
    }
    .model-card:nth-child(1) { animation-delay: 0.05s; }
    .model-card:nth-child(2) { animation-delay: 0.1s; }
    .model-card:nth-child(3) { animation-delay: 0.15s; }
    .model-card:nth-child(4) { animation-delay: 0.2s; }
    .model-card:nth-child(5) { animation-delay: 0.25s; }
    .model-card:nth-child(6) { animation-delay: 0.3s; }
    .model-card:hover {
        transform: translateY(-4px) scale(1.03);
        box-shadow: 0 6px 18px rgba(102, 126, 234, 0.2);
    }
    .model-card.active {
        border-color: #667eea;
        background: linear-gradient(135deg, #f0f4ff 0%, #e8edff 100%);
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.2);
        animation: scaleIn 0.5s ease-out both, pulseGlow 2s ease-in-out infinite;
    }
    .model-card .card-icon {
        font-size: 1.6rem;
        margin-bottom: 0.3rem;
    }
    .model-card .card-name {
        font-size: 0.8rem;
        font-weight: 600;
        color: #1e3a5f;
        line-height: 1.2;
        min-height: 2rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .model-card .card-acc {
        font-size: 1.1rem;
        font-weight: 700;
        color: #667eea;
        margin-top: 0.2rem;
    }
    .model-card .card-acc-label {
        font-size: 0.65rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    /* Responsive model cards grid */
    .model-cards-grid {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 0.6rem;
        margin-top: 0.5rem;
    }
    @media (max-width: 992px) {
        .model-cards-grid {
            grid-template-columns: repeat(3, 1fr);
        }
    }
    @media (max-width: 576px) {
        .model-cards-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    .selected-model-banner {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin-top: 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        animation: slideInRight 0.6s ease-out both;
    }
    .selected-model-banner .banner-label {
        font-size: 0.75rem;
        opacity: 0.85;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .selected-model-banner .banner-name {
        font-size: 1.15rem;
        font-weight: 700;
    }
    .selected-model-banner .banner-badge {
        background: rgba(255,255,255,0.2);
        border-radius: 20px;
        padding: 0.3rem 0.8rem;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* Selectbox styling */
    .stSelectbox > div[data-baseweb="select"] {
        border-radius: 10px;
        border: 2px solid #667eea;
    }
    .stSelectbox > div[data-baseweb="select"] > div {
        background-color: #f8f9fa;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Cached functions imported from model/cached.py ───
# (Separated to avoid inspect.getsource TokenError with complex CSS in this file)


def render_metric_card(label, value, color="#667eea"):
    """Render a styled metric card."""
    if isinstance(value, float):
        display = f"{value:.4f}"
    else:
        display = str(value)
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value" style="color: {color};">{display}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """Plot a styled confusion matrix."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Malignant', 'Benign'],
        yticklabels=['Malignant', 'Benign'],
        linewidths=0.5, linecolor='white',
        annot_kws={"size": 14, "weight": "bold"},
        ax=ax
    )
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    return fig


def plot_roc_curves(results, X_test, y_test):
    """Plot ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#667eea', '#f093fb', '#4facfe', '#43e97b', '#fa709a', '#fee140']

    for i, (name, res) in enumerate(results.items()):
        model = res['model']
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i % len(colors)],
                    lw=2, label=f'{name} (AUC={roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    ax.set_title('ROC Curves — All Models', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_metric_comparison(comp_df):
    """Plot bar chart comparing all metrics across models."""
    metrics_cols = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']
    df_melted = comp_df.melt(id_vars='Model', value_vars=metrics_cols,
                              var_name='Metric', value_name='Score')
    df_melted['Score'] = pd.to_numeric(df_melted['Score'], errors='coerce')

    fig, ax = plt.subplots(figsize=(12, 5))
    palette = ['#667eea', '#f093fb', '#4facfe', '#43e97b', '#fa709a', '#fee140']
    sns.barplot(data=df_melted, x='Metric', y='Score', hue='Model',
                palette=palette, ax=ax, edgecolor='white')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylim(0, 1.1)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════
#  MAIN APP — SINGLE PAGE LAYOUT
# ════════════════════════════════════════════════════════════

# Title
st.markdown('<p class="main-title">🔬 ML Classification Model Comparison</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Breast Cancer Wisconsin (Diagnostic) Dataset &nbsp;|&nbsp; 6 Models &times; 6 Metrics</p>', unsafe_allow_html=True)

# Load trained results
results, X_test, y_test, feature_names, target_names, scaler = get_trained_results()
comp_df = get_comparison_df(results)


# ─── Sidebar: Data Upload ───
with st.sidebar:
    st.markdown("### 📂 Data Source")
    st.markdown("**📤 Upload Custom Test Data**")
    st.markdown(
        "Upload a CSV with the same **30 features**. "
        "Include a `target` column to evaluate all models on your data.\n\n"
        "A sample file is also available in the GitHub **`data/`** folder."
    )
    sample_csv_path = os.path.join(os.path.dirname(__file__), 'data', 'dataset_for_app.csv')
    if os.path.exists(sample_csv_path):
        with open(sample_csv_path, 'rb') as f:
            sample_csv_bytes = f.read()
        st.download_button(
            label="⬇️ Download Sample CSV",
            data=sample_csv_bytes,
            file_name="dataset_for_app.csv",
            mime="text/csv",
        )
    uploaded_file = st.file_uploader(
        "Choose a CSV file", type=['csv'],
        help="30 numeric features. Include 'target' column for evaluation."
    )
    if uploaded_file is not None:
        # Reset run state if a different file is uploaded
        if st.session_state.get('_upload_name') != uploaded_file.name:
            st.session_state['_upload_run'] = False
            st.session_state['_upload_name'] = uploaded_file.name
        st.markdown(
            """<style>
            div.stSidebar button[kind="secondary"] { background-color: #43e97b; color: #1e3a5f;
            font-weight: 600; border: none; }
            div.stSidebar button[kind="secondary"]:hover { background-color: #38d96e; }
            </style>""", unsafe_allow_html=True
        )
        if st.button("▶️ Run Predictions", use_container_width=True):
            st.session_state['_upload_run'] = True

# ─── Determine Active Data Source ───
active_results = results
active_X_test = X_test
active_y_test = y_test
active_comp_df = comp_df
data_mode = "default"
upload_predictions = {}

if uploaded_file is not None and st.session_state.get('_upload_run', False):
    try:
        df_upload = pd.read_csv(uploaded_file)
        expected_features = list(feature_names)

        # Auto-detect Kaggle-style column names (e.g. radius_mean → mean radius)
        # and rename them to match sklearn's load_breast_cancer() feature names
        if 'radius_mean' in df_upload.columns and 'mean radius' not in df_upload.columns:
            kaggle_to_sklearn = {
                'radius_mean': 'mean radius', 'texture_mean': 'mean texture',
                'perimeter_mean': 'mean perimeter', 'area_mean': 'mean area',
                'smoothness_mean': 'mean smoothness', 'compactness_mean': 'mean compactness',
                'concavity_mean': 'mean concavity', 'concave points_mean': 'mean concave points',
                'symmetry_mean': 'mean symmetry', 'fractal_dimension_mean': 'mean fractal dimension',
                'radius_se': 'radius error', 'texture_se': 'texture error',
                'perimeter_se': 'perimeter error', 'area_se': 'area error',
                'smoothness_se': 'smoothness error', 'compactness_se': 'compactness error',
                'concavity_se': 'concavity error', 'concave points_se': 'concave points error',
                'symmetry_se': 'symmetry error', 'fractal_dimension_se': 'fractal dimension error',
                'radius_worst': 'worst radius', 'texture_worst': 'worst texture',
                'perimeter_worst': 'worst perimeter', 'area_worst': 'worst area',
                'smoothness_worst': 'worst smoothness', 'compactness_worst': 'worst compactness',
                'concavity_worst': 'worst concavity', 'concave points_worst': 'worst concave points',
                'symmetry_worst': 'worst symmetry', 'fractal_dimension_worst': 'worst fractal dimension',
            }
            df_upload.rename(columns=kaggle_to_sklearn, inplace=True)

        # Auto-convert Kaggle 'diagnosis' column (M/B) to numeric 'target' (0/1)
        if 'diagnosis' in df_upload.columns and 'target' not in df_upload.columns:
            df_upload['target'] = df_upload['diagnosis'].map({'M': 0, 'B': 1})
            df_upload.drop(columns=['diagnosis'], inplace=True)

        feature_cols = [c for c in expected_features if c in df_upload.columns]

        if len(feature_cols) == 0:
            with st.sidebar:
                st.error("No matching features found in uploaded CSV.")
        else:
            X_upload = df_upload[feature_cols]
            if len(feature_cols) < len(expected_features):
                X_upload = X_upload.reindex(columns=expected_features, fill_value=0)
                with st.sidebar:
                    st.warning(f"Missing {len(expected_features) - len(feature_cols)} features (filled with 0).")

            X_upload_scaled = scaler.transform(X_upload)
            has_target = 'target' in df_upload.columns

            if has_target:
                y_upload = df_upload['target'].values
                upload_results = {}
                for name, res_item in results.items():
                    m = res_item['model']
                    u_metrics, u_cm, u_report, u_pred = evaluate_model(m, X_upload_scaled, y_upload)
                    upload_results[name] = {
                        'model': m,
                        'metrics': u_metrics,
                        'confusion_matrix': u_cm,
                        'classification_report': u_report,
                        'predictions': u_pred
                    }
                active_results = upload_results
                active_X_test = X_upload_scaled
                active_y_test = y_upload
                rows = []
                for name, res in upload_results.items():
                    rows.append({'Model': name, **res['metrics']})
                active_comp_df = pd.DataFrame(rows)
                data_mode = "uploaded"
            else:
                for name, res_item in results.items():
                    m = res_item['model']
                    upload_predictions[name] = m.predict(X_upload_scaled)
                data_mode = "uploaded_no_target"

            with st.sidebar:
                st.success(f"✅ {df_upload.shape[0]} rows × {df_upload.shape[1]} cols")
                with st.expander("Preview Uploaded Data", expanded=False):
                    st.dataframe(df_upload.head(10))

    except Exception as e:
        with st.sidebar:
            st.error(f"Error: {str(e)}")


# ════════════════════════════════════════════════════════════
#  SECTION 1: OVERVIEW
# ════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Problem Statement</div>', unsafe_allow_html=True)
st.markdown("""
Breast cancer is one of the most common cancers worldwide. Early and accurate diagnosis
is critical for effective treatment. This project applies **6 different machine learning
classification models** to the Breast Cancer Wisconsin (Diagnostic) dataset to classify
tumours as **Malignant** or **Benign** based on 30 numeric features derived from
digitized images of fine needle aspirate (FNA) of breast masses.
""")

col1, col2, col3, col4 = st.columns(4)
with col1:
    render_metric_card("Features", 30, "#667eea")
with col2:
    render_metric_card("Instances", 569, "#764ba2")
with col3:
    render_metric_card("Models Trained", 6, "#4facfe")
with col4:
    render_metric_card("Metrics per Model", 6, "#43e97b")

st.markdown('<div class="section-header">Dataset Description</div>', unsafe_allow_html=True)

X, y, _, _ = load_dataset()
col_a, col_b = st.columns([3, 2])
with col_a:
    st.markdown("**Feature List** (computed from cell nuclei images):")
    feat_df = pd.DataFrame({
        'Feature': list(feature_names),
        'Mean': [f"{X[f].mean():.3f}" for f in feature_names],
        'Std': [f"{X[f].std():.3f}" for f in feature_names],
    })
    st.dataframe(feat_df, width='stretch', height=350)
with col_b:
    st.markdown("**Class Distribution:**")
    fig_cls, ax_cls = plt.subplots(figsize=(4, 3))
    counts = y.value_counts()
    colors_pie = ['#fa709a', '#43e97b']
    ax_cls.pie(counts, labels=['Benign (1)', 'Malignant (0)'],
                autopct='%1.1f%%', colors=colors_pie,
                startangle=90, textprops={'fontsize': 10})
    ax_cls.set_title('Target Distribution', fontsize=11, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_cls)

    st.markdown("**Quick Statistics:**")
    st.markdown(f"""
- Malignant samples: **{(y == 0).sum()}** ({(y == 0).mean()*100:.1f}%)
- Benign samples: **{(y == 1).sum()}** ({(y == 1).mean()*100:.1f}%)
- Train/Test split: **85/15** (stratified)
""")

st.markdown('<div class="section-header">Models Implemented</div>', unsafe_allow_html=True)
model_info = {
    "Logistic Regression": "A linear model that uses a logistic/sigmoid function to model binary outcomes. Fast, interpretable, and works well when features are roughly linearly separable.",
    "Decision Tree": "A tree-based model that recursively partitions the feature space using axis-aligned splits. Highly interpretable but prone to overfitting.",
    "K-Nearest Neighbors (kNN)": "An instance-based learner that classifies based on the majority vote of k closest training samples. Simple but sensitive to feature scaling.",
    "Naive Bayes (Gaussian)": "A probabilistic model based on Bayes' theorem assuming feature independence. Very fast and works surprisingly well on many real-world problems.",
    "Random Forest (Ensemble)": "An ensemble of decision trees trained on bootstrap samples with random feature selection. Reduces overfitting and improves generalization.",
    "XGBoost (Ensemble)": "A gradient boosting framework that builds trees sequentially, each correcting the errors of its predecessor. State-of-the-art on many tabular datasets."
}
cols = st.columns(3)
icons = ["📈", "🌳", "👥", "🎯", "🌲", "⚡"]
for i, (name, desc) in enumerate(model_info.items()):
    with cols[i % 3]:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf4 100%);
                    border-radius: 12px; padding: 1rem; margin-bottom: 1rem;
                    min-height: 160px; box-shadow: 0 2px 6px rgba(0,0,0,0.06);">
            <div style="font-size: 1.5rem; margin-bottom: 0.3rem;">{icons[i]}</div>
            <div style="font-weight: 600; color: #1e3a5f; margin-bottom: 0.4rem;">{name}</div>
            <div style="font-size: 0.85rem; color: #6b7280;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)


st.markdown("---")

# ─── Data Source Indicator ───
if data_mode == "uploaded":
    st.markdown("""
    <div style="background: linear-gradient(90deg, #43e97b, #38f9d7); color: #1e3a5f;
                border-radius: 10px; padding: 0.6rem 1.2rem; margin-bottom: 1rem;
                font-weight: 600; text-align: center;">
        📊 Showing results for <strong>Uploaded Data</strong> — all models re-evaluated
    </div>
    """, unsafe_allow_html=True)
elif data_mode == "uploaded_no_target":
    st.markdown("""
    <div style="background: linear-gradient(90deg, #ffecd2, #fcb69f); color: #1e3a5f;
                border-radius: 10px; padding: 0.6rem 1.2rem; margin-bottom: 1rem;
                font-weight: 600; text-align: center;">
        ⚠️ Uploaded data has no <code>target</code> column — showing default test metrics. Predictions available in model analysis below.
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  SECTION 2: MODEL COMPARISON
# ════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Evaluation Metrics — All Models</div>', unsafe_allow_html=True)

# Styled comparison table
st.dataframe(
    active_comp_df.style.format({
        'Accuracy': '{:.4f}', 'AUC': '{:.4f}',
        'Precision': '{:.4f}', 'Recall': '{:.4f}',
        'F1 Score': '{:.4f}', 'MCC': '{:.4f}'
    }).background_gradient(cmap='Blues', subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']),
    width='stretch', hide_index=True
)

# Bar chart comparison
st.markdown('<div class="section-header">Visual Comparison</div>', unsafe_allow_html=True)
fig_bar = plot_metric_comparison(active_comp_df)
st.pyplot(fig_bar)


# ROC Curves
if data_mode != "uploaded_no_target":
    st.markdown('<div class="section-header">ROC Curves</div>', unsafe_allow_html=True)
    fig_roc = plot_roc_curves(active_results, active_X_test, active_y_test)
    st.pyplot(fig_roc)

# Best model highlight
st.markdown('<div class="section-header">Best Model per Metric</div>', unsafe_allow_html=True)
best_cols = st.columns(6)
metric_names = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']
metric_colors = ['#667eea', '#764ba2', '#4facfe', '#43e97b', '#fa709a', '#fee140']
for i, metric in enumerate(metric_names):
    with best_cols[i]:
        numeric_col = pd.to_numeric(active_comp_df[metric], errors='coerce')
        best_idx = numeric_col.idxmax()
        best_model = active_comp_df.loc[best_idx, 'Model']
        best_val = numeric_col.max()
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f5f7fa, #e8ecf4);
                    border-radius: 10px; padding: 0.8rem; text-align: center;
                    border-top: 3px solid {metric_colors[i]};">
            <div style="font-size: 0.75rem; color: #9ca3af;">{metric}</div>
            <div style="font-size: 1.3rem; font-weight: 700; color: #1e3a5f;">{best_val:.4f}</div>
            <div style="font-size: 0.7rem; color: #6b7280; margin-top: 4px;">{best_model}</div>
        </div>
        """, unsafe_allow_html=True)


st.markdown("---")

# ════════════════════════════════════════════════════════════
#  SECTION 3: INDIVIDUAL MODEL ANALYSIS
# ════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Select a Model to Analyze</div>', unsafe_allow_html=True)

# ─── Model Selection Control Panel ───
model_icons = {"Logistic Regression": "📈", "Decision Tree": "🌳",
               "K-Nearest Neighbors (kNN)": "👥", "Naive Bayes (Gaussian)": "🎯",
               "Random Forest (Ensemble)": "🌲", "XGBoost (Ensemble)": "⚡"}

with st.container():
    st.markdown('<div class="panel-title">🎛️ Model Control Panel</div>', unsafe_allow_html=True)

    # Dropdown selector
    model_name = st.selectbox(
        "Choose a classification model:",
        list(active_results.keys()),
        index=0,
        key="model_selector"
    )

    # Model preview cards row (rendered after selectbox so active state matches selection)
    cards_html = '<div class="model-cards-grid">'
    for name, res_item in active_results.items():
        icon = model_icons.get(name, "🤖")
        acc = res_item['metrics'].get('Accuracy', 0)
        is_active = "active" if name == model_name else ""
        cards_html += f"""
        <div class="model-card {is_active}">
            <div class="card-icon">{icon}</div>
            <div class="card-name">{name.split('(')[0].strip()}</div>
            <div class="card-acc">{acc:.2%}</div>
            <div class="card-acc-label">Accuracy</div>
        </div>"""
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

# Selected model banner
sel_icon = model_icons.get(model_name, "🤖")
sel_acc = active_results[model_name]['metrics'].get('Accuracy', 0)
st.markdown(f"""
<div class="selected-model-banner">
    <div>
        <div class="banner-label">Currently Analyzing</div>
        <div class="banner-name">{sel_icon} {model_name}</div>
    </div>
    <div class="banner-badge">Accuracy: {sel_acc:.4f}</div>
</div>
""", unsafe_allow_html=True)

res = active_results[model_name]
metrics = res['metrics']

# Metric cards
st.markdown(f'<div class="section-header">Metrics — {model_name}</div>', unsafe_allow_html=True)
cols = st.columns(6)
m_colors = ['#667eea', '#764ba2', '#4facfe', '#43e97b', '#fa709a', '#f6d365']
for i, (key, val) in enumerate(metrics.items()):
    with cols[i]:
        render_metric_card(key, val, m_colors[i])

st.markdown("")  # spacer

# Confusion Matrix + Classification Report side by side
col_cm, col_cr = st.columns([1, 1])

with col_cm:
    st.markdown(f'<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
    fig_cm = plot_confusion_matrix(res['confusion_matrix'], title=model_name)
    st.pyplot(fig_cm)

with col_cr:
    st.markdown(f'<div class="section-header">Classification Report</div>', unsafe_allow_html=True)
    st.code(res['classification_report'], language='text')

# Prediction distribution
st.markdown('<div class="section-header">Prediction Distribution</div>', unsafe_allow_html=True)
col_pred1, col_pred2 = st.columns(2)

with col_pred1:
    fig_dist, ax_dist = plt.subplots(figsize=(5, 3.5))
    pred_counts = pd.Series(res['predictions']).value_counts().sort_index()
    true_counts = pd.Series(np.array(active_y_test)).value_counts().sort_index()
    x_pos = np.arange(2)
    width = 0.35
    ax_dist.bar(x_pos - width/2, true_counts.values, width, label='Actual', color='#667eea', alpha=0.8)
    ax_dist.bar(x_pos + width/2, pred_counts.values, width, label='Predicted', color='#fa709a', alpha=0.8)
    ax_dist.set_xticks(x_pos)
    ax_dist.set_xticklabels(['Malignant', 'Benign'])
    ax_dist.set_ylabel('Count', fontweight='bold')
    ax_dist.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
    ax_dist.legend()
    ax_dist.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_dist)

with col_pred2:
    fig_err, ax_err = plt.subplots(figsize=(5, 3.5))
    y_test_arr = np.array(active_y_test)
    y_pred_arr = res['predictions']
    correct = (y_test_arr == y_pred_arr).sum()
    incorrect = (y_test_arr != y_pred_arr).sum()
    ax_err.pie([correct, incorrect], labels=['Correct', 'Incorrect'],
               autopct='%1.1f%%', colors=['#43e97b', '#fa709a'],
               startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax_err.set_title('Prediction Accuracy', fontsize=12, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_err)


# ─── Predictions on Uploaded Data (no target) ───
if data_mode == "uploaded_no_target" and model_name in upload_predictions:
    st.markdown("---")
    st.markdown('<div class="section-header">Predictions on Uploaded Data</div>', unsafe_allow_html=True)
    pred_vals = upload_predictions[model_name]
    pred_labels = ['Benign' if p == 1 else 'Malignant' for p in pred_vals]

    col_pr1, col_pr2 = st.columns([2, 1])
    with col_pr1:
        pred_df = pd.DataFrame({
            'Sample': range(1, len(pred_vals) + 1),
            'Prediction': pred_vals,
            'Label': pred_labels
        })
        st.dataframe(pred_df, width='stretch', hide_index=True)
    with col_pr2:
        pred_summary = pd.Series(pred_labels).value_counts()
        fig_ps, ax_ps = plt.subplots(figsize=(4, 3))
        ax_ps.pie(pred_summary.values, labels=pred_summary.index,
                  autopct='%1.1f%%', colors=['#43e97b', '#fa709a'], startangle=90)
        ax_ps.set_title('Prediction Summary', fontsize=11, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_ps)

    csv_out = pred_df.to_csv(index=False)
    st.download_button("⬇️ Download Predictions CSV", csv_out, "predictions.csv", "text/csv")


# ─── Footer ───
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #9ca3af; font-size: 0.85rem;'>"
    "ML Classification Dashboard &bull; Breast Cancer Wisconsin Dataset &bull; "
    "Built with Streamlit, scikit-learn & XGBoost"
    "</div>",
    unsafe_allow_html=True
)
