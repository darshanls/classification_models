"""
Wine Quality Classification - Streamlit Web Application
A machine learning classification app demonstrating 6 different ML models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ML Classification App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None

def clean_dataframe_for_arrow(df):
    """Clean dataframe to avoid Arrow serialization issues"""
    for col in df.columns:
        # Convert object columns to numeric when possible
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
        
        # Convert pandas nullable dtypes to standard dtypes
        if hasattr(df[col].dtype, 'na_value'):
            if df[col].dtype.name.startswith('float'):
                df[col] = df[col].astype('float64')
            elif df[col].dtype.name.startswith('int'):
                df[col] = df[col].astype('int64')
        
        # Handle any remaining object columns by converting to string
        if df[col].dtype == 'object':
            # For categorical columns like 'type', convert to category then to string codes
            if col == 'type' or df[col].nunique() < 10:
                df[col] = df[col].astype('category').cat.codes
            else:
                df[col] = df[col].astype(str)
    
    return df

def load_sample_data():
    """Load sample wine quality dataset"""
    try:
        red_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        white_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
        
        red_wine = pd.read_csv(red_wine_url, sep=';')
        white_wine = pd.read_csv(white_wine_url, sep=';')
        
        red_wine['wine_type'] = 0
        white_wine['wine_type'] = 1
        
        wine_data = pd.concat([red_wine, white_wine], axis=0, ignore_index=True)
        wine_data['quality_label'] = (wine_data['quality'] >= 6).astype(int)
        
        # Clean data to avoid Arrow serialization issues
        wine_data = clean_dataframe_for_arrow(wine_data)
        
        return wine_data
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return None

def train_models(X_train, X_test, y_train, y_test, scaler):
    """Train all 6 classification models"""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, max_depth=6, 
                                  use_label_encoder=False, eval_metric='logloss')
    }
    
    trained_models = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (name, model) in enumerate(models.items()):
        status_text.text(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        progress_bar.progress((i + 1) / len(models))
    
    status_text.text("All models trained successfully!")
    progress_bar.empty()
    
    return trained_models

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate all evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_prob),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1 Score': f1_score(y_true, y_pred, average='weighted'),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    return metrics

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Create confusion matrix visualization"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Bad Wine (0)', 'Good Wine (1)'],
        y=['Bad Wine (0)', 'Good Wine (1)'],
        color_continuous_scale='Blues',
        title=f'Confusion Matrix - {model_name}'
    )
    fig.update_layout(width=500, height=400)
    
    # Add text annotations
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            fig.add_annotation(
                x=j, y=i,
                text=str(cm[i][j]),
                showarrow=False,
                font=dict(color='white' if cm[i][j] > cm.max()/2 else 'black', size=16)
            )
    
    return fig

def main():
    # Header
    st.markdown('<p class="main-header">🍷 Wine Quality Classification</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Machine Learning Classification with 6 Different Models</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📊 Configuration")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Select Data Source:",
        ["Use Sample Dataset", "Upload CSV File"]
    )
    
    df = None
    
    if data_source == "Upload CSV File":
        st.sidebar.markdown("---")
        st.sidebar.subheader("📁 Upload Test Data")
        uploaded_file = st.sidebar.file_uploader(
            "Upload your CSV file",
            type=['csv'],
            help="Upload a CSV file with the same features as the training data"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                # Clean data to avoid Arrow serialization issues
                df = clean_dataframe_for_arrow(df)
                
                st.sidebar.success(f"✅ File uploaded: {uploaded_file.name}")
                st.sidebar.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            except Exception as e:
                st.sidebar.error(f"Error reading file: {e}")
    else:
        with st.spinner("Loading sample Wine Quality dataset..."):
            df = load_sample_data()
        if df is not None:
            st.sidebar.success("✅ Sample dataset loaded!")
            st.sidebar.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    if df is not None:
        # Display dataset info
        st.header("📋 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", df.shape[0])
        with col2:
            st.metric("Total Features", df.shape[1] - 1)
        with col3:
            if 'quality_label' in df.columns:
                st.metric("Good Wine (1)", df['quality_label'].sum())
        with col4:
            if 'quality_label' in df.columns:
                st.metric("Bad Wine (0)", len(df) - df['quality_label'].sum())
        
        # Show data preview
        with st.expander("🔍 View Dataset Preview"):
            st.dataframe(df.head(10),width="stretch")
        
        # Feature statistics
        with st.expander("📊 Feature Statistics"):
            st.dataframe(df.describe(), width="stretch")
        
        st.markdown("---")
        
        # Model Training Section
        st.header("🎯 Model Training & Evaluation")
        
        # Prepare data
        if 'quality_label' in df.columns:
            feature_cols = [col for col in df.columns if col not in ['quality', 'quality_label']]
            X = df[feature_cols]
            y = df['quality_label']
        else:
            # For uploaded data, assume last column is target
            feature_cols = df.columns[:-1].tolist()
            X = df[feature_cols]
            y = df.iloc[:, -1]
        
        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models button
        if st.button("🚀 Train All Models", type="primary"):
            st.session_state.models = train_models(X_train_scaled, X_test_scaled, y_train, y_test, scaler)
            st.session_state.scaler = scaler
            st.session_state.X_test = X_test_scaled
            st.session_state.y_test = y_test
            st.session_state.feature_names = feature_cols
            st.session_state.models_trained = True
            st.success("✅ All models trained successfully!")
        
        # Model Selection and Evaluation
        if st.session_state.models_trained:
            st.markdown("---")
            st.header("📈 Model Evaluation")
            
            # Model selection dropdown
            selected_model = st.selectbox(
                "🔧 Select Model for Detailed Analysis:",
                list(st.session_state.models.keys())
            )
            
            # Get predictions for selected model
            model = st.session_state.models[selected_model]
            y_pred = model.predict(st.session_state.X_test)
            y_prob = model.predict_proba(st.session_state.X_test)[:, 1]
            
            # Calculate metrics
            metrics = calculate_metrics(st.session_state.y_test, y_pred, y_prob)
            
            # Display metrics in columns
            st.subheader(f"📊 Evaluation Metrics - {selected_model}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                st.metric("AUC Score", f"{metrics['AUC']:.4f}")
            with col2:
                st.metric("Precision", f"{metrics['Precision']:.4f}")
                st.metric("Recall", f"{metrics['Recall']:.4f}")
            with col3:
                st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
                st.metric("MCC Score", f"{metrics['MCC']:.4f}")
            
            # Confusion Matrix and Classification Report
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔢 Confusion Matrix")
                fig = plot_confusion_matrix(st.session_state.y_test, y_pred, selected_model)
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                st.subheader("📋 Classification Report")
                report = classification_report(st.session_state.y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format("{:.4f}"), width='stretch')
            
            # Model Comparison Table
            st.markdown("---")
            st.header("📊 Model Comparison Table")
            
            all_metrics = {}
            for model_name, model in st.session_state.models.items():
                y_pred_temp = model.predict(st.session_state.X_test)
                y_prob_temp = model.predict_proba(st.session_state.X_test)[:, 1]
                all_metrics[model_name] = calculate_metrics(st.session_state.y_test, y_pred_temp, y_prob_temp)
            
            comparison_df = pd.DataFrame(all_metrics).T
            comparison_df = comparison_df.round(4)
            
            # Style the dataframe
            st.dataframe(
                comparison_df.style.highlight_max(axis=0, color='lightgreen'),
                width='stretch'
            )
            
            # Bar chart comparison
            st.subheader("📈 Visual Comparison")
            
            metric_to_plot = st.selectbox(
                "Select metric to visualize:",
                ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']
            )
            
            fig = px.bar(
                x=list(all_metrics.keys()),
                y=[all_metrics[m][metric_to_plot] for m in all_metrics.keys()],
                labels={'x': 'Model', 'y': metric_to_plot},
                title=f'{metric_to_plot} Comparison Across Models',
                color=[all_metrics[m][metric_to_plot] for m in all_metrics.keys()],
                color_continuous_scale='Viridis'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')
            
            # Model Observations
            st.markdown("---")
            st.header("📝 Model Observations")
            
            observations = {
                'Logistic Regression': "Linear model that works well for linearly separable data. Fast training and interpretable coefficients. May underperform on complex non-linear relationships.",
                'Decision Tree': "Non-linear model that captures complex patterns. Prone to overfitting without proper depth constraints. Easy to interpret and visualize.",
                'K-Nearest Neighbors': "Instance-based learning that relies on distance metrics. Performance depends on k value and feature scaling. Can be slow for large datasets.",
                'Naive Bayes': "Probabilistic classifier assuming feature independence. Fast and works well with high-dimensional data. May underperform when independence assumption is violated.",
                'Random Forest': "Ensemble of decision trees that reduces overfitting. Robust and handles non-linear relationships well. Provides feature importance rankings.",
                'XGBoost': "Gradient boosting algorithm with regularization. Often achieves state-of-the-art performance. Handles missing values and provides feature importance."
            }
            
            for model_name in st.session_state.models.keys():
                with st.expander(f"📌 {model_name}"):
                    st.write(observations.get(model_name, "No observation available."))
                    st.write(f"**Performance on this dataset:**")
                    st.write(f"- Accuracy: {all_metrics[model_name]['Accuracy']:.4f}")
                    st.write(f"- Best for: {'High accuracy' if all_metrics[model_name]['Accuracy'] == max([m['Accuracy'] for m in all_metrics.values()]) else 'Balanced performance'}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🎓 Machine Learning Classification Assignment</p>
        <p>Built with Streamlit | Scikit-learn | XGBoost</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
