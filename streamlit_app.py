import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
import sys
import logging
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set up logging to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Streamlit app starting...")

# Add model directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
from ml_models import MLClassifier

# Page configuration
st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .model-performance {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🤖 Machine Learning Classification Dashboard</h1>', unsafe_allow_html=True)
st.markdown("""
This interactive dashboard demonstrates the performance of 6 different classification models on a synthetic dataset.
Upload your own CSV file or use the default dataset to explore model performance.
""")

# Sidebar
st.sidebar.header("📊 Configuration")

# Model selection
model_options = ['Logistic Regression', 'Decision Tree', 'KNN', 'Naive Bayes', 'Random Forest', 'XGBoost']
selected_model = st.sidebar.selectbox("Select Model", model_options, index=4)  # Default to Random Forest

# Data upload section
st.sidebar.header("📁 Data Upload")
use_default_data = st.sidebar.checkbox("Use Default Dataset (Wine QualityN)", value=True)

@st.cache_data
def load_default_data():
    """Load the default dataset"""
    try:
        logger.info("Loading processed wine dataset...")
        df = pd.read_csv('data/dataset_for_app.csv')
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Clean dataframe to avoid Arrow serialization issues
        for col in df.columns:
            if df[col].dtype == 'object':
                # For categorical columns like 'type', convert to category then to codes
                if col == 'type' or df[col].nunique() < 10:
                    df[col] = df[col].astype('category').cat.codes.astype('int64')
                else:
                    df[col] = df[col].astype(str)
            # Convert float64 to float32 to reduce precision issues
            elif df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
        
        logger.info(f"Data preprocessing completed. Final shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error("Processed dataset not found.")
        st.error("Default dataset not found. Please upload a CSV file.")
        return None
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        st.error(f"Error loading dataset: {e}")
        return None

@st.cache_data
def load_uploaded_data(uploaded_file):
    """Load uploaded CSV data"""
    try:
        df = pd.read_csv(uploaded_file)
        # Clean dataframe to avoid Arrow serialization issues
        for col in df.columns:
            if df[col].dtype == 'object':
                # For categorical columns, convert to category then to codes
                if df[col].nunique() < 10:
                    df[col] = df[col].astype('category').cat.codes.astype('int64')
                else:
                    df[col] = df[col].astype(str)
            # Convert float64 to float32 to reduce precision issues
            elif df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Load data
if use_default_data:
    df = load_default_data()
    if df is not None:
        st.success(f"Default dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        df = load_uploaded_data(uploaded_file)
        if df is not None:
            st.success(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    else:
        df = None

# Main content
if df is not None:
    # Data preview
    st.header("📋 Data Preview")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Dataset Sample")
        st.dataframe(df.head(10), width='stretch')
    
    with col2:
        st.subheader("Dataset Info")
        st.write(f"**Rows:** {df.shape[0]}")
        st.write(f"**Columns:** {df.shape[1]}")
        st.write(f"**Target Column:** quality")
        
        # Show data types
        st.subheader("Data Types")
        dtype_df = pd.DataFrame(df.dtypes, columns=['Type'])
        # Convert data types to strings to avoid Arrow serialization issues
        dtype_df['Type'] = dtype_df['Type'].astype(str)
        st.dataframe(dtype_df, width='stretch')
    
    # Initialize ML classifier and load models
    @st.cache_resource
    def load_models():
        classifier = MLClassifier()
        try:
            classifier.load_models('model/')
            return classifier
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return None
    
    classifier = load_models()
    
    if classifier is not None:
        # Model performance section
        st.header("🎯 Model Performance")
        
        # Load comparison metrics
        try:
            comparison_df = pd.read_csv('data/model_comparison.csv', index_col=0)
            
            # Display comparison table
            st.subheader("Model Comparison Table")
            st.dataframe(comparison_df.round(4), width='stretch')
            
            # Performance visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Accuracy comparison
                fig_acc = px.bar(
                    x=comparison_df.index,
                    y=comparison_df['Accuracy'],
                    title="Model Accuracy Comparison",
                    labels={'x': 'Model', 'y': 'Accuracy'},
                    color=comparison_df['Accuracy'],
                    color_continuous_scale='viridis'
                )
                fig_acc.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_acc, width='stretch')
            
            with col2:
                # F1 Score comparison
                fig_f1 = px.bar(
                    x=comparison_df.index,
                    y=comparison_df['F1 Score'],
                    title="Model F1 Score Comparison",
                    labels={'x': 'Model', 'y': 'F1 Score'},
                    color=comparison_df['F1 Score'],
                    color_continuous_scale='plasma'
                )
                fig_f1.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_f1, width='stretch')
            
            # Selected model detailed analysis
            st.header(f"🔍 Detailed Analysis: {selected_model}")
            
            # Get metrics for selected model
            selected_metrics = comparison_df.loc[selected_model]
            
            # Display metrics in cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Accuracy</h3>
                    <h2>{selected_metrics['Accuracy']:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>F1 Score</h3>
                    <h2>{selected_metrics['F1 Score']:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>AUC Score</h3>
                    <h2>{selected_metrics['AUC']:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # All metrics for selected model
            st.subheader("Complete Metrics")
            metrics_df = pd.DataFrame(selected_metrics)
            metrics_df.columns = ['Metric Value']
            st.dataframe(metrics_df.round(4), width='stretch')
            
            # Prediction section
            st.header("🔮 Make Predictions")
            
            # Allow users to select test data for prediction
            if not use_default_data and uploaded_file is not None:
                st.info("Using your uploaded data for predictions")
                test_data = df.copy()
            else:
                st.info("Using a sample of the default dataset for predictions")
                test_data = df.sample(min(10, len(df)), random_state=42)
            
            if st.button("Generate Predictions"):
                with st.spinner("Generating predictions..."):
                    try:
                        # Make predictions
                        predictions, probabilities, class_names = classifier.predict_new_data(
                            selected_model, test_data.drop('quality', axis=1, errors='ignore')
                        )
                        
                        # Display predictions
                        results_df = test_data.copy()
                        results_df['Predicted'] = predictions
                        
                        # Add probability columns
                        for i, class_name in enumerate(class_names):
                            results_df[f'Prob_{class_name}'] = probabilities[:, i]
                        
                        st.subheader("Prediction Results")
                        st.dataframe(results_df, width='stretch')
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv,
                            file_name=f'predictions_{selected_model.replace(" ", "_")}.csv',
                            mime='text/csv'
                        )
                        
                    except Exception as e:
                        st.error(f"Error making predictions: {e}")
            
            # Confusion Matrix (for default dataset)
            if use_default_data:
                st.header("📊 Confusion Matrix")
                
                # Generate confusion matrix
                try:
                    # Load and preprocess data using the same method as during training
                    df_default = load_default_data()
                    if df_default is not None:
                        # Use the loaded classifier's preprocessing to ensure consistency
                        temp_classifier = MLClassifier()
                        temp_classifier.load_and_preprocess_data(df_default, 'quality')
                        
                        # Load the saved feature names to ensure consistency
                        import pickle
                        with open('model/feature_names.pkl', 'rb') as f:
                            saved_feature_names = pickle.load(f)
                        
                        # Convert X_test to DataFrame if it's not already
                        if not hasattr(temp_classifier.X_test, 'columns'):
                            X_test_df = pd.DataFrame(temp_classifier.X_test, columns=temp_classifier.feature_names)
                        else:
                            X_test_df = temp_classifier.X_test.copy()
                        
                        # Ensure the test data has the same features as during training
                        for feature in saved_feature_names:
                            if feature not in X_test_df.columns:
                                X_test_df[feature] = 0
                        
                        # Reorder columns to match training
                        X_test_df = X_test_df[saved_feature_names]
                        
                        # Scale the data using the loaded classifier's scaler
                        X_test_scaled = classifier.scaler.transform(X_test_df)
                        
                        # Get test predictions
                        model = classifier.models[selected_model]
                        y_pred = model.predict(X_test_scaled)
                        cm = confusion_matrix(temp_classifier.y_test, y_pred)
                        
                        # Plot confusion matrix
                        fig_cm = px.imshow(
                            cm,
                            text_auto=True,
                            title=f"Confusion Matrix - {selected_model}",
                            labels=dict(x="Predicted", y="Actual", color="Count"),
                            color_continuous_scale='Blues'
                        )
                        st.plotly_chart(fig_cm, width='stretch')
                        
                        # Classification report
                        st.subheader("Classification Report")
                        report = classification_report(temp_classifier.y_test, y_pred, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.round(4), width='stretch')
                    
                except Exception as e:
                    st.error(f"Error generating confusion matrix: {e}")
            
        except FileNotFoundError:
            st.error("Model comparison file not found. Please train the models first.")
            st.info("Run `python model/train_models.py` to train and save models.")
    
    # Dataset statistics
    st.header("📈 Dataset Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Target Distribution")
        if 'quality' in df.columns:
            target_counts = df['quality'].value_counts()
            fig_target = px.pie(
                values=target_counts.values,
                names=target_counts.index,
                title="Target Class Distribution"
            )
            st.plotly_chart(fig_target, width='stretch')
    
    with col2:
        st.subheader("Feature Correlations")
        # Select only numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            # Create heatmap
            fig_corr = px.imshow(
                corr_matrix,
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu',
                aspect="auto"
            )
            st.plotly_chart(fig_corr, width='stretch')

# Instructions section
st.sidebar.markdown("---")
st.sidebar.header("📖 Instructions")
st.sidebar.markdown("""
1. **Select Model**: Choose from 6 different classification algorithms
2. **Upload Data**: Use default dataset or upload your own CSV
3. **View Performance**: Compare metrics across all models
4. **Make Predictions**: Generate predictions on your data
5. **Download Results**: Export predictions as CSV

**Required CSV Format:**
- Target column named 'quality'
- Features in remaining columns
- No missing values in quality column
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Made with ❤️ using Streamlit | Machine Learning Classification Dashboard</p>
</div>
""", unsafe_allow_html=True)
