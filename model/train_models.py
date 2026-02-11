import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import pickle
import os
from ml_models import MLClassifier

def create_sample_dataset():
    """Create a sample classification dataset that meets requirements"""
    # Create a synthetic dataset with 15 features and 1000 samples
    X, y = make_classification(
        n_samples=1000,
        n_features=15,
        n_informative=10,
        n_redundant=3,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=2,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Add some categorical features for more realistic dataset
    df['category_A'] = np.random.choice(['Type1', 'Type2', 'Type3'], size=len(df))
    df['category_B'] = np.random.choice(['GroupX', 'GroupY'], size=len(df))
    
    return df

def main():
    """Main function to train and save all models"""
    print("Loading wine dataset...")
    # Load the wine dataset
    df = pd.read_csv('data/winequalityN.csv')
    
    # Preprocess the wine dataset
    print("Preprocessing wine dataset...")
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
    
    # Remove rows with missing values
    df = df.dropna()
    print(f"Dataset shape after cleaning: {df.shape}")
    print(f"Features: {len(df.columns) - 1}")
    print(f"Samples: {len(df)}")
    print(f"Target column: quality")
    print(f"Quality distribution: {df['quality'].value_counts().sort_index()}")
    
    # Initialize ML classifier
    ml_classifier = MLClassifier()
    
    # Load and preprocess data
    print("\nPreprocessing data for ML...")
    X_train, X_test, y_train, y_test = ml_classifier.load_and_preprocess_data(df, 'quality')
    
    # Train all models
    print("Training all models...")
    ml_classifier.train_all_models()
    
    # Evaluate all models
    print("\nEvaluating models...")
    all_metrics, _, _, _ = ml_classifier.get_all_metrics()
    
    # Print results
    print("\n" + "="*80)
    print("MODEL COMPARISON TABLE")
    print("="*80)
    
    # Create comparison table
    metrics_df = pd.DataFrame(all_metrics).T
    print(metrics_df.round(4))
    
    # Save comparison table to data folder for Streamlit app
    metrics_df.to_csv('data/model_comparison.csv')
    print("\nComparison table saved as 'data/model_comparison.csv'")
    
    # Save models
    print("\nSaving models...")
    ml_classifier.save_models()
    print("All models saved successfully!")
    
    # Save the processed dataset for Streamlit app
    df.to_csv('data/dataset_for_app.csv', index=False)
    print("Processed dataset for app saved as 'data/dataset_for_app.csv'")
    
    return ml_classifier, all_metrics

if __name__ == "__main__":
    classifier, metrics = main()
