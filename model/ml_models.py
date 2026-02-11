import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, classification_report, confusion_matrix
from xgboost import XGBClassifier
import pickle
import os

class MLClassifier:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.target_name = None
        self.target_encoded = False
        
    def load_and_preprocess_data(self, df, target_column):
        """Load and preprocess the dataset"""
        # Store feature names and target name
        self.target_name = target_column
        self.feature_names = [col for col in df.columns if col != target_column]
        
        # Separate features and target
        X = df[self.feature_names]
        y = df[target_column]
        
        # Handle categorical variables
        X_encoded = X.copy()
        for col in X.select_dtypes(include=['object']).columns:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
            X_encoded = pd.concat([X_encoded.drop(col, axis=1), dummies], axis=1)
        X = X_encoded
        
        # Encode target if it's categorical or if numeric classes don't start from 0
        if y.dtype == 'object' or (y.dtype in ['int64', 'int32'] and y.min() != 0):
            y = self.label_encoder.fit_transform(y)
            self.target_encoded = True
        else:
            self.target_encoded = False
        
        # Update feature names after one-hot encoding
        self.feature_names = X.columns.tolist()
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_all_models(self):
        """Train all 6 classification models"""
        # 1. Logistic Regression
        self.models['Logistic Regression'] = LogisticRegression(random_state=42, max_iter=1000)
        self.models['Logistic Regression'].fit(self.X_train_scaled, self.y_train)
        
        # 2. Decision Tree Classifier
        self.models['Decision Tree'] = DecisionTreeClassifier(random_state=42)
        self.models['Decision Tree'].fit(self.X_train_scaled, self.y_train)
        
        # 3. K-Nearest Neighbor Classifier
        self.models['KNN'] = KNeighborsClassifier(n_neighbors=5)
        self.models['KNN'].fit(self.X_train_scaled, self.y_train)
        
        # 4. Naive Bayes Classifier
        self.models['Naive Bayes'] = GaussianNB()
        self.models['Naive Bayes'].fit(self.X_train_scaled, self.y_train)
        
        # 5. Random Forest Classifier
        self.models['Random Forest'] = RandomForestClassifier(random_state=42, n_estimators=100)
        self.models['Random Forest'].fit(self.X_train_scaled, self.y_train)
        
        # 6. XGBoost Classifier
        self.models['XGBoost'] = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        self.models['XGBoost'].fit(self.X_train_scaled, self.y_train)
    
    def evaluate_model(self, model_name):
        """Evaluate a specific model and return all metrics"""
        model = self.models[model_name]
        y_pred = model.predict(self.X_test_scaled)
        
        # Handle binary vs multiclass for AUC
        if len(np.unique(self.y_test)) == 2:
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
        else:
            y_pred_proba = model.predict_proba(self.X_test_scaled)
            auc_score = roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr')
        
        metrics = {
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'AUC': auc_score,
            'Precision': precision_score(self.y_test, y_pred, average='weighted'),
            'Recall': recall_score(self.y_test, y_pred, average='weighted'),
            'F1 Score': f1_score(self.y_test, y_pred, average='weighted'),
            'MCC': matthews_corrcoef(self.y_test, y_pred)
        }
        
        return metrics, y_pred, classification_report(self.y_test, y_pred), confusion_matrix(self.y_test, y_pred)
    
    def get_all_metrics(self):
        """Get evaluation metrics for all models"""
        all_metrics = {}
        all_predictions = {}
        all_reports = {}
        all_confusion_matrices = {}
        
        for model_name in self.models.keys():
            metrics, y_pred, report, conf_matrix = self.evaluate_model(model_name)
            all_metrics[model_name] = metrics
            all_predictions[model_name] = y_pred
            all_reports[model_name] = report
            all_confusion_matrices[model_name] = conf_matrix
        
        return all_metrics, all_predictions, all_reports, all_confusion_matrices
    
    def predict_new_data(self, model_name, new_data):
        """Make predictions on new data"""
        model = self.models[model_name]
        
        # Ensure new_data has the same columns as training data
        for col in self.feature_names:
            if col not in new_data.columns:
                new_data[col] = 0
        
        # Select only the training columns
        new_data_processed = new_data[self.feature_names]
        
        # Scale the data
        new_data_scaled = self.scaler.transform(new_data_processed)
        
        # Make predictions
        predictions = model.predict(new_data_scaled)
        probabilities = model.predict_proba(new_data_scaled)
        
        # Convert predictions back to original labels if needed
        if hasattr(self.label_encoder, 'classes_'):
            predictions = self.label_encoder.inverse_transform(predictions)
            class_names = self.label_encoder.classes_
        else:
            class_names = [f'Class_{i}' for i in range(len(model.classes_))]
        
        return predictions, probabilities, class_names
    
    def save_models(self, path='model/'):
        """Save all trained models"""
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.models.items():
            filename = os.path.join(path, f"{name.lower().replace(' ', '_')}.pkl")
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
        
        # Save scaler and label encoder
        with open(os.path.join(path, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(os.path.join(path, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save feature names
        with open(os.path.join(path, 'feature_names.pkl'), 'wb') as f:
            pickle.dump(self.feature_names, f)
    
    def load_models(self, path='model/'):
        """Load all trained models"""
        model_files = {
            'Logistic Regression': 'logistic_regression.pkl',
            'Decision Tree': 'decision_tree.pkl',
            'KNN': 'knn.pkl',
            'Naive Bayes': 'naive_bayes.pkl',
            'Random Forest': 'random_forest.pkl',
            'XGBoost': 'xgboost.pkl'
        }
        
        for name, filename in model_files.items():
            filepath = os.path.join(path, filename)
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    self.models[name] = pickle.load(f)
        
        # Load scaler and label encoder
        if os.path.exists(os.path.join(path, 'scaler.pkl')):
            with open(os.path.join(path, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
        
        if os.path.exists(os.path.join(path, 'label_encoder.pkl')):
            with open(os.path.join(path, 'label_encoder.pkl'), 'rb') as f:
                self.label_encoder = pickle.load(f)
        
        # Load feature names
        if os.path.exists(os.path.join(path, 'feature_names.pkl')):
            with open(os.path.join(path, 'feature_names.pkl'), 'rb') as f:
                self.feature_names = pickle.load(f)
