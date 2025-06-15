#!/usr/bin/env python3
"""
Model training script for WSSV outbreak prediction.

This module contains functions to train and evaluate different machine learning models
for predicting WSSV outbreaks based on preprocessed data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import joblib
from time import time
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import shap

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Define data and model directories
ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"

# Ensure directory exists
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_processed_data(base_filename):
    """
    Load processed data from X and y CSV files
    
    Parameters:
    -----------
    base_filename : str or Path
        Base filename for the processed data (without .X.csv or .y.csv extension)
        
    Returns:
    --------
    tuple
        X, y dataframes and preprocessor object
    """
    try:
        X_path = Path(str(base_filename) + '.X.csv')
        y_path = Path(str(base_filename) + '.y.csv')
        preprocessor_path = Path(str(base_filename) + '.preprocessor.pkl')
        
        X = pd.read_csv(X_path)
        y = pd.read_csv(y_path).iloc[:, 0]  # Extract the first column
        preprocessor = joblib.load(preprocessor_path)
        
        logging.info(f"Loaded processed data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y, preprocessor
    
    except Exception as e:
        logging.error(f"Error loading processed data: {e}")
        return None, None, None

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logging.info(f"Split data: train={X_train.shape[0]} samples, test={X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train):
    """Train a logistic regression model (baseline)"""
    logging.info("Training logistic regression model...")
    
    # Define parameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'class_weight': [None, 'balanced']
    }
    
    # Initialize model
    lr = LogisticRegression(random_state=42, max_iter=1000)
    
    # Grid search
    grid_search = GridSearchCV(
        lr, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
    )
    
    # Train model
    start_time = time()
    grid_search.fit(X_train, y_train)
    train_time = time() - start_time
    
    # Get best model
    best_model = grid_search.best_estimator_
    logging.info(f"Best logistic regression parameters: {grid_search.best_params_}")
    logging.info(f"Training time: {train_time:.2f} seconds")
    
    return best_model, grid_search

def train_random_forest(X_train, y_train):
    """Train a random forest model"""
    logging.info("Training random forest model...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': [None, 'balanced']
    }
    
    # Initialize model
    rf = RandomForestClassifier(random_state=42)
    
    # Grid search
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
    )
    
    # Train model
    start_time = time()
    grid_search.fit(X_train, y_train)
    train_time = time() - start_time
    
    # Get best model
    best_model = grid_search.best_estimator_
    logging.info(f"Best random forest parameters: {grid_search.best_params_}")
    logging.info(f"Training time: {train_time:.2f} seconds")
    
    return best_model, grid_search

def train_xgboost(X_train, y_train):
    """Train an XGBoost model"""
    logging.info("Training XGBoost model...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'scale_pos_weight': [1, sum(y_train == 0) / sum(y_train == 1)]  # For imbalanced datasets
    }
    
    # Initialize model
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        random_state=42
    )
    
    # Grid search
    grid_search = GridSearchCV(
        xgb_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
    )
    
    # Train model
    start_time = time()
    grid_search.fit(X_train, y_train, verbose=False)
    train_time = time() - start_time
    
    # Get best model
    best_model = grid_search.best_estimator_
    logging.info(f"Best XGBoost parameters: {grid_search.best_params_}")
    logging.info(f"Training time: {train_time:.2f} seconds")
    
    return best_model, grid_search

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance on test data"""
    logging.info(f"Evaluating {model_name}...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Log results
    logging.info(f"{model_name} performance:")
    logging.info(f"  Accuracy:  {accuracy:.4f}")
    logging.info(f"  Precision: {precision:.4f}")
    logging.info(f"  Recall:    {recall:.4f}")
    logging.info(f"  F1 Score:  {f1:.4f}")
    logging.info(f"  ROC AUC:   {roc_auc:.4f}")
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Return metrics as dictionary
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'y_prob': y_prob
    }
    
    return results

def analyze_feature_importance(model, X, model_name):
    """Analyze feature importance of the model"""
    logging.info(f"Analyzing feature importance for {model_name}...")
    
    feature_names = X.columns
    
    # For Random Forest and XGBoost
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Log top 10 features
        logging.info(f"Top 10 features for {model_name}:")
        for i, idx in enumerate(indices[:10]):
            logging.info(f"  {i+1}. {feature_names[idx]} - {importances[idx]:.4f}")
            
        return {
            'feature_names': feature_names,
            'importances': importances,
            'indices': indices
        }
    
    # For Logistic Regression
    elif hasattr(model, 'coef_'):
        coefs = model.coef_[0]
        abs_coefs = np.abs(coefs)
        indices = np.argsort(abs_coefs)[::-1]
        
        # Log top 10 features
        logging.info(f"Top 10 features for {model_name}:")
        for i, idx in enumerate(indices[:10]):
            logging.info(f"  {i+1}. {feature_names[idx]} - {coefs[idx]:.4f}")
            
        return {
            'feature_names': feature_names,
            'importances': coefs,
            'indices': indices
        }
    
    else:
        logging.warning(f"Feature importance not available for {model_name}")
        return None

def explain_predictions_with_shap(model, X_test, model_name):
    """Generate SHAP explanations for model predictions"""
    logging.info(f"Generating SHAP values for {model_name}...")
    
    try:
        # Create SHAP explainer based on model type
        if model_name == 'XGBoost':
            explainer = shap.TreeExplainer(model)
        elif model_name == 'Random Forest':
            explainer = shap.TreeExplainer(model)
        elif model_name == 'Logistic Regression':
            explainer = shap.LinearExplainer(model, X_test)
        else:
            logging.warning(f"SHAP explainer not implemented for {model_name}")
            return None
        
        # Calculate SHAP values
        # Use a small sample if dataset is large to avoid memory issues
        sample_size = min(40, X_test.shape[0])
        X_sample = X_test.iloc[:sample_size]
        
        shap_values = explainer.shap_values(X_sample)
        
        # For tree-based models with binary classification, shap returns a list of two arrays
        # (one for each class). We want the one for the positive class (index 1)
        if isinstance(shap_values, list):
            if model_name == 'Logistic Regression':
                shap_values = shap_values[0]  # For logistic regression
            else:
                shap_values = shap_values[1]  # For tree-based models, get positive class
        
        # Ensure shap_values is 2D and matches the number of columns
        if len(shap_values.shape) > 2:
            logging.warning(f"SHAP values have unexpected shape {shap_values.shape}. Skipping SHAP analysis.")
            return None
        
        # Verify shape matches
        if shap_values.shape[1] != X_sample.shape[1]:
            logging.warning(f"SHAP values shape {shap_values.shape} doesn't match data shape {X_sample.shape}. Skipping SHAP analysis.")
            return None
        
        return {
            'explainer': explainer,
            'shap_values': shap_values,
            'data': X_sample
        }
    except Exception as e:
        logging.warning(f"Error calculating SHAP values: {e}")
        return None

def save_model(model, model_name, metrics, feature_importance, shap_data=None):
    """Save the trained model and related information"""
    logging.info(f"Saving {model_name} model and results...")
    
    # Create model directory
    model_path = MODEL_DIR / f"{model_name.lower().replace(' ', '_')}"
    model_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, model_path / "model.pkl")
    
    # Save metrics
    metrics_df = pd.DataFrame({k: [v] for k, v in metrics.items() 
                               if k not in ['confusion_matrix', 'y_prob']})
    metrics_df.to_csv(model_path / "metrics.csv", index=False)
    
    # Save feature importance
    if feature_importance is not None:
        importance_df = pd.DataFrame({
            'feature': feature_importance['feature_names'],
            'importance': feature_importance['importances']
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df.to_csv(model_path / "feature_importance.csv", index=False)
    
    # Save SHAP data if available and valid
    if shap_data is not None:
        try:
            shap_values = shap_data['shap_values']
            data_columns = shap_data['data'].columns
            
            # Ensure shap_values is 2D and matches the number of columns
            if len(shap_values.shape) == 2 and shap_values.shape[1] == len(data_columns):
                shap_df = pd.DataFrame(shap_values, columns=data_columns)
                shap_df.to_csv(model_path / "shap_values.csv", index=False)
                logging.info(f"Saved SHAP values with shape {shap_values.shape}")
            else:
                logging.warning(f"SHAP values shape {shap_values.shape} doesn't match data columns ({len(data_columns)}). Skipping SHAP save.")
        except Exception as e:
            logging.warning(f"Error saving SHAP values: {e}")
    
    logging.info(f"Saved {model_name} model and results to {model_path}")

def main():
    """Main function to run model training"""
    logging.info("Starting model training")
    
    # Find processed data files
    processed_files = list(PROCESSED_DATA_DIR.glob("*.X.csv"))
    if not processed_files:
        logging.error("No processed data files found. Run data_preprocessing.py first.")
        return
    
    # Use the first processed data file
    base_filename = processed_files[0].stem[:-2]  # Remove .X suffix
    base_path = PROCESSED_DATA_DIR / base_filename
    
    # Load data
    X, y, preprocessor = load_processed_data(base_path)
    if X is None:
        return
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train models
    models = []
    
    # Logistic Regression
    lr_model, lr_cv = train_logistic_regression(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    lr_importance = analyze_feature_importance(lr_model, X, "Logistic Regression")
    lr_shap = explain_predictions_with_shap(lr_model, X_test, "Logistic Regression")
    save_model(lr_model, "Logistic Regression", lr_metrics, lr_importance, lr_shap)
    models.append(("Logistic Regression", lr_model, lr_metrics))
    
    # Random Forest
    rf_model, rf_cv = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    rf_importance = analyze_feature_importance(rf_model, X, "Random Forest")
    rf_shap = explain_predictions_with_shap(rf_model, X_test, "Random Forest")
    save_model(rf_model, "Random Forest", rf_metrics, rf_importance, rf_shap)
    models.append(("Random Forest", rf_model, rf_metrics))
    
    # XGBoost
    xgb_model, xgb_cv = train_xgboost(X_train, y_train)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    xgb_importance = analyze_feature_importance(xgb_model, X, "XGBoost")
    xgb_shap = explain_predictions_with_shap(xgb_model, X_test, "XGBoost")
    save_model(xgb_model, "XGBoost", xgb_metrics, xgb_importance, xgb_shap)
    models.append(("XGBoost", xgb_model, xgb_metrics))
    
    # Find best model
    best_model_name, best_model_obj, best_metrics = max(
        models, key=lambda x: x[2]['roc_auc']
    )
    logging.info(f"Best model: {best_model_name} (ROC AUC: {best_metrics['roc_auc']:.4f})")
    
    # Save best model as final model
    joblib.dump(best_model_obj, MODEL_DIR / "final_model.pkl")
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.pkl")
    
    logging.info("Model training completed")

if __name__ == "__main__":
    main() 