#!/usr/bin/env python3
"""
Visualization utilities for the WSSV prediction model.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Define directories
ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "models"

def plot_feature_importance(model, feature_names, top_n=10):
    """Plot feature importance for a trained model"""
    plt.figure(figsize=(10, 6))
    
    # Get feature importances based on model type
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        top_indices = indices[:top_n]
        top_importances = importances[top_indices]
        top_features = [feature_names[i] for i in top_indices]
        
        sns.barplot(x=top_importances, y=top_features)
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        
    elif hasattr(model, 'coef_'):
        coefs = model.coef_[0]
        abs_coefs = np.abs(coefs)
        indices = np.argsort(abs_coefs)[::-1]
        
        top_indices = indices[:top_n]
        top_coefs = coefs[top_indices]
        top_features = [feature_names[i] for i in top_indices]
        
        colors = ['blue' if c > 0 else 'red' for c in top_coefs]
        sns.barplot(x=top_coefs, y=top_features, palette=colors)
        plt.title(f'Top {top_n} Feature Coefficients')
        plt.xlabel('Coefficient')
        plt.ylabel('Feature')
    
    else:
        logging.warning("Model does not have feature_importances_ or coef_ attribute")
        return
    
    plt.tight_layout()
    plt.show()

def plot_roc_curve(models, X_test, y_test, model_names=None):
    """Plot ROC curves for multiple models"""
    from sklearn.metrics import roc_curve, auc
    
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    
    if model_names is None:
        model_names = [f'Model {i+1}' for i in range(len(models))]
    
    for model, name in zip(models, model_names):
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, normalize=False):
    """Plot confusion matrix"""
    from sklearn.metrics import confusion_matrix
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=['No Outbreak', 'Outbreak'],
                yticklabels=['No Outbreak', 'Outbreak'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_prediction_gauge(probability):
    """Create a gauge plot for outbreak probability"""
    fig, ax = plt.subplots(figsize=(8, 2))
    plt.axis('off')
    
    gauge = np.linspace(0, 1, 100)
    gauge_y = np.zeros_like(gauge)
    
    plt.scatter(gauge, gauge_y, c=gauge, cmap='RdYlGn_r', s=1000, marker='|')
    plt.scatter([probability], [0], c='black', s=500, marker='v')
    plt.xlim(0, 1)
    plt.ylim(-0.5, 0.5)
    plt.annotate(f"Risk: {probability:.2f}", xy=(probability, 0.1), 
                 xytext=(probability, 0.1), ha='center', fontsize=12)
    
    return fig

def main():
    """Main function to demonstrate visualizations"""
    logging.info("Generating model visualizations...")
    
    # Check if models exist
    final_model_path = MODEL_DIR / "final_model.pkl"
    
    if final_model_path.exists():
        model = joblib.load(final_model_path)
        logging.info(f"Loaded model: {type(model).__name__}")
        
        # Find feature importance file
        feature_dirs = ["random_forest", "logistic_regression", "xgboost"]
        for dir_name in feature_dirs:
            feature_file = MODEL_DIR / dir_name / "feature_importance.csv"
            if feature_file.exists():
                feature_data = pd.read_csv(feature_file)
                logging.info(f"Found feature importance data: {len(feature_data)} features")
                
                # Plot top 10 features
                plt.figure(figsize=(10, 6))
                top_features = feature_data.head(10)
                sns.barplot(x='importance', y='feature', data=top_features)
                plt.title(f'Top 10 Feature Importances - {dir_name.replace("_", " ").title()}')
                plt.tight_layout()
                plt.show()
                break
        else:
            logging.warning("No feature importance data found")
    else:
        logging.warning("No trained models found. Please train models first.")

if __name__ == "__main__":
    main()