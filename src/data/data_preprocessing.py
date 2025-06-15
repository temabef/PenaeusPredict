#!/usr/bin/env python3
"""
Data preprocessing utilities for WSSV outbreak prediction model.

This module contains functions to clean, transform, and prepare data for modeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Define data directories
ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

def load_dataset(file_path):
    """Load dataset from specified path"""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded dataset from {file_path} with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return None

def explore_data(df):
    """Perform initial data exploration"""
    results = {}
    
    # Basic info
    results['shape'] = df.shape
    results['columns'] = df.columns.tolist()
    results['dtypes'] = df.dtypes.to_dict()
    
    # Missing values
    results['missing_values'] = df.isnull().sum().to_dict()
    results['missing_percent'] = (df.isnull().sum() / len(df) * 100).to_dict()
    
    # Summary statistics
    results['numeric_summary'] = df.describe().to_dict()
    
    # Target distribution (if exists)
    if 'wssv_outbreak' in df.columns:
        results['target_distribution'] = df['wssv_outbreak'].value_counts().to_dict()
    
    logging.info("Completed data exploration")
    return results

def clean_data(df):
    """Clean the dataset by handling missing values, outliers, etc."""
    logging.info("Starting data cleaning process")
    
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Handle missing values
    for col in df_clean.columns:
        missing = df_clean[col].isnull().sum()
        if missing > 0:
            logging.info(f"Column {col} has {missing} missing values")
            
            # For numeric columns, impute with median
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                logging.info(f"  - Imputed with median value: {median_val}")
            
            # For categorical columns, impute with mode
            else:
                mode_val = df_clean[col].mode()[0]
                df_clean[col].fillna(mode_val, inplace=True)
                logging.info(f"  - Imputed with mode value: {mode_val}")
    
    # Handle outliers in numeric columns
    numeric_cols = df_clean.select_dtypes(include=['number']).columns
    
    for col in numeric_cols:
        # Skip the target variable
        if col in ['wssv_outbreak', 'outbreak_probability']:
            continue
            
        # Calculate IQR
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
        
        if outliers > 0:
            logging.info(f"Column {col} has {outliers} outliers")
            
            # Cap outliers
            df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
            df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
            logging.info(f"  - Capped outliers to range [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    logging.info("Data cleaning completed")
    return df_clean

def feature_engineering(df):
    """Create new features that might improve model performance"""
    logging.info("Starting feature engineering")
    
    # Create a copy to avoid modifying the original
    df_featured = df.copy()
    
    # Example feature engineering:
    
    # 1. Temperature-salinity interaction (both affect disease risk)
    if 'water_temperature' in df_featured.columns and 'salinity' in df_featured.columns:
        df_featured['temp_salinity_interaction'] = df_featured['water_temperature'] * df_featured['salinity']
        logging.info("Created temperature-salinity interaction feature")
    
    # 2. Water quality index (combine pH, DO, ammonia)
    water_quality_cols = ['ph', 'dissolved_oxygen', 'ammonia']
    if all(col in df_featured.columns for col in water_quality_cols):
        # Normalize variables to 0-1 scale for combining
        ph_score = (df_featured['ph'] - 7).abs() / 2  # 0 = ideal pH of 7, 1 = bad (pH 5 or 9)
        do_score = 1 - (df_featured['dissolved_oxygen'] / 10)  # 0 = high DO (good), 1 = low DO (bad)
        ammonia_score = np.minimum(df_featured['ammonia'] / 3, 1)  # 0 = no ammonia, 1 = high ammonia (≥ 3 mg/L)
        
        # Combine to water quality index (higher = worse water quality)
        df_featured['water_quality_index'] = (ph_score + do_score + ammonia_score) / 3
        logging.info("Created water quality index feature")
    
    # 3. Farm risk score (combine stocking density and history)
    if 'stocking_density' in df_featured.columns and 'wssv_history' in df_featured.columns:
        # Normalize stocking density to 0-1 range (assuming 100 PL/m² is high)
        density_score = np.minimum(df_featured['stocking_density'] / 100, 1)
        
        # Combine with history (0-1 already)
        df_featured['farm_risk_score'] = (density_score + df_featured['wssv_history']) / 2
        logging.info("Created farm risk score feature")
    
    # 4. Season encoding - convert to cyclical features
    if 'month' in df_featured.columns:
        df_featured['month_sin'] = np.sin(2 * np.pi * df_featured['month'] / 12)
        df_featured['month_cos'] = np.cos(2 * np.pi * df_featured['month'] / 12)
        logging.info("Created cyclical month features")
    
    logging.info("Feature engineering completed")
    return df_featured

def encode_categorical_features(df, target_column='wssv_outbreak'):
    """Encode categorical features for machine learning"""
    logging.info("Encoding categorical features")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    
    logging.info(f"Found {len(categorical_cols)} categorical columns and {len(numeric_cols)} numeric columns")
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)
    
    logging.info(f"Encoded data shape: {X_processed.shape}")
    
    # Get feature names after transformation
    feature_names = numeric_cols.copy()  # Start with numeric column names
    
    # Add categorical feature names if any exist
    if categorical_cols:
        try:
            ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
            categorical_features = ohe.get_feature_names_out(categorical_cols).tolist()
            feature_names.extend(categorical_features)
        except Exception as e:
            logging.warning(f"Could not get categorical feature names: {e}")
    
    # Return as DataFrame to preserve feature names
    # If feature names don't match the shape, use generic column names
    if len(feature_names) != X_processed.shape[1]:
        logging.warning(f"Feature names count ({len(feature_names)}) doesn't match data shape ({X_processed.shape[1]}). Using generic column names.")
        feature_names = [f'feature_{i}' for i in range(X_processed.shape[1])]
    
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
    
    return X_processed_df, y, preprocessor

def preprocess_data(input_file, output_file=None):
    """Main preprocessing function that applies all steps"""
    logging.info(f"Starting data preprocessing for {input_file}")
    
    # Load the data
    df = load_dataset(input_file)
    if df is None:
        return None, None, None
    
    # Explore data
    explore_data(df)
    
    # Clean data
    df_clean = clean_data(df)
    
    # Feature engineering
    df_featured = feature_engineering(df_clean)
    
    # Encode categorical features
    X, y, preprocessor = encode_categorical_features(df_featured)
    
    # Save processed data if output file is provided
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save X and y separately
        X.to_csv(output_path.with_suffix('.X.csv'), index=False)
        y.to_csv(output_path.with_suffix('.y.csv'), index=False)
        
        # Save preprocessor
        joblib.dump(preprocessor, output_path.with_suffix('.preprocessor.pkl'))
        
        logging.info(f"Saved processed data to {output_path}")
    
    return X, y, preprocessor

def main():
    """Main function to run data preprocessing"""
    logging.info("Starting data preprocessing")
    
    # Check if raw data exists
    raw_data_files = list(RAW_DATA_DIR.glob("*.csv"))
    if not raw_data_files:
        logging.error("No raw data files found. Run data_collection.py first.")
        return
    
    # Process each raw data file
    for input_file in raw_data_files:
        filename = input_file.stem
        output_file = PROCESSED_DATA_DIR / filename
        
        logging.info(f"Processing {input_file}")
        X, y, preprocessor = preprocess_data(input_file, output_file)
        
        if X is not None:
            logging.info(f"Successfully processed {filename} with {X.shape[0]} samples and {X.shape[1]} features")
    
    logging.info("Data preprocessing completed")

if __name__ == "__main__":
    main()
