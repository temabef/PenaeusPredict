#!/usr/bin/env python3
"""
Script to combine real-world data and retrain the WSSV prediction model.

This script:
1. Combines all real data collected through the app
2. Merges it with existing training data (if any)
3. Preprocesses the combined data
4. Retrains the model
5. Evaluates and saves the new model
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import glob
import shutil
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Define paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
REAL_DATA_DIR = DATA_DIR / "real_data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "models"

def combine_real_data():
    """Combine all real data files into a single dataset"""
    logging.info("Combining real data files...")
    
    # Check if real data directory exists
    if not REAL_DATA_DIR.exists():
        logging.warning("Real data directory not found. No real data to combine.")
        return None
    
    # Find all CSV files in the real data directory
    data_files = list(REAL_DATA_DIR.glob("*.csv"))
    
    if not data_files:
        logging.warning("No real data files found.")
        return None
    
    logging.info(f"Found {len(data_files)} real data files.")
    
    # Combine all data files
    all_data = []
    for file in data_files:
        try:
            df = pd.read_csv(file)
            all_data.append(df)
        except Exception as e:
            logging.error(f"Error reading file {file}: {e}")
    
    if not all_data:
        logging.warning("No valid data files found.")
        return None
    
    # Combine into a single DataFrame
    combined_data = pd.concat(all_data, ignore_index=True)
    logging.info(f"Combined {combined_data.shape[0]} records with {combined_data.shape[1]} columns.")
    
    # Filter out records without actual outbreak status
    if 'wssv_outbreak_actual' in combined_data.columns:
        valid_data = combined_data.dropna(subset=['wssv_outbreak_actual'])
        logging.info(f"Found {valid_data.shape[0]} records with known outbreak status.")
        
        # Rename column to match training data
        valid_data = valid_data.rename(columns={'wssv_outbreak_actual': 'wssv_outbreak'})
        
        # Keep only necessary columns
        required_columns = [
            'water_temperature', 'salinity', 'ph', 'dissolved_oxygen', 'ammonia', 
            'rainfall', 'stocking_density', 'pond_size', 'water_exchange_rate', 
            'wssv_history', 'probiotics_used', 'antibiotics_used', 'culture_duration',
            'country', 'season', 'month', 'wssv_outbreak'
        ]
        
        # Keep only columns that exist in the data
        columns_to_keep = [col for col in required_columns if col in valid_data.columns]
        valid_data = valid_data[columns_to_keep]
        
        return valid_data
    else:
        logging.warning("No records with actual outbreak status found.")
        return None

def merge_with_existing_data(real_data):
    """Merge real data with existing training data"""
    logging.info("Merging with existing training data...")
    
    # Check if sample data exists
    sample_data_path = RAW_DATA_DIR / "sample_wssv_data.csv"
    if not sample_data_path.exists():
        logging.warning("No existing training data found. Using only real data.")
        return real_data
    
    # Load existing data
    try:
        existing_data = pd.read_csv(sample_data_path)
        logging.info(f"Loaded existing data with {existing_data.shape[0]} records.")
        
        # Ensure column compatibility
        common_columns = list(set(existing_data.columns) & set(real_data.columns))
        logging.info(f"Found {len(common_columns)} common columns between datasets.")
        
        # If too few common columns, might need manual intervention
        if len(common_columns) < 10:
            logging.warning("Too few common columns between datasets. Manual intervention may be needed.")
            return real_data
        
        # Combine datasets
        combined_data = pd.concat([
            existing_data[common_columns], 
            real_data[common_columns]
        ], ignore_index=True)
        
        logging.info(f"Combined dataset has {combined_data.shape[0]} records.")
        return combined_data
        
    except Exception as e:
        logging.error(f"Error merging data: {e}")
        return real_data

def save_for_training(combined_data):
    """Save the combined data for training"""
    logging.info("Saving combined data for training...")
    
    # Create a backup of the original sample data if it exists
    sample_data_path = RAW_DATA_DIR / "sample_wssv_data.csv"
    if sample_data_path.exists():
        backup_path = RAW_DATA_DIR / f"sample_wssv_data_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        shutil.copy(sample_data_path, backup_path)
        logging.info(f"Created backup of original data at {backup_path}")
    
    # Save the combined data
    combined_data.to_csv(sample_data_path, index=False)
    logging.info(f"Saved combined data to {sample_data_path}")
    
    return sample_data_path

def run_training():
    """Run the training pipeline"""
    logging.info("Running training pipeline...")
    
    try:
        # Use the run.py script to run preprocessing and training
        import subprocess
        result = subprocess.run(
            ["python", "run.py", "--steps", "preprocess", "train"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logging.info("Training completed successfully!")
            logging.info(result.stdout)
            return True
        else:
            logging.error("Training failed.")
            logging.error(result.stderr)
            return False
            
    except Exception as e:
        logging.error(f"Error running training: {e}")
        return False

def main():
    """Main function to run the retraining process"""
    logging.info("Starting model retraining with real data...")
    
    # Step 1: Combine real data
    real_data = combine_real_data()
    if real_data is None or real_data.empty:
        logging.error("No valid real data found. Exiting.")
        return 1
    
    # Step 2: Merge with existing data
    combined_data = merge_with_existing_data(real_data)
    
    # Step 3: Save for training
    save_for_training(combined_data)
    
    # Step 4: Run training
    success = run_training()
    
    if success:
        logging.info("Model successfully retrained with real data!")
        logging.info("You can now use the updated model in the app.")
        return 0
    else:
        logging.error("Model retraining failed. Please check the logs.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 