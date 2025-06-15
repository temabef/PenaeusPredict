#!/usr/bin/env python3
"""
Main execution script for the PenaeusPredict project.

This script runs the complete pipeline from data collection to model training and
Streamlit application deployment.
"""

import os
import argparse
import subprocess
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run the PenaeusPredict WSSV outbreak prediction pipeline."
    )
    parser.add_argument(
        "--steps", 
        type=str, 
        default="all",
        choices=["collect", "preprocess", "train", "app", "all"],
        help="Specify which steps to run: collect, preprocess, train, app, or all"
    )
    parser.add_argument(
        "--sample", 
        action="store_true",
        help="Use sample data instead of attempting to collect real data"
    )
    return parser.parse_args()

def run_data_collection(use_sample=False):
    """Run data collection step"""
    logging.info("Starting data collection...")
    
    try:
        from src.data.data_collection import create_sample_data, main as collect_main
        
        if use_sample:
            logging.info("Using sample data as requested")
            create_sample_data()
        else:
            collect_main()  # Will try to collect real data, fall back to sample if none found
            
        logging.info("Data collection completed successfully")
        return True
    except Exception as e:
        logging.error(f"Error in data collection: {e}")
        return False

def run_data_preprocessing():
    """Run data preprocessing step"""
    logging.info("Starting data preprocessing...")
    
    try:
        from src.data.data_preprocessing import main as preprocess_main
        preprocess_main()
        logging.info("Data preprocessing completed successfully")
        return True
    except Exception as e:
        logging.error(f"Error in data preprocessing: {e}")
        return False

def run_model_training():
    """Run model training step"""
    logging.info("Starting model training...")
    
    try:
        from src.models.train_model import main as train_main
        train_main()
        logging.info("Model training completed successfully")
        return True
    except Exception as e:
        logging.error(f"Error in model training: {e}")
        return False

def run_streamlit_app():
    """Run Streamlit application"""
    logging.info("Starting Streamlit application...")
    
    try:
        # Use subprocess to run Streamlit in a separate process
        app_path = Path(__file__).resolve().parent / "app" / "app.py"
        subprocess.run(["streamlit", "run", str(app_path)])
        return True
    except Exception as e:
        logging.error(f"Error running Streamlit app: {e}")
        return False

def main():
    """Main function to run the pipeline"""
    args = parse_args()
    
    # Determine which steps to run
    run_collect = args.steps in ["collect", "all"]
    run_preprocess = args.steps in ["preprocess", "all"]
    run_train = args.steps in ["train", "all"]
    run_app = args.steps in ["app", "all"]
    
    # Execute the pipeline steps
    if run_collect:
        if not run_data_collection(use_sample=args.sample):
            logging.error("Data collection failed. Stopping pipeline.")
            return
    
    if run_preprocess:
        if not run_data_preprocessing():
            logging.error("Data preprocessing failed. Stopping pipeline.")
            return
    
    if run_train:
        if not run_model_training():
            logging.error("Model training failed. Stopping pipeline.")
            return
    
    if run_app:
        run_streamlit_app()

if __name__ == "__main__":
    main() 