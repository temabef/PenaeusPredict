#!/usr/bin/env python3
"""
Initialization script for the PenaeusPredict project.

This script creates the necessary directory structure and verifies the setup.
"""

import os
from pathlib import Path
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Define project structure
PROJECT_STRUCTURE = {
    "data": ["raw", "processed"],
    "models": [],
    "notebooks": [],
    "src": ["data", "features", "models", "visualization"],
    "app": []
}

def create_directory_structure():
    """Create the project directory structure"""
    logging.info("Creating project directory structure...")
    
    root_dir = Path(__file__).resolve().parent
    
    for directory, subdirectories in PROJECT_STRUCTURE.items():
        dir_path = root_dir / directory
        dir_path.mkdir(exist_ok=True)
        logging.info(f"Created directory: {dir_path}")
        
        for subdirectory in subdirectories:
            subdir_path = dir_path / subdirectory
            subdir_path.mkdir(exist_ok=True)
            logging.info(f"Created subdirectory: {subdir_path}")

def verify_python_packages():
    """Verify that required Python packages are installed"""
    logging.info("Verifying Python packages...")
    
    required_packages = [
        "pandas",
        "numpy",
        "sklearn",
        "xgboost",
        "shap",
        "matplotlib",
        "seaborn",
        "streamlit",
        "jupyter",
        "requests"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logging.info(f"Package {package} is installed")
        except ImportError:
            logging.warning(f"Package {package} is NOT installed")
            missing_packages.append(package)
    
    if missing_packages:
        logging.warning(
            f"Missing packages: {', '.join(missing_packages)}. "
            f"Please install them using 'pip install -r requirements.txt'"
        )
        return False
    
    return True

def verify_files_exist():
    """Verify that critical files exist"""
    logging.info("Verifying critical files exist...")
    
    root_dir = Path(__file__).resolve().parent
    
    critical_files = [
        "requirements.txt",
        "README.md",
        "src/data/data_collection.py",
        "src/data/data_preprocessing.py",
        "src/models/train_model.py",
        "app/app.py",
        "run.py"
    ]
    
    missing_files = []
    
    for file_path in critical_files:
        full_path = root_dir / file_path
        if not full_path.exists():
            logging.warning(f"Critical file missing: {file_path}")
            missing_files.append(file_path)
        else:
            logging.info(f"Critical file exists: {file_path}")
    
    if missing_files:
        logging.warning(
            f"Missing critical files: {', '.join(missing_files)}. "
            f"Please make sure all required files are in place."
        )
        return False
    
    return True

def main():
    """Main function to initialize the project"""
    logging.info("Initializing PenaeusPredict project...")
    
    # Create directory structure
    create_directory_structure()
    
    # Verify Python packages
    packages_verified = verify_python_packages()
    
    # Verify critical files
    files_verified = verify_files_exist()
    
    if packages_verified and files_verified:
        logging.info("Project initialization successful!")
        logging.info("You can now run the project using 'python run.py'")
    else:
        logging.warning(
            "Project initialization completed with warnings. "
            "Please address the issues mentioned above."
        )
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 