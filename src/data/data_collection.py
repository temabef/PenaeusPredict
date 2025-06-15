"""
Data collection utilities for WSSV outbreak prediction model.

This module contains functions to scrape, download, and process data from various sources:
- SEAFDEC reports
- FAO Aquastat / ASFIS / FishStat
- Research papers with supplementary datasets
- Government fisheries departments
- Open data platforms (Kaggle, Zenodo, GitHub)
"""

import os
import pandas as pd
import requests
from pathlib import Path
import logging

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

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_data_from_url(url, output_file, desc=None):
    """
    Download data from a URL and save to specified location
    
    Parameters:
    -----------
    url : str
        URL to download data from
    output_file : str or Path
        Path to save downloaded data
    desc : str, optional
        Description of the data being downloaded
    """
    try:
        logging.info(f"Downloading data from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        logging.info(f"Successfully downloaded data to {output_file}")
        return True
    except Exception as e:
        logging.error(f"Error downloading data: {e}")
        return False

def search_seafdec_reports():
    """
    Search SEAFDEC reports for WSSV data
    
    This function should be expanded to programmatically search for and extract
    data from SEAFDEC reports.
    """
    logging.info("Searching SEAFDEC reports for WSSV data")
    # TODO: Implement SEAFDEC report scraping
    pass

def search_fao_data():
    """
    Search FAO datasets for relevant aquaculture data
    
    This function should be expanded to query FAO APIs or download datasets.
    """
    logging.info("Searching FAO datasets for aquaculture data")
    # TODO: Implement FAO data collection
    pass

def check_kaggle_datasets():
    """
    Search Kaggle for relevant WSSV datasets
    
    Requires kaggle API credentials to be set up.
    """
    try:
        import kaggle
        logging.info("Searching Kaggle for WSSV datasets")
        # TODO: Implement Kaggle dataset search and download
        # Example: kaggle.api.dataset_download_files('dataset/wssv-data', path=RAW_DATA_DIR)
    except ImportError:
        logging.error("Kaggle package not installed. Run 'pip install kaggle'")
    except Exception as e:
        logging.error(f"Error accessing Kaggle API: {e}")

def search_zenodo_datasets():
    """
    Search Zenodo for relevant WSSV datasets
    """
    logging.info("Searching Zenodo for WSSV datasets")
    # TODO: Implement Zenodo dataset search and download
    pass

def compile_research_paper_data():
    """
    Compile data from research papers with supplementary datasets
    
    This function should be expanded to include specific papers and their datasets.
    """
    logging.info("Compiling data from research papers")
    # TODO: Implement research paper data compilation
    # Example papers to check:
    papers = [
        "https://doi.org/10.1016/j.aquaculture.2019.734577",
        "https://doi.org/10.1016/j.jip.2018.02.019",
        # Add more relevant papers
    ]
    pass

def create_sample_data():
    """
    Create sample dataset based on literature review (when real data is not available)
    
    This function creates a synthetic dataset for initial model development
    based on known parameters and relationships from literature.
    """
    logging.info("Creating sample dataset for initial model development")
    
    # Define parameters for sample data
    n_samples = 200
    
    # Create random data with realistic distributions
    import numpy as np
    
    # Environmental factors
    water_temp = np.random.normal(28, 3, n_samples)  # Mean: 28°C, Std: 3°C
    salinity = np.random.normal(15, 5, n_samples)    # Mean: 15ppt, Std: 5ppt
    ph = np.random.normal(7.8, 0.5, n_samples)      # Mean: 7.8, Std: 0.5
    do = np.random.normal(5, 1.2, n_samples)        # Dissolved oxygen (mg/L)
    ammonia = np.random.gamma(2, 0.5, n_samples)    # Ammonia (mg/L)
    rainfall = np.random.gamma(3, 10, n_samples)    # Rainfall (mm/month)
    
    # Farm practices
    stocking_density = np.random.gamma(5, 10, n_samples)  # PL/m²
    pond_size = np.random.gamma(4, 500, n_samples)  # m²
    water_exchange = np.random.beta(2, 5, n_samples) * 30  # % per day
    history_wssv = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])  # 0=No, 1=Yes
    probiotics = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])  # 0=No, 1=Yes
    antibiotics = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])  # 0=No, 1=Yes
    culture_duration = np.random.normal(100, 20, n_samples)  # Days
    
    # Metadata
    country = np.random.choice(['Thailand', 'Vietnam', 'Indonesia', 'India', 'Philippines'], n_samples)
    season = np.random.choice(['Dry', 'Wet'], n_samples)
    year = np.random.choice(range(2013, 2024), n_samples)
    month = np.random.choice(range(1, 13), n_samples)
    
    # Generate outbreak status (target variable)
    # This is a simplified model based on literature where:
    # - Higher temperature increases risk
    # - Sudden temperature changes increase risk
    # - Higher stocking density increases risk
    # - History of WSSV increases risk
    # - Wet season increases risk
    
    # Create some temperature variations
    temp_variation = np.random.normal(0, 2, n_samples)
    
    # Calculate risk score
    risk_score = (
        0.1 * (water_temp - 27) ** 2 +      # Optimal temp around 27°C
        0.2 * np.abs(temp_variation) +      # Temperature variations
        0.15 * (stocking_density / 50) +    # Stocking density effect
        0.3 * history_wssv +                # History of WSSV
        0.1 * (season == 'Wet') +           # Season effect
        0.05 * (ammonia > 2) +              # Ammonia effect
        0.1 * (do < 3.5) -                  # Low DO effect
        0.1 * probiotics                    # Protective effect of probiotics
    )
    
    # Convert to probability and then binary outcome
    # Use a threshold that ensures a more balanced dataset
    outbreak_prob = 1 / (1 + np.exp(-risk_score))
    
    # Adjust the threshold to get approximately 40% positive cases
    threshold = np.percentile(outbreak_prob, 60)
    outbreak = (outbreak_prob > threshold).astype(int)
    
    # Verify we have both classes
    if np.sum(outbreak) == 0 or np.sum(outbreak) == n_samples:
        logging.warning("Generated dataset has only one class. Forcing balanced classes.")
        # Force approximately 40% positive cases
        sorted_indices = np.argsort(outbreak_prob)
        outbreak = np.zeros(n_samples, dtype=int)
        outbreak[sorted_indices[-int(n_samples * 0.4):]] = 1
    
    logging.info(f"Created dataset with {np.sum(outbreak)} positive cases and {n_samples - np.sum(outbreak)} negative cases")
    
    # Create DataFrame
    df = pd.DataFrame({
        # Environmental factors
        'water_temperature': water_temp,
        'salinity': salinity,
        'ph': ph,
        'dissolved_oxygen': do,
        'ammonia': ammonia,
        'rainfall': rainfall,
        
        # Farm practices
        'stocking_density': stocking_density,
        'pond_size': pond_size,
        'water_exchange_rate': water_exchange,
        'wssv_history': history_wssv,
        'probiotics_used': probiotics,
        'antibiotics_used': antibiotics,
        'culture_duration': culture_duration,
        
        # Metadata
        'country': country,
        'season': season,
        'year': year,
        'month': month,
        
        # Target
        'wssv_outbreak': outbreak,
        'outbreak_probability': outbreak_prob
    })
    
    # Save to CSV
    output_path = RAW_DATA_DIR / "sample_wssv_data.csv"
    df.to_csv(output_path, index=False)
    
    logging.info(f"Created sample dataset with {n_samples} samples at {output_path}")
    return df

def main():
    """Main function to run data collection process"""
    logging.info("Starting WSSV data collection process")
    
    # Check if we already have data
    if list(RAW_DATA_DIR.glob("*.csv")):
        logging.info("Data files already exist in raw data directory")
    else:
        logging.info("No data files found. Attempting to collect data...")
        
        # Try to collect data from various sources
        search_seafdec_reports()
        search_fao_data()
        check_kaggle_datasets()
        search_zenodo_datasets()
        compile_research_paper_data()
        
        # If no data could be collected, create sample data
        if not list(RAW_DATA_DIR.glob("*.csv")):
            logging.warning("No real data could be collected. Creating sample data for development.")
            create_sample_data()

if __name__ == "__main__":
    main() 