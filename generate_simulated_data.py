#!/usr/bin/env python3
"""
Comprehensive multi-disease data generation for shrimp aquaculture disease prediction.

This script generates realistic simulated data for five major shrimp diseases:
1. White Spot Syndrome Virus (WSSV)
2. Early Mortality Syndrome (EMS)/Acute Hepatopancreatic Necrosis Disease (AHPND)
3. Infectious Hypodermal and Hematopoietic Necrosis Virus (IHHNV)
4. Yellow Head Virus (YHV)
5. Infectious Myonecrosis Virus (IMNV)

Each disease has unique risk factors and environmental triggers based on scientific literature.
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Create data directory structure
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

for dir_path in [RAW_DIR, PROCESSED_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Define parameters for data generation
NUM_SAMPLES = 2000  # Increased sample size for better model training
COUNTRIES = ['Thailand', 'Vietnam', 'Indonesia', 'India', 'Philippines', 'Malaysia', 'China']
SEASONS = ['Dry', 'Wet']
YEARS = list(range(2018, 2024))
MONTHS = list(range(1, 13))

# Disease-specific parameters and risk factors
DISEASE_CONFIGS = {
    'wssv': {
        'name': 'White Spot Syndrome Virus',
        'base_risk': 0.15,
        'temp_range': (25, 32),  # Optimal temperature range
        'temp_risk_factor': 0.3,
        'salinity_range': (10, 25),
        'salinity_risk_factor': 0.2,
        'stocking_density_threshold': 80,
        'density_risk_factor': 0.25,
        'water_quality_importance': 0.3,
        'seasonal_effect': 0.15,
        'history_impact': 0.4
    },
    'ems_ahpnd': {
        'name': 'Early Mortality Syndrome/AHPND',
        'base_risk': 0.12,
        'temp_range': (28, 35),  # Higher temperature preference
        'temp_risk_factor': 0.35,
        'salinity_range': (5, 20),
        'salinity_risk_factor': 0.15,
        'stocking_density_threshold': 100,
        'density_risk_factor': 0.3,
        'water_quality_importance': 0.4,
        'seasonal_effect': 0.1,
        'history_impact': 0.25
    },
    'ihhnv': {
        'name': 'Infectious Hypodermal and Hematopoietic Necrosis Virus',
        'base_risk': 0.08,
        'temp_range': (22, 30),
        'temp_risk_factor': 0.2,
        'salinity_range': (15, 30),
        'salinity_risk_factor': 0.25,
        'stocking_density_threshold': 60,
        'density_risk_factor': 0.2,
        'water_quality_importance': 0.25,
        'seasonal_effect': 0.05,
        'history_impact': 0.3
    },
    'yhv': {
        'name': 'Yellow Head Virus',
        'base_risk': 0.10,
        'temp_range': (26, 33),
        'temp_risk_factor': 0.25,
        'salinity_range': (12, 25),
        'salinity_risk_factor': 0.2,
        'stocking_density_threshold': 90,
        'density_risk_factor': 0.35,
        'water_quality_importance': 0.35,
        'seasonal_effect': 0.2,
        'history_impact': 0.35
    },
    'imnv': {
        'name': 'Infectious Myonecrosis Virus',
        'base_risk': 0.06,
        'temp_range': (24, 31),
        'temp_risk_factor': 0.2,
        'salinity_range': (10, 28),
        'salinity_risk_factor': 0.15,
        'stocking_density_threshold': 70,
        'density_risk_factor': 0.25,
        'water_quality_importance': 0.3,
        'seasonal_effect': 0.1,
        'history_impact': 0.2
    }
}

def random_float(min_val, max_val):
    """Generate random float within range"""
    return min_val + (max_val - min_val) * random.random()

def calculate_disease_probability(params, disease_config):
    """Calculate disease outbreak probability based on disease-specific factors"""
    prob = disease_config['base_risk']
    
    # Temperature effect (disease-specific)
    temp_range = disease_config['temp_range']
    temp_risk_factor = disease_config['temp_risk_factor']
    
    if temp_range[0] <= params['water_temp'] <= temp_range[1]:
        prob += temp_risk_factor
    elif params['water_temp'] > temp_range[1]:
        prob += temp_risk_factor * 0.5  # Reduced risk at very high temps
    
    # Salinity effect
    salinity_range = disease_config['salinity_range']
    salinity_risk_factor = disease_config['salinity_risk_factor']
    
    if salinity_range[0] <= params['salinity'] <= salinity_range[1]:
        prob += salinity_risk_factor
    
    # Stocking density effect
    density_threshold = disease_config['stocking_density_threshold']
    density_risk_factor = disease_config['density_risk_factor']
    
    if params['stocking_density'] > density_threshold:
        prob += density_risk_factor
    
    # Water quality effects (disease-specific importance)
    water_quality_importance = disease_config['water_quality_importance']
    
    if params['dissolved_oxygen'] < 4:
        prob += water_quality_importance * 0.3
    if params['ammonia'] > 1:
        prob += water_quality_importance * 0.4
    if abs(params['ph'] - 7.5) > 0.5:
        prob += water_quality_importance * 0.2
    
    # History effect
    if params['disease_history'] == 1:
        prob += disease_config['history_impact']
    
    # Seasonal effect
    if params['season'] == 'Wet':
        prob += disease_config['seasonal_effect']
    
    # Probiotics reduce risk
    if params['probiotics_used'] == 1:
        prob -= 0.1
    
    # Antibiotics can increase risk (due to stress)
    if params['antibiotics_used'] == 1:
        prob += 0.05
    
    # Cap probability between 0 and 1
    return max(0, min(prob, 1))

def generate_disease_data(disease_code, disease_config):
    """Generate data for a specific disease"""
    print(f"Generating data for {disease_config['name']}...")
    
    data = []
    
    for i in range(NUM_SAMPLES):
        # Generate random parameters
        water_temp = round(random_float(20, 35), 2)
        salinity = round(random_float(0, 35), 2)
        ph = round(random_float(6.0, 9.0), 2)
        dissolved_oxygen = round(random_float(2.0, 10.0), 2)
        ammonia = round(random_float(0, 5.0), 3)
        rainfall = round(random_float(0, 500), 2)
        
        stocking_density = round(random_float(10, 200), 2)
        pond_size = round(random_float(100, 5000), 2)
        water_exchange = round(random_float(0, 30), 2)
        disease_history = random.choice([0, 1])
        probiotics_used = random.choice([0, 1])
        antibiotics_used = random.choice([0, 1])
        culture_duration = random.randint(30, 180)
        
        country = random.choice(COUNTRIES)
        season = random.choice(SEASONS)
        year = random.choice(YEARS)
        month = random.choice(MONTHS)
        
        # Calculate water quality index
        ph_score = abs(ph - 7.5) / 2
        do_score = 1 - (dissolved_oxygen / 10)
        ammonia_score = min(ammonia / 3, 1)
        water_quality_index = round((ph_score + do_score + ammonia_score) / 3, 3)
        
        # Calculate disease-specific outbreak probability
        params = {
            'water_temp': water_temp,
            'salinity': salinity,
            'ph': ph,
            'dissolved_oxygen': dissolved_oxygen,
            'ammonia': ammonia,
            'stocking_density': stocking_density,
            'disease_history': disease_history,
            'season': season,
            'probiotics_used': probiotics_used,
            'antibiotics_used': antibiotics_used
        }
        
        outbreak_probability = calculate_disease_probability(params, disease_config)
        
        # Determine if outbreak occurs based on probability
        outbreak_occurred = 1 if random.random() < outbreak_probability else 0
        
        # Add to dataset
        data.append({
            'water_temperature': water_temp,
            'salinity': salinity,
            'ph': ph,
            'dissolved_oxygen': dissolved_oxygen,
            'ammonia': ammonia,
            'rainfall': rainfall,
            'stocking_density': stocking_density,
            'pond_size': pond_size,
            'water_exchange_rate': water_exchange,
            'disease_history': disease_history,
            'probiotics_used': probiotics_used,
            'antibiotics_used': antibiotics_used,
            'culture_duration': culture_duration,
            'country': country,
            'season': season,
            'month': month,
            'year': year,
            'water_quality_index': water_quality_index,
            'outbreak_probability': round(outbreak_probability, 3),
            f'{disease_code}_outbreak': outbreak_occurred,
            'disease_type': disease_code,
            'data_source': 'simulated'
        })
    
    return data

def generate_combined_dataset():
    """Generate combined dataset with all diseases"""
    print("Generating combined multi-disease dataset...")
    
    all_data = []
    
    for disease_code, disease_config in DISEASE_CONFIGS.items():
        disease_data = generate_disease_data(disease_code, disease_config)
        all_data.extend(disease_data)
    
    # Convert to DataFrame
    combined_df = pd.DataFrame(all_data)
    
    # Save combined dataset
    combined_file = RAW_DIR / "multi_disease_data.csv"
    combined_df.to_csv(combined_file, index=False)
    print(f"Saved combined dataset with {len(combined_df)} samples to {combined_file}")
    
    return combined_df

def generate_individual_datasets():
    """Generate individual datasets for each disease"""
    print("Generating individual disease datasets...")
    
    for disease_code, disease_config in DISEASE_CONFIGS.items():
        disease_data = generate_disease_data(disease_code, disease_config)
        df = pd.DataFrame(disease_data)
        
        # Save individual dataset
        individual_file = RAW_DIR / f"{disease_code}_data.csv"
        df.to_csv(individual_file, index=False)
        print(f"Saved {disease_code} dataset with {len(df)} samples to {individual_file}")

def main():
    """Main function to generate all datasets"""
    print("Starting multi-disease data generation...")
    print(f"Generating {NUM_SAMPLES} samples per disease...")
    
    # Generate individual datasets
    generate_individual_datasets()
    
    # Generate combined dataset
    combined_df = generate_combined_dataset()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("DATA GENERATION SUMMARY")
    print("="*50)
    
    for disease_code, disease_config in DISEASE_CONFIGS.items():
        disease_data = combined_df[combined_df['disease_type'] == disease_code]
        outbreak_rate = disease_data[f'{disease_code}_outbreak'].mean()
        print(f"{disease_config['name']}: {outbreak_rate:.1%} outbreak rate")
    
    print(f"\nTotal samples generated: {len(combined_df)}")
    print("Data generation completed successfully!")
    print("\nNote: This data is simulated and should be marked as such in any publication.")

if __name__ == "__main__":
    main()
