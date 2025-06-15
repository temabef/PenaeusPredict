"""
Simple script to generate sample data for shrimp disease prediction.
This is a simplified version that should work with minimal dependencies.
"""

import random
import csv
import os
from datetime import datetime

# Create data directory if it doesn't exist
os.makedirs("data/raw", exist_ok=True)

# Define parameters for data generation
NUM_SAMPLES = 1000
COUNTRIES = ['Thailand', 'Vietnam', 'Indonesia', 'India', 'Philippines']
SEASONS = ['Dry', 'Wet']
YEARS = list(range(2018, 2024))
MONTHS = list(range(1, 13))

# Function to generate a random float within a range
def random_float(min_val, max_val):
    return min_val + (max_val - min_val) * random.random()

# Function to determine disease outbreak probability based on parameters
def calculate_probability(params):
    # Start with base probability
    prob = 0.1
    
    # Temperature effect (higher temp = higher risk)
    if 25 <= params['water_temp'] <= 32:
        prob += 0.3
    
    # Stocking density effect
    if params['stocking_density'] > 80:
        prob += 0.2
    
    # Water quality effects
    if params['dissolved_oxygen'] < 4:
        prob += 0.15
    if params['ammonia'] > 1:
        prob += 0.15
    
    # History effect
    if params['wssv_history'] == 1:
        prob += 0.3
    
    # Season effect
    if params['season'] == 'Wet':
        prob += 0.1
    
    # Probiotics reduce risk
    if params['probiotics_used'] == 1:
        prob -= 0.1
    
    # Cap probability between 0 and 1
    return max(0, min(prob, 1))

# Generate data
print(f"Generating {NUM_SAMPLES} simulated data points...")
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
    wssv_history = random.choice([0, 1])
    probiotics_used = random.choice([0, 1])
    antibiotics_used = random.choice([0, 1])
    culture_duration = random.randint(30, 180)
    
    country = random.choice(COUNTRIES)
    season = random.choice(SEASONS)
    year = random.choice(YEARS)
    month = random.choice(MONTHS)
    
    # Calculate water quality index
    ph_score = abs(ph - 7) / 2
    do_score = 1 - (dissolved_oxygen / 10)
    ammonia_score = min(ammonia / 3, 1)
    water_quality_index = round((ph_score + do_score + ammonia_score) / 3, 3)
    
    # Calculate outbreak probability
    params = {
        'water_temp': water_temp,
        'salinity': salinity,
        'dissolved_oxygen': dissolved_oxygen,
        'ammonia': ammonia,
        'stocking_density': stocking_density,
        'wssv_history': wssv_history,
        'season': season,
        'probiotics_used': probiotics_used
    }
    
    outbreak_probability = calculate_probability(params)
    
    # Determine if outbreak occurs based on probability
    wssv_outbreak = 1 if random.random() < outbreak_probability else 0
    
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
        'wssv_history': wssv_history,
        'probiotics_used': probiotics_used,
        'antibiotics_used': antibiotics_used,
        'culture_duration': culture_duration,
        'country': country,
        'season': season,
        'month': month,
        'year': year,
        'water_quality_index': water_quality_index,
        'outbreak_probability': round(outbreak_probability, 3),
        'wssv_outbreak': wssv_outbreak,
        'data_source': 'simulated'
    })

# Write to CSV
output_file = "data/raw/sample_wssv_data.csv"
with open(output_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)

print(f"Successfully generated {NUM_SAMPLES} data points and saved to {output_file}")
print("Note: This data is simulated and should be marked as such in any publication.") 