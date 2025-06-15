# PenaeusPredict: Multi-Disease Prediction for Shrimp Aquaculture

A machine learning application for predicting disease outbreaks in shrimp aquaculture based on environmental and farm management parameters.

## Features

- **Multi-Disease Prediction**: Forecast risk for five major shrimp diseases:
  - White Spot Syndrome Virus (WSSV)
  - Early Mortality Syndrome (EMS)/Acute Hepatopancreatic Necrosis Disease (AHPND)
  - Infectious Hypodermal and Hematopoietic Necrosis Virus (IHHNV)
  - Yellow Head Virus (YHV)
  - Infectious Myonecrosis Virus (IMNV)

- **User-Friendly Interface**: Easy-to-use web application for farmers and aquaculture specialists
- **Risk Assessment**: Get probability scores for disease outbreaks
- **Key Risk Factors**: Identify the most important factors contributing to disease risk
- **Data Collection**: Submit your own data to improve the models
- **Simulated Data**: Includes realistic simulated data for training and demonstration

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/PenaeusPredict.git
cd PenaeusPredict

# Install dependencies
pip install -r requirements.txt

# Run the application
cd app
streamlit run app.py
```

## Usage

1. Navigate to the "Prediction" page
2. Select the disease you want to predict
3. Enter environmental parameters (temperature, salinity, pH, etc.)
4. Enter farm management practices (stocking density, probiotics use, etc.)
5. Click "Predict" to get the disease risk assessment

## Data

This project uses simulated data that mimics real-world relationships between environmental factors and disease outbreaks. The data generation scripts are included:

- `generate_simulated_data.py`: Comprehensive script for generating data for all diseases
- `generate_simple_data.py`: Simplified script with minimal dependencies

## Project Structure

```
PenaeusPredict/
├── app/                      # Streamlit web application
├── data/                     # Data directory
│   ├── raw/                  # Raw data files
│   └── simulated/            # Simulated data for each disease
├── models/                   # Trained models
│   ├── wssv/                 # WSSV-specific models
│   ├── ems_ahpnd/            # EMS/AHPND-specific models
│   └── ...                   # Other disease models
├── src/                      # Source code
│   ├── data/                 # Data processing scripts
│   ├── models/               # Model training scripts
│   └── visualization/        # Visualization utilities
├── notebooks/                # Jupyter notebooks for analysis
├── tests/                    # Test files
├── generate_simulated_data.py # Data generation script
├── retrain_with_real_data.py # Script for retraining with real data
└── README.md                 # This file
```

## Documentation

- [Multi-Disease Design Document](MULTI_DISEASE_DESIGN.md): Technical design for the multi-disease system
- [Portfolio Enhancement Summary](PORTFOLIO_ENHANCEMENT_SUMMARY.md): Overview of project enhancements
- [Simulated Data README](README_SIMULATED_DATA.md): Information about the simulated data generation

## Disclaimer

This project is for demonstration and educational purposes. The predictions are based on simulated data and should not be used for actual farm management decisions without validation against real-world data.

## Future Work

- Geographic visualization of disease risk
- Time-series forecasting for seasonal risk
- Mobile app for field use
- Image recognition for disease symptom identification

## License

This project is licensed under the MIT License - see the LICENSE file for details. 