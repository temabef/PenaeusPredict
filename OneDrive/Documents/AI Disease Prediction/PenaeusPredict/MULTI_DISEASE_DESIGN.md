# Multi-Disease Prediction System Design

This document outlines the design for extending PenaeusPredict to support multiple shrimp diseases beyond WSSV.

## 1. Overview

The current system predicts White Spot Syndrome Virus (WSSV) outbreaks based on environmental and farm management parameters. We will extend this to support multiple diseases that affect shrimp aquaculture.

## 2. Diseases to Support

1. **White Spot Syndrome Virus (WSSV)** - Already implemented
2. **Early Mortality Syndrome (EMS)/Acute Hepatopancreatic Necrosis Disease (AHPND)**
3. **Infectious Hypodermal and Hematopoietic Necrosis Virus (IHHNV)**
4. **Yellow Head Virus (YHV)**
5. **Infectious Myonecrosis Virus (IMNV)**

## 3. Data Structure Changes

### 3.1 Combined Dataset
- Create a combined dataset with a `disease_type` column to identify each disease
- Rename disease-specific outbreak columns to a generic `disease_outbreak` column
- Add a `data_source` column to identify real vs. simulated data

### 3.2 Disease-Specific Datasets
- Maintain separate datasets for each disease for training individual models
- Ensure consistent column naming across all datasets

## 4. Model Architecture

### 4.1 Multiple Model Approach
- Train a separate model for each disease
- Store models in disease-specific directories:
  ```
  models/
  ├── wssv/
  │   ├── final_model.pkl
  │   └── preprocessor.pkl
  ├── ems_ahpnd/
  │   ├── final_model.pkl
  │   └── preprocessor.pkl
  └── ...
  ```

### 4.2 Model Selection
- At prediction time, select the appropriate model based on the disease chosen by the user

## 5. UI Changes

### 5.1 Disease Selection
- Add a disease selection dropdown to the prediction page
- Update UI labels to reflect the selected disease

### 5.2 Disease-Specific Information
- Add disease-specific information panels
- Show unique risk factors for each disease

### 5.3 Comparative Analysis
- Add an option to compare risk across multiple diseases for the same parameters

## 6. Backend Changes

### 6.1 Model Loading
```python
def load_model(disease_type="wssv"):
    """Load model and preprocessor for the specified disease"""
    model_dir = MODEL_DIR / disease_type
    model_path = model_dir / "final_model.pkl"
    preprocessor_path = model_dir / "preprocessor.pkl"
    
    if not model_path.exists() or not preprocessor_path.exists():
        return None, None, None, None
        
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    # Get model type and feature importance
    model_type = type(model).__name__
    feature_importance = pd.read_csv(model_dir / "feature_importance.csv")
    
    return model, preprocessor, model_type, feature_importance
```

### 6.2 Prediction Function
```python
def predict_disease_risk(input_data, disease_type="wssv"):
    """Predict risk for the specified disease"""
    model, preprocessor, model_type, feature_importance = load_model(disease_type)
    
    if model is None:
        return None, None, None
    
    X_processed = preprocessor.transform(input_data)
    outbreak_prob = model.predict_proba(X_processed)[0, 1]
    outbreak_pred = 1 if outbreak_prob >= 0.5 else 0
    
    return outbreak_prob, outbreak_pred, feature_importance
```

## 7. Data Collection Changes

### 7.1 Disease-Specific Data Collection
- Update data collection form to include disease type
- Store collected data in disease-specific directories

### 7.2 Batch Upload
- Update batch upload to support multiple disease types
- Provide templates for each disease

## 8. Training Pipeline Changes

### 8.1 Disease-Specific Training
```python
def train_models_for_all_diseases():
    """Train models for all supported diseases"""
    diseases = ["wssv", "ems_ahpnd", "ihhnv", "yhv", "imnv"]
    
    for disease in diseases:
        print(f"Training model for {disease}...")
        train_model_for_disease(disease)
```

### 8.2 Model Evaluation
- Add comparative evaluation across disease models
- Report performance metrics for each disease

## 9. Implementation Plan

### Phase 1: Data Generation and Structure
1. Create the data generation script for multiple diseases
2. Establish the new data structure
3. Generate simulated datasets

### Phase 2: Backend Changes
1. Modify model loading and prediction functions
2. Implement disease-specific training pipeline
3. Create model evaluation for multiple diseases

### Phase 3: UI Changes
1. Add disease selection to prediction page
2. Update visualization for multiple diseases
3. Implement comparative analysis

### Phase 4: Testing and Documentation
1. Test with simulated data for all diseases
2. Document the multi-disease capabilities
3. Create user guides for the extended functionality

## 10. Future Extensions

- Add geographic visualization of disease risk
- Implement time-series forecasting for seasonal risk
- Create a mobile app version for field use
- Add image recognition for disease symptom identification 