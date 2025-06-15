#!/usr/bin/env python3
"""
Streamlit web application for White Spot Syndrome Virus (WSSV) outbreak prediction
in Southeast Asian shrimp aquaculture.

This app allows users to input environmental and farm management parameters
to predict the risk of WSSV outbreaks in shrimp ponds.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shap
import os
import datetime
import uuid

# Set page configuration
st.set_page_config(
    page_title="PenaeusPredict: WSSV Outbreak Risk Prediction",
    page_icon="🦐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
REAL_DATA_DIR = DATA_DIR / "real_data"

# Ensure real data directory exists
REAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load model and preprocessor
@st.cache_resource
def load_model():
    try:
        model_path = MODEL_DIR / "final_model.pkl"
        preprocessor_path = MODEL_DIR / "preprocessor.pkl"
        
        if not model_path.exists():
            st.error(f"Model file not found at {model_path}")
            return None, None, None, None
            
        if not preprocessor_path.exists():
            st.error(f"Preprocessor file not found at {preprocessor_path}")
            return None, None, None, None
        
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        
        # Get additional data for app
        model_type = type(model).__name__
        
        # Get feature importances
        if model_type in ["RandomForestClassifier", "XGBClassifier"]:
            feature_dir = MODEL_DIR / (
                "random_forest" if model_type == "RandomForestClassifier" else "xgboost"
            )
        else:
            feature_dir = MODEL_DIR / "logistic_regression"
            
        feature_importance_path = feature_dir / "feature_importance.csv"
        if not feature_dir.exists() or not feature_importance_path.exists():
            st.warning(f"Feature importance file not found. Some visualizations will be limited.")
            return model, preprocessor, model_type, None
            
        feature_importance = pd.read_csv(feature_importance_path)
        feature_importance = feature_importance.sort_values("importance", ascending=False)
        
        return model, preprocessor, model_type, feature_importance
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

def get_color_for_risk(risk_value):
    """Return color based on risk value (0-1)"""
    if risk_value < 0.3:
        return "green"
    elif risk_value < 0.7:
        return "orange"
    else:
        return "red"

def save_real_data(input_data, outbreak_actual=None):
    """Save real-world data with actual outbreak status for model improvement"""
    # Add timestamp and unique ID
    input_data['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    input_data['data_id'] = str(uuid.uuid4())[:8]
    
    if outbreak_actual is not None:
        input_data['wssv_outbreak_actual'] = outbreak_actual
    
    # Create filename with timestamp
    filename = f"real_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = REAL_DATA_DIR / filename
    
    # Save data
    input_data.to_csv(filepath, index=False)
    return filepath

def main():
    # Sidebar for navigation
    st.sidebar.title("🦐 PenaeusPredict")
    page = st.sidebar.radio("Navigation", ["Prediction Tool", "Data Collection", "About"])
    
    if page == "Prediction Tool":
        prediction_page()
    elif page == "Data Collection":
        data_collection_page()
    else:
        about_page()

def prediction_page():
    # Header
    st.title("🦐 PenaeusPredict: WSSV Outbreak Risk Prediction")
    st.markdown("""
    ### AI-Based Prediction Tool for White Spot Syndrome Virus Outbreaks
    
    This tool predicts the risk of WSSV outbreaks in *Litopenaeus vannamei* (Pacific white shrimp) farms 
    in Southeast Asia based on environmental and farm management parameters.
    
    ---
    """)
    
    # Load model and preprocessor
    model, preprocessor, model_type, feature_importance = load_model()
    
    if model is None:
        st.warning("Model not loaded. Please train the model first by running `python run.py --steps train`.")
        st.info("If you've already trained the model, check that the model files exist in the 'models' directory.")
        return
        
    # Create layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🌊 Environmental Parameters")
        
        # Water parameters
        water_temp = st.slider("Water Temperature (°C)", 20.0, 35.0, 28.0, 0.1)
        salinity = st.slider("Salinity (ppt)", 0.0, 35.0, 15.0, 0.5)
        ph = st.slider("pH", 6.0, 9.0, 7.8, 0.1)
        do = st.slider("Dissolved Oxygen (mg/L)", 2.0, 10.0, 5.0, 0.1)
        ammonia = st.slider("Ammonia Level (mg/L)", 0.0, 5.0, 0.5, 0.01)
        rainfall = st.slider("Monthly Rainfall (mm)", 0.0, 500.0, 100.0, 10.0)
        
        st.subheader("🗓️ Season & Location")
        
        # Season and location
        country = st.selectbox(
            "Country", 
            ["Thailand", "Vietnam", "Indonesia", "India", "Philippines"]
        )
        season = st.selectbox("Season", ["Dry", "Wet"])
        month = st.slider("Month (1-12)", 1, 12, 6)
        
    with col2:
        st.subheader("🏊‍♀️ Farm Management Parameters")
        
        # Farm parameters
        stocking_density = st.slider("Stocking Density (PL/m²)", 10.0, 200.0, 80.0, 5.0)
        pond_size = st.slider("Pond Size (m²)", 100.0, 5000.0, 1000.0, 100.0)
        water_exchange = st.slider("Water Exchange Rate (%/day)", 0.0, 30.0, 10.0, 1.0)
        history_wssv = st.selectbox("History of WSSV in Area", ["No", "Yes"])
        probiotics = st.selectbox("Using Probiotics", ["No", "Yes"])
        antibiotics = st.selectbox("Using Antibiotics", ["No", "Yes"])
        culture_duration = st.slider("Culture Duration (days)", 30, 180, 100, 5)
        
        st.subheader("⚙️ Actions")
        predict_button = st.button("Predict WSSV Risk", type="primary", use_container_width=True)
    
    # Process binary/categorical features
    history_wssv_binary = 1 if history_wssv == "Yes" else 0
    probiotics_binary = 1 if probiotics == "Yes" else 0
    antibiotics_binary = 1 if antibiotics == "Yes" else 0
    
    # Create dataframe from inputs
    input_data = pd.DataFrame({
        # Environmental factors
        'water_temperature': [water_temp],
        'salinity': [salinity],
        'ph': [ph],
        'dissolved_oxygen': [do],
        'ammonia': [ammonia],
        'rainfall': [rainfall],
        
        # Farm practices
        'stocking_density': [stocking_density],
        'pond_size': [pond_size],
        'water_exchange_rate': [water_exchange],
        'wssv_history': [history_wssv_binary],
        'probiotics_used': [probiotics_binary],
        'antibiotics_used': [antibiotics_binary],
        'culture_duration': [culture_duration],
        
        # Metadata
        'country': [country],
        'season': [season],
        'month': [month],
        'year': [datetime.datetime.now().year],  # Add current year
        
        # Add default outbreak probability (will be predicted)
        'outbreak_probability': [0.5]  # Default value, will be replaced by prediction
    })
    
    # Feature engineering (same as in training)
    input_data['temp_salinity_interaction'] = input_data['water_temperature'] * input_data['salinity']
    
    # Water quality index
    ph_score = (input_data['ph'] - 7).abs() / 2
    do_score = 1 - (input_data['dissolved_oxygen'] / 10)
    ammonia_score = np.minimum(input_data['ammonia'] / 3, 1)
    input_data['water_quality_index'] = (ph_score + do_score + ammonia_score) / 3
    
    # Farm risk score
    density_score = np.minimum(input_data['stocking_density'] / 100, 1)
    input_data['farm_risk_score'] = (density_score + input_data['wssv_history']) / 2
    
    # Month encoding
    input_data['month_sin'] = np.sin(2 * np.pi * input_data['month'] / 12)
    input_data['month_cos'] = np.cos(2 * np.pi * input_data['month'] / 12)
    
    # Make prediction
    if predict_button:
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        # Create prediction display columns
        pred_col1, pred_col2 = st.columns([1, 1])
        
        # Prepare data for prediction
        try:
            # Apply preprocessing
            try:
                X_processed = preprocessor.transform(input_data)
            except ValueError as column_error:
                if "columns are missing" in str(column_error):
                    st.error(f"Error: {column_error}")
                    st.info("This error occurs because the model was trained with columns that are not in your input data. Please retrain the model with 'python retrain_with_real_data.py' or use the following workaround:")
                    
                    # Add any missing columns that the error mentions
                    error_msg = str(column_error)
                    if "outbreak_probability" in error_msg and "year" in error_msg:
                        st.info("Adding missing columns 'outbreak_probability' and 'year' to input data...")
                        input_data['year'] = datetime.datetime.now().year
                        input_data['outbreak_probability'] = 0.5  # Default value
                        X_processed = preprocessor.transform(input_data)
                    else:
                        raise column_error
                else:
                    raise column_error
            
            # Get probability and class prediction
            outbreak_prob = model.predict_proba(X_processed)[0, 1]
            outbreak_pred = 1 if outbreak_prob >= 0.5 else 0
            
            with pred_col1:
                # Display risk gauge
                risk_color = get_color_for_risk(outbreak_prob)
                st.markdown(f"""
                ### Risk Level: <span style="color:{risk_color}">{'High' if outbreak_prob >= 0.7 else 'Moderate' if outbreak_prob >= 0.3 else 'Low'}</span>
                """, unsafe_allow_html=True)
                
                # Risk gauge visualization
                fig, ax = plt.subplots(figsize=(8, 2))
                plt.axis('off')
                
                # Create gauge chart
                gauge = np.linspace(0, 1, 100)
                gauge_y = np.zeros_like(gauge)
                
                # Create the colored gauge background
                plt.scatter(gauge, gauge_y, c=gauge, cmap='RdYlGn_r', s=1000, marker='|')
                plt.scatter([outbreak_prob], [0], c='black', s=500, marker='v')
                plt.xlim(0, 1)
                plt.ylim(-0.5, 0.5)
                plt.annotate(f"Risk: {outbreak_prob:.2f}", xy=(outbreak_prob, 0.1), 
                             xytext=(outbreak_prob, 0.1), ha='center', fontsize=12)
                
                st.pyplot(fig)
                
                # Risk interpretation
                st.markdown(f"""
                ### Prediction: {outbreak_pred}
                - **Probability**: {outbreak_prob:.2f} (or {outbreak_prob*100:.1f}%)
                - **Interpretation**: {'High risk of WSSV outbreak' if outbreak_prob >= 0.7 else 'Moderate risk of WSSV outbreak' if outbreak_prob >= 0.3 else 'Low risk of WSSV outbreak'}
                """)
                
                # Option to save this prediction with actual outcome (for real data collection)
                st.markdown("### Record Actual Outcome")
                st.markdown("If this is a real case, you can record the actual outcome to help improve the model:")
                
                col1a, col1b, col1c = st.columns([1, 1, 1])
                with col1a:
                    if st.button("No Outbreak Occurred", key="no_outbreak"):
                        filepath = save_real_data(input_data, outbreak_actual=0)
                        st.success(f"Data saved to {filepath}")
                with col1b:
                    if st.button("Outbreak Occurred", key="yes_outbreak"):
                        filepath = save_real_data(input_data, outbreak_actual=1)
                        st.success(f"Data saved to {filepath}")
                with col1c:
                    if st.button("Save Without Outcome", key="no_outcome"):
                        filepath = save_real_data(input_data)
                        st.success(f"Data saved to {filepath}")
                
            with pred_col2:
                # Top factors affecting prediction
                st.markdown("### Key Risk Factors")
                
                # Display top 5 contributing features based on model type
                if feature_importance is not None and model_type in ["RandomForestClassifier", "XGBClassifier"]:
                    top_features = feature_importance.head(5)
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.barplot(x="importance", y="feature", data=top_features, palette="viridis")
                    plt.title("Top Features Contributing to Risk")
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Recommendations based on top features
                    st.markdown("### Risk Mitigation Recommendations")
                    recommendations = []
                    
                    # Temperature recommendations
                    if 'water_temperature' in top_features['feature'].values:
                        if water_temp > 30:
                            recommendations.append("- Reduce water temperature below 30°C if possible")
                        elif water_temp < 25:
                            recommendations.append("- Maintain water temperature between 28-30°C")
                    
                    # Stocking density recommendations
                    if 'stocking_density' in top_features['feature'].values:
                        if stocking_density > 100:
                            recommendations.append("- Consider reducing stocking density below 100 PL/m²")
                    
                    # Water quality recommendations
                    if any(col in top_features['feature'].values for col in ['water_quality_index', 'ph', 'dissolved_oxygen', 'ammonia']):
                        if do < 4:
                            recommendations.append("- Increase aeration to improve dissolved oxygen levels")
                        if ammonia > 1:
                            recommendations.append("- Reduce feeding and increase water exchange to lower ammonia")
                        if abs(ph - 7.5) > 0.5:
                            recommendations.append("- Adjust pH closer to optimal range (7.5-8.0)")
                    
                    # Farm management recommendations
                    if any(col in top_features['feature'].values for col in ['farm_risk_score', 'wssv_history']):
                        if history_wssv_binary == 1:
                            recommendations.append("- Implement stricter biosecurity measures for high-risk areas")
                    
                    # If no specific recommendations, give general advice
                    if not recommendations:
                        recommendations = [
                            "- Maintain good water quality through regular monitoring",
                            "- Practice biosecurity protocols to prevent disease introduction",
                            "- Consider using SPF (specific pathogen free) post-larvae",
                            "- Monitor shrimp behavior regularly for early disease detection"
                        ]
                    
                    for rec in recommendations:
                        st.markdown(rec)
                else:
                    st.warning("Feature importance data not available. Unable to show key risk factors.")
                
        except Exception as e:
            st.error(f"Error making prediction: {e}")

def data_collection_page():
    st.title("📊 Data Collection for Research")
    st.markdown("""
    ### Record Real-World WSSV Outbreak Data
    
    This form allows you to record actual data from shrimp farms for research purposes.
    All data collected will be stored securely and can be used to improve the model.
    
    ---
    """)
    
    # Create tabs for different data collection methods
    tab1, tab2, tab3 = st.tabs(["Manual Entry", "Batch Upload", "View Collected Data"])
    
    with tab1:
        st.subheader("Enter Farm Data Manually")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Environmental parameters
            st.subheader("Environmental Parameters")
            water_temp = st.number_input("Water Temperature (°C)", 20.0, 35.0, 28.0, 0.1)
            salinity = st.number_input("Salinity (ppt)", 0.0, 35.0, 15.0, 0.5)
            ph = st.number_input("pH", 6.0, 9.0, 7.8, 0.1)
            do = st.number_input("Dissolved Oxygen (mg/L)", 2.0, 10.0, 5.0, 0.1)
            ammonia = st.number_input("Ammonia Level (mg/L)", 0.0, 5.0, 0.5, 0.01)
            rainfall = st.number_input("Monthly Rainfall (mm)", 0.0, 500.0, 100.0, 10.0)
            
            # Season and location
            st.subheader("Season & Location")
            country = st.selectbox(
                "Country", 
                ["Thailand", "Vietnam", "Indonesia", "India", "Philippines"],
                key="country_data"
            )
            season = st.selectbox("Season", ["Dry", "Wet"], key="season_data")
            month = st.number_input("Month (1-12)", 1, 12, 6, 1)
            
        with col2:
            # Farm parameters
            st.subheader("Farm Management")
            stocking_density = st.number_input("Stocking Density (PL/m²)", 10.0, 200.0, 80.0, 5.0)
            pond_size = st.number_input("Pond Size (m²)", 100.0, 5000.0, 1000.0, 100.0)
            water_exchange = st.number_input("Water Exchange Rate (%/day)", 0.0, 30.0, 10.0, 1.0)
            history_wssv = st.selectbox("History of WSSV in Area", ["No", "Yes"], key="history_data")
            probiotics = st.selectbox("Using Probiotics", ["No", "Yes"], key="probiotics_data")
            antibiotics = st.selectbox("Using Antibiotics", ["No", "Yes"], key="antibiotics_data")
            culture_duration = st.number_input("Culture Duration (days)", 30, 180, 100, 5)
            
            # Outcome (actual)
            st.subheader("Actual Outcome")
            outbreak_actual = st.radio("Did WSSV Outbreak Occur?", ["Yes", "No", "Unknown"])
            
        # Submit button
        if st.button("Submit Data", type="primary"):
            # Process binary/categorical features
            history_wssv_binary = 1 if history_wssv == "Yes" else 0
            probiotics_binary = 1 if probiotics == "Yes" else 0
            antibiotics_binary = 1 if antibiotics == "Yes" else 0
            
            # Convert outbreak_actual to numeric
            if outbreak_actual == "Yes":
                outbreak_actual_value = 1
            elif outbreak_actual == "No":
                outbreak_actual_value = 0
            else:
                outbreak_actual_value = None
            
            # Create dataframe from inputs
            input_data = pd.DataFrame({
                # Environmental factors
                'water_temperature': [water_temp],
                'salinity': [salinity],
                'ph': [ph],
                'dissolved_oxygen': [do],
                'ammonia': [ammonia],
                'rainfall': [rainfall],
                
                # Farm practices
                'stocking_density': [stocking_density],
                'pond_size': [pond_size],
                'water_exchange_rate': [water_exchange],
                'wssv_history': [history_wssv_binary],
                'probiotics_used': [probiotics_binary],
                'antibiotics_used': [antibiotics_binary],
                'culture_duration': [culture_duration],
                
                # Metadata
                'country': [country],
                'season': [season],
                'month': [month],
                'year': [datetime.datetime.now().year],  # Add current year
                
                # Add default outbreak probability (will be predicted)
                'outbreak_probability': [0.5]  # Default value, will be replaced by prediction
            })
            
            # Save data
            filepath = save_real_data(input_data, outbreak_actual_value)
            st.success(f"Data saved successfully to {filepath}")
    
    with tab2:
        st.subheader("Upload Multiple Records")
        st.markdown("""
        You can upload a CSV file with multiple records. The file should have the following columns:
        
        - water_temperature
        - salinity
        - ph
        - dissolved_oxygen
        - ammonia
        - rainfall
        - stocking_density
        - pond_size
        - water_exchange_rate
        - wssv_history (0 or 1)
        - probiotics_used (0 or 1)
        - antibiotics_used (0 or 1)
        - culture_duration
        - country
        - season
        - month
        - wssv_outbreak_actual (0 or 1, optional)
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.write(f"Uploaded data preview ({data.shape[0]} records):")
                st.dataframe(data.head())
                
                if st.button("Process and Save Data"):
                    # Process each row
                    for i, row in data.iterrows():
                        # Create a DataFrame for this row
                        row_df = pd.DataFrame([row])
                        
                        # Extract outbreak_actual if it exists
                        outbreak_actual = None
                        if 'wssv_outbreak_actual' in row_df.columns:
                            outbreak_actual = row_df['wssv_outbreak_actual'].iloc[0]
                            row_df = row_df.drop(columns=['wssv_outbreak_actual'])
                        
                        # Save this record
                        save_real_data(row_df, outbreak_actual)
                    
                    st.success(f"Successfully processed and saved {data.shape[0]} records")
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    with tab3:
        st.subheader("View Collected Data")
        
        # Check if any data files exist
        data_files = list(REAL_DATA_DIR.glob("*.csv"))
        
        if not data_files:
            st.info("No data has been collected yet.")
        else:
            st.write(f"Found {len(data_files)} data files.")
            
            # Option to combine all data
            if st.button("Combine All Data"):
                all_data = []
                for file in data_files:
                    df = pd.read_csv(file)
                    all_data.append(df)
                
                if all_data:
                    combined_data = pd.concat(all_data, ignore_index=True)
                    st.write(f"Combined data: {combined_data.shape[0]} records")
                    st.dataframe(combined_data)
                    
                    # Download option
                    csv = combined_data.to_csv(index=False)
                    st.download_button(
                        label="Download Combined Data",
                        data=csv,
                        file_name="combined_wssv_data.csv",
                        mime="text/csv"
                    )

def about_page():
    st.title("🦐 About PenaeusPredict")
    st.markdown("""
    ### AI-Based WSSV Prediction Tool
    
    **PenaeusPredict** is a machine learning-based tool designed to predict White Spot Syndrome Virus (WSSV) outbreaks in shrimp aquaculture.
    
    #### Project Structure
    
    The project is organized into the following components:
    
    1. **Data Collection**: Scripts to gather or generate data about shrimp pond conditions and WSSV outbreaks.
    
    2. **Data Preprocessing**: Tools to clean, transform, and prepare data for modeling.
    
    3. **Model Training**: Algorithms that learn patterns from the data to predict WSSV outbreaks.
    
    4. **Web Application**: This interactive tool that allows users to input farm parameters and get predictions.
    
    #### For Research Publication
    
    This tool includes features specifically designed for research purposes:
    
    - **Real Data Collection**: The "Data Collection" page allows you to systematically collect real-world data.
    
    - **Data Export**: All collected data can be exported for analysis and inclusion in research papers.
    
    - **Model Retraining**: As you collect more real data, you can retrain the model to improve its accuracy.
    
    #### How to Retrain the Model with Real Data
    
    1. Collect real-world data using the "Data Collection" page.
    
    2. Export the combined data from the "View Collected Data" tab.
    
    3. Place the CSV file in the `data/raw` directory.
    
    4. Run the following command to retrain the model:
       ```
       python run.py --steps preprocess train
       ```
    
    5. The model will be updated with your real-world data.
    
    #### Publication Information
    
    When publishing research using this tool, please cite the following:
    
    ```
    [Your Name], et al. "PenaeusPredict: An AI-Based Tool for Predicting WSSV Outbreaks in Shrimp Aquaculture"
    [Journal Name], [Year]
    ```
    """)

if __name__ == "__main__":
    main()