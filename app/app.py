#!/usr/bin/env python3
"""
Streamlit web application for Multi-Disease Shrimp Outbreak Prediction
in Southeast Asian shrimp aquaculture.

This app allows users to input environmental and farm management parameters
to predict the risk of multiple shrimp diseases: WSSV, IHHNV, and IMNV.
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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="PenaeusPredict: Multi-Disease Shrimp Outbreak Prediction",
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

# Define supported diseases
SUPPORTED_DISEASES = {
    'WSSV': {
        'full_name': 'White Spot Syndrome Virus',
        'description': 'A highly contagious viral disease causing white spots on shrimp',
        'color': '#FF6B6B',
        'icon': '🦐'
    },
    'IHHNV': {
        'full_name': 'Infectious Hypodermal and Hematopoietic Necrosis Virus',
        'description': 'A viral disease affecting shrimp growth and survival',
        'color': '#4ECDC4',
        'icon': '🦐'
    },
    'IMNV': {
        'full_name': 'Infectious Myonecrosis Virus',
        'description': 'A viral disease causing muscle necrosis in shrimp',
        'color': '#45B7D1',
        'icon': '🦐'
    }
}

# Load models for all diseases
@st.cache_resource
def load_all_models():
    """Load models and preprocessors for all supported diseases"""
    models = {}
    preprocessors = {}
    feature_importances = {}
    
    for disease in SUPPORTED_DISEASES.keys():
        disease_dir = MODEL_DIR / disease.lower()
        model_path = disease_dir / "model.pkl"
        preprocessor_path = disease_dir / "preprocessor.pkl"
        feature_importance_path = disease_dir / "feature_importance.csv"
        
        try:
            if model_path.exists() and preprocessor_path.exists():
                models[disease] = joblib.load(model_path)
                preprocessors[disease] = joblib.load(preprocessor_path)
                
                # Load feature importance if available
                if feature_importance_path.exists():
                    feature_importances[disease] = pd.read_csv(feature_importance_path)
                    feature_importances[disease] = feature_importances[disease].sort_values("importance", ascending=False)
                else:
                    feature_importances[disease] = None
                    
                # Model loaded successfully (no message to avoid cluttering the UI)
            else:
                # Model files not found (no message to avoid cluttering the UI)
                models[disease] = None
                preprocessors[disease] = None
                feature_importances[disease] = None
                
        except Exception as e:
            # Error loading model (no message to avoid cluttering the UI)
            models[disease] = None
            preprocessors[disease] = None
            feature_importances[disease] = None
    
    return models, preprocessors, feature_importances

def get_color_for_risk(risk_value):
    """Return color based on risk value (0-1)"""
    if risk_value < 0.3:
        return "#2ECC71"  # Green
    elif risk_value < 0.7:
        return "#F39C12"  # Orange
    else:
        return "#E74C3C"  # Red

def get_risk_level(risk_value):
    """Return risk level text based on risk value"""
    if risk_value < 0.3:
        return "Low Risk"
    elif risk_value < 0.7:
        return "Moderate Risk"
    else:
        return "High Risk"

def save_real_data(input_data, disease_predictions=None):
    """Save real-world data with actual outbreak status for model improvement"""
    # Add timestamp and unique ID
    input_data['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    input_data['data_id'] = str(uuid.uuid4())[:8]
    
    # Add disease predictions if provided
    if disease_predictions:
        for disease, prob in disease_predictions.items():
            input_data[f'{disease.lower()}_outbreak_probability'] = prob
    
    # Create filename with timestamp
    filename = f"real_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = REAL_DATA_DIR / filename
    
    # Save data
    input_data.to_csv(filepath, index=False)
    return filepath

def create_risk_gauge(risk_value, disease_name, disease_color):
    """Create a beautiful risk gauge using Plotly"""
    fig = go.Figure()
    
    # Create gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=risk_value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{disease_name} Risk", 'font': {'size': 20}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': disease_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#2ECC71'},
                {'range': [30, 70], 'color': '#F39C12'},
                {'range': [70, 100], 'color': '#E74C3C'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_feature_importance_chart(feature_importance, disease_name, top_n=5):
    """Create feature importance chart using Plotly"""
    if feature_importance is None:
        return None
    
    top_features = feature_importance.head(top_n)
    
    fig = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title=f"Top {top_n} Risk Factors for {disease_name}",
        color='importance',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Importance",
        yaxis_title="Features"
    )
    
    return fig

def get_recommendations(feature_importance, input_data, disease_name):
    """Generate recommendations based on top features and input values"""
    if feature_importance is None:
        return [
            "Maintain good water quality through regular monitoring",
            "Practice biosecurity protocols to prevent disease introduction",
            "Consider using SPF (specific pathogen free) post-larvae",
            "Monitor shrimp behavior regularly for early disease detection"
        ]
    
    recommendations = []
    top_features = feature_importance.head(5)['feature'].values
    
    # Water temperature recommendations
    if 'water_temperature' in top_features:
        temp = input_data['water_temperature'].iloc[0]
        if temp > 30:
            recommendations.append("🌡️ Reduce water temperature below 30°C if possible")
        elif temp < 25:
            recommendations.append("🌡️ Maintain water temperature between 28-30°C")
    
    # Stocking density recommendations
    if 'stocking_density' in top_features:
        density = input_data['stocking_density'].iloc[0]
        if density > 100:
            recommendations.append("🐟 Consider reducing stocking density below 100 PL/m²")
    
    # Water quality recommendations
    if any(col in top_features for col in ['water_quality_index', 'ph', 'dissolved_oxygen', 'ammonia']):
        do = input_data['dissolved_oxygen'].iloc[0]
        ammonia = input_data['ammonia'].iloc[0]
        ph = input_data['ph'].iloc[0]
        
        if do < 4:
            recommendations.append("💧 Increase aeration to improve dissolved oxygen levels")
        if ammonia > 1:
            recommendations.append("🧪 Reduce feeding and increase water exchange to lower ammonia")
        if abs(ph - 7.5) > 0.5:
            recommendations.append("⚗️ Adjust pH closer to optimal range (7.5-8.0)")
    
    # Farm management recommendations
    if any(col in top_features for col in ['farm_risk_score', 'wssv_history']):
        history = input_data['wssv_history'].iloc[0]
        if history == 1:
            recommendations.append("🛡️ Implement stricter biosecurity measures for high-risk areas")
    
    # Disease-specific recommendations
    if disease_name == "WSSV":
        recommendations.extend([
            "🦐 Use SPF (Specific Pathogen Free) post-larvae",
            "🧼 Disinfect equipment and vehicles entering the farm",
            "🚫 Avoid introducing wild shrimp or other crustaceans"
        ])
    elif disease_name == "IHHNV":
        recommendations.extend([
            "📏 Monitor shrimp growth rates regularly",
            "🔍 Check for signs of stunted growth or deformities",
            "🧬 Use IHHNV-free broodstock and post-larvae"
        ])
    elif disease_name == "IMNV":
        recommendations.extend([
            "💪 Monitor for signs of muscle weakness or necrosis",
            "🏥 Isolate affected ponds immediately",
            "🧪 Test for IMNV before introducing new stock"
        ])
    
    # If no specific recommendations, give general advice
    if not recommendations:
        recommendations = [
            "🌊 Maintain good water quality through regular monitoring",
            "🛡️ Practice biosecurity protocols to prevent disease introduction",
            "🔬 Consider using SPF (specific pathogen free) post-larvae",
            "👀 Monitor shrimp behavior regularly for early disease detection"
        ]
    
    return recommendations

def main():
    # Load all models
    models, preprocessors, feature_importances = load_all_models()
    
    # Sidebar for navigation
    st.sidebar.title("🦐 PenaeusPredict")
    st.sidebar.markdown("### Multi-Disease Prediction System")
    
    # Disease selection
    selected_diseases = st.sidebar.multiselect(
        "Select Diseases to Predict",
        list(SUPPORTED_DISEASES.keys()),
        default=list(SUPPORTED_DISEASES.keys()),
        help="Choose which diseases to include in your prediction"
    )
    
    if not selected_diseases:
        st.warning("Please select at least one disease to predict.")
        return
    
    # Navigation
    page = st.sidebar.radio("Navigation", ["Prediction Tool", "Data Collection", "About"])
    
    if page == "Prediction Tool":
        prediction_page(models, preprocessors, feature_importances, selected_diseases)
    elif page == "Data Collection":
        data_collection_page()
    else:
        about_page()

def prediction_page(models, preprocessors, feature_importances, selected_diseases):
    # Header
    st.title("🦐 PenaeusPredict: Multi-Disease Shrimp Outbreak Prediction")
    st.markdown("""
    ### AI-Based Prediction Tool for Multiple Shrimp Diseases
    
    This tool predicts the risk of multiple shrimp diseases in Southeast Asia farms based on environmental and farm management parameters.
    
    ---
    """)
    
    # Check if models are loaded
    available_diseases = [d for d in selected_diseases if models[d] is not None]
    if not available_diseases:
        st.error("No models are available for the selected diseases. Please train the models first.")
        return
    
    # Create input form
    with st.form("prediction_form"):
        st.subheader("📊 Farm Parameters")
        
        # Create columns for better layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🌊 Environmental Parameters")
            water_temp = st.slider("Water Temperature (°C)", 20.0, 35.0, 28.0, 0.1)
            salinity = st.slider("Salinity (ppt)", 0.0, 35.0, 15.0, 0.5)
            ph = st.slider("pH", 6.0, 9.0, 7.8, 0.1)
            do = st.slider("Dissolved Oxygen (mg/L)", 2.0, 10.0, 5.0, 0.1)
            ammonia = st.slider("Ammonia Level (mg/L)", 0.0, 5.0, 0.5, 0.01)
            rainfall = st.slider("Monthly Rainfall (mm)", 0.0, 500.0, 100.0, 10.0)
        
        with col2:
            st.markdown("#### 🏊‍♀️ Farm Management")
            stocking_density = st.slider("Stocking Density (PL/m²)", 10.0, 200.0, 80.0, 5.0)
            pond_size = st.slider("Pond Size (m²)", 100.0, 5000.0, 1000.0, 100.0)
            water_exchange = st.slider("Water Exchange Rate (%/day)", 0.0, 30.0, 10.0, 1.0)
            history_wssv = st.selectbox("History of WSSV in Area", ["No", "Yes"])
            probiotics = st.selectbox("Using Probiotics", ["No", "Yes"])
            antibiotics = st.selectbox("Using Antibiotics", ["No", "Yes"])
            culture_duration = st.slider("Culture Duration (days)", 30, 180, 100, 5)
        
        with col3:
            st.markdown("#### 🗓️ Season & Location")
            country = st.selectbox(
                "Country", 
                ["Thailand", "Vietnam", "Indonesia", "India", "Philippines"]
            )
            season = st.selectbox("Season", ["Dry", "Wet"])
            month = st.slider("Month (1-12)", 1, 12, 6)
        
        # Submit button
        predict_button = st.form_submit_button("🚀 Predict Disease Risk", type="primary", use_container_width=True)
    
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
        'year': [datetime.datetime.now().year],
        
        # Add default outbreak probability (will be predicted)
        'outbreak_probability': [0.5],
        
        # Disease-specific outbreak columns (required by models)
        'wssv_outbreak': [0],  # Default to 0, will be predicted
        'ihhnv_outbreak': [0],  # Default to 0, will be predicted
        'imnv_outbreak': [0],   # Default to 0, will be predicted
        'ems_ahpnd_outbreak': [0],  # Default to 0
        'yhv_outbreak': [0],    # Default to 0
        
        # Additional metadata columns
        'disease_type': ['wssv'],  # Default, will be overridden for each disease
        'data_source': ['simulated'],  # Default source
        'disease_history': [history_wssv_binary]  # Use WSSV history as general disease history
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
    
    # Make predictions
    if predict_button:
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        # Store predictions
        disease_predictions = {}
        
        # Create results display
        results_col1, results_col2 = st.columns([2, 1])
        
        with results_col1:
            # Create comparison chart
            st.markdown("### 🎯 Disease Risk Comparison")
            
            # Make predictions for each disease
            prediction_data = []
            for disease in available_diseases:
                try:
                    # Apply preprocessing
                    X_processed = preprocessors[disease].transform(input_data)
                    
                    # Get probability
                    outbreak_prob = models[disease].predict_proba(X_processed)[0, 1]
                    disease_predictions[disease] = outbreak_prob
                    
                    prediction_data.append({
                        'Disease': disease,
                        'Risk Level': get_risk_level(outbreak_prob),
                        'Probability': outbreak_prob,
                        'Color': SUPPORTED_DISEASES[disease]['color']
                    })
                    
                except Exception as e:
                    st.error(f"Error predicting {disease}: {e}")
                    continue
            
            if prediction_data:
                # Create comparison chart
                df_predictions = pd.DataFrame(prediction_data)
                
                fig = px.bar(
                    df_predictions,
                    x='Disease',
                    y='Probability',
                    color='Disease',
                    color_discrete_map={row['Disease']: row['Color'] for row in prediction_data},
                    title="Disease Risk Comparison",
                    labels={'Probability': 'Risk Probability'},
                    text='Risk Level'
                )
                
                fig.update_traces(textposition='outside')
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    xaxis_title="Disease",
                    yaxis_title="Risk Probability"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with results_col2:
            st.markdown("### 📈 Risk Summary")
            
            # Display individual risk levels
            for disease in available_diseases:
                if disease in disease_predictions:
                    prob = disease_predictions[disease]
                    risk_level = get_risk_level(prob)
                    color = get_color_for_risk(prob)
                    
                    st.markdown(f"""
                    **{SUPPORTED_DISEASES[disease]['full_name']}**
                    - Risk: {risk_level}
                    - Probability: {prob:.1%}
                    """)
                    
                    # Create mini gauge
                    gauge_fig = create_risk_gauge(prob, disease, SUPPORTED_DISEASES[disease]['color'])
                    st.plotly_chart(gauge_fig, use_container_width=True)
        
        # Detailed analysis for each disease
        st.markdown("---")
        st.subheader("🔍 Detailed Analysis")
        
        # Create tabs for each disease
        disease_tabs = st.tabs([f"{SUPPORTED_DISEASES[d]['icon']} {d}" for d in available_diseases])
        
        for i, disease in enumerate(available_diseases):
            with disease_tabs[i]:
                if disease in disease_predictions:
                    prob = disease_predictions[disease]
                    
                    # Create two columns for detailed view
                    detail_col1, detail_col2 = st.columns([1, 1])
                    
                    with detail_col1:
                        st.markdown(f"### {SUPPORTED_DISEASES[disease]['full_name']}")
                        st.markdown(f"**Description**: {SUPPORTED_DISEASES[disease]['description']}")
                        st.markdown(f"**Risk Level**: {get_risk_level(prob)}")
                        st.markdown(f"**Probability**: {prob:.1%}")
                        
                        # Feature importance chart
                        if feature_importances[disease] is not None:
                            importance_fig = create_feature_importance_chart(
                                feature_importances[disease], 
                                disease,
                                top_n=5
                            )
                            if importance_fig:
                                st.plotly_chart(importance_fig, use_container_width=True)
                    
                    with detail_col2:
                        st.markdown("### 💡 Recommendations")
                        recommendations = get_recommendations(
                            feature_importances[disease], 
                            input_data, 
                            disease
                        )
                        
                        for rec in recommendations:
                            st.markdown(f"• {rec}")
                        
                        # Risk interpretation
                        st.markdown("### 📊 Risk Interpretation")
                        if prob >= 0.7:
                            st.error(f"⚠️ **High Risk**: Immediate action required to prevent {disease} outbreak")
                        elif prob >= 0.3:
                            st.warning(f"⚠️ **Moderate Risk**: Monitor closely and implement preventive measures")
                        else:
                            st.success(f"✅ **Low Risk**: Current conditions are favorable, maintain good practices")
        
        # Data collection option
        st.markdown("---")
        st.subheader("📝 Record This Prediction")
        st.markdown("Help improve our models by recording this prediction with actual outcomes:")
        
        col1a, col1b, col1c = st.columns([1, 1, 1])
        with col1a:
            if st.button("💾 Save Prediction Only", key="save_prediction"):
                filepath = save_real_data(input_data, disease_predictions)
                st.success(f"✅ Prediction saved to {filepath}")
        with col1b:
            if st.button("📊 Record Outcomes", key="record_outcomes"):
                st.info("Feature coming soon: Record actual disease outcomes to improve model accuracy")

def data_collection_page():
    st.title("📊 Data Collection for Research")
    st.markdown("""
    ### Record Real-World Multi-Disease Outbreak Data
    
    This form allows you to record actual data from shrimp farms for research purposes.
    All data collected will be stored securely and can be used to improve the models.
    
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
            
            # Disease outcomes
            st.subheader("Disease Outcomes")
            st.markdown("Record actual disease outbreaks:")
            wssv_outbreak = st.radio("WSSV Outbreak?", ["No", "Yes", "Unknown"])
            ihhnv_outbreak = st.radio("IHHNV Outbreak?", ["No", "Yes", "Unknown"])
            imnv_outbreak = st.radio("IMNV Outbreak?", ["No", "Yes", "Unknown"])
            
        # Submit button
        if st.button("Submit Data", type="primary"):
            # Process binary/categorical features
            history_wssv_binary = 1 if history_wssv == "Yes" else 0
            probiotics_binary = 1 if probiotics == "Yes" else 0
            antibiotics_binary = 1 if antibiotics == "Yes" else 0
            
            # Convert outbreak outcomes to numeric
            outbreak_mapping = {"Yes": 1, "No": 0, "Unknown": None}
            wssv_outbreak_value = outbreak_mapping[wssv_outbreak]
            ihhnv_outbreak_value = outbreak_mapping[ihhnv_outbreak]
            imnv_outbreak_value = outbreak_mapping[imnv_outbreak]
            
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
                'year': [datetime.datetime.now().year],
                
                # Add default outbreak probability
                'outbreak_probability': [0.5],
                
                # Disease-specific outbreak columns (required by models)
                'wssv_outbreak': [wssv_outbreak_value if wssv_outbreak_value is not None else 0],
                'ihhnv_outbreak': [ihhnv_outbreak_value if ihhnv_outbreak_value is not None else 0],
                'imnv_outbreak': [imnv_outbreak_value if imnv_outbreak_value is not None else 0],
                'ems_ahpnd_outbreak': [0],  # Default to 0
                'yhv_outbreak': [0],    # Default to 0
                
                # Additional metadata columns
                'disease_type': ['multi'],  # Multi-disease data collection
                'data_source': ['real'],  # Real data collection
                'disease_history': [history_wssv_binary]  # Use WSSV history as general disease history
            })
            
            # Save data with disease outcomes
            disease_outcomes = {}
            if wssv_outbreak_value is not None:
                disease_outcomes['wssv_outbreak_actual'] = wssv_outbreak_value
            if ihhnv_outbreak_value is not None:
                disease_outcomes['ihhnv_outbreak_actual'] = ihhnv_outbreak_value
            if imnv_outbreak_value is not None:
                disease_outcomes['imnv_outbreak_actual'] = imnv_outbreak_value
            
            filepath = save_real_data(input_data, disease_outcomes)
            st.success(f"✅ Data saved successfully to {filepath}")
    
    with tab2:
        st.subheader("Upload Multiple Records")
        st.markdown("""
        You can upload a CSV file with multiple records. The file should include all the parameters
        and can optionally include actual disease outcomes for model improvement.
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.write(f"📊 Uploaded data preview ({data.shape[0]} records):")
                st.dataframe(data.head())
                
                if st.button("Process and Save Data"):
                    # Process each row
                    for i, row in data.iterrows():
                        # Create a DataFrame for this row
                        row_df = pd.DataFrame([row])
                        
                        # Extract disease outcomes if they exist
                        disease_outcomes = {}
                        for disease in ['wssv', 'ihhnv', 'imnv']:
                            outcome_col = f'{disease}_outbreak_actual'
                            if outcome_col in row_df.columns:
                                disease_outcomes[outcome_col] = row_df[outcome_col].iloc[0]
                                row_df = row_df.drop(columns=[outcome_col])
                        
                        # Save this record
                        save_real_data(row_df, disease_outcomes)
                    
                    st.success(f"✅ Successfully processed and saved {data.shape[0]} records")
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
    
    with tab3:
        st.subheader("View Collected Data")
        
        # Check if any data files exist
        data_files = list(REAL_DATA_DIR.glob("*.csv"))
        
        if not data_files:
            st.info("📝 No data has been collected yet.")
        else:
            st.write(f"📁 Found {len(data_files)} data files.")
            
            # Option to combine all data
            if st.button("Combine All Data"):
                all_data = []
                for file in data_files:
                    df = pd.read_csv(file)
                    all_data.append(df)
                
                if all_data:
                    combined_data = pd.concat(all_data, ignore_index=True)
                    st.write(f"📊 Combined data: {combined_data.shape[0]} records")
                    st.dataframe(combined_data)
                    
                    # Download option
                    csv = combined_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Combined Data",
                        data=csv,
                        file_name="combined_multi_disease_data.csv",
                        mime="text/csv"
                    )

def about_page():
    st.title("🦐 About PenaeusPredict")
    st.markdown("""
    ### AI-Based Multi-Disease Shrimp Prediction Tool
    
    **PenaeusPredict** is a machine learning-based tool designed to predict multiple shrimp diseases
    in aquaculture, including White Spot Syndrome Virus (WSSV), Infectious Hypodermal and Hematopoietic 
    Necrosis Virus (IHHNV), and Infectious Myonecrosis Virus (IMNV).
    
    #### Supported Diseases
    
    """)
    
    # Display disease information
    for disease, info in SUPPORTED_DISEASES.items():
        st.markdown(f"""
        **{info['icon']} {disease} ({info['full_name']})**
        - {info['description']}
        """)
    
    st.markdown("""
    #### Project Structure
    
    The project is organized into the following components:
    
    1. **Data Collection**: Scripts to gather or generate data about shrimp pond conditions and disease outbreaks.
    
    2. **Data Preprocessing**: Tools to clean, transform, and prepare data for modeling.
    
    3. **Model Training**: Algorithms that learn patterns from the data to predict disease outbreaks.
    
    4. **Web Application**: This interactive tool that allows users to input farm parameters and get predictions.
    
    #### For Research Publication
    
    This tool includes features specifically designed for research purposes:
    
    - **Real Data Collection**: The "Data Collection" page allows you to systematically collect real-world data.
    
    - **Multi-Disease Support**: Predict and collect data for multiple shrimp diseases simultaneously.
    
    - **Data Export**: All collected data can be exported for analysis and inclusion in research papers.
    
    - **Model Retraining**: As you collect more real data, you can retrain the models to improve their accuracy.
    
    #### How to Retrain the Models with Real Data
    
    1. Collect real-world data using the "Data Collection" page.
    
    2. Export the combined data from the "View Collected Data" tab.
    
    3. Place the CSV file in the `data/raw` directory.
    
    4. Run the following command to retrain the models:
       ```
       python run.py --steps preprocess train
       ```
    
    5. The models will be updated with your real-world data.
    
    #### Publication Information
    
    When publishing research using this tool, please cite the following:
    
    ```
    [Your Name], et al. "PenaeusPredict: An AI-Based Multi-Disease Prediction Tool for Shrimp Aquaculture"
    [Journal Name], [Year]
    ```
    """)

if __name__ == "__main__":
    main()