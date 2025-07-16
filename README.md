# PenaeusPredict: Multi-Disease Shrimp Outbreak Prediction

A comprehensive machine learning application for predicting multiple shrimp disease outbreaks in Southeast Asian aquaculture based on environmental and farm management parameters.

## 🦐 **Supported Diseases**

Currently implemented and fully functional:
- **🦐 White Spot Syndrome Virus (WSSV)** - Highly contagious viral disease
- **🦐 Infectious Hypodermal and Hematopoietic Necrosis Virus (IHHNV)** - Affects shrimp growth and survival
- **🦐 Infectious Myonecrosis Virus (IMNV)** - Causes muscle necrosis in shrimp

*Note: EMS/AHPND and YHV models are planned for future implementation*

## ✨ **Key Features**

### **🎯 Multi-Disease Prediction**
- **Real-time risk assessment** for three major shrimp diseases
- **Interactive disease selection** - choose which diseases to predict
- **Comparative analysis** - see risk levels across multiple diseases simultaneously

### **📊 Advanced Visualizations**
- **Interactive risk gauges** with color-coded risk levels
- **Disease comparison charts** showing relative risk across diseases
- **Feature importance analysis** for each disease
- **Real-time recommendations** based on input parameters

### **🔬 Research-Grade Features**
- **Data collection system** for real-world validation
- **Model interpretability** with SHAP values and feature importance
- **Domain-specific recommendations** for each disease
- **Export capabilities** for research and analysis

### **🌐 User-Friendly Interface**
- **Modern Streamlit frontend** with intuitive design
- **Responsive layout** that works on different screen sizes
- **Real-time feedback** with immediate predictions
- **Professional UI** with clear visual hierarchy

## 🚀 **Quick Start**

### **Installation**

```bash
# Clone the repository
git clone https://github.com/temabef/PenaeusPredict.git
cd PenaeusPredict

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app/app.py
```

### **Usage**

1. **Navigate to the app** (usually at `http://localhost:8501`)
2. **Select diseases** to predict from the sidebar
3. **Enter farm parameters**:
   - Environmental factors (temperature, salinity, pH, etc.)
   - Farm management practices (stocking density, probiotics, etc.)
   - Location and seasonal data
4. **Click "Predict Disease Risk"** to get comprehensive results
5. **View detailed analysis** with risk gauges, recommendations, and feature importance

## 📈 **Model Performance**

Our trained models achieve excellent performance across all supported diseases:

| Disease | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| WSSV    | 87.5%    | 85.2%     | 89.1%  | 87.1%    |
| IHHNV   | 86.3%    | 84.7%     | 87.9%  | 86.3%    |
| IMNV    | 85.9%    | 83.4%     | 88.2%  | 85.7%    |

*Performance metrics based on simulated data with realistic disease-environment relationships*

## 🏗️ **System Architecture**

### **Multi-Disease Model Structure**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   WSSV Model    │    │  IHHNV Model    │    │   IMNV Model    │
│   (XGBoost)     │    │   (XGBoost)     │    │   (XGBoost)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Streamlit UI   │
                    │  (Frontend)     │
                    └─────────────────┘
```

### **Data Pipeline**
```
Raw Data → Preprocessing → Feature Engineering → Model Training → Prediction API
```

### **Key Technical Features**
- **Disease-specific preprocessing** pipelines
- **Advanced feature engineering** (water quality indices, risk scores)
- **Model interpretability** with SHAP values
- **Real-time caching** for optimal performance
- **Error handling** and validation

## 📁 **Project Structure**

```
PenaeusPredict/
├── app/                      # Streamlit web application
│   └── app.py               # Main application file
├── data/                     # Data directory
│   ├── raw/                  # Raw data files
│   ├── processed/            # Processed data for each disease
│   └── real_data/           # Real-world data collection
├── models/                   # Trained models
│   ├── wssv/                # WSSV-specific models and metrics
│   ├── ihhnv/               # IHHNV-specific models and metrics
│   └── imnv/                # IMNV-specific models and metrics
├── src/                      # Source code
│   ├── data/                # Data processing scripts
│   ├── models/              # Model training scripts
│   └── visualization/       # Visualization utilities
├── notebooks/               # Jupyter notebooks for analysis
├── generate_simulated_data.py # Multi-disease data generation
├── retrain_with_real_data.py # Script for retraining with real data
└── requirements.txt         # Python dependencies
```

## 🔬 **Technical Implementation**

### **Machine Learning Stack**
- **XGBoost** for disease prediction (chosen for interpretability and performance)
- **Scikit-learn** for preprocessing and evaluation
- **SHAP** for model interpretability
- **Streamlit** for web interface
- **Plotly** for interactive visualizations

### **Feature Engineering**
- **Water Quality Index**: Composite metric combining pH, DO, and ammonia
- **Farm Risk Score**: Based on stocking density and disease history
- **Temporal Features**: Cyclical encoding of seasonal patterns
- **Interaction Terms**: Temperature-salinity interactions for WSSV

### **Model Training Pipeline**
1. **Data Generation**: Realistic simulated data for each disease
2. **Feature Engineering**: Domain-specific feature creation
3. **Model Training**: XGBoost with hyperparameter optimization
4. **Evaluation**: Comprehensive metrics and SHAP analysis
5. **Deployment**: Streamlit integration with caching

## 📊 **Data Collection & Research**

### **Real Data Collection**
The application includes a comprehensive data collection system:
- **Manual entry** for individual farm records
- **Batch upload** for multiple records
- **Multi-disease outcomes** tracking
- **Export capabilities** for research analysis

### **Simulated Data**
- **1,800+ data points** across all diseases
- **Realistic disease-environment relationships**
- **Disease-specific parameter correlations**
- **Seasonal and geographic variations**

## 🎯 **Use Cases**

### **For Aquaculture Farmers**
- **Risk assessment** before stocking ponds
- **Preventive measures** based on current conditions
- **Management decisions** informed by AI predictions

### **For Researchers**
- **Data collection** for disease studies
- **Model validation** with real-world data
- **Comparative analysis** across diseases

### **For Industry Professionals**
- **Training tool** for disease management
- **Decision support** for farm operations
- **Risk communication** with stakeholders

## 🚀 **Deployment Options**

### **Local Development**
```bash
streamlit run app/app.py
```

### **Streamlit Cloud** (Recommended for demos)
1. Connect your GitHub repository
2. Deploy automatically from main branch
3. Get a public URL for sharing

### **Docker Deployment**
```bash
docker build -t penaeuspredict .
docker run -p 8501:8501 penaeuspredict
```

## 📚 **Documentation**

- **[Multi-Disease Design](MULTI_DISEASE_DESIGN.md)**: Technical architecture and design decisions
- **[Portfolio Enhancement](PORTFOLIO_ENHANCEMENT_SUMMARY.md)**: Project evolution and portfolio value
- **[Simulated Data Guide](README_SIMULATED_DATA.md)**: Data generation and validation

## 🔮 **Future Enhancements**

- **Geographic visualization** of disease risk
- **Time-series forecasting** for seasonal risk
- **Mobile app** for field use
- **Image recognition** for disease symptom identification
- **Additional diseases** (EMS/AHPND, YHV)
- **Real-time sensor integration**

## ⚠️ **Disclaimer**

This project uses simulated data for demonstration purposes. While the models are based on realistic disease-environment relationships, predictions should not be used for actual farm management decisions without validation against real-world data.

## 🤝 **Contributing**

This is a research project for PhD studies. For academic collaboration or research partnerships, please contact the author.

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ for sustainable aquaculture and disease prevention**
