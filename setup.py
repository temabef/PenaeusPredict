#!/usr/bin/env python3
"""
Setup script for the PenaeusPredict project.
"""

from setuptools import setup, find_packages

setup(
    name="penaeus-predict",
    version="0.1.0",
    description="AI-Based Prediction Tool for WSSV Outbreaks in Southeast Asian Shrimp Aquaculture",
    author="PenaeusPredict Team",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "shap>=0.40.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "streamlit>=1.0.0",
        "jupyter>=1.0.0",
        "requests>=2.25.0",
    ],
) 