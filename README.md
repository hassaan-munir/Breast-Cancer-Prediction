Breast Cancer Prediction System
📌 Project Description
A machine learning-based web application that predicts breast cancer malignancy (benign/malignant) with 97% accuracy. The system offers two modes:

Quick Prediction (10 Key Features): Uses only the most important features for faster diagnosis

Comprehensive Prediction (30 Features): Uses all available features for maximum accuracy

Built with:

Python + Scikit-learn (ML models)

Streamlit (Web interface)

Joblib (Model serialization)

🚀 Key Features
Dual Prediction Modes: Toggle between 10 or 30 input features

Medical-Grade Accuracy: 97% test accuracy on Wisconsin Breast Cancer Dataset

User-Friendly Interface: Interactive sliders for all diagnostic parameters

Instant Results: Clear benign/malignant classification with visual indicators

📂 Repository Structure
text
breast-cancer-prediction/
├── app.py                # Streamlit web application
├── models/               # Trained models
│   ├── logistic_regression_selected.pkl
│   ├── scaler.pkl
│   └── selector.pkl
├── training_notebook.ipynb  # Model training notebook
└── README.md             # This file
🛠️ Installation
Clone repository:

bash
git clone https://github.com/yourusername/breast-cancer-prediction.git
cd breast-cancer-prediction
Install requirements:

bash
pip install -r requirements.txt
Run the app:

bash
streamlit run app.py
📊 Technical Highlights
Feature Selection: ANOVA F-test for identifying top 10 most significant features

Model Comparison:

Logistic Regression: 97% accuracy

Random Forest: 96% accuracy

Preprocessing: StandardScaler for normalized inputs

🌟 Potential Applications
Preliminary diagnosis tool for clinics

Medical education resource

Research benchmarking tool
