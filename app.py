import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Load dataset for min/max values
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Load models
try:
    lr_model = joblib.load('logistic_regression_selected.pkl')
    scaler = joblib.load('scaler.pkl')
    selector = joblib.load('selector.pkl')
except FileNotFoundError:
    st.error("Model files not found! Please train models first.")
    st.stop()

# App UI
st.title("Breast Cancer Predictor 🎗️")

# Feature selection
option = st.radio("Feature Mode:", ["Selected Features (10)", "All Features (30)"])

# Input sliders
st.header("Enter Patient Values")
input_features = []

if option == "Selected Features (10)":
    # Selected 10 features (replace with your actual selected features)
    selected_feature_names = [
        'mean radius', 'mean perimeter', 'mean area', 'mean concavity',
        'mean concave points', 'worst radius', 'worst perimeter', 
        'worst area', 'worst concavity', 'worst concave points'
    ]
    
    for feature in selected_feature_names:
        value = st.slider(
            f"{feature}",
            float(df[feature].min()),
            float(df[feature].max()),
            float(df[feature].mean())
        )
        input_features.append(value)
else:
    # All 30 features
    for feature in data.feature_names:
        value = st.slider(
            f"{feature}",
            float(df[feature].min()),
            float(df[feature].max()),
            float(df[feature].mean())
        )
        input_features.append(value)

# Prediction
if st.button("Predict"):
    input_array = np.array([input_features])
    
    # Scale the input
    scaled_data = scaler.transform(input_array)
    
    if option == "Selected Features (10)":
        # Select features if in selected mode
        selected_data = selector.transform(scaled_data)
        prediction = lr_model.predict(selected_data)[0]
    else:
        # Use all features
        prediction = lr_model.predict(scaled_data)[0]
    
    # Show result
    if prediction == 1:
        st.success("✅ Result: Benign (No Cancer)")
    else:
        st.error("⚠️ Result: Malignant (Cancer) - Consult Doctor Immediately!")