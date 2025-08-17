# Breast Cancer Prediction System

A machine learning-based web application that predicts breast cancer malignancy (benign/malignant) with up to 97% accuracy. The system supports two prediction modes for flexibility and efficiency.

## Project Description

This project provides a user-friendly platform for quick and comprehensive breast cancer diagnosis using machine learning models. It is designed for medical professionals, researchers, and students seeking accurate predictive insights.

Prediction Modes:

* Quick Prediction (10 Features): Uses the most significant features for faster diagnosis
* Comprehensive Prediction (30 Features): Uses all available features for maximum accuracy

Technology Stack:

* Python + Scikit-learn (Machine Learning)
* Streamlit (Web Interface)
* Joblib (Model Serialization)

## Key Features

* Dual Prediction Modes: Switch between 10-feature quick prediction or full 30-feature comprehensive prediction
* High Accuracy: 97% test accuracy on the Wisconsin Breast Cancer Dataset
* Interactive Interface: Sliders for all diagnostic parameters
* Instant Results: Clear benign/malignant classification with visual cues

## Repository Structure

```
breast-cancer-prediction/
├── app.py                        # Streamlit web application
├── models/                       # Pre-trained models
│   ├── logistic_regression_selected.pkl
│   ├── scaler.pkl
│   └── selector.pkl
├── training_notebook.ipynb       # Model training notebook
└── README.md                     # Project overview
```

## Installation & Setup

Clone the repository:

```bash
git clone https://github.com/yourusername/breast-cancer-prediction.git
cd breast-cancer-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the web application:

```bash
streamlit run app.py
```

## Technical Highlights

* Feature Selection: ANOVA F-test to identify the top 10 significant features
* Model Comparison:

  * Logistic Regression: 97% accuracy
  * Random Forest: 96% accuracy
* Preprocessing: StandardScaler for input normalization

## Potential Applications

* Preliminary diagnostic support in clinical settings
* Educational tool for medical students
* Research benchmarking for predictive models

## Contact Me

Connect with me on LinkedIn: [Your LinkedIn Profile](https://www.linkedin.com/in/muhammad-hassaan-munir-79b5b2327/)

