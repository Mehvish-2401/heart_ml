# Heart Disease Risk Prediction

A machine learning web application that predicts the risk of heart disease in patients using the Cleveland Heart Disease dataset.

## Overview
This project uses Logistic Regression to classify whether a patient is at risk of heart disease based on clinical features. The model achieves 88% prediction accuracy with a ROC-AUC score of 0.95. The application is deployed as an interactive web app using Streamlit.

## Features
- Predicts heart disease risk based on patient data
- Real-time individual patient prediction
- Batch inference support
- Interactive and user-friendly Streamlit interface

## Dataset
- Source: Cleveland Heart Disease Dataset
- Features include age, sex, chest pain type, blood pressure, cholesterol, and more
- Binary classification: presence or absence of heart disease

## Tech Stack
- Python
- Scikit-learn (Logistic Regression, model evaluation)
- Pandas and NumPy (data preprocessing)
- Matplotlib and Seaborn (data visualization)
- Streamlit (web deployment)

## Model Performance
- Accuracy: 88%
- ROC-AUC: 0.95
- Evaluation metrics: Confusion Matrix, Classification Report

## Project Structure
- app.py — Streamlit web application
- heart.py — model training and preprocessing
- model.pkl — saved trained model
- test.csv — sample test data

## How to Run
1. Install dependencies: pip install -r requirements.txt
2. Run the app: streamlit run app.py

## Author
Mehvish Nazneen
B.E. Artificial Intelligence and Data Science
