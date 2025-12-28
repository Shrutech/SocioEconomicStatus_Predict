import streamlit as st
import pandas as pd
import pickle
import os

BASE_DIR = os.path.dirname(__file__)

st.title("Socio-Economic Status Prediction")
st.write("Enter normalized feature values (0 to 1):")

# User inputs
EduScore = st.number_input("Education Score (Normalized)", 0.0, 1.0, 0.60)
IncomeScore = st.number_input("Income Score (Normalized)", 0.0, 1.0, 0.50)
Deprivation = st.number_input("Total Deprivation (Normalized)", 0.0, 1.0, 0.32)
TechAccess = st.number_input("Technology Access (Normalized)", 0.0, 1.0, 0.55)
SCST = st.number_input("SC/ST Population (Normalized)", 0.0, 1.0, 0.30)

# Load model and encoder using BASE_DIR
model_path = os.path.join(BASE_DIR, "stack_model.pkl")
le_path = os.path.join(BASE_DIR, "label_encoder.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(le_path, "rb") as f:
    le = pickle.load(f)

if st.button("Predict SES"):

    # Prepare input dataframe
    new_district = pd.DataFrame({
        'EduScore_Normalized': [EduScore],
        'IncomeScore_Normalized': [IncomeScore],
        'Total_Deprivation_Norm': [Deprivation],
        'TechAccess_Normalized': [TechAccess],
        'SCST_Normalized': [SCST]
    })

    # Add interaction features
    new_district['EduIncome_Interaction'] = new_district['EduScore_Normalized'] * new_district['IncomeScore_Normalized']
    new_district['TechEdu_Interaction'] = new_district['TechAccess_Normalized'] * new_district['EduScore_Normalized']
    new_district['SCST_Income_Interaction'] = new_district['SCST_Normalized'] * new_district['IncomeScore_Normalized']

    # Make prediction
    prediction = model.predict(new_district)
    predicted_label = le.inverse_transform(prediction)[0]

    st.success(f"Predicted Socio-Economic Status: {predicted_label}")
