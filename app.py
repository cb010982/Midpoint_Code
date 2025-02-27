import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained Random Forest model
rf_model = joblib.load("diabetes_random_forest_model.pkl") 

# Streamlit UI
st.title("🔍 Diabetes Prediction")
st.write("Enter patient details to predict the likelihood of diabetes.")

# Sidebar - User Inputs
st.sidebar.header("Input Features")
pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.sidebar.number_input("Glucose Level", min_value=0, max_value=200, value=120)
blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.sidebar.number_input("Insulin Level", min_value=0, max_value=900, value=79)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.2f")
dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, format="%.3f")
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)

# input data
user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

# Medical Rule: If glucose ≥ 97, classify as "at risk"
if st.button("Predict Diabetes"):
    if glucose >= 97:
        st.error(f"**High Risk Detected:** Glucose level is {glucose} mg/dL. Potential diabetes risk, get diet recommendation.")
        if st.button("Proceed to Sign In to Get Diet Recommendations"):
            st.success("Redirecting to sign-in page...")  # You can add actual redirection logic here
    else:    
        prediction = rf_model.predict(user_input)  # Use the model to predict
        pred_prob = rf_model.predict_proba(user_input)[:, 1]  # Probability of having diabetes
        if prediction[0] == 1:
            st.error(f"The model predicts **diabetes** with a probability of **{pred_prob[0]:.2f}**.")
            if st.button("Proceed to Sign In to Get Diet Recommendations"):
                st.success("Redirecting to sign-in page...")  # Add actual redirection logic
        else:
            st.success(f"The model predicts **no diabetes** with a probability of **{1 - pred_prob[0]:.2f}**.")
