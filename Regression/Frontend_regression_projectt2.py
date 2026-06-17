import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('f_model.sav')
scaler = joblib.load('scaler.sav')

# List your feature columns (update as needed)
feature_columns = [
    'Month No', 'Mortgage Rate', 'Inflation', 'Housing Price Index', 'Treasury Yield',
    'Unemployment Rate', 'GDP', 'Business Confidence Index', 'Consumer Confidence Index',
    'Initial Unemployment Claim', 'Disposable Income'
]

st.title("Refinance Volume Prediction")

st.write("Please enter values for all input features:")

# Create input fields for each feature
user_input = []
for col in feature_columns:
    val = st.number_input(f"{col}", value=0.0, format="%.4f")
    user_input.append(val)

if st.button("Predict"):
    # Convert input to DataFrame
    input_df = pd.DataFrame([user_input], columns=feature_columns)
    # Standardize input
    input_scaled = scaler.transform(input_df)
    # Predict
    prediction = model.predict(input_scaled)[0]
    st.success(f"Predicted Refinance Volume: {prediction:.2f}")
