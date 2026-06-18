import pathlib
import streamlit as st
import pandas as pd
import joblib

app_dir = pathlib.Path(__file__).resolve().parent

# Model is stored in repository root
model_path = app_dir.parent / "fmodel.sav"

scaler_paths = [
    app_dir.parent / "scaler.sav",
    app_dir / "scaler.sav",
]
# Load model and optional scaler
model = joblib.load(model_path)
scaler = None
for scaler_path in scaler_paths:
	if scaler_path.exists():
		scaler = joblib.load(scaler_path)
		break

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
	# Standardize input if scaler available
	if scaler is not None:
		input_for_model = scaler.transform(input_df)
	else:
		input_for_model = input_df
	# Predict
	prediction = model.predict(input_for_model)[0]
	if scaler is None:
		st.warning("Note: scaler.sav not found — inputs are used without scaling.")
	st.success(f"Predicted Refinance Volume: {prediction:.2f}")

