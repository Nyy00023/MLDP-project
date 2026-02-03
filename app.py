import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ===============================
# Load trained model
# ===============================
model = joblib.load("used_car_price_model.pkl")

st.title("Used Car Price Prediction App 🚗")
st.write("Predict the estimated price of a used car using a trained Random Forest model.")

# ===============================
# User Inputs
# ===============================

milage = st.number_input("Mileage (km)", min_value=0, step=1000)
engine_liters = st.number_input("Engine Size (Liters)", min_value=0.5, step=0.1)
car_age = st.number_input("Car Age (years)", min_value=0, step=1)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Gasoline", "Diesel", "Hybrid", "Electric"]
)

transmission = st.selectbox(
    "Transmission",
    ["Automatic", "Manual"]
)

has_accident = st.selectbox(
    "Accident History",
    ["No", "Yes"]
)

# ===============================
# Feature Engineering (must match training)
# ===============================

mileage_per_year = milage / (car_age + 1)
has_accident_flag = 1 if has_accident == "Yes" else 0

# Create input dataframe
input_data = pd.DataFrame({
    "milage": [milage],
    "engine_liters": [engine_liters],
    "car_age": [car_age],
    "mileage_per_year": [mileage_per_year],
    "has_accident": [has_accident_flag],
    "fuel_type_" + fuel_type: [1],
    "transmission_" + transmission: [1]
})

# Ensure missing dummy columns are filled with 0
model_features = model.feature_names_in_
for col in model_features:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[model_features]

# ===============================
# Prediction
# ===============================

if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Used Car Price: ${prediction:,.2f}")