import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="Used Car Price Estimator",
    page_icon="🚗",
    layout="centered"
)

# ===============================
# Load trained model
# ===============================
model = joblib.load("used_car_price_model.pkl")

# ===============================
# Header
# ===============================
st.title("🚗 Used Car Price Estimator")
st.write(
    """
    This application estimates the **market price of a used car**
    using a **Random Forest regression model** trained on real used-car listings.
    
    Adjust the car details in the sidebar and click **Estimate Price**.
    """
)

st.divider()

# ===============================
# Sidebar Inputs
# ===============================
st.sidebar.header("Car Details")

milage = st.sidebar.number_input(
    "Mileage (km)",
    min_value=0,
    max_value=500_000,
    step=1_000,
    value=50_000
)

engine_liters = st.sidebar.number_input(
    "Engine Size (Liters)",
    min_value=0.5,
    max_value=8.0,
    step=0.1,
    value=2.0
)

CURRENT_YEAR = 2026
model_year = st.sidebar.number_input(
    "Model Year",
    min_value=1990,
    max_value=CURRENT_YEAR,
    step=1,
    value=2018
)

car_age = CURRENT_YEAR - model_year

transmission = st.sidebar.selectbox(
    "Transmission",
    ["Automatic", "Manual"]
)
# ===============================
# Brand Selection (IMPORTANT)
# ===============================
brand_columns = sorted([
    col.replace("brand_", "")
    for col in model.feature_names_in_
    if col.startswith("brand_")
])

brand = st.sidebar.selectbox(
    "Car Brand",
    brand_columns
)

interior_color = st.sidebar.selectbox(
    "Interior Color",
    ["Black", "Gray"]
)

# ===============================
# Feature Engineering
# ===============================
mileage_per_year = milage / (car_age + 1)
# Interior colour one-hot encoding
int_col_black = 1 if interior_color == "Black" else 0
int_col_gray = 1 if interior_color == "Gray" else 0

# Base input row
input_data = pd.DataFrame({
    "milage": [milage],
    "engine_liters": [engine_liters],
    "car_age": [car_age],
    "model_year": [model_year],
    "mileage_per_year": [mileage_per_year],
    f"transmission_{transmission}": [1],
    f"brand_{brand}": [1],
    "int_col_Black": [int_col_black],
    "int_col_Gray": [int_col_gray]
})

# ===============================
# Align with training features
# ===============================
for col in model.feature_names_in_:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[model.feature_names_in_]

# ===============================
# Prediction Section
# ===============================
st.divider()
st.subheader("Estimated Price")

if st.button("Estimate Price", use_container_width=True):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Used Car Price: **${prediction:,.2f}**")

st.divider()
