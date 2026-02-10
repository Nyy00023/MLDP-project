import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Used Car Price Estimator",
    page_icon="🚙",
    layout="centered"
)

model = joblib.load("used_car_price_model.pkl")

st.markdown(
    """
    <h1 style="text-align:center;">🚙 Used Car Price Estimator</h1>
    <p style="text-align:center; color:gray;">
        Get an estimated market price using a machine-learning model trained on real used-car data
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

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
    "Transmission Type",
    ["Automatic", "Manual"]
)

brand_columns = sorted(
    col.replace("brand_", "")
    for col in model.feature_names_in_
    if col.startswith("brand_")
)

brand = st.sidebar.selectbox(
    "Car Brand",
    brand_columns
)

interior_color = st.sidebar.selectbox(
    "Interior Colour",
    ["Black", "Gray"]
)

st.subheader("Vehicle Summary")

c1, c2, c3 = st.columns(3)
c1.metric("Mileage", f"{milage:,} km")
c2.metric("Engine", f"{engine_liters:.1f} L")
c3.metric("Car Age", f"{car_age} years")

if car_age == 0 and milage > 20_000:
    st.warning("⚠️ Very high mileage for a new car. Please double-check.")

mileage_per_year = milage / (car_age + 1)

int_col_black = 1 if interior_color == "Black" else 0
int_col_gray = 1 if interior_color == "Gray" else 0

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

for col in model.feature_names_in_:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[model.feature_names_in_]

show_details = st.checkbox(
    "Show calculated fields & model inputs",
    help="View derived features and exact values sent to the model"
)

if show_details:
    st.subheader("Input & Calculated Fields")

    calc_df = pd.DataFrame({
        "Field": [
            "Mileage (km)",
            "Engine Size (Liters)",
            "Car Age (years)",
            "Mileage per Year (km/year)",
            "Transmission",
            "Car Brand",
            "Interior Colour",
        ],
        "Value": [
            f"{milage:,}",
            f"{engine_liters:.1f}",
            car_age,
            f"{mileage_per_year:,.0f}",
            transmission,
            brand,
            interior_color,
        ]
    })

    st.table(calc_df)

    with st.expander("Model Input Vector (Advanced)"):
        st.write(
            "This table shows the exact feature values passed to the machine learning model "
            "after feature engineering and encoding."
        )
        st.dataframe(input_data)

st.divider()
st.subheader("Estimated Market Price")

if input_data.isnull().any().any():
    st.error("Invalid input detected. Please check your selections.")
    st.stop()

if st.button("Estimate Price", use_container_width=True):
    prediction = model.predict(input_data)[0]

    st.markdown(
        f"""
        <div style="
            background-color:#f0f9f4;
            padding:25px;
            border-radius:12px;
            text-align:center;
            border:1px solid #cceee0;
        ">
            <h2 style="color:#1b7f5f;">${prediction:,.0f}</h2>
            <p style="color:gray;">
                Estimated selling price based on similar vehicles
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()