import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("car_price_model.pkl")

# Page settings
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Car Price Prediction")
st.write("Enter the car details below to predict its selling price.")

# User inputs
year = st.number_input(
    "Manufacturing Year",
    min_value=2000,
    max_value=2026,
    value=2018
)

present_price = st.number_input(
    "Present Price (Lakhs)",
    min_value=0.0,
    value=8.5,
    step=0.1
)

driven_kms = st.number_input(
    "Driven Kilometers",
    min_value=0,
    value=30000,
    step=1000
)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "CNG"]
)

selling_type = st.selectbox(
    "Selling Type",
    ["Dealer", "Individual"]
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

owner = st.selectbox(
    "Number of Previous Owners",
    [0, 1, 2, 3]
)

# Calculate car age
car_age = 2026 - year

# Prediction button
if st.button("Predict Car Price"):
    
    input_data = pd.DataFrame({
        "Year": [year],
        "Present_Price": [present_price],
        "Driven_kms": [driven_kms],
        "Fuel_Type": [fuel_type],
        "Selling_type": [selling_type],
        "Transmission": [transmission],
        "Owner": [owner],
        "Car_Age": [car_age]
    })

    prediction = model.predict(input_data)

    st.success(
        f"Estimated Selling Price: ₹{prediction[0]:.2f} Lakhs"
    )