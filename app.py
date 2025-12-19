import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the model, scaler, and feature info
model = joblib.load('car_price_model.pkl')
scaler = joblib.load('car_scaler.pkl')
feature_info = joblib.load('feature_info.pkl')

# App title
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")
st.title("🚗 Car Price Prediction")
st.markdown("### Estimate the market value of your car")
st.write("Fill in the details below and get an instant price estimate!")

# Sidebar for additional info
with st.sidebar:
    st.header("ℹ️ About This Tool")
    st.write("""
    This prediction tool uses machine learning to estimate car prices based on:
    - Manufacturing Year
    - Mileage
    - Engine Specifications
    - Fuel Type
    - Transmission
    - Ownership History
    
    The model is trained on synthetic car market data.
    """)
    st.markdown("---")
    st.caption("💰 Prices are estimated in thousands (e.g., 500 = ₹5,00,000)")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Car Specifications")
    
    # Manufacturing Year
    manufacturing_year = st.slider(
        "Manufacturing Year",
        min_value=2000,
        max_value=2024,
        value=2018,
        help="Year when the car was manufactured"
    )
    
    # Mileage
    mileage_km = st.number_input(
        "Mileage (km)",
        min_value=0.0,
        max_value=500000.0,
        value=50000.0,
        step=1000.0,
        help="Total distance the car has traveled"
    )
    
    # Engine Size
    engine_size_cc = st.selectbox(
        "Engine Size (cc)",
        options=[800, 1000, 1200, 1500, 1800, 2000, 2500, 3000, 3500],
        index=3,
        help="Engine displacement in cubic centimeters"
    )
    
    # Horsepower
    horsepower = st.slider(
        "Horsepower (HP)",
        min_value=50,
        max_value=400,
        value=120,
        help="Power output of the engine"
    )

with col2:
    st.subheader("Additional Details")
    
    # Fuel Type
    fuel_type = st.selectbox(
        "Fuel Type",
        options=list(feature_info['fuel_mapping'].values()),
        help="Type of fuel the car uses"
    )
    
    # Transmission
    transmission = st.selectbox(
        "Transmission",
        options=list(feature_info['transmission_mapping'].values()),
        index=1,
        help="Type of transmission"
    )
    
    # Owner Count
    owner_count = st.selectbox(
        "Number of Previous Owners",
        options=[1, 2, 3, 4, 5],
        help="How many owners has the car had?"
    )
    
    # Car Age (calculated)
    car_age = 2024 - manufacturing_year
    st.info(f"📅 Car Age: {car_age} years")

# Convert categorical inputs to numerical
fuel_type_encoded = list(feature_info['fuel_mapping'].values()).index(fuel_type)
transmission_encoded = list(feature_info['transmission_mapping'].values()).index(transmission)

# Prepare feature array
features = np.array([[
    manufacturing_year,
    mileage_km,
    engine_size_cc,
    horsepower,
    fuel_type_encoded,
    transmission_encoded,
    owner_count
]])

# Scale features
features_scaled = scaler.transform(features)

# Prediction button
st.markdown("---")
col_button1, col_button2, col_button3 = st.columns([1, 2, 1])
with col_button2:
    predict_button = st.button("🚀 Predict Price", type="primary", use_container_width=True)

if predict_button:
    # Make prediction
    with st.spinner("Analyzing car details..."):
        prediction = model.predict(features_scaled)[0]
    
    # Display result
    st.success(f"### Estimated Price: ₹{prediction:,.2f} thousand")
    
    # Convert to lakhs for Indian context
    price_lakhs = prediction / 100  # Convert thousands to lakhs
    st.metric(
        label="Price in Lakhs", 
        value=f"₹{price_lakhs:.2f} L",
        delta=None
    )
    
    # Additional insights
    st.markdown("---")
    st.subheader("📊 Insights & Factors")
    
    # Create insights based on inputs
    insights = []
    
    if car_age < 5:
        insights.append("✅ Car is relatively new (good for price)")
    elif car_age < 10:
        insights.append("⚠️ Moderate age affecting price")
    else:
        insights.append("❌ Older car may have lower value")
    
    if mileage_km < 50000:
        insights.append("✅ Low mileage (positive impact)")
    elif mileage_km < 150000:
        insights.append("⚠️ Moderate mileage")
    else:
        insights.append("❌ High mileage affecting value")
    
    if owner_count == 1:
        insights.append("✅ Single owner (preferred by buyers)")
    elif owner_count <= 2:
        insights.append("⚠️ Multiple owners may affect price")
    else:
        insights.append("❌ Many previous owners reduces value")
    
    # Display insights
    for insight in insights:
        st.write(insight)

# Footer
st.markdown("---")
st.caption("Note: This is an estimation tool. Actual market prices may vary based on additional factors like brand, model, condition, and location.")