import streamlit as st
import numpy as np
import joblib
import os

# Simple error checking first
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")

# Check for required files
required_files = ['car_price_model.pkl', 'car_scaler.pkl', 'feature_info.pkl']
missing_files = []

for file in required_files:
    if not os.path.exists(file):
        missing_files.append(file)

if missing_files:
    st.error(f"⚠️ Missing files: {', '.join(missing_files)}")
    
    st.info("""
    **To create the missing files:**
    
    1. **Make sure you have `train_model.py` in the same folder**
    2. **Open terminal/command prompt in this folder**
    3. **Run this command:**
    ```bash
    python train_model.py
    ```
    4. **Wait for it to finish (should take 10-30 seconds)**
    5. **Then refresh this page**
    """)
    
    # Show current directory contents
    st.write("📁 **Current folder contents:**")
    files_in_folder = os.listdir('.')
    for f in files_in_folder:
        if f.endswith('.py') or f.endswith('.pkl'):
            st.write(f"   - {f}")
    
    if st.button("🔄 Check Again"):
        st.rerun()
    
    st.stop()

# If all files exist, load them
try:
    model = joblib.load('car_price_model.pkl')
    scaler = joblib.load('car_scaler.pkl')
    feature_info = joblib.load('feature_info.pkl')
    
    # Get mappings
    fuel_mapping = feature_info['fuel_mapping']
    transmission_mapping = feature_info['transmission_mapping']
    
    # Reverse mappings for display
    fuel_types = {v: k for k, v in fuel_mapping.items()}
    transmission_types = {v: k for k, v in transmission_mapping.items()}
    
except Exception as e:
    st.error(f"Error loading files: {str(e)}")
    st.stop()

# Main App
st.title("🚗 Car Price Prediction")
st.write("Estimate your car's market value in seconds!")

# Input fields in columns
col1, col2 = st.columns(2)

with col1:
    manufacturing_year = st.slider(
        "Manufacturing Year",
        min_value=2000,
        max_value=2024,
        value=2018
    )
    
    mileage_km = st.number_input(
        "Mileage (km)",
        min_value=0.0,
        max_value=500000.0,
        value=50000.0,
        step=1000.0
    )
    
    engine_size_cc = st.selectbox(
        "Engine Size (cc)",
        options=[800, 1000, 1200, 1500, 1800, 2000, 2500, 3000, 3500]
    )
    
    horsepower = st.slider(
        "Horsepower",
        min_value=50,
        max_value=400,
        value=120
    )

with col2:
    # Fuel type dropdown
    fuel_options = list(fuel_mapping.keys())
    fuel_type = st.selectbox(
        "Fuel Type",
        options=fuel_options
    )
    
    # Transmission radio
    transmission_options = list(transmission_mapping.keys())
    transmission = st.radio(
        "Transmission",
        options=transmission_options,
        horizontal=True
    )
    
    owner_count = st.selectbox(
        "Number of Previous Owners",
        options=[1, 2, 3, 4, 5]
    )
    
    # Display car age
    car_age = 2024 - manufacturing_year
    st.info(f"Car Age: {car_age} years")

# Prepare features for prediction
fuel_type_encoded = fuel_mapping[fuel_type]
transmission_encoded = transmission_mapping[transmission]

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

# Predict button
if st.button("🔮 Predict Price", type="primary"):
    try:
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Display results
        st.success("### Prediction Result")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.metric(
                "Estimated Price",
                f"₹{prediction:,.0f} thousand",
                help="Price in thousands (e.g., 500 = ₹5,00,000)"
            )
        
        with col_b:
            price_lakhs = prediction / 100
            st.metric(
                "Price in Lakhs",
                f"₹{price_lakhs:.2f} L",
                help="Price in lakhs (e.g., 5.00 L = ₹5,00,000)"
            )
        
        # Show details
        with st.expander("📋 View Details"):
            st.write(f"- **Manufacturing Year:** {manufacturing_year}")
            st.write(f"- **Mileage:** {mileage_km:,.0f} km")
            st.write(f"- **Engine Size:** {engine_size_cc} cc")
            st.write(f"- **Horsepower:** {horsepower} HP")
            st.write(f"- **Fuel Type:** {fuel_type}")
            st.write(f"- **Transmission:** {transmission}")
            st.write(f"- **Owners:** {owner_count}")
            st.write(f"- **Car Age:** {car_age} years")
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Footer
st.markdown("---")
st.caption("Note: This is an estimation tool. Actual prices may vary.")
