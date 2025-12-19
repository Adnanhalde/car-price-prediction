import streamlit as st
import numpy as np
import joblib
import pandas as pd
import sys
import os

# App configuration
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #F0F9FF;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin: 1rem 0;
    }
    .feature-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #E2E8F0;
    }
    .stButton button {
        width: 100%;
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
    }
    .stButton button:hover {
        background-color: #2563EB;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🚗 Car Price Prediction Tool</h1>', unsafe_allow_html=True)
st.markdown("### Estimate Your Car's Market Value Instantly")

# Check if model files exist
model_files = ['car_price_model.pkl', 'car_scaler.pkl', 'feature_info.pkl']
missing_files = [f for f in model_files if not os.path.exists(f)]

if missing_files:
    st.error(f"⚠️ Model files not found: {', '.join(missing_files)}")
    st.info("""
    **To fix this:**
    
    1. Run the model training script first:
    ```bash
    python train_model.py
    ```
    
    2. Make sure all these files are in the same directory:
    - `car_price_model.pkl`
    - `car_scaler.pkl` 
    - `feature_info.pkl`
    
    3. Then refresh this page or restart the app.
    """)
    
    if st.button("📁 Check Files Again"):
        st.rerun()
    
    st.stop()

# Load the model, scaler, and feature info
try:
    model = joblib.load('car_price_model.pkl')
    scaler = joblib.load('car_scaler.pkl')
    feature_info = joblib.load('feature_info.pkl')
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model files: {str(e)}")
    st.stop()

# Sidebar for additional info
with st.sidebar:
    st.header("📊 About This Tool")
    st.write("""
    This AI-powered tool predicts car prices using a **Random Forest** machine learning model trained on thousands of car listings.
    
    **Factors considered:**
    - Manufacturing Year & Age
    - Mileage & Usage
    - Engine Specifications
    - Fuel Type & Transmission
    - Ownership History
    
    💡 *For best results, provide accurate information about your car.*
    """)
    
    st.markdown("---")
    st.subheader("💰 Price Format")
    st.write("""
    **Thousands Format:**
    - 500 = ₹5,00,000
    - 1,000 = ₹10,00,000
    
    **Lakhs Format:**
    - 5.00 L = ₹5,00,000
    - 10.00 L = ₹10,00,000
    """)
    
    st.markdown("---")
    st.caption("⚙️ Version 1.0 | Made with Streamlit")

# Create tabs for better organization
tab1, tab2 = st.tabs(["🏎️ Price Prediction", "📈 Market Insights"])

with tab1:
    # Create two columns for input layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("Car Specifications")
        
        # Manufacturing Year
        manufacturing_year = st.slider(
            "**Manufacturing Year**",
            min_value=2000,
            max_value=2024,
            value=2018,
            help="Select the year your car was manufactured"
        )
        
        # Car Age (calculated)
        car_age = 2024 - manufacturing_year
        st.metric("Car Age", f"{car_age} years")
        
        # Mileage
        mileage_km = st.number_input(
            "**Mileage (km)**",
            min_value=0.0,
            max_value=500000.0,
            value=50000.0,
            step=1000.0,
            help="Total distance traveled by the car"
        )
        
        # Mileage category
        if mileage_km < 30000:
            st.caption("✅ Low mileage - Good for resale")
        elif mileage_km < 100000:
            st.caption("⚠️ Average mileage")
        else:
            st.caption("📈 High mileage - Affects price")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("Technical Details")
        
        # Engine Size
        engine_size_cc = st.selectbox(
            "**Engine Size (cc)**",
            options=[800, 1000, 1200, 1500, 1800, 2000, 2500, 3000, 3500],
            index=3,
            help="Engine displacement in cubic centimeters"
        )
        
        # Horsepower
        horsepower = st.slider(
            "**Horsepower (HP)**",
            min_value=50,
            max_value=400,
            value=120,
            help="Engine power output"
        )
        
        # Fuel Type
        fuel_type = st.selectbox(
            "**Fuel Type**",
            options=['Petrol', 'Diesel', 'Hybrid', 'Electric'],
            help="Type of fuel used by the car"
        )
        
        # Transmission
        transmission = st.radio(
            "**Transmission**",
            options=['Manual', 'Automatic'],
            horizontal=True,
            help="Type of transmission"
        )
        
        # Owner Count
        owner_count = st.selectbox(
            "**Number of Previous Owners**",
            options=[1, 2, 3, 4, 5],
            index=0,
            help="Total number of owners the car has had"
        )
        
        # Owner impact
        if owner_count == 1:
            st.caption("✅ Single owner - Best for resale value")
        elif owner_count <= 2:
            st.caption("⚠️ 1-2 owners - Acceptable")
        else:
            st.caption("📉 Multiple owners - Reduces value")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction section
    st.markdown("---")
    
    # Convert categorical inputs to numerical
    fuel_mapping = {'Petrol': 0, 'Diesel': 1, 'Hybrid': 2, 'Electric': 3}
    transmission_mapping = {'Manual': 0, 'Automatic': 1}
    
    fuel_type_encoded = fuel_mapping[fuel_type]
    transmission_encoded = transmission_mapping[transmission]
    
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
    
    # Display features summary
    with st.expander("📋 Review Your Inputs"):
        st.write(f"- **Manufacturing Year:** {manufacturing_year}")
        st.write(f"- **Car Age:** {car_age} years")
        st.write(f"- **Mileage:** {mileage_km:,.0f} km")
        st.write(f"- **Engine Size:** {engine_size_cc} cc")
        st.write(f"- **Horsepower:** {horsepower} HP")
        st.write(f"- **Fuel Type:** {fuel_type}")
        st.write(f"- **Transmission:** {transmission}")
        st.write(f"- **Previous Owners:** {owner_count}")
    
    # Prediction button
    st.markdown("### Ready to Predict?")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button("🚀 **PREDICT CAR PRICE**", type="primary", use_container_width=True)
    
    if predict_button:
        try:
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Make prediction
            with st.spinner("🤖 Analyzing car specifications..."):
                prediction = model.predict(features_scaled)[0]
            
            # Display result
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.metric(
                    label="Estimated Price",
                    value=f"₹{prediction:,.0f} thousand",
                    delta=None
                )
                
                # Show in lakhs
                price_lakhs = prediction / 100
                st.write(f"**In Lakhs:** ₹{price_lakhs:.2f} L")
            
            with col_res2:
                # Price range estimation
                lower_bound = prediction * 0.85
                upper_bound = prediction * 1.15
                
                st.write("**Expected Range:**")
                st.write(f"₹{lower_bound:,.0f}k - ₹{upper_bound:,.0f}k")
                st.write(f"(₹{lower_bound/100:.1f}L - ₹{upper_bound/100:.1f}L)")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional insights
            st.markdown("---")
            st.subheader("📊 Price Analysis")
            
            insights_col1, insights_col2 = st.columns(2)
            
            with insights_col1:
                # Factors affecting price
                st.write("**Positive Factors (+):**")
                if car_age < 5:
                    st.write("✅ Recent model year")
                if mileage_km < 50000:
                    st.write("✅ Low mileage")
                if owner_count == 1:
                    st.write("✅ Single owner")
                if fuel_type in ['Hybrid', 'Electric']:
                    st.write("✅ Modern fuel type")
                if transmission == 'Automatic':
                    st.write("✅ Automatic transmission")
            
            with insights_col2:
                st.write("**Negative Factors (-):**")
                if car_age > 10:
                    st.write("❌ Older vehicle")
                if mileage_km > 150000:
                    st.write("❌ High mileage")
                if owner_count > 2:
                    st.write("❌ Multiple owners")
                if engine_size_cc < 1200:
                    st.write("⚠️ Smaller engine")
            
            # Export option
            st.markdown("---")
            st.write("💾 **Save this estimate:**")
            estimate_data = {
                'Manufacturing Year': manufacturing_year,
                'Mileage (km)': mileage_km,
                'Engine Size (cc)': engine_size_cc,
                'Horsepower': horsepower,
                'Fuel Type': fuel_type,
                'Transmission': transmission,
                'Previous Owners': owner_count,
                'Estimated Price (thousands)': prediction,
                'Estimated Price (lakhs)': price_lakhs
            }
            
            df_estimate = pd.DataFrame([estimate_data])
            st.download_button(
                label="📥 Download as CSV",
                data=df_estimate.to_csv(index=False),
                file_name=f"car_price_estimate_{manufacturing_year}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("Please check your inputs and try again.")

with tab2:
    st.header("📈 Car Market Insights")
    
    st.write("""
    ### How Car Features Affect Price
    
    1. **Manufacturing Year & Age**
    - Newer cars (0-3 years) depreciate slower
    - Cars older than 10 years see accelerated depreciation
    
    2. **Mileage Impact**
    - < 50,000 km: Minimal impact on price
    - 50,000 - 150,000 km: Moderate depreciation
    - > 150,000 km: Significant price reduction
    
    3. **Fuel Type Trends**
    - Electric & Hybrid: Higher resale value
    - Diesel: Good for long-distance drivers
    - Petrol: Most common, stable prices
    
    4. **Transmission Preference**
    - Automatic: Increasing demand, higher value
    - Manual: Traditional preference, lower maintenance
    
    5. **Ownership History**
    - Single owner: Highest value
    - Each additional owner reduces value by ~5-10%
    """)
    
    # Price trends chart (simulated)
    st.subheader("📊 Average Price by Car Age")
    
    # Simulate price depreciation data
    ages = list(range(0, 21, 2))
    base_price = 1000  # thousand
    depreciation = [base_price * (0.9 ** age) for age in ages]
    
    chart_data = pd.DataFrame({
        'Age (Years)': ages,
        'Price (thousands)': [round(p, 0) for p in depreciation]
    })
    
    st.line_chart(chart_data.set_index('Age (Years)'))
    
    st.caption("Note: Chart shows simulated depreciation trend. Actual values vary by model and condition.")

# Footer
st.markdown("---")
st.caption("""
⚠️ **Disclaimer:** This tool provides estimates based on machine learning models and market trends. 
Actual prices may vary based on car condition, brand, model, location, and market conditions. 
Always consult with professional appraisers for accurate valuations.
""")
