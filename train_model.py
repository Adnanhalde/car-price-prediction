import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

print("🚗 Training car price prediction model...")

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic car data
n_samples = 2000
print(f"Generating {n_samples} synthetic car samples...")

# Generate features
manufacturing_year = np.random.randint(2000, 2024, n_samples)
mileage_km = np.random.uniform(1000, 300000, n_samples)
engine_size_cc = np.random.choice([800, 1000, 1200, 1500, 1800, 2000, 2500, 3000, 3500], n_samples)
horsepower = np.random.randint(50, 400, n_samples)
fuel_types = np.random.choice(['Petrol', 'Diesel', 'Hybrid', 'Electric'], n_samples, p=[0.4, 0.3, 0.2, 0.1])
transmission_types = np.random.choice(['Manual', 'Automatic'], n_samples, p=[0.4, 0.6])
owner_count = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.4, 0.3, 0.15, 0.1, 0.05])

# Create price based on features
age = 2024 - manufacturing_year
base_price = (
    2000 * (2024 - manufacturing_year) +  # Depreciation with age
    -0.03 * mileage_km +                 # Depreciation with mileage
    0.8 * engine_size_cc +               # Value from engine size
    15 * horsepower +                    # Value from horsepower
    np.random.normal(0, 50000, n_samples)  # Random noise
)

# Adjust for fuel type
fuel_multiplier = {
    'Petrol': 1.0,
    'Diesel': 1.1,
    'Hybrid': 1.3,
    'Electric': 1.5
}
base_price *= np.array([fuel_multiplier[ft] for ft in fuel_types])

# Adjust for transmission
transmission_bonus = np.where(transmission_types == 'Automatic', 50000, 0)

# Adjust for owner count
owner_discount = (owner_count - 1) * 20000

# Final price in thousands
price_thousands = (base_price + transmission_bonus - owner_discount) / 1000

# Ensure prices are positive and reasonable
price_thousands = np.maximum(price_thousands, 50)
price_thousands = np.minimum(price_thousands, 2000)

# Convert categorical to numerical for modeling
fuel_type_encoded = pd.factorize(fuel_types)[0]
transmission_encoded = pd.factorize(transmission_types)[0]

# Create feature matrix
X = np.column_stack([
    manufacturing_year,
    mileage_km,
    engine_size_cc,
    horsepower,
    fuel_type_encoded,
    transmission_encoded,
    owner_count
])

y = price_thousands

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_scaled, y)

# Evaluate model
score = model.score(X_scaled, y)
print(f"✅ Model trained successfully!")
print(f"📊 Model R² score: {score:.3f}")

# Save model and scaler
joblib.dump(model, 'car_price_model.pkl')
joblib.dump(scaler, 'car_scaler.pkl')

# Save feature names and mappings
feature_info = {
    'feature_names': ['manufacturing_year', 'mileage_km', 'engine_size_cc', 
                     'horsepower', 'fuel_type', 'transmission', 'owner_count'],
    'fuel_mapping': dict(enumerate(['Petrol', 'Diesel', 'Hybrid', 'Electric'])),
    'transmission_mapping': dict(enumerate(['Manual', 'Automatic']))
}
joblib.dump(feature_info, 'feature_info.pkl')

# Test prediction
sample_features = np.array([[2018, 50000, 1500, 120, 0, 1, 1]])  # 2018 Petrol Automatic, 1 owner
sample_scaled = scaler.transform(sample_features)
sample_prediction = model.predict(sample_scaled)[0]

print(f"📝 Sample prediction for 2018 car with 50,000 km:")
print(f"   Estimated Price: ₹{sample_prediction:,.2f} thousand")
print(f"   (Approximately ₹{sample_prediction/100:.2f} Lakhs)")

print("\n✅ All files saved:")
print("   - car_price_model.pkl (trained model)")
print("   - car_scaler.pkl (feature scaler)")
print("   - feature_info.pkl (feature mappings)")
print("\n🎯 Now you can run the app with: streamlit run app.py")
