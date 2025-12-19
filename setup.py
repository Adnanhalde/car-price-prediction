import subprocess
import sys
import os

print("🔧 Car Price Prediction Setup")
print("=" * 50)

# Check if required packages are installed
required_packages = ['streamlit', 'scikit-learn', 'numpy', 'pandas', 'joblib']

print("Checking dependencies...")
for package in required_packages:
    try:
        __import__(package.replace('-', '_'))
        print(f"✓ {package}")
    except ImportError:
        print(f"✗ {package} - installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

print("\n📊 Creating model files...")
# Run the training script
if os.path.exists('train_model.py'):
    subprocess.check_call([sys.executable, "train_model.py"])
else:
    print("Error: train_model.py not found!")
    print("Please make sure all files are in the same directory.")
    sys.exit(1)

print("\n✅ Setup complete!")
print("\n🎯 To run the app, use:")
print("   streamlit run app.py")
print("\nOr click the button below if you're in an IDE that supports it.")
