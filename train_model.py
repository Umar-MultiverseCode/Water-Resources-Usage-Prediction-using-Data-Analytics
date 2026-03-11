import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'clean_water_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'water_prediction_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'feature_scaler.pkl')
SAMPLE_DATA_PATH = os.path.join(BASE_DIR, 'data', 'sample_data.csv')

def main():
    print("====================================")
    print(" HydroMind AI - Model Training ")
    print("====================================")
    
    print("\n1. Loading dataset...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return
        
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} records.")
    
    print("\n2. Handling missing values...")
    # Convert month to numeric if present
    if 'month' in df.columns:
        df['month'] = pd.to_numeric(df['month'], errors='coerce')

    columns_to_keep = ['USO2013', 'TU', 'month', 'temperature', 'rainfall', 'population', 'water_consumption']
    actual_cols = [c for c in columns_to_keep if c in df.columns]
    
    # Required for training
    required_cols = ['temperature', 'rainfall', 'population', 'water_consumption']
    if not all(c in df.columns for c in required_cols):
        print(f"Error: Dataset is missing some of the explicitly required columns: {required_cols}")
        return
        
    df = df[actual_cols]
    
    # Ensure numerics
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    initial_len = len(df)
    df.dropna(subset=required_cols, inplace=True)
    print(f"Dropped {initial_len - len(df)} rows with missing required values.")
    
    # Downsample huge datasets to prevent memory crash during model training
    max_samples = 300000
    if len(df) > max_samples:
        print(f"Downsampling dataset from {len(df)} to {max_samples} to prevent MemoryError...")
        df = df.sample(n=max_samples, random_state=42)
    
    print("\n3. Exploratory Data Analysis Structure...")
    print("Dataset Summary:")
    print(df[required_cols].describe())
    
    print("\n4. Splitting dataset...")
    X = df[['temperature', 'rainfall', 'population']]
    y = df['water_consumption']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    print("\n5. Feature scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n6. Training RandomForestRegressor...")
    print("Parameters: n_estimators=200, max_depth=15, random_state=42")
    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    print("\n7. Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"-> Mean Absolute Error (MAE): {mae:.2f}")
    print(f"-> R² Score: {r2:.4f}")
    
    print("\n8. Saving trained model, scaler, and sample data for dashboard...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    # Save a sample of the data for faster loading in Streamlit
    sample_size = min(10000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    sample_df.to_csv(SAMPLE_DATA_PATH, index=False)
    
    print(f"Model saved at: {MODEL_PATH}")
    print(f"Scaler saved at: {SCALER_PATH}")
    print(f"Sample data saved at: {SAMPLE_DATA_PATH}")
    print("\nTraining completed successfully! You can now view the Streamlit dashboard.")

if __name__ == '__main__':
    main()
