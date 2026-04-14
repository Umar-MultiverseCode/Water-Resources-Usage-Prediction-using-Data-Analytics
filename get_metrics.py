import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'clean_water_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'water_prediction_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'feature_scaler.pkl')

print("Loading data and model...")
df = pd.read_csv(DATA_PATH)

required_cols = ['temperature', 'rainfall', 'population', 'water_consumption']
for col in required_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(subset=required_cols, inplace=True)

max_samples = 300000
if len(df) > max_samples:
    df = df.sample(n=max_samples, random_state=42)

X = df[['temperature', 'rainfall', 'population']]
y = df['water_consumption']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)

X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("\n" + "="*45)
print("   TABLE 1: Model Performance Metrics")
print("="*45)
print(f"  Metric                              Value")
print(f"  Mean Absolute Error (MAE)         : {mae:.2f}")
print(f"  Root Mean Square Error (RMSE)     : {rmse:.2f}")
print(f"  Coefficient of Determination (R²) : {r2:.2f}")
print("="*45)
