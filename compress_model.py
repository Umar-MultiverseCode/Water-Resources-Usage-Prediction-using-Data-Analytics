import joblib
import os

print("Loading model...")
model = joblib.load('models/water_prediction_model.pkl')
print("Model loaded. Compressing...")
joblib.dump(model, 'models/water_prediction_model_compressed.pkl', compress=3)
print("Compression complete.")

size = os.path.getsize('models/water_prediction_model_compressed.pkl') / (1024 * 1024)
print(f"New size: {size:.2f} MB")
