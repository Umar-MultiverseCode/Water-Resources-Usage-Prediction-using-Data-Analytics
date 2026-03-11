# HydroMind AI – Water Resource Usage Prediction using Data Analytics

## Overview
HydroMind AI is a comprehensive data analytics and machine learning project aimed at predicting water consumption based on key environmental and demographic factors. By modeling consumption behavior, the project seeks to support smart city initiatives, optimize water distribution, and aid drought or peak demand preparedness.

## Features
- **Exploratory Data Analysis (EDA):** Understand historical trends of water usage, correlations with weather conditions, and seasonal patterns.
- **Machine Learning Model:** Utilizes a standard `RandomForestRegressor` to accurately predict water demand based on real-world inputs: Temperature, Rainfall, and Population.
- **Modern Dashboard:** Built with Streamlit, the user-friendly interface allows users to view data visualizations and perform interactive predictions.
- **Explainable AI:** Highlights the feature importance to indicate what components drive water utility shifts the most.
- **Scenario Simulation:** Test extreme scenarios such as heatwaves, droughts, or population booms.

## Project Structure
- `data/` - Contains the dataset (e.g., `clean_water_data.csv`)
- `scripts/` - Pipeline scripts like model training (`train_model.py`)
- `dashboard/` - Streamlit application files for UI (`app.py`)

## Setup Instructions

1. **Install Requirements:**
```bash
pip install -r requirements.txt
```

2. **Train the Model:**
Run the training script to generate the regression model and sample dataset:
```bash
python scripts/train_model.py
```

3. **Run the Dashboard:**
Launch the Streamlit interactive dashboard:
```bash
streamlit run dashboard/app.py
```
