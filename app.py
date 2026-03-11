import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# Configuration for page layout
st.set_page_config(page_title="HydroMind AI", page_icon="💧", layout="wide", initial_sidebar_state="expanded")

# Inject Custom CSS for modern looks
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'water_prediction_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'feature_scaler.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'sample_data.csv')

# Load Model
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

# Load Scaler
@st.cache_resource
def load_scaler():
    if os.path.exists(SCALER_PATH):
        return joblib.load(SCALER_PATH)
    return None

# Load Sample Data
@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()

model = load_model()
scaler = load_scaler()
df = load_data()

# Navigation Sidebar
st.sidebar.title("💧 HydroMind AI")
st.sidebar.markdown("**Water Resource Usage Prediction**")
st.sidebar.divider()
menu = ["Overview", "Data Visualization", "Prediction Tool", "Explainable AI"]
choice = st.sidebar.radio("Navigation", menu)
st.sidebar.divider()
st.sidebar.info("Developed for predictive analytics and smart city water management.")

# --- ROUTING ---
if choice == "Overview":
    st.title("Welcome to HydroMind AI")
    st.markdown("""
    HydroMind AI is a predictive analytics system designed to optimize water resource planning.
    
    By analyzing historical data encompassing temperature, rainfall, and population demographics, the platform aims to accurately deduce water consumption trends. 
    Through this intelligence, city planners and municipal services can preemptively address water demand and execute conservation strategies.
    
    ### Project Goals:
    - 🔍 **Understand Patterns**: Explore water usage vis-a-vis various environmental components.
    - 🤖 **Predictive Regressions**: A trained `RandomForestRegressor` provides accurate estimates based on scaled inputs.
    - 💡 **Explainability**: View feature importance to demystify real-world impact.
    """)
    
    st.divider()
    if not df.empty:
        st.subheader("Dataset Summary (Sample)")
        st.dataframe(df.describe().T, use_container_width=True)
        st.write("Recent Data Snippet:")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.warning("Sample dataset not found. Please train the model and generate the sample data first.")

elif choice == "Data Visualization":
    st.title("Data Visualization & EDA")
    if not df.empty:
        st.markdown("Insights into the key driving correlations to Water Consumption.")
        st.divider()
        
        # Month or TU Trend
        trend_col = 'month' if 'month' in df.columns else 'TU'
        if trend_col in df.columns:
            st.subheader(f"Water Consumption Trend ({trend_col.capitalize()})")
            trend_avg = df.groupby(trend_col)['water_consumption'].mean().reset_index()
            fig_trend = px.line(trend_avg, x=trend_col, y='water_consumption', title="Water Consumption Trend", markers=True)
            fig_trend.update_layout(plot_bgcolor="rgba(0,0,0,0)", line=dict(color='#8b5cf6', width=3))
            st.plotly_chart(fig_trend, use_container_width=True)
            
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Temperature vs Consumption")
            fig_temp = px.scatter(df, x="temperature", y="water_consumption", title="Temperature vs Consumption", opacity=0.6, color_discrete_sequence=['#ef4444'], trendline="ols")
            fig_temp.update_layout(plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_temp, use_container_width=True)
            
            st.subheader("Population vs Consumption")
            fig_pop = px.scatter(df, x="population", y="water_consumption", title="Population vs Consumption", opacity=0.6, color_discrete_sequence=['#f59e0b'], trendline="ols")
            fig_pop.update_layout(plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pop, use_container_width=True)
            
        with col2:
            st.subheader("Rainfall vs Consumption")
            fig_rain = px.scatter(df, x="rainfall", y="water_consumption", title="Rainfall vs Consumption", opacity=0.6, color_discrete_sequence=['#10b981'], trendline="ols")
            fig_rain.update_layout(plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_rain, use_container_width=True)

    else:
        st.warning("Data not found. Execute the training script first.")

elif choice == "Prediction Tool":
    st.title("Interactive Demand Prediction")
    st.markdown("Input environmental variables to instantly forecast expected water consumption.")
    
    if model is not None and scaler is not None:
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            # Enforce validation bounds on inputs
            temp = st.slider("Temperature (°C)", min_value=10.0, max_value=45.0, value=25.0, step=0.1)
            rainfall = st.slider("Rainfall (mm)", min_value=0.0, max_value=200.0, value=50.0, step=0.1)
        
        with col2:
            population = st.number_input("Population Demographics", min_value=10000, max_value=50000, value=25000, step=1000)
            
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Predict Consumption", use_container_width=True):
            # Form DataFrame in correct feature order: temperature, rainfall, population
            feature_cols = ['temperature', 'rainfall', 'population']
            input_df = pd.DataFrame([[temp, rainfall, population]], columns=feature_cols)
            
            # Predict
            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)[0]
            
            st.success(f"### Predicted Water Consumption: {prediction:,.2f} units")
            st.balloons()
    else:
        st.error("Model engine or feature scaler missing. Please train the dataset first to activate the engine.")

elif choice == "Explainable AI":
    st.title("Explainable AI (Feature Importance)")
    st.markdown("Understanding what shifts water demand using algorithmic explainability.")
    
    if model is not None:
        st.divider()
        feature_names = ['Temperature', 'Rainfall', 'Population']
        importances = model.feature_importances_
        
        max_idx = np.argmax(importances)
        st.info(f"💡 According to the Random Forest model, **{feature_names[max_idx]}** represents the highest contributor impacting water usage predictions.")
        
        color_scale = [importances[i] for i in range(len(importances))]
        fig_feat = px.bar(
            x=importances, 
            y=feature_names, 
            orientation='h',
            title="Feature Importance Distribution",
            labels={'x': 'Relative Importance', 'y': 'Feature Input'},
            color=color_scale,
            color_continuous_scale="Mint"
        )
        fig_feat.update_layout(plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_feat, use_container_width=True)
    else:
        st.error("No model found. Retrain network components.")
