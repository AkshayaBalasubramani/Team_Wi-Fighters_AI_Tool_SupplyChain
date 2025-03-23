import streamlit as st
import requests
from PIL import Image
import io

# FastAPI server URL
FASTAPI_URL = "http://127.0.0.1:8000"  # Change if hosted elsewhere

st.set_page_config(page_title="Logistics Forecasting & EDA Dashboard", layout="wide")

st.title("📊 Sales Forecasting & EDA")

# Tabs for navigation
tab1, tab2 = st.tabs(["🔮 Predictions", "📊 EDA Visualizations"])

# 🔮 Sales Forecasting
tab1.header("🔮 Prediction Models")
model = tab1.selectbox("Select a Model", ["arima", "gru", "prophet"], key="model")
if tab1.button("Generate Prediction Graph"):
    response = requests.get(f"{FASTAPI_URL}/plot?model_name={model}")
    if response.status_code == 200:
        img = Image.open(io.BytesIO(response.content))
        tab1.image(img, caption=f"{model.upper()} Predictions")
    else:
        tab1.error("Failed to fetch prediction graph. Check API status.")

# 📊 EDA Visualizations
tab2.header("📊 Exploratory Data Analysis")
graph_type = tab2.selectbox("Select EDA Graph", ["histogram", "correlation", "trend"], key="graph")
if tab2.button("Generate EDA Graph"):
    response = requests.get(f"{FASTAPI_URL}/eda?graph_type={graph_type}")
    if response.status_code == 200:
        img = Image.open(io.BytesIO(response.content))
        tab2.image(img, caption=f"{graph_type.capitalize()} Visualization")
    else:
        tab2.error("Failed to fetch EDA graph. Check API status.")

# 📌 Footer
st.markdown("---")
st.markdown("🚀 Built with **FastAPI** & **Streamlit** | ✨ AI-powered sales forecasting dashboard")
