import streamlit as st
import requests
from PIL import Image
import io

# FastAPI server URL
FASTAPI_URL = "http://127.0.0.1:8000"  # Update if running on a different host/port

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

st.title("📈 Supply Forecasting & EDA")

# Tabs for navigation
tab1, tab2 = st.tabs(["🔮 Predictions", "📊 EDA Visualizations"])

# 🔮 Prediction Graphs
with tab1:
    st.header("🔮 Prediction Models")

    model = st.selectbox("Select a Model", ["arima", "gru", "prophet"])
    if st.button("Generate Prediction Graph"):
        response = requests.get(f"{FASTAPI_URL}/plot?model_name={model}")
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
            st.image(img, caption=f"{model.upper()} Predictions")
        else:
            st.error("Failed to fetch the prediction graph. Check API status.")

# 📊 EDA Visualizations
with tab2:
    st.header("📊 Exploratory Data Analysis")

    graph_type = st.selectbox("Select EDA Graph", ["histogram", "correlation", "trend"])
    if st.button("Generate EDA Graph"):
        response = requests.get(f"{FASTAPI_URL}/eda?graph_type={graph_type}")
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
            st.image(img, caption=f"{graph_type.capitalize()} Visualization")
        else:
            st.error("Failed to fetch EDA graph. Check API status.")

# 📌 Footer
st.markdown("---")
st.markdown("🚀 Built with **FastAPI** & **Streamlit** | ✨ Developed for AI-powered sales forecasting.")
