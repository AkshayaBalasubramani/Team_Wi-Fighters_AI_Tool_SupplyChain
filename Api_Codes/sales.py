from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
import pandas as pd
import numpy as np
import io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
import uvicorn
import os

app = FastAPI()

# ✅ Load dataset internally
df = pd.read_csv(r"C:\Users\Akshaya\OneDrive\Desktop\Xcelerate\chat_gpt\sales_data.csv")  # Replace with actual dataset path
df['sales_date'] = pd.to_datetime(df['sales_date'])
df = df.sort_values(by='sales_date')

# ✅ Cache for predictions
results_cache = {}

# ✅ Function to create dataset for GRU
def create_dataset(data, look_back=3):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

# ✅ GRU Model Class
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  
        return out

# ✅ Run Predictions and Cache Results
def generate_predictions():
    # ARIMA Model
    model_arima = ARIMA(df['total_sales'], order=(5,1,0))
    model_fit = model_arima.fit()
    arima_predictions = model_fit.forecast(steps=len(df)).tolist()

    # GRU Model
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['total_sales']].values)
    look_back = 5
    X, Y = create_dataset(scaled_data, look_back)
    X = torch.tensor(X, dtype=torch.float32).reshape(X.shape[0], X.shape[1], 1)
    Y = torch.tensor(Y, dtype=torch.float32)

    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    input_size = 1
    hidden_size = 50
    output_size = 1
    model_gru = GRUModel(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_gru.parameters(), lr=0.01)

    num_epochs = 50
    for epoch in range(num_epochs):
        for batch_X, batch_Y in dataloader:
            outputs = model_gru(batch_X)
            loss = criterion(outputs, batch_Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model_gru.eval()
    with torch.no_grad():
        train_predict = model_gru(X).numpy()
    train_predict = scaler.inverse_transform(train_predict).tolist()

    # Prophet Model
    prophet_df = df.rename(columns={"sales_date": "ds", "total_sales": "y"})
    model_prophet = Prophet()
    model_prophet.fit(prophet_df)
    future = model_prophet.make_future_dataframe(periods=30, freq='D')
    forecast = model_prophet.predict(future)
    prophet_predictions = forecast[['ds', 'yhat']]

    # ✅ Store predictions in cache
    results_cache["arima"] = (df['sales_date'], df['total_sales'], arima_predictions)
    results_cache["gru"] = (df['sales_date'][look_back:], df['total_sales'][look_back:], train_predict)
    results_cache["prophet"] = (prophet_predictions["ds"], prophet_df["y"], prophet_predictions["yhat"])

# ✅ Generate predictions at startup
generate_predictions()

# ✅ Prediction Graphs API
@app.get("/plot")
async def plot_graph(model_name: str = Query(..., description="Choose model: arima, gru, or prophet")):
    if model_name not in results_cache:
        raise HTTPException(status_code=404, detail="Model not found!")

    dates, actual, predicted = results_cache[model_name]

    plt.figure(figsize=(10, 5))
    plt.plot(dates, actual, label="Actual Sales", marker='o', linestyle='dashed')
    plt.plot(dates, predicted, label=f"{model_name.upper()} Predictions", marker='x')
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.title(f"{model_name.upper()} Sales Predictions")
    plt.legend()
    plt.grid()

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png")
    img_buf.seek(0)
    plt.close()

    return Response(content=img_buf.getvalue(), media_type="image/png")

# ✅ EDA Visualizations API
@app.get("/eda")
async def eda_visualization(graph_type: str = Query(..., description="Choose graph: histogram, correlation, trend")):
    plt.figure(figsize=(10, 5))

    if graph_type == "histogram":
        sns.histplot(df['total_sales'], bins=30, kde=True)
        plt.title("Sales Distribution (Histogram)")

    elif graph_type == "correlation":
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix")

    elif graph_type == "trend":
        plt.plot(df["sales_date"], df["total_sales"], marker="o", linestyle="-")
        plt.xlabel("Date")
        plt.ylabel("Total Sales")
        plt.title("Sales Trend Over Time")

    else:
        raise HTTPException(status_code=400, detail="Invalid graph type! Choose 'histogram', 'correlation', or 'trend'.")

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png")
    img_buf.seek(0)
    plt.close()

    return Response(content=img_buf.getvalue(), media_type="image/png")


