from fastapi import FastAPI
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import uvicorn

app = FastAPI()

# Load data
def load_data():
    df = pd.read_csv("supplies_data.csv")
    df["sales_date"] = pd.to_datetime(df["sales_date"])
    df = df.sort_values("sales_date")
    return df

@app.get("/forecast/prophet")
def forecast_prophet():
    df = load_data()
    prophet_df = df[["sales_date", "total_sales"]].rename(columns={"sales_date": "ds", "total_sales": "y"})
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=10)
    forecast = model.predict(future)
    return forecast.tail(10)[["ds", "yhat"]].to_dict(orient="records")

@app.get("/forecast/arima")
def forecast_arima():
    df = load_data()
    model = ARIMA(df["total_sales"], order=(2,1,2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=10)
    return {"forecast": forecast.tolist()}

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])

@app.get("/forecast/gru")
def forecast_gru():
    df = load_data()
    scaler = MinMaxScaler()
    scaled_sales = scaler.fit_transform(df["total_sales"].values.reshape(-1, 1))
    
    seq_length = 3
    def create_sequences(data, seq_length):
        sequences, labels = [], []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i + seq_length])
            labels.append(data[i + seq_length])
        return np.array(sequences), np.array(labels)
    
    X, y = create_sequences(scaled_sales, seq_length)
    X_train, y_train = torch.Tensor(X), torch.Tensor(y)
    X_train = X_train.reshape(X_train.shape[0], seq_length, 1)
    
    model = GRUModel(input_size=1, hidden_size=64, output_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train.unsqueeze(-1))
        loss.backward()
        optimizer.step()
    
    X_test = torch.Tensor(scaled_sales[-seq_length:].reshape(1, seq_length, 1))
    forecast = model(X_test).detach().numpy()
    forecast = scaler.inverse_transform(forecast.reshape(-1, 1))
    return {"forecast": forecast.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
