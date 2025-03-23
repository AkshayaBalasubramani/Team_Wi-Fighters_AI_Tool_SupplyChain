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

def load_data():
    df = pd.read_csv(r"C:\Users\Akshaya\OneDrive\Desktop\Xcelerate\chat_gpt\inventory_data.csv")
    df['sales_date'] = pd.to_datetime(df['sales_date'])
    df = df.sort_values('sales_date')
    return df

@app.get("/forecast/arima")
def arima_forecast():
    df = load_data()
    sales_series = df.set_index('sales_date')['quantity_sold']
    model = ARIMA(sales_series, order=(5,1,0))
    model_fit = model.fit()
    pred = model_fit.forecast(steps=10)
    return {"forecast": pred.to_dict()}

@app.get("/forecast/prophet")
def prophet_forecast():
    df = load_data()
    df_prophet = df[['sales_date', 'quantity_sold']].rename(columns={'sales_date': 'ds', 'quantity_sold': 'y'})
    m = Prophet()
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=10)
    forecast = m.predict(future)
    return {"forecast": forecast[['ds', 'yhat']].to_dict(orient='records')}

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])

@app.get("/forecast/gru")
def gru_forecast():
    df = load_data()
    scaler = MinMaxScaler()
    df['scaled_sales'] = scaler.fit_transform(df[['quantity_sold']])
    X, y = [], []
    seq_length = 5
    for i in range(len(df) - seq_length):
        X.append(df['scaled_sales'].iloc[i:i+seq_length].values)
        y.append(df['scaled_sales'].iloc[i+seq_length])
    X = np.array(X)
    X = torch.Tensor(X).unsqueeze(-1)
    
    model = GRUModel(input_size=1, hidden_size=64, output_size=1)
    model.eval()
    with torch.no_grad():
        gru_forecast = model(X[-1:]).detach().numpy()
    forecast = scaler.inverse_transform(gru_forecast.reshape(-1, 1))
    return {"forecast": forecast.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003, reload=True)
