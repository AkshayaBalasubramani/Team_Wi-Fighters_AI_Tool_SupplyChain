from fastapi import FastAPI
import pandas as pd
import statsmodels.api as sm
from prophet import Prophet
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import uvicorn

app = FastAPI()

data_path = "logistics_data.csv"  # Replace with your dataset path
df = pd.read_csv(data_path, parse_dates=["date"], index_col="date")

# --- EDA Endpoints ---
@app.get("/eda/summary")
def summary_statistics():
    return df.describe().to_dict()

@app.get("/eda/trends")
def sales_trends():
    plt.figure(figsize=(10,5))
    sns.lineplot(data=df, x=df.index, y="total_sales")
    plt.title("Sales Trend Over Time")
    plt.savefig("sales_trend.png")
    return {"message": "Sales trend saved as sales_trend.png"}

# --- Forecasting Models ---
@app.get("/forecast/arima")
def arima_forecast():
    model = sm.tsa.ARIMA(df["total_sales"], order=(5,1,0))
    results = model.fit()
    forecast = results.forecast(steps=30)
    return forecast.to_dict()

@app.get("/forecast/prophet")
def prophet_forecast():
    prophet_df = df.reset_index().rename(columns={"date": "ds", "total_sales": "y"})
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(30).to_dict()

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
    X = torch.tensor(df["total_sales"].values).float().view(-1, 1, 1)
    model = GRUModel(1, 10, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for epoch in range(100):
        optimizer.zero_grad()
        y_pred = model(X[:-1])
        loss = loss_fn(y_pred, X[1:])
        loss.backward()
        optimizer.step()
    future_input = X[-1].unsqueeze(0)
    forecast = model(future_input).detach().numpy()
    return {"gru_forecast": forecast.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=True)

