import os
print("Current working directory:", os.getcwd())

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Load data
df = pd.read_csv('data/sales_data_cleaned.csv')

# Ensure the data is stationary by differencing
df['sales_diff'] = df['sales'].diff().dropna()

# Prepare data for ARIMA
arima_df = df.set_index('date')
arima_df.index = pd.to_datetime(arima_df.index)
arima_df.dropna(inplace=True)

# Initialize and fit the model
model = ARIMA(arima_df['sales_diff'], order=(5,1,0))
model_fit = model.fit()

# Create a dataframe to hold predictions
start_index = len(arima_df)
end_index = start_index + 29  # Forecast for 30 periods
forecast = model_fit.predict(start=start_index, end=end_index, typ='levels')

# Reverse the differencing to get the actual forecast
forecast_cumsum = forecast.cumsum()
forecast_final = arima_df['sales'].iloc[-1] + forecast_cumsum

# Save predictions with valid dates
forecast_dates = pd.date_range(start=arima_df.index[-1] + pd.Timedelta(days=1), periods=len(forecast), freq='D')
forecast_df = pd.DataFrame({'date': forecast_dates, 'forecast_arima': forecast_final})

print(f"Forecast dates: {forecast_dates}")

forecast_df.to_csv('data/sales_data_forecast_arima.csv', index=False)

print("ARIMA model trained and forecast saved to 'data/sales_data_forecast_arima.csv'")
