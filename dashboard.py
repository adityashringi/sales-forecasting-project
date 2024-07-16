import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Sales Forecasting Dashboard")
st.write("""
This dashboard visualizes actual sales data and forecasts from ARIMA, Linear Regression, and Prophet models.
""")

# Load the data
sales_data = pd.read_csv('../data/sales_data_cleaned.csv')
forecast_arima = pd.read_csv('../data/sales_data_forecast_arima.csv')
forecast_linear_regression = pd.read_csv('../data/sales_data_forecast_linear_regression.csv')
forecast_prophet = pd.read_csv('../data/sales_data_forecast_prophet.csv')

# Convert date columns to datetime
sales_data['date'] = pd.to_datetime(sales_data['date'])
forecast_arima['date'] = pd.to_datetime(forecast_arima['date'])
forecast_linear_regression['date'] = pd.to_datetime(forecast_linear_regression['date'])
forecast_prophet['ds'] = pd.to_datetime(forecast_prophet['ds'])

# Fill missing forecast values with 0 if necessary
forecast_arima['forecast_arima'] = forecast_arima['forecast_arima'].fillna(0)
forecast_linear_regression['forecast_linear_regression'] = forecast_linear_regression['forecast_linear_regression'].fillna(0)
forecast_prophet['yhat'] = forecast_prophet['yhat'].fillna(0)

# Debugging
st.write("Checking data...")
st.write("Sales Data:")
st.dataframe(sales_data.head())

st.write("ARIMA Forecast:")
st.dataframe(forecast_arima.head())

st.write("Linear Regression Forecast:")
st.dataframe(forecast_linear_regression.head())

st.write("Prophet Forecast:")
st.dataframe(forecast_prophet.head())

# Plot historical sales data
st.subheader("Historical Sales Data")
st.line_chart(sales_data.set_index('date')['sales'])

# Plot ARIMA forecast
st.subheader("ARIMA Forecast")
plt.figure(figsize=(10, 5))
plt.plot(sales_data['date'], sales_data['sales'], label='Actual Sales')
plt.plot(forecast_arima['date'], forecast_arima['forecast_arima'], label='ARIMA Forecast')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('ARIMA Sales Forecast')
st.pyplot(plt)

# Plot Linear Regression forecast
st.subheader("Linear Regression Forecast")
plt.figure(figsize=(10, 5))
plt.plot(sales_data['date'], sales_data['sales'], label='Actual Sales')
plt.plot(forecast_linear_regression['date'], forecast_linear_regression['forecast_linear_regression'], label='Linear Regression Forecast')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Linear Regression Sales Forecast')
st.pyplot(plt)

# Plot Prophet forecast with historical data
st.subheader("Prophet Forecast")
plt.figure(figsize=(10, 5))
plt.plot(sales_data['date'], sales_data['sales'], label='Actual Sales')
plt.plot(forecast_prophet['ds'], forecast_prophet['yhat'], label='Prophet Forecast')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Prophet Sales Forecast')
st.pyplot(plt)

# Display evaluation metrics
st.subheader("Model Evaluation Metrics")
evaluation_metrics = pd.DataFrame({
    'Model': ['ARIMA', 'Linear Regression', 'Prophet'],
    'MAE': [676.92, 676.92, 596.15],  # Replace these with actual calculated metrics
    'RMSE': [798.88, 798.88, 778.52]  # Replace these with actual calculated metrics
})
st.table(evaluation_metrics)

st.write("Dashboard created with Streamlit")
