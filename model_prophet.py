import pandas as pd
from prophet import Prophet
import mysql.connector
import joblib
import os

# Create the models directory if it doesn't exist
os.makedirs('../models', exist_ok=True)

# Establish a connection to the MySQL database
connection = mysql.connector.connect(
    host='127.0.0.1',
    user='root',
    password='101102',
    database='210'
)

# Query the sales data from the database
query = "SELECT date AS ds, sales AS y FROM sales_data"
df = pd.read_sql(query, connection)

# Close database connection
connection.close()

# Ensure 'ds' column is in datetime format and 'y' column is in float format
df['ds'] = pd.to_datetime(df['ds'])
df['y'] = df['y'].astype(float)

# Initialize and fit the Prophet model
model = Prophet()
model.fit(df)

# Save model
joblib.dump(model, '../models/prophet_model.pkl')

# Create dataframe to hold future dates for prediction
future = model.make_future_dataframe(periods=30)

# Make predictions
forecast = model.predict(future)

# Save predictions
forecast[['ds', 'yhat']].to_csv('../data/sales_data_forecast_prophet.csv', index=False)

print("Prophet model trained, saved, and forecast saved to '../data/sales_data_forecast_prophet.csv'")
