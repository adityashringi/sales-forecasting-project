import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv('data/sales_data_cleaned.csv')

# Ensure 'date' is datetime
df['date'] = pd.to_datetime(df['date'])

# Prepare  data for linear regression
df['days'] = (df['date'] - df['date'].min()).dt.days
X = df[['days']]
y = df['sales']

# Initialize and fit the model
model = LinearRegression()
model.fit(X, y)

# Create a dataframe to hold predictions
future_days = np.arange(df['days'].max() + 1, df['days'].max() + 31).reshape(-1, 1)
future_dates = pd.date_range(df['date'].max() + pd.Timedelta(days=1), periods=30)
forecast = model.predict(future_days)

# Save predictions
forecast_df = pd.DataFrame({'date': future_dates, 'forecast_linear_regression': forecast})
forecast_df.to_csv('data/sales_data_forecast_linear_regression.csv', index=False)

print("Linear regression model trained and forecast saved to 'data/sales_data_forecast_linear_regression.csv'")
