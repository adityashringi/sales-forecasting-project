# verify_forecast.py

import pandas as pd

# Load the forecasted data
df = pd.read_csv('../data/sales_data_forecast_prophet.csv')

# Display the dataframe
print(df)
