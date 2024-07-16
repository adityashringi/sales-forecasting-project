# generate_sales_data.py
import pandas as pd
import numpy as np
import os

# Generate synthetic data
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
sales = np.random.randint(100, 1000, size=(100,))
data = pd.DataFrame({'date': dates, 'sales': sales})

# Ensure the directory exists
if not os.path.exists('data'):
    os.makedirs('data')

# Save to CSV
data.to_csv('data/sales_data_cleaned.csv', index=False)
print("Synthetic data saved to 'data/sales_data_cleaned.csv'")
