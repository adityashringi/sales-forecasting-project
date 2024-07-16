import pandas as pd
import numpy as np

def clean_data(df):
    # Data Exploration
    print("Data Exploration:")
    print(df.head())
    print(df.info())
    print(df.describe())

    # Handle Missing Values
    print("\nHandling Missing Values:")
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    print(df.info())

    # Detect and Handle Outliers wit IQR
    print("\nDetecting Outliers:")
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum().sum()
    df_cleaned = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    print(f"Removed {outliers} outliers")
    print(df_cleaned.info())

    # Feature Engineering
    print("\nFeature Engineering:")
    df_cleaned['sales_diff'] = df_cleaned['sales'].diff()
    df_cleaned.dropna(inplace=True)
    print(df_cleaned.head())

    return df_cleaned

if __name__ == "__main__":
    df = pd.read_csv('data/sales_data.csv')
    df_cleaned = clean_data(df)
    df_cleaned.to_csv('data/sales_data_cleaned.csv', index=False)
    print("Data cleaned and saved to 'data/sales_data_cleaned.csv'")
