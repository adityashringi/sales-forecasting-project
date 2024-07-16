import pandas as pd

def create_features(df):
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    return df

if __name__ == "__main__":
    df = pd.read_csv('data/sales_data_cleaned.csv')
    df_features = create_features(df)
    df_features.to_csv('data/sales_data_features.csv', index=False)
    print("Features created and saved to 'data/sales_data_features.csv'")
