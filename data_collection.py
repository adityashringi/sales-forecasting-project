import mysql.connector
import pandas as pd

# Function to generate synthetic data
def generate_synthetic_data():
    # Generate some example data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    sales = pd.Series(range(1000, 1100))
    df = pd.DataFrame({'date': dates, 'sales': sales})
    return df

# Function to save data to MySQL
def save_data_to_mysql(df):
    connection = mysql.connector.connect(
        host='127.0.0.1',
        user='root',  # Update with  MySQL username
        password='101102',  # Update with  MySQL password
        database='210'
    )

    cursor = connection.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS sales_data (date DATE, sales INT)")

    for i, row in df.iterrows():
        cursor.execute("INSERT INTO sales_data (date, sales) VALUES (%s, %s)", (row['date'], row['sales']))
    connection.commit()
    cursor.close()
    connection.close()

# Main script
if __name__ == '__main__':
    df = generate_synthetic_data()
    df.to_csv('../data/sales_data.csv', index=False)
    save_data_to_mysql(df)
    print("Data collection completed and saved to MySQL and CSV.")
