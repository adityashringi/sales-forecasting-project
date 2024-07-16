import mysql.connector

try:
    connection = mysql.connector.connect(
        host='127.0.0.1',
        user='root',
        password='101102'
    )

    if connection.is_connected():
        print("Successfully connected to MySQL database")

except mysql.connector.Error as err:
    print(f"Error: {err}")

finally:
    if connection.is_connected():
        connection.close()
        print("MySQL connection is closed")
