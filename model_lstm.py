import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import LambdaCallback
import tensorflow as tf
import os

# Print TensorFlow and GPU information
print(f"TensorFlow version: {tf.__version__}")
print(f"Is GPU available: {tf.test.is_gpu_available()}")

# Generate synthetic data
print("Generating synthetic data...")
dates = pd.date_range(start="2023-01-01", periods=50, freq='D')
sales = np.random.randint(100, 1000, size=(50,))
month = dates.month
day_of_week = dates.dayofweek
data = pd.DataFrame({'date': dates, 'sales': sales, 'month': month, 'day_of_week': day_of_week})

# Use a larger subset for this test
subset_data = data.head(50)
train_data = subset_data[:35]
test_data = subset_data[35:]

X_train = train_data[['sales', 'month', 'day_of_week']].values
y_train = train_data['sales'].values
X_test = test_data[['sales', 'month', 'day_of_week']].values
y_test = test_data['sales'].values

X_train = X_train.reshape((35, 3, 1))
X_test = X_test.reshape((15, 3, 1))

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Build the LSTM model
print("Building the model...")
model = Sequential()
model.add(LSTM(10, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Logging function to print training progress
def on_epoch_end(epoch, logs):
    print(f"Epoch {epoch+1} completed. Loss: {logs['loss']}")

time_callback = LambdaCallback(on_epoch_end=on_epoch_end, verbose=1)

# Train model
print("Training the model...")
model.fit(X_train, y_train, batch_size=1, epochs=2, callbacks=[time_callback], verbose=1)
print("Model training completed.")

# Predict future values
print("Making predictions...")
predictions = model.predict(X_test)
print(f"Predictions shape: {predictions.shape}")
print("Predictions:", predictions)

# Save predictions
forecast_dates = pd.date_range(start=data['date'].iloc[-1], periods=len(predictions) + 1)[1:]
forecast_df = pd.DataFrame(predictions, columns=['forecast_lstm'])
forecast_df['date'] = forecast_dates

print("Forecast DataFrame before saving:")
print(forecast_df)

# Writing to CSV file
try:
    forecast_df.to_csv('data/sales_data_forecast_lstm.csv', index=False)
    print("Forecast saved to 'data/sales_data_forecast_lstm.csv'")
    
    # Check if file exists and print contents
    if os.path.exists('data/sales_data_forecast_lstm.csv'):
        print("File exists. Printing file contents:")
        with open('data/sales_data_forecast_lstm.csv', 'r') as file:
            print(file.read())
    else:
        print("File does not exist after attempting to save.")
except Exception as e:
    print("Error saving forecast to CSV:", e)
