import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

print("Starting the evaluation script...")

# Load actual data
print("Loading actual data from '../data/sales_data_cleaned.csv'...")
try:
    df_actual = pd.read_csv('../data/sales_data_cleaned.csv')
    df_actual['date'] = pd.to_datetime(df_actual['date'])
    actuals = df_actual.set_index('date')['sales']
    print("Actual data loaded successfully.")
    print(f"Actual data date range: {actuals.index.min()} to {actuals.index.max()}")
except Exception as e:
    print(f"Error loading actual data: {e}")

# Load ARIMA predictions
print("Loading ARIMA predictions from '../data/sales_data_forecast_arima.csv'...")
try:
    df_arima = pd.read_csv('../data/sales_data_forecast_arima.csv')
    df_arima['date'] = pd.to_datetime(df_arima['date'])
    df_arima.set_index('date', inplace=True)
    arima_preds = df_arima['forecast_arima'].dropna()
    print("ARIMA predictions loaded successfully.")
    print(f"ARIMA predictions date range: {arima_preds.index.min()} to {arima_preds.index.max()}")
except Exception as e:
    print(f"Error loading ARIMA predictions: {e}")

# Load Linear Regression predictions
print("Loading Linear Regression predictions from '../data/sales_data_forecast_linear_regression.csv'...")
try:
    df_lr = pd.read_csv('../data/sales_data_forecast_linear_regression.csv')
    df_lr['date'] = pd.to_datetime(df_lr['date'])
    df_lr.set_index('date', inplace=True)
    lr_preds = df_lr['forecast_linear_regression'].dropna()
    print("Linear Regression predictions loaded successfully.")
    print(f"Linear Regression predictions date range: {lr_preds.index.min()} to {lr_preds.index.max()}")
except Exception as e:
    print(f"Error loading Linear Regression predictions: {e}")

# Load Prophet predictions
print("Loading Prophet predictions from '../data/sales_data_forecast_prophet.csv'...")
try:
    df_prophet = pd.read_csv('../data/sales_data_forecast_prophet.csv')
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    df_prophet.set_index('ds', inplace=True)
    prophet_preds = df_prophet['yhat'].dropna()
    print("Prophet predictions loaded successfully.")
    print(f"Prophet predictions date range: {prophet_preds.index.min()} to {prophet_preds.index.max()}")
except Exception as e:
    print(f"Error loading Prophet predictions: {e}")

# To make sure dates align correctly .important for accurate evaluation.
try:
    # Adjusting predictions to match actual date range
    common_dates = actuals.index.union(arima_preds.index).union(lr_preds.index).union(prophet_preds.index)
    actuals = actuals.reindex(common_dates).ffill()
    arima_preds = arima_preds.reindex(common_dates).fillna(0)
    lr_preds = lr_preds.reindex(common_dates).fillna(0)
    prophet_preds = prophet_preds.reindex(common_dates).fillna(0)
    print(f"Common dates range: {common_dates.min()} to {common_dates.max()}")
    if len(common_dates) > 0:
        actuals = actuals.loc[common_dates]
        arima_preds = arima_preds.loc[common_dates]
        lr_preds = lr_preds.loc[common_dates]
        prophet_preds = prophet_preds.loc[common_dates]
        print("Dates aligned successfully.")
    else:
        raise ValueError("No common dates between actuals and predictions.")
except Exception as e:
    print(f"Error aligning dates: {e}")

# ARIMA Predictions
try:
    if len(actuals) > 0 and len(arima_preds) > 0:
        print("Actuals:")
        print(actuals)
        print("ARIMA Predictions:")
        print(arima_preds)
        mae_arima = mean_absolute_error(actuals, arima_preds)
        rmse_arima = np.sqrt(mean_squared_error(actuals, arima_preds))
        print(f"ARIMA MAE: {mae_arima}, RMSE: {rmse_arima}")
    else:
        raise ValueError("Empty actuals or predictions after alignment.")
except Exception as e:
    print(f"Error calculating ARIMA metrics: {e}")

# Linear Regression Predictions
try:
    if len(actuals) > 0 and len(lr_preds) > 0:
        print("Linear Regression Predictions:")
        print(lr_preds)
        mae_lr = mean_absolute_error(actuals, lr_preds)
        rmse_lr = np.sqrt(mean_squared_error(actuals, lr_preds))
        print(f"Linear Regression MAE: {mae_lr}, RMSE: {rmse_lr}")
    else:
        raise ValueError("Empty actuals or predictions after alignment.")
except Exception as e:
    print(f"Error calculating Linear Regression metrics: {e}")

# Prophet Predictions
try:
    if len(actuals) > 0 and len(prophet_preds) > 0:
        print("Prophet Predictions:")
        print(prophet_preds)
        mae_prophet = mean_absolute_error(actuals, prophet_preds)
        rmse_prophet = np.sqrt(mean_squared_error(actuals, prophet_preds))
        print(f"Prophet MAE: {mae_prophet}, RMSE: {rmse_prophet}")
    else:
        raise ValueError("Empty actuals or predictions after alignment.")
except Exception as e:
    print(f"Error calculating Prophet metrics: {e}")

print("Evaluation script completed.")
