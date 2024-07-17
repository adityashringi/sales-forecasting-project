# Sales Forecasting Project

This project aims to forecast sales for an e-commerce platform using different time series forecasting models.

## Project Objective

The goal of the project is to develop a time series forecasting model to predict future sales based on historical sales data from an e-commerce platform.

## Models Used

- ARIMA
- Prophet
- Linear Regression

## Project Structure

- `data/`: Contains data files used in the project.
- `notebook/`: Contains Jupyter notebooks used for exploration and model building.
- `src/`: Contains all the Python scripts for data processing, model training, and evaluation.
  - `data_cleaning.py`
  - `data_collection.py`
  - `feature_engineering.py`
  - `generate_sales_data.py`
  - `model_arima.py`
  - `model_linear_regression.py`
  - `model_prophet.py`
  - `evaluate_models.py`
  - `dashboard.py`
  - `api.py`
  - `test_mysql_connection.py`
  - `verify_forecast.py`
- `requirements.txt`: List of dependencies required to run the project.

## How to Run the Project

1. **Clone the repository**:
  

2. **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the data processing and model training scripts**:
    ```bash
    cd src
    python data_cleaning.py
    python data_collection.py
    python feature_engineering.py
    python generate_sales_data.py
    python model_arima.py
    python model_linear_regression.py
    python model_prophet.py
    python evaluate_models.py
    ```

5. **Run the dashboard**:
    ```bash
    streamlit run dashboard.py
    ```

6. **Run the API**:
    ```bash
    python api.py
    ```

## Project Results

The project evaluates the models using MAE and RMSE metrics. The results and visualizations can be seen in the Streamlit dashboard.

