from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('../models/prophet_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame(data)
    df['ds'] = pd.to_datetime(df['ds'])

    # Make predictions
    forecast = model.predict(df)
    predictions = forecast[['ds', 'yhat']].to_dict(orient='records')

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Using port 5001 instead of 5000
