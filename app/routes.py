from flask import Blueprint, request, jsonify, render_template
import joblib
import pandas as pd

main = Blueprint('main', __name__)

# Load trained model and feature columns
model = joblib.load("src/house_price_model.pkl")
model_columns = joblib.load("src/model_columns.pkl")  
locations = joblib.load("src/location_list.pkl")

@main.route('/')
def home():
    return render_template('index.html', locations=locations)

@main.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        df = pd.DataFrame([data])

        # One-hot encode and align columns with training data
        df = pd.get_dummies(df, columns=['location'])
        df = df.reindex(columns=model_columns, fill_value=0)

        # Predict
        prediction = model.predict(df)[0]

        # Convert numpy float32 -> Python float
        prediction = float(prediction)

        return jsonify({'predicted_price': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': str(e)})
