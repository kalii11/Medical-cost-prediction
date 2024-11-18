from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)


app = Flask(__name__)

# Load the model and scaler
model = joblib.load('modesl.joblib')
scaler = joblib.load('scalers.joblib')


# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction page route
@app.route('/predict')
def predict_page():

    return render_template('predict.html')

# Prediction API route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("Received data:", data)

        # Prepare the input data
        input_data = {
            'age': data.get('age', 0),
            'sex': 1 if data.get('sex') == 'male' else 0,
            'bmi': data.get('bmi', 0),
            'children': data.get('children', 0),
            'smoker': 1 if data.get('smoker') == 'yes' else 0,
            'region': {'northeast': 0, 'southeast': 1, 'southwest': 2, 'northwest': 3}.get(data.get('region'), 0)
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Scale the features
        input_scaled = scaler.transform(input_df)

        # Make a prediction
        predicted_value = model.predict(input_scaled)[0]

        # Create response
        response = {
            "prediction": 'High Charges' if predicted_value > 0.5 else 'Low Charges',
            "predicted_value": round(predicted_value, 4)
        }
        return jsonify(response)

    except Exception as e:
        print("Error in prediction endpoint:", e)
        return jsonify({"error": str(e)}), 500

@app.route('/status')
def status_page():
    return render_template('status.html')

@app.route('/aboutus')
def aboutus_page():
    return render_template('aboutus.html')

@app.route('/learnmore')
def learnmore_page():
    return render_template('learnmore.html')

@app.route('/explain')
def explain_page():
    return render_template('explain.html')

@app.route('/tips')
def tips_page():
    return render_template('tips.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555, debug=True)








