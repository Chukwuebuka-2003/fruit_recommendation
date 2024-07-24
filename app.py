from flask import Flask, request, jsonify
import joblib
import pandas as pd


# Flask application
app = Flask(__name__)

# Load the model and preprocessors
model = joblib.load('model.sav')
le_gender = joblib.load('le_gender.sav')
le_health = joblib.load('le_health.sav')
le_fruit = joblib.load('le_fruit.sav')
le_vegetable = joblib.load('le_vegetable.sav')
scaler = joblib.load('scaler.sav')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    age = data['age']
    gender = data['gender']
    health_condition = data['health_condition']

    # Preprocess input
    age_scaled = scaler.transform([[age]])[0][0]
    gender_encoded = le_gender.transform([gender])[0]
    health_encoded = le_health.transform([health_condition])[0]

    # Make prediction
    prediction = model.predict([[age_scaled, gender_encoded, health_encoded]])

    # Decode prediction
    fruit = le_fruit.inverse_transform(prediction[:, 0])[0]
    vegetable = le_vegetable.inverse_transform(prediction[:, 1])[0]

    return jsonify({
        'recommended_fruit': fruit,
        'recommended_vegetable': vegetable
    })

if __name__ == '__main__':
    app.run(debug=True)
