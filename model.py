from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Enable CORS if needed
from flask_cors import CORS
CORS(app)
with open('Capstone-Transport-Demand-Prediction-main\Route_management.pkl', 'rb') as file:
    model = pickle.load(file)

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.get_json()

    # Assuming the data is passed as a list of features for prediction
    features = np.array(data['features']).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)

    # Return the prediction as JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
