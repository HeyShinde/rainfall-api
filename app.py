from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the request data is in JSON format
        if request.is_json:
            data = request.get_json()
        else:
            # If not JSON, assume form data
            data = {
                'min_temp': request.form.get('min_temp', None),
                'max_temp': request.form.get('max_temp', None),
                'min_humidity': request.form.get('min_humidity', None),
                'max_humidity': request.form.get('max_humidity', None),
                'min_wind_speed': request.form.get('min_wind_speed', None),
                'max_wind_speed': request.form.get('max_wind_speed', None),
                'month': request.form.get('month', None)
            }

        # Check if any of the required fields are missing or empty
        if None in data.values():
            return jsonify({'error': 'Missing or empty input fields.'}), 400

        # Convert the input values to the appropriate data types
        data = {
            'min_temp': float(data['min_temp']),
            'max_temp': float(data['max_temp']),
            'min_humidity': float(data['min_humidity']),
            'max_humidity': float(data['max_humidity']),
            'min_wind_speed': float(data['min_wind_speed']),
            'max_wind_speed': float(data['max_wind_speed']),
            'month': int(data['month'])
        }

        # Prepare the input data as a NumPy array
        input_query = np.array([[
            data['min_temp'],
            data['max_temp'],
            data['min_humidity'],
            data['max_humidity'],
            data['min_wind_speed'],
            data['max_wind_speed'],
            data['month']
        ]])

        # Make a prediction using the model
        result = model.predict(input_query)[0]

        return jsonify({'prediction': float(result)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
