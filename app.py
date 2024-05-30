from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('C:\\Users\\NAMO\\Desktop\\heart_disease_app\\heart_disease_model.pkl')
scaler = joblib.load('C:\\Users\\NAMO\\Desktop\\heart_disease_app\\Hscaler.pkl')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect input data from the form
        features = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal'])
        ]
        
        # Convert features to array
        features_array = np.array(features).reshape(1, -1)
        
        # Scale the input features
        scaled_features = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(scaled_features)
        
        # Determine prediction text
        prediction_text = "Heart Disease Risk: {}".format('High' if prediction[0] else 'Low')
        
        # Render the template with prediction result
        return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
