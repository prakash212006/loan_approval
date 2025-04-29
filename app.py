import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the saved model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form inputs
    education = float(request.form['education'])
    income_annum = float(request.form['income_annum'])
    self_employed = float(request.form['self_employed'])
    bank_asset_value = float(request.form['bank_asset_value'])

    # Create input array
    input_features = np.array([[education, income_annum, self_employed, bank_asset_value]])

    # Predict
    prediction = model.predict(input_features)[0]

    # Convert prediction to readable form
    result = "Approved " if prediction == 1 else "Rejected "

    return render_template('result.html', prediction_text=f"Loan Status: {result}")

if __name__ == "__main__":
    app.run(debug=True)
