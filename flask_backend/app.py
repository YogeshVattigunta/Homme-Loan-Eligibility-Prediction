from flask import Flask, request, jsonify
import numpy as np
import pickle
from flask_cors import CORS

# Load model and scaler
with open('loan_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Bank data with interest rates and processing fees
banks_info = {
    "Union Bank of India": {
        "Starting Interest Rate": "8.10% onwards",
        "Processing Fees": "0.50% of the loan amount"
    },
    "Bank of Maharashtra": {
        "Starting Interest Rate": "8.10% onwards",
        "Processing Fees": "Not specified"
    },
    "Central Bank of India": {
        "Starting Interest Rate": "8.10% onwards",
        "Processing Fees": "0.50% up to Rs.20,000 Plus GST (waived till 31 March 2024)"
    },
    "Bank of Baroda": {
        "Starting Interest Rate": "8.15% onwards",
        "Processing Fees": "No processing fee; discounted upfront fee"
    },
    "Canara Bank": {
        "Starting Interest Rate": "8.15% onwards",
        "Processing Fees": "0.50% of the loan amount"
    },
    "Kotak Mahindra Bank": {
        "Starting Interest Rate": "8.75% onwards",
        "Processing Fees": "Salaried: 0.5% Plus taxes; Self-Employed/Commercial: 1.0% Plus taxes"
    },
    "Axis Bank": {
        "Starting Interest Rate": "8.75% onwards",
        "Processing Fees": "Up to 1% or min. Rs.10,000 Plus GST"
    }
}

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.json
        LoanAmount = float(data['LoanAmount'])
        DownPayment = float(data['DownPayment'])
        AnnualIncome = float(data['AnnualIncome'])
        CreditScore = float(data['CreditScore'])
        YearsEmployed = float(data['YearsEmployed'])
        PropertyType = int(data['PropertyType'])  # 0 = Apartment, 1 = House

        # Prepare input (Ensure 2D shape)
        input_data = np.array([LoanAmount, DownPayment, AnnualIncome, CreditScore, YearsEmployed, PropertyType]).reshape(1, -1)
        input_data = scaler.transform(input_data)  # Scale input data

        # Predict
        prediction = model.predict(input_data)
        result = "Approved" if prediction[0] == 1 else "Denied"

        # Return JSON response with bank information
        return jsonify({"result": result, "banks": banks_info})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)