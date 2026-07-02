# ==============================================================================
# SMART LENDER: END-TO-END MACHINE LEARNING & WEB APPLICATION
# PHASE 4: FLASK SERVER IMPLEMENTATION
# ==============================================================================

import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import pickle

# Initialize Flask application structure
app = Flask(__name__)

# Load the pre-trained model and scaler generated during Phase 2
try:
    with open('rdf.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scale1.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    print("Backend artifacts ('rdf.pkl' & 'scale1.pkl') loaded successfully!")
except FileNotFoundError:
    print("Error: Missing required model artifacts. Run the training script first.")

# 1. Route for Home Page (Renders the Application Form)
@app.route('/')
def home():
    return render_template('index.html')

# 2. Route for Handling Form Submission and Generating Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract inputs from form fields
        gender = request.form['Gender']
        married = request.form['Married']
        dependents = request.form['Dependents']
        education = request.form['Education']
        self_employed = request.form['Self_Employed']
        applicant_income = float(request.form['ApplicantIncome'])
        coapplicant_income = float(request.form['CoapplicantIncome'])
        loan_amount = float(request.form['LoanAmount'])
        loan_term = float(request.form['Loan_Amount_Term'])
        credit_history = float(request.form['Credit_History'])
        property_area = request.form['Property_Area']
        
        # Transform Categorical values to their matching training numeric values
        gender_mapped = 1 if gender == 'Male' else 0
        married_mapped = 1 if married == 'Yes' else 0
        
        dep_map = {'0': 0, '1': 1, '2': 2, '3+': 3}
        dependents_mapped = dep_map.get(dependents, 0)
        
        education_mapped = 1 if education == 'Graduate' else 0
        self_employed_mapped = 1 if self_employed == 'Yes' else 0
        
        prop_map = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}
        property_area_mapped = prop_map.get(property_area, 1)
        
        # Synthesize into an ordered numerical feature array matching training shape
        feature_values = np.array([[
            gender_mapped, married_mapped, dependents_mapped, education_mapped,
            self_employed_mapped, applicant_income, coapplicant_income,
            loan_amount, loan_term, credit_history, property_area_mapped
        ]])
        
        # Scale the feature values using the saved scaler structure
        scaled_features = scaler.transform(feature_values)
        
        # Run live model prediction (0 = Rejected, 1 = Approved)
        prediction_outcome = int(model.predict(scaled_features)[0])
        
        # Send result over to the output template page
        return render_template('submit.html', prediction=prediction_outcome)

if __name__ == '__main__':
    # Start the local development web server on port 5000
    app.run(debug=True, port=5000)