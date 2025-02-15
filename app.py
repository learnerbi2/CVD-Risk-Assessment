import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, request
import io
import base64
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')  # Fix GUI issue

# Load Model & Scaler
model = pickle.load(open("Modifiedmodel.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Ensure correct column order
feature_columns = [
    'age', 'sex', 'smoking_status', 'exercise_frequency', 'alcohol_consumption', 'diet', 'high_blood_pressure',
    'high_cholesterol', 'diabetes', 'heart_conditions', 'family_history_heart_disease', 'shortness_of_breath', 'palpitations'
]

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('predictionpage.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_name = request.form['name']
        # Get user input from form
        user_data = {col: int(request.form[col]) for col in feature_columns}

        # Convert to DataFrame with proper column names
        user_df = pd.DataFrame([user_data], columns=feature_columns)

        # Apply StandardScaler (transform)
        user_scaled = scaler.transform(user_df)

        # Predict probability
        probability = model.predict_proba(user_scaled)[0][1] * 100
        probability = round(probability, 2)  # Round to 2 decimal places

        if probability > 70:
            graph_color='red'
            risk_level = "High Risk"
        elif 40 < probability <= 70:
            graph_color= 'orange'
            risk_level = "Moderate Risk"
        else:
            graph_color='green'
            risk_level = "low Risk"

        # Generate the bar graph
        plt.figure(figsize=(7,7))
        plt.bar(["Heart Disease Risk"], [probability], color=graph_color)
        plt.ylim([0, 100])
        plt.ylabel('Risk Percentage')
        plt.title('Heart Disease Prediction')
        plt.text(0, -10, risk_level, 
        ha='center', va='top', 
        fontsize=15, fontweight='bold',
        color=graph_color
        ) 

        # Convert the plot to an image
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        # Convert image to base64
        graph_url = "data:image/png;base64," + base64.b64encode(img.getvalue()).decode()

        # Redirect based on probability range
        if probability > 70:
            return render_template('high_risk.html', name=user_name,probability=probability, graph_url=graph_url)
        elif 40 < probability <= 70:
            return render_template('moderate_risk.html', name=user_name, probability=probability, graph_url=graph_url)
        else:
            return render_template('low_risk.html', name=user_name, probability=probability, graph_url=graph_url)

    except Exception as e:
        return f"Error: {e}", 400

if __name__ == "__main__":
    app.run(debug=True)
