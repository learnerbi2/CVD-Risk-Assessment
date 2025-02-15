import pandas as pd
import pickle
from flask import Flask, render_template, request, jsonify
import io
import base64
import matplotlib
import matplotlib.pyplot as plt
import traceback
import os
# Fix GUI issue for servers
matplotlib.use('Agg')  

app = Flask(__name__)
try:
    model_path = "Modifiedmodel.pkl"
    scaler_path = "scaler.pkl"
#missing file error handle
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(" Model or Scaler file is missing!")

    model = pickle.load(open(model_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))

except Exception as e:
    print(f"Error loading model/scaler: {e}")
    exit() 

# column values
feature_columns = [
    'age', 'sex', 'smoking_status', 'exercise_frequency', 'alcohol_consumption', 'diet',
    'high_blood_pressure', 'high_cholesterol', 'diabetes', 'heart_conditions',
    'family_history_heart_disease', 'shortness_of_breath', 'palpitations'
]

@app.route('/')
def index():
    return render_template('predictionpage.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure required fields exist in form data
        missing_fields = [col for col in feature_columns if col not in request.form]
        if missing_fields:
            raise ValueError(f" Missing form fields: {', '.join(missing_fields)}")

        user_name = request.form.get('name', 'User')

# Collect user inputs from column
        user_data = {}
        for col in feature_columns:
            try:
                user_data[col] = int(request.form[col])
            except ValueError:
                raise ValueError(f" Invalid input for {col}. Please enter a valid number.")

# Convert to DataFrame
        user_df = pd.DataFrame([user_data], columns=feature_columns)

# Apply Scaler
        user_scaled = scaler.transform(user_df)

# Predict probability bnaya
        probability = model.predict_proba(user_scaled)[0][1] * 100
        probability = round(probability, 2)  # Round to 2 decimal places

# Risk check
        if probability > 70:
            graph_color = 'red'
            risk_level = "High Risk"
        elif 50 < probability <= 70:
            graph_color = 'orange'
            risk_level = "Moderate Risk"
        else:
            graph_color = 'green'
            risk_level = "Low Risk"

        # graph generate
        try:
            plt.figure(figsize=(5, 5))
            plt.bar(["Heart Disease Risk"], [probability], color=graph_color)
            plt.ylim([0, 100])
            plt.ylabel('Risk Percentage')
            plt.title('Heart Disease Prediction')
            plt.text(0, -10, risk_level, ha='center', va='top', fontsize=15, fontweight='bold', color=graph_color)

        #convert graph to image and save
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plt.close()

        # Encode image
            graph_url = "data:image/png;base64," + base64.b64encode(img.getvalue()).decode()

        except Exception as e:
            raise RuntimeError(f"Error generating graph: {e}")

        #render in three pages
        if probability > 70:
            return render_template('high_risk.html', name=user_name, probability=probability, graph_url=graph_url)
        elif 50 < probability <= 70:
            return render_template('moderate_risk.html', name=user_name, probability=probability, graph_url=graph_url)
        else:
            return render_template('low_risk.html', name=user_name, probability=probability, graph_url=graph_url)

    except Exception as e:
        error_message = f"❌ Error: {str(e)}\n{traceback.format_exc()}"
       # Log full traceback for debugging
        print(error_message)  
       # Return JSON error response
        return jsonify({"error": str(e)}), 400 

if __name__ == "__main__":
    app.run(debug=True)
