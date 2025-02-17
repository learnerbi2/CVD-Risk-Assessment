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

# Load Model and Scaler with Better Handling
model_path = "Modifiedmodel.pkl"
scaler_path = "scaler.pkl"

def load_pickle_file(file_path):
    """Load pickle files safely."""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

model = load_pickle_file(model_path)
scaler = load_pickle_file(scaler_path)

if not model or not scaler:
    print("❌ Model or Scaler is missing/corrupted! Ensure files exist.")
    exit()  # Stop execution if critical files are missing

# Feature Columns
feature_columns = [
    'age', 'sex', 'smoking_status', 'exercise_frequency', 'alcohol_consumption', 'diet',
    'high_blood_pressure', 'high_cholesterol', 'diabetes', 'heart_conditions',
    'family_history_heart_disease', 'shortness_of_breath', 'palpitations'
]

@app.route('/')
def index():
    return render_template('predictionpage.html')

def determine_risk_level(probability):
    """Determine risk category and graph color based on probability."""
    if probability > 70:
        return "High Risk", "red", "high_risk.html"
    elif 50 < probability <= 70:
        return "Moderate Risk", "orange", "moderate_risk.html"
    else:
        return "Low Risk", "green", "low_risk.html"

def generate_graph(probability, risk_level, graph_color):
    """Generate and return the graph URL."""
    try:
        plt.figure(figsize=(5, 5))
        plt.bar(["Heart Disease Risk"], [probability], color=graph_color)
        plt.ylim([0, 100])
        plt.ylabel('Risk Percentage')
        plt.title('Heart Disease Prediction')
        plt.text(0, -10, risk_level, ha='center', va='top', fontsize=15, fontweight='bold', color=graph_color)
        plt.grid(True, linestyle="--", alpha=0.6)

        # Convert to image and encode
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        return "data:image/png;base64," + base64.b64encode(img.getvalue()).decode()
    
    except Exception as e:
        print(f"Error generating graph: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate Input Fields
        missing_fields = [col for col in feature_columns if col not in request.form]
        if missing_fields:
            return jsonify({"error": f"Missing form fields: {', '.join(missing_fields)}"}), 400

        user_name = request.form.get('name', 'User')

        # Parse and validate input values
        user_data = {}
        for col in feature_columns:
            try:
                user_data[col] = int(request.form[col])
            except ValueError:
                return jsonify({"error": f"Invalid input for {col}. Enter a valid number."}), 400

        # Convert to DataFrame & Scale
        user_df = pd.DataFrame([user_data], columns=feature_columns)
        user_scaled = scaler.transform(user_df)

        # Predict Probability
        probability = round(model.predict_proba(user_scaled)[0][1] * 100, 2)

        # Determine Risk Level
        risk_level, graph_color, template = determine_risk_level(probability)

        # Generate Graph
        graph_url = generate_graph(probability, risk_level, graph_color)
        if not graph_url:
            return jsonify({"error": "Failed to generate graph."}), 500

        # Render Template
        return render_template(template, name=user_name, probability=probability, graph_url=graph_url)

    except Exception as e:
        error_message = f"❌ Error: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
