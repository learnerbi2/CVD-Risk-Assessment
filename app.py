import os
import pandas as pd

import pickle
from flask import Flask, render_template, request,redirect, url_for
import io
import base64
import matplotlib
import matplotlib.pyplot as plt
# fix gui issue here
matplotlib.use('Agg')

try:
    model_path = "Modifiedmodel.pkl"
    scaler_path = "scaler.pkl"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(" Model or Scaler file is missing!")

    model = pickle.load(open(model_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))

except Exception as e:
    print(f"Error loading model/scaler: {e}")
    exit() 

# Ensure correct column order
feature_columns = [
    'age', 'sex', 'smoking_status', 'exercise_frequency', 'alcohol_consumption', 'diet', 'high_blood_pressure',
    'high_cholesterol', 'diabetes', 'heart_conditions', 'family_history_heart_disease', 'shortness_of_breath', 'palpitations'
]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/registration', methods=['GET', 'POST'])
def registration():
    if request.method == 'POST':
        # Handle registration logic here (e.g., store user data)
        return redirect(url_for('login'))  # Redirect to login after successful registration
    return render_template('registration.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Handle login logic here (e.g., verify credentials)
        return redirect(url_for('profile'))  # Redirect to profile after successful login
    return render_template('login.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        # Handle feedback submission here
        return render_template('feedback_submitted.html')  # Redirect to thank you page
    return render_template('feedback.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            user_name = request.form['name']
            user_data = {}
            for col in feature_columns:
                try:
                    user_data[col] = int(request.form[col])
                except ValueError:
                    return f"Invalid input for {col}. Please enter a number.", 400

            user_df = pd.DataFrame([user_data], columns=feature_columns)
            user_scaled = scaler.transform(user_df)
            probability = model.predict_proba(user_scaled)[0][1] * 100
            probability = round(probability, 2)
            
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


            if probability > 70:
                return redirect( url_for('high_risk', name=user_name, probability=probability, graph_url=graph_url))
            elif 40 < probability <= 70:
                return redirect( url_for('moderate_risk', name=user_name, probability=probability, graph_url=graph_url))
            else:
                return redirect( url_for('low_risk', name=user_name, probability=probability, graph_url=graph_url))

        except KeyError as e:
            return f"Missing form field: {e}", 400
        except ValueError as e:  # Catch potential ValueError during int conversion
            return f"Invalid input: {e}", 400
        except Exception as e:
            return f"An error occurred: {e}", 500

    return render_template('predictionpage.html')  # Display the form on GET request

@app.route('/high_risk')
def high_risk():
    name = request.args.get('name')
    probability = request.args.get('probability')
    graph_url = request.args.get('graph_url')
    return render_template('high_risk.html', name=name, probability=probability, graph_url=graph_url)

@app.route('/moderate_risk')
def moderate_risk():
    name = request.args.get('name')
    probability = request.args.get('probability')
    graph_url = request.args.get('graph_url')
    return render_template('moderate_risk.html', name=name, probability=probability, graph_url=graph_url)

@app.route('/low_risk')
def low_risk():
    name = request.args.get('name')
    probability = request.args.get('probability')
    graph_url = request.args.get('graph_url')
    return render_template('low_risk.html', name=name, probability=probability, graph_url=graph_url)

if __name__ == "__main__":
    app.run(debug=True)
