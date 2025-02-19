@app.route('/predict', methods=['GET','POST'])
# def predict():
    
#      if request.method == 'POST':
#         try:
#               user_name = request.form['name']
#               # Get user input from form
#               user_data = {col: int(request.form[col]) for col in feature_columns}

#               # Convert to DataFrame with proper column names
#               user_df = pd.DataFrame([user_data], columns=feature_columns)

#               # Apply StandardScaler (transform)
#               user_scaled = scaler.transform(user_df)

#               # Predict probability
#               probability = model.predict_proba(user_scaled)[0][1] * 100
#               probability = round(probability, 2)  # Round to 2 decimal places

#               if probability > 70:
#                 graph_color='red'
#                 risk_level = "High Risk"
#               elif 40 < probability <= 70:
#                 graph_color= 'orange'
#                 risk_level = "Moderate Risk"
#               else:
#                 graph_color='green'
#                 risk_level = "low Risk"

              
#               # Redirect based on probability range
#               if probability > 70:
#                    return redirect(url_for('high_risk', name=user_name, probability=probability, graph_url=graph_url))
#               elif 40 < probability <= 70:
#                    return redirect(url_for('moderate_risk', name=user_name, probability=probability, graph_url=graph_url))
#               else:
#                    return redirect(url_for('low_risk', name=user_name, probability=probability, graph_url=graph_url))
#         except Exception as e:
#             return f"Error: {e}", 400
#      return render_template('predictionpage.html')

