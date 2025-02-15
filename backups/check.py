import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# dataset add kiya
try:
    dataset_path = r"heartdiseasepredicModify\new_modified_heart_disease_data.csv"
    columns = [
        'age', 'sex', 'smoking_status', 'exercise_frequency', 'alcohol_consumption', 'diet', 'high_blood_pressure',
        'high_cholesterol', 'diabetes', 'heart_conditions', 'family_history_heart_disease', 'shortness_of_breath', 'palpitations', 'target'
    ]
    data = pd.read_csv(dataset_path, header=None, names=columns)
    
# data correctness check kiya
    data = data.apply(pd.to_numeric, errors='coerce')
    
#missing values
    data.dropna(inplace=True)
    
# Prepare data for training
    X = data.drop('target', axis=1)
    y = data['target'].apply(lambda x: 1 if x > 0 else 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
# Train kiya model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
#Check Model accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
except Exception as e:
    print(f"Error loading or processing dataset: {e}")
    exit()

# User inputs add
print("\nPlease answer the following health-related questions:")
try:
    user_data = {}
    input_prompts = {
        "age": "What is your age? ",
        "sex": "Are you male or female? (Enter 1 for Male, 0 for Female): ",
        "smoking_status": "Do you smoke? (Enter 1 for Yes, 0 for No): ",
        "exercise_frequency": "How often do you exercise? (0-4 scale): ",
        "alcohol_consumption": "How often do you drink alcohol? (0-4 scale): ",
        "diet": "How healthy is your diet? (0-4 scale): ",
        "high_blood_pressure": "Do you have high blood pressure? (1 for Yes, 0 for No, -1 for Don't Know): ",
        "high_cholesterol": "Do you have high cholesterol? (1 for Yes, 0 for No, -1 for Don't Know): ",
        "diabetes": "Do you have diabetes? (1 for Yes, 0 for No, -1 for Don't Know): ",
        "heart_conditions": "Do you have any heart conditions? (1 for Yes, 0 for No, -1 for Don't Know): ",
        "family_history_heart_disease": "Family history of heart disease? (1 for Yes, 0 for No, -1 for Don't Know): ",
        "shortness_of_breath": "Do you experience shortness of breath? (1 for Yes, 0 for No): ",
        "palpitations": "Do you experience that your heart is thumping, racing or fluttering? (1 for Yes, 0 for No): "
    }
    
    for key, prompt in input_prompts.items():
        while True:
            try:
                user_data[key] = int(input(prompt))
                break
            except ValueError:
                print("Invalid input! Please enter a valid number.")
    
# Convert user input to DataFrame
    user_df = pd.DataFrame([user_data], columns=X.columns)
    
# Predict kari probability ko
    probability = model.predict_proba(user_df)[0][1] * 100
    print(f"\nYour estimated risk of heart disease is: {probability:.2f}%")
    
#Display matplot lib 
    labels = ['No Heart Disease', 'Heart Disease']
    probabilities = [100 - probability, probability]
    colors = ['green', 'red']
    plt.figure(figsize=(6, 6))
    plt.pie(probabilities, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
    plt.title('Heart Disease Risk Probability')
    plt.show()
    
# Provide health suggestions
    if probability > 70:
        print("\n⚠ High risk! Consult a doctor immediately.")
    elif 40 < probability <= 70:
        print("\n⚠ Moderate risk! Consider lifestyle changes and consult a doctor.")
    else:
        print("\n✅ Low risk! Maintain a healthy lifestyle.")
    
    if probability > 40:
        print("\n💡 Health Tips:")
        print("- 🥗 Eat a balanced diet.")
        print("- 🚶‍♂ Exercise regularly.")
        print("- 🚭 Avoid smoking and excessive alcohol.")
        print("- 🧘‍♀ Manage stress.")
        print("- ⚖ Maintain a healthy weight.")
except Exception as e:
    print(f"\n❌ Error: {e}")

pickle.dump(model,open("Modifiedmodel.pkl","wb"))
