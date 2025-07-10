# coding: utf-8

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, render_template
import pickle

app = Flask("__name__")

# Load base training dataset
df_1 = pd.read_csv("first_telc.csv")




# Drop auto-added index column if present
if 'Unnamed: 0' in df_1.columns:
    df_1.drop(columns=['Unnamed: 0'], inplace=True)




@app.route("/")
def loadPage():
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def predict():
    # Load model
    model = pickle.load(open("model.sav", "rb"))

    # Read user inputs
    inputs = [request.form[f'query{i}'] for i in range(1, 20)]

    # Build dataframe from form data
    input_columns = ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender',
                     'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                     'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                     'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                     'PaymentMethod', 'tenure']

    new_df = pd.DataFrame([inputs], columns=input_columns)

    # Append to original dataset
    df_2 = pd.concat([df_1, new_df], ignore_index=True)

    # Convert numeric columns BEFORE further processing
    numeric_cols = ['MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'tenure']
    df_2[numeric_cols] = df_2[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Group tenure into ranges
    tenure_labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    df_2['tenure_group'] = pd.cut(df_2['tenure'], range(1, 80, 12), right=False, labels=tenure_labels)

    # Drop unused columns
    df_2.drop(columns=['tenure'], inplace=True)

    # One-hot encode
    df_dummies = pd.get_dummies(df_2)

    # Fill any missing values
    df_dummies.fillna(0, inplace=True)

    # Align with model training columns (important if training used different dummy columns)
    # Load training column names used in model
    try:
        with open("model_columns.pkl", "rb") as f:
            model_columns = pickle.load(f)
        # Add any missing columns
        for col in model_columns:
            if col not in df_dummies.columns:
                df_dummies[col] = 0
        # Reorder to match model
        df_dummies = df_dummies[model_columns]
    except:
        pass  # If you didn’t save model_columns.pkl, this block will be skipped

    # Get last row (new input)
    input_data = df_dummies.tail(1)

    # Make prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1]

    if prediction[0] == 1:
        o1 = "This customer is likely to be churned!!"
    else:
        o1 = "This customer is likely to continue!!"

    o2 = f"Confidence: {probability[0]*100:.2f}%"

    return render_template('home.html', output1=o1, output2=o2,
                           **{f'query{i}': request.form[f'query{i}'] for i in range(1, 20)})

app.run(debug=True, port=8000)
