import joblib
import pandas as pd
model = joblib.load("models/xgboost_model.pkl")
def predict_fraud(amount, transaction_type, entry_mode, time, country):
    input_df = pd.DataFrame([{
        "Amount": amount,
        "Type of Transaction": transaction_type,
        "Entry Mode": entry_mode,
        "Time": time,
        "Country of Transaction": country
    }])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    return prediction, probability