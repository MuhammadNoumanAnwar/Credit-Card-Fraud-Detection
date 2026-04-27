from flask import Flask, render_template, request
from trainedModels import predict_fraud
app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    try:
        amount = float(request.form["amount"])
        transaction_type = request.form["transaction_type"]
        entry_mode = request.form["entry_mode"]
        time = int(request.form["time"])
        country = request.form["country"]
        if amount < 0 or time < 0:
            return render_template(
                "index.html",
                error="Amount and time must be non-negative."
            )
        
        prediction, fraud_prob = predict_fraud(
            amount, transaction_type, entry_mode, time, country
        )
        fraud_prob = round(fraud_prob * 100, 2)
        safe_prob = round(100 - fraud_prob, 2)
        result = "Fraud" if prediction == 1 else "Not Fraud"
        feature_contributions = {
            "Amount": 0.35,
            "Time": 0.25,
            "Transaction Type": 0.15,
            "Entry Mode": 0.10,
            "Country": 0.15
        }
        return render_template(
            "index.html",
            pred=result,
            fraud_prob=fraud_prob,
            safe_prob=safe_prob,
            feature_contributions=feature_contributions
        )
    except Exception as e:
        return render_template(
            "index.html",
            error="Invalid input. Please try again."
        )
if __name__ == "__main__":
    app.run(debug=True)