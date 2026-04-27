import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib
import os
DATA_PATH = "CreditCardData.csv"
data = pd.read_csv(DATA_PATH)
required_columns = [
    "Amount",
    "Type of Transaction",
    "Entry Mode",
    "Time",
    "Country of Transaction",
    "Fraud"
]
data = data[required_columns]
data['Amount'] = data['Amount'].replace('[£$,]', '', regex=True).astype(float)
X = data.drop("Fraud", axis=1)
y = data["Fraud"]
numeric_features = ["Amount", "Time"]
categorical_features = [
    "Type of Transaction",
    "Entry Mode",
    "Country of Transaction"
]
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(eval_metric='logloss'))
])
print("Training model...")
model_pipeline.fit(X_train, y_train)
print("Training complete!")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")
joblib.dump(model_pipeline, MODEL_PATH)
print(f"Model saved at: {MODEL_PATH}")