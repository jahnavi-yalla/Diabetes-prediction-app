import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load CSV WITHOUT headers
data = pd.read_csv("diabetes.csv", header=None)

# Assign correct column names
data.columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

# Split features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "best_model.joblib")

print("Model fixed and saved successfully")
