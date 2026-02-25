import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

joblib.dump(model, "best_model.joblib")
