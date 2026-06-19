import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_json("cleaned_students.json")

# Fill missing values
df["attendance"] = df["attendance"].fillna(df["attendance"].mean())
df["study_hours"] = df["study_hours"].fillna(df["study_hours"].mean())

# Create pass/fail target
df["result"] = np.where(df["average"] >= 75, 1, 0)

# Features and target
X = df[["attendance", "study_hours"]]
y = df["result"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# Compare actual vs predicted
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})

print("\nActual vs Predicted:")
print(results)
