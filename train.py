
import csv
import joblib
import numpy as np

from pathlib import Path
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Locate project files
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "gesture_data.csv"
MODEL_PATH = PROJECT_DIR / "gesture_model.pkl"

# Load training data
features = []
labels = []

with open(DATA_PATH, "r", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        coordinates = [
            float(row[f"feature_{i}"])
            for i in range(63)
        ]

        features.append(coordinates)
        labels.append(row["label"])

X = np.array(features, dtype=np.float32)
y = np.array(labels)

# Show how many samples we collected
print("Dataset summary:")
for gesture, count in Counter(y).items():
    print(f"{gesture}: {count} samples")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create the AI model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")

# Train the model
model.fit(X_train, y_train)

# Test the trained model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save trained model
joblib.dump(model, MODEL_PATH)

print("\nModel saved as gesture_model.pkl")
