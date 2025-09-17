import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import json
import os

# Load dataset
df = pd.read_csv("data/wine.csv")
X = df.drop(columns=["target"])
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Models to train
models = {
    "logistic_regression": LogisticRegression(max_iter=200),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "svm": SVC(kernel="linear", probability=True, random_state=42)
}

metrics = {}

# Ensure models folder exists
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Train and evaluate each model
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Save metrics
    metrics[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }

    # Save model
    with open(f"models/{name}.pkl", "wb") as f:
        pickle.dump(model, f)

# Save metrics to JSON
with open("results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Training complete! Models saved in /models and metrics in /results/metrics.json")
print("\n=== Model Comparison Results ===")
for model_name, vals in metrics.items():
    print(f"\n{model_name}:")
    for k, v in vals.items():
        print(f"  {k}: {v:.4f}")

