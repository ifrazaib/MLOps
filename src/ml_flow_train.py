import pandas as pd
import pickle
import os
import json
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv("data/wine.csv")   # <- Wine dataset
X = df.drop(columns=["target"])
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models dictionary
models = {
    "logistic_regression": LogisticRegression(max_iter=200),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "svm": SVC(kernel="linear", probability=True, random_state=42)
}

metrics = {}

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Start MLflow experiment
mlflow.set_experiment("Wine-Model-Comparison")

for name, model in models.items():
    print(f"\n=== Training {name} ===")

    with mlflow.start_run(run_name=name):
        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        metrics[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}

        # Print metrics in terminal
        print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}")

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Log parameters
        if name == "logistic_regression":
            mlflow.log_param("max_iter", 200)
        elif name == "random_forest":
            mlflow.log_param("n_estimators", 100)
        elif name == "svm":
            mlflow.log_param("kernel", "linear")

        # Save pickle model
        model_path = f"models/{name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Log model artifact
        mlflow.sklearn.log_model(model, name + "_model")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        cm_path = f"results/{name}_confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

# Save metrics JSON
with open("results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# === Grouped Bar Chart of Metrics ===
models_list = list(metrics.keys())
scores = {m: [] for m in ["accuracy", "precision", "recall", "f1_score"]}

for model_name in models_list:
    for metric in scores.keys():
        scores[metric].append(metrics[model_name][metric])

x = np.arange(len(models_list))
width = 0.2

plt.figure(figsize=(10,6))
plt.bar(x - 0.3, scores["accuracy"], width, label="Accuracy")
plt.bar(x - 0.1, scores["precision"], width, label="Precision")
plt.bar(x + 0.1, scores["recall"], width, label="Recall")
plt.bar(x + 0.3, scores["f1_score"], width, label="F1-score")

plt.xticks(x, models_list)
plt.ylabel("Score")
plt.title("Model Comparison on Wine Dataset")
plt.ylim(0, 1.1)
plt.legend()
plt.tight_layout()

chart_path = "results/comparison_chart.png"
plt.savefig(chart_path)
plt.close()

# Log chart as artifact in MLflow
with mlflow.start_run(run_name="comparison_chart"):
    mlflow.log_artifact(chart_path)

print("\nTraining complete! Metrics saved in results/metrics.json, models in /models, and all logs in MLflow UI.")
