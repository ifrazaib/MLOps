import pandas as pd
import pickle
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


df = pd.read_csv("data/wine.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (Poly)": SVC(kernel='poly', probability=True),
    "Neural Network": MLPClassifier(
        hidden_layer_sizes=(50, 30),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )
}

metrics = {}

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)



mlflow.set_experiment("Mops Assignment 01")

for name, model in models.items():
    print(f"Training {name}...")

    with mlflow.start_run(run_name=name):
      
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        metrics[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        }

    
        model_path = f"models/{name.replace(' ', '_')}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        mlflow.log_params(model.get_params())
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        cm_path = f"results/confusion_matrix_{name.replace(' ', '_')}.png"
        plt.savefig(cm_path)
        plt.close()

        mlflow.log_artifact(cm_path)
        mlflow.sklearn.log_model(model, name.replace(" ", "_"))

with open("results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)


metrics_df = pd.DataFrame(metrics).T

plt.figure(figsize=(10, 6))
metrics_df.plot(kind="bar")
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=45)
plt.legend(loc="lower right")
plt.tight_layout()
comparison_path = "results/model_comparison_all_metrics.png"
plt.savefig(comparison_path)
plt.close()
mlflow.log_artifact(comparison_path)


for metric in ["accuracy", "precision", "recall", "f1_score"]:
    plt.figure(figsize=(8, 5))
    sns.barplot(x=metrics_df.index, y=metrics_df[metric])
    plt.title(f"{metric.capitalize()} Comparison Across Models")
    plt.ylabel(metric.capitalize())
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45)
    plt.tight_layout()
    metric_path = f"results/{metric}_comparison.png"
    plt.savefig(metric_path)
    plt.close()
    mlflow.log_artifact(metric_path)


print("\n Training complete! Models saved in /models and results in /results/")
print("\n=== Model Comparison Results ===")
for model_name, vals in metrics.items():
    print(f"\n{model_name}:")
    for k, v in vals.items():
        print(f"  {k}: {v:.4f}")


best_model_name = max(metrics, key=lambda m: metrics[m]["accuracy"])
print(f"\n Best Model: {best_model_name}")


mlflow.end_run()  #
with mlflow.start_run(run_name=f"{best_model_name}_Registration") as run:
    best_model = models[best_model_name]
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        registered_model_name="WineClassifier"
    )
    print(f" Registered {best_model_name} as 'WineClassifier' in MLflow Registry")
