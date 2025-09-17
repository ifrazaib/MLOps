import json
import matplotlib.pyplot as plt
import numpy as np

# Load metrics JSON
with open("results/metrics.json", "r") as f:
    metrics = json.load(f)

models = list(metrics.keys())
scores = {m: [] for m in ["accuracy", "precision", "recall", "f1_score"]}

for model_name in models:
    for metric in scores.keys():
        scores[metric].append(metrics[model_name][metric])

x = np.arange(len(models))
width = 0.2

plt.figure(figsize=(10,6))

plt.bar(x - 0.3, scores["accuracy"], width, label="Accuracy")
plt.bar(x - 0.1, scores["precision"], width, label="Precision")
plt.bar(x + 0.1, scores["recall"], width, label="Recall")
plt.bar(x + 0.3, scores["f1_score"], width, label="F1-score")

plt.xticks(x, models)
plt.ylabel("Score")
plt.title("Model Comparison on Wine Dataset")
plt.ylim(0, 1.1)
plt.legend()
plt.tight_layout()

# Save chart to results/
plt.savefig("results/comparison_chart.png")
plt.show()
