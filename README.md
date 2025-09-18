# 🍷 Wine Classification with MLflow – MLOps Assignment 1

## 1️⃣ Problem Statement
The goal of this assignment is to build, train, and compare multiple machine learning models on the **Wine dataset**.  
We aim to:
- Identify the best-performing model.  
- Track all experiments using **MLflow**.  
- Save metrics, plots, and model artifacts.  
- Register the best model in **MLflow Model Registry**.  

---

## 2️⃣ Dataset Description
- **Dataset Name:** Wine dataset (from scikit-learn)  
- **Number of samples:** 178  
- **Features:** 13 physicochemical attributes of wine samples (e.g., alcohol, malic acid, flavanoids).  
- **Target:** Wine class (3 categories)  

Dataset saved as `data/wine.csv` via `src/save_dataset.py`.

---

## 3️⃣ Model Selection & Comparison
We trained four models:

| Model               | Description |
|--------------------|-------------|
| Logistic Regression | Linear model for classification |
| Random Forest       | Ensemble of decision trees |
| SVM (Polynomial)    | Non-linear classification using polynomial kernel |
| Neural Network      | MLPClassifier with hidden layers (50, 30) |

**Evaluation Metrics:**
- Accuracy  
- Precision (weighted)  
- Recall (weighted)  
- F1-score (weighted)

**Example Results:**

| Model               | Accuracy | Precision | Recall | F1-score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.9722   | 0.9722    | 0.9722 | 0.9722   |
| Random Forest       | 1.0000   | 1.0000    | 1.0000 | 1.0000   |
| SVM (Poly)          | 0.9444   | 0.9444    | 0.9444 | 0.9444   |
| Neural Network      | 0.6389   | 0.4613    | 0.6389 | 0.5357   |

Charts for comparison are saved in `/results` folder.

---

## 4️⃣ MLflow Logging
All model runs were tracked in MLflow:  
- Logged **parameters** (hyperparameters)  
- Logged **metrics** (accuracy, precision, recall, F1-score)  
- Logged **artifacts** (comparison charts, confusion matrices, trained models)

### MLflow UI Screenshots (Placeholders)
- **Runs Comparison:**
![MLflow Runs](results/screenshots/mlflow_runs.png)  
- **Metrics Visualization:**
![MLflow Metrics](results/screenshots/mlflow_metrics.png)  
- **Confusion Matrices / Plots:**
![Artifacts](results/screenshots/artifacts.png)

---

## 5️⃣ Best Model Selection & Registration
- Best model determined by **highest accuracy** → Random Forest in this example.  
- Registered in **MLflow Model Registry** as `WineClassifier`.

### Registration Screenshot Placeholder
![Registered Model](results/screenshots/mlflow_registry.png)

**Code snippet for registration:**

mlflow.end_run()  # ensure no active run
with mlflow.start_run(run_name=f"{best_model_name}_Registration") as run:
    best_model = models[best_model_name]
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        registered_model_name="WineClassifier"
    )
--- 
