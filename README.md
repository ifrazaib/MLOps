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
<img width="1000" height="600" alt="comparison_chart" src="https://github.com/user-attachments/assets/97b3eaac-e455-4096-9c2d-a4c8cdc8cc28" />

- **Metrics Visualization:**
<img width="500" height="400" alt="random_forest_confusion_matrix" src="https://github.com/user-attachments/assets/a6908715-eb30-448c-ae16-3d87834dad5c" />

- **Confusion Matrices / Plots:**
<img width="500" height="400" alt="logistic_regression_confusion_matrix" src="https://github.com/user-attachments/assets/11bdcc34-9efb-4d44-b36e-6a32fc8e08c9" />


---

## 5️⃣ Best Model Selection & Registration
- Best model determined by **highest accuracy** → Random Forest in this example.  
- Registered in **MLflow Model Registry** as `WineClassifier`.

### Registration Screenshot Placeholder
<img width="975" height="494" alt="image" src="https://github.com/user-attachments/assets/f3fae75e-c500-4a40-ba23-e2c1a30628a6" />
<img width="975" height="496" alt="image" src="https://github.com/user-attachments/assets/35d4ec31-8d4d-4fa9-8d34-de4124a26ae5" />


----

- mlflow.end_run()  # ensure no active run
- with mlflow.start_run(run_name=f"{best_model_name}_Registration") as run:
-  best_model = models[best_model_name]
-  mlflow.sklearn.log_model(
-       sk_model=best_model,
-      artifact_path="model",
-    registered_model_name="WineClassifier"
-    )
     --- 
## 📂 Dataset
- **Dataset Name:** Wine dataset (`data/wine.csv`)  
- **Target:** `target` column (wine classes)  
- **Features:** Various physicochemical properties of wine samples.  

---

## ⚙️ Project Structure
- <img width="447" height="284" alt="image" src="https://github.com/user-attachments/assets/a72baa97-b04a-45ec-9704-a8c8a7fc5245" />
---
## 🚀 How to Run the Project

### 1️⃣ Clone Repository
- git clone <your-repo-link>
- cd MLOps_Wine_Project
## 2️⃣ Create Environment
- pip install -r requirements.txt
- conda env create -f environment.yml
- conda activate mlops-assignment
## 3️⃣ Run Training
- python src/ml_flow_train.py
## 4️⃣ Start MLflow UI
- mlflow ui
- Open http://127.0.0.1:5000 -> to explore runs and compare models.
---

## 📊 MLflow Tracking
- Each run logs:

- Parameters: model hyperparameters

- Metrics: accuracy, precision, recall, F1-score

- Artifacts: plots (comparison charts, per-metric barplots), confusion matrices, model .pkl files

- Registered Model: best-performing model is registered in MLflow Model Registry with version tracking
---

## Best Model Selection

- After all runs, the script selects the best-performing model based on accuracy.

- This model is registered in MLflow with version control.
---
## 1️⃣1️⃣ Conclusion
- Summarize results: best model, metrics, MLflow tracking, and model registration.

- Optionally mention reproducibility and GitHub repository link.
## 1️⃣2️⃣ Credits
- This project was entirely developed and implemented by Ifra Zaib .
- All code, MLflow setup, training, evaluation, and documentation are done by the author.

