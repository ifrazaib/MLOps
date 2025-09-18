# MLOps Assignment 1 – Wine Dataset Model Tracking with MLflow

## 📌 Problem Statement
The goal of this assignment is to:
- Train and compare multiple ML models on the Wine dataset.
- Track experiments, metrics, and artifacts using MLflow.
- Register the best-performing model in the MLflow Model Registry.
- Maintain reproducibility with GitHub and structured project workflow.

---

## 📊 Dataset
- **Dataset:** Wine dataset (from scikit-learn).
- **Features:** 13 numerical features describing wine chemical properties.
- **Target:** Wine type (3 classes).

---

## ⚙️ Project Structure
mlops-assignment-1/
├── data/ # dataset
│ └── wine.csv
├── models/ # saved trained models (.pkl)
├── notebooks/ # (optional) exploratory notebooks
├── results/ # metrics.json + charts
├── src/ # training + MLflow scripts
│ ├── save_dataset.py
│ └── ml_flow_train.py
└── README.md # documentation
## 🤖 Models Trained
1. Logistic Regression  
2. Random Forest  
3. SVM (Polynomial Kernel)  
4. Neural Network (MLPClassifier)  

---

## 📈 Results

The models were compared on **accuracy, precision, recall, F1-score**.

Example:

| Model               | Accuracy | Precision | Recall | F1-score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.9722   | 0.9722    | 0.9722 | 0.9722   |
| Random Forest       | 1.0000   | 1.0000    | 1.0000 | 1.0000   |
| SVM (Poly)          | 0.9444   | 0.9444    | 0.9444 | 0.9444   |
| Neural Network      | 0.6389   | 0.4613    | 0.6389 | 0.5357   |

---

## 📊 Visualizations

- Model comparison charts (`accuracy`, `precision`, `recall`, `f1_score`)
- Stored in `/results` folder  

**Example Charts:**
- `results/model_comparison_all_metrics.png`  
- `results/accuracy_comparison.png`  
- `results/precision_comparison.png`  
- `results/recall_comparison.png`  
- `results/f1_score_comparison.png`  

---

## 📦 MLflow Tracking
- All experiments logged with parameters, metrics, and artifacts.
- Launch UI with:
  ```bash
  mlflow ui
  
