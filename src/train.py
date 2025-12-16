import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
import os

# Print current directory for debugging
print(f"Current working directory: {os.getcwd()}")
print(f"Checking if data/dataset.csv exists: {os.path.exists('data/dataset.csv')}")

# Load data
data = pd.read_csv('data/dataset.csv')
print(f"Data loaded. Shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
print(f"First 3 rows:\n{data.head(3)}")

# Define feature and target columns based on your dataset
feature_cols = ['feature1', 'feature2', 'feature3']
target_col = 'label'

print(f"\nUsing feature columns: {feature_cols}")
print(f"Using target column: {target_col}")

X = data[feature_cols].values
y = data[target_col].values

print(f"\nX shape: {X.shape}, y shape: {y.shape}")

# Train a simple model
model = LinearRegression()
model.fit(X, y)
print(f"\nModel trained successfully!")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_:.2f}")

# Save model
os.makedirs('models', exist_ok=True)  # Ensure models directory exists
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\nModel saved to models/model.pkl")

# Make a sample prediction
sample = X[0:1]  # First row
prediction = model.predict(sample)
print(f"\nSample prediction for first row: {prediction[0]:.2f}")
print(f"Actual value for first row: {y[0]}")