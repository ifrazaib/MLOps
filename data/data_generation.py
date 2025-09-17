# src/save_dataset.py
import pandas as pd
from sklearn.datasets import load_wine

# Load Wine dataset
wine = load_wine(as_frame=True)
df = wine.frame  # includes features + target column

# Save to CSV
df.to_csv("data/wine.csv", index=False)
print("Wine dataset saved to data/wine.csv")
