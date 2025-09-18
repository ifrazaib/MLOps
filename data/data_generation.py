
import pandas as pd
from sklearn.datasets import load_wine


wine = load_wine(as_frame=True)
df = wine.frame  


df.to_csv("data/wine.csv", index=False)
print("Wine dataset saved to data/wine.csv")
