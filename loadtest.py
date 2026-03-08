import pandas as pd

df=pd.read_csv("train.csv")

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)
print("\nColumns: ")
print(df.columns)
print("\nFirst 5 rows: ")
print(df.head())