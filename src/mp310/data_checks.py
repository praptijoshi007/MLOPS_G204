
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/raw/train.csv")

df = pd.read_csv(DATA_PATH)
print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== FIRST FIVE ROWS ==========")
print(df.head())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== NUMERICAL SUMMARY ==========")
print(df.describe(include="all").T)

print("\n========== UNIQUE VALUES ==========")
for col in df.columns:
    print(f"\n{col}: {df[col].nunique(dropna=False)} unique values")
