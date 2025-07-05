import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import joblib
import os


df = sns.load_dataset("titanic")

df = df[["age", "fare", "sex", "pclass", "sibsp", "survived"]].dropna()
df["sex"] = df["sex"].map({"male": 1, "female": 0})

X = df[["age", "fare", "sex", "pclass", "sibsp"]]
y = df["survived"]


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

os.makedirs("../model_training", exist_ok=True)
joblib.dump(model, "../model_training/loan_model.pkl")
joblib.dump(X.columns.tolist(), "../model_training/train_columns.pkl")
joblib.dump(X.mean().to_dict(), "../model_training/train_mean.pkl")

print("Model, columns, and feature means saved.")
