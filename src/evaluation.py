import pandas as pd
import os
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 1️⃣ Charger le modèle
path = "models/pipeline_churn.pkl"
with open(path, "rb") as f:
    pipeline = pickle.load(f)

# 2️⃣ Charger les données test
data_path = "Data/processed/ML_data.csv"
df = pd.read_csv(data_path)

X_test = df.drop(columns=["Exited"])
Y_test = df["Exited"]

# 3️⃣ Prédictions
y_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_proba > 0.3).astype(int)


# Évaluation
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy :", accuracy)

confusion_m = confusion_matrix(Y_test, y_pred)
print("Confusion Matrix :\n", confusion_m)

classif_report = classification_report(Y_test, y_pred)
print("Classification Report :\n", classif_report)