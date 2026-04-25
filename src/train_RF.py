import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from model import get_pipeline  # ton pipeline défini dans model.py

# 1️⃣ Chemin du dataset
data_path = "Data/processed/ML_data.csv"

# 2️⃣ Lire le CSV avec encodage Windows (si UTF-8 plante)
df = pd.read_csv(data_path, encoding='cp1252')  # ou 'latin1' si ça plante encore

# Vérifier les colonnes
print(df.head())

# 3️⃣ Séparer features et cible
X = df.drop(columns=["Exited"])
y = df["Exited"]

# 4️⃣ Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5️⃣ Obtenir le pipeline
pipeline = get_pipeline()

# 6️⃣ Entraîner le pipeline
pipeline.fit(X_train, y_train)

# 7️⃣ Sauvegarder le pipeline
model_path = "models/pipeline_churn.pkl"
with open(model_path, "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Pipeline sauvegardé à :", os.path.abspath(model_path))