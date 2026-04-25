from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def get_pipeline():
    """
    Renvoie un pipeline complet avec scaling + Random Forest
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # met toutes les features à la même échelle
        ('rfc', RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42
        ))
    ])
    return pipeline


