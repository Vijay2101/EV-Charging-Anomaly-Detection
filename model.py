import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ZSCORE_THRESHOLD = 3.5
POWER_DISC_THRESH = 2.0
CONTAMINATION = 0.03


def apply_hard_rules(df):
    flags = np.zeros(len(df), dtype=bool)

    flags |= df['has_error'].astype(bool).values
    flags |= (df['power_kw'] < 0).values

    for col in ['voltage_zscore', 'current_zscore', 'temperature_c_zscore']:
        if col in df.columns:
            flags |= (df[col].abs() > ZSCORE_THRESHOLD).values

    if 'power_discrepancy' in df.columns:
        flags |= (df['power_discrepancy'] > POWER_DISC_THRESH).values

    return flags


def build_model():
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('iforest', IsolationForest(
            n_estimators=200,
            contamination=CONTAMINATION,
            random_state=42,
            n_jobs=-1
        ))
    ])
    return model


def train_model(X, model_path="model.joblib"):
    model = build_model()
    model.fit(X)
    joblib.dump(model, model_path)
    return model


def load_model(model_path="model.joblib"):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"{model_path} not found.")
    return joblib.load(model_path)


def predict(model, X, df):
    preds = model.predict(X)
    scores = model.score_samples(X)

    model_flags = (preds == -1)
    rule_flags = apply_hard_rules(df)

    is_anomaly = (model_flags | rule_flags).astype(int)

    return is_anomaly, scores