import argparse
from pathlib import Path
import pandas as pd

from features import (
    basic_preprocess,
    add_rolling_features,
    add_session_features,
    add_station_baseline_features,
    build_feature_matrix
)

from model import load_model, apply_hard_rules


FEATURE_COLS = [
    # Raw signals
    'voltage', 'current', 'power_kw', 'temperature_c',
    'duration_sec', 'energy_kwh', 'error_code', 'has_error',

    # Derived
    'power_discrepancy', 'energy_rate', 'power_calc_kw',

    # Temporal
    'hour', 'dayofweek', 'is_weekend',

    # Rolling deltas
    'voltage_delta', 'current_delta', 'power_kw_delta', 'temperature_c_delta',
    'voltage_roll_std', 'current_roll_std', 'power_kw_roll_std', 'temperature_c_roll_std',

    # Session
    'session_error_rate', 'session_event_count', 'session_total_energy',
    'session_avg_power', 'session_max_temp',

    # Station z-scores
    'voltage_zscore', 'current_zscore', 'power_kw_zscore', 'temperature_c_zscore',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_path", default="model.joblib")
    args = parser.parse_args()

    # Load Original (Preserve Order)
    original = pd.read_csv(args.input)
    df = original.copy()

    # Keep track of original order
    df["__orig_index"] = df.index

    # Feature Engineering (sorting happens inside)
    df = basic_preprocess(df)
    df = add_rolling_features(df)
    df = add_session_features(df)
    df = add_station_baseline_features(df)

    # Build Feature Matrix
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = build_feature_matrix(df, feat_cols)

    # Load Model
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    model = load_model(args.model_path)

    # Model Prediction
    raw_preds = model.predict(X)              # -1 / 1
    scores = model.score_samples(X)           # continuous
    model_flags = (raw_preds == -1)

    # Apply hard rules explicitly
    rule_flags = apply_hard_rules(df)

    # Final anomaly label
    is_anomaly = (model_flags | rule_flags).astype(int)

    # Restore Original Row Order
    df["is_anomaly"] = is_anomaly
    df["anomaly_score"] = scores

    df = df.sort_values("__orig_index")

    # Align back to original
    original["is_anomaly"] = df["is_anomaly"].values
    original["anomaly_score"] = df["anomaly_score"].values

    # Save
    original.to_csv(args.output, index=False)

    print(f"Predictions saved → {args.output}")
    print(f"Flagged {original['is_anomaly'].sum():,} / {len(original):,} "
          f"({original['is_anomaly'].mean():.2%})")


if __name__ == "__main__":
    main()