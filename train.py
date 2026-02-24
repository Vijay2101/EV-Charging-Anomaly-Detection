import argparse
import pandas as pd
import joblib

from features import (
    basic_preprocess,
    add_rolling_features,
    add_session_features,
    add_station_baseline_features,
    build_feature_matrix
)

from model import train_model, predict


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
    parser.add_argument("--model_path", default="model.joblib")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    # Feature Engineering
    df = basic_preprocess(df)
    df = add_rolling_features(df)
    df = add_session_features(df)
    df = add_station_baseline_features(df)

    X = build_feature_matrix(df, FEATURE_COLS)

    # Train
    model = train_model(X, args.model_path)
    print(f"Model saved → {args.model_path}")

    # Predict on training data
    is_anomaly, scores = predict(model, X, df)
    df['is_anomaly'] = is_anomaly
    df['anomaly_score'] = scores

    # Save anomaly summary
    summary = df[df['is_anomaly'] == 1].sort_values('anomaly_score')[
        ['station_id', 'timestamp', 'session_id',
         'voltage', 'current', 'power_kw', 'temperature_c',
         'error_code', 'message', 'anomaly_score', 'is_anomaly']
    ]

    summary.to_csv('anomaly_summary.csv', index=False)
    print(f"Anomaly summary saved → anomaly_summary.csv ({len(summary):,} rows)")

    df.to_csv('charging_logs_with_predictions.csv', index=False)
    print("Full predictions saved → charging_logs_with_predictions.csv")


if __name__ == "__main__":
    main()