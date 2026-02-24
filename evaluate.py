import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from features import (
    basic_preprocess,
    add_rolling_features,
    add_session_features,
    add_station_baseline_features,
    build_feature_matrix,
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

CONTAMINATION = 0.03


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model_path", default="model.joblib")
    parser.add_argument("--output_dir", default="evaluation_outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    df = pd.read_csv(args.input)

    # Feature Engineering
    df = basic_preprocess(df)
    df = add_rolling_features(df)
    df = add_session_features(df)
    df = add_station_baseline_features(df)

    X = build_feature_matrix(df, FEATURE_COLS)

    model = load_model(args.model_path)

    raw_preds = model.predict(X)
    scores = model.score_samples(X)
    model_flags = (raw_preds == -1)

    rule_flags = apply_hard_rules(df)
    is_anomaly = (model_flags | rule_flags).astype(int)

    df['is_anomaly'] = is_anomaly
    df['anomaly_score'] = scores

    # Anomaly Score Distribution + Station Rate
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    normal_scores = scores[is_anomaly == 0]
    anomaly_scores = scores[is_anomaly == 1]

    axes[0].hist(normal_scores, bins=80, alpha=0.7,
                 color='#55A868', label='Normal', density=True)
    axes[0].hist(anomaly_scores, bins=80, alpha=0.7,
                 color='#C44E52', label='Anomaly', density=True)
    axes[0].set_xlabel('Anomaly Score (lower = more anomalous)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Score Distribution by Label', fontweight='bold')
    axes[0].legend()

    station_anomaly = df.groupby('station_id')['is_anomaly'] \
        .mean().sort_values(ascending=False).head(20)

    station_anomaly.plot(kind='bar', ax=axes[1],
                         color='#C44E52', edgecolor='none')
    axes[1].set_title('Anomaly Rate by Station (Top 20)', fontweight='bold')
    axes[1].set_ylabel('Fraction Flagged')
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "score_distribution_and_station_rate.png")
    plt.show()

    # Hour Rate + Contamination Sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    df.groupby('hour')['is_anomaly'].mean().plot(
        ax=axes[0], color='#C44E52', marker='o', linewidth=2)
    axes[0].set_title('Anomaly Rate by Hour of Day', fontweight='bold')
    axes[0].set_xlabel('Hour (UTC)')
    axes[0].set_ylabel('Fraction Flagged')
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    contamination_vals = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
    flagged_counts = []

    for c in contamination_vals:
        m = Pipeline([
            ('scaler', StandardScaler()),
            ('iforest', IsolationForest(
                n_estimators=100,
                contamination=c,
                random_state=42,
                n_jobs=-1))
        ]).fit(X)

        preds = m.predict(X)
        combined = ((preds == -1) | rule_flags)
        flagged_counts.append(combined.mean() * 100)

    axes[1].plot([c*100 for c in contamination_vals],
                 flagged_counts,
                 marker='o',
                 color='#4C72B0',
                 linewidth=2)

    axes[1].set_xlabel('Contamination Parameter (%)')
    axes[1].set_ylabel('% Events Flagged')
    axes[1].set_title('Sensitivity to Contamination Parameter', fontweight='bold')
    axes[1].axvline(CONTAMINATION*100,
                    color='red',
                    linestyle='--',
                    label=f'Current ({CONTAMINATION*100:.0f}%)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "contamination_sensitivity.png")
    plt.show()

    # Top Anomalous Events
    top_anomalies = (
        df[df['is_anomaly'] == 1]
        .sort_values('anomaly_score')
        [['station_id', 'timestamp', 'session_id',
          'voltage', 'current', 'power_kw', 'temperature_c',
          'error_code', 'message', 'anomaly_score']]
        .head(20)
    )

    print(f'\nTotal anomalous events: {df["is_anomaly"].sum():,}')
    print('\nTop 20 most anomalous events:')
    print(top_anomalies)

    # Feature Importance
    score_corr = X.corrwith(
        pd.Series(scores, name='anomaly_score')
    ).sort_values()

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#C44E52' if v < 0 else '#55A868' for v in score_corr]

    score_corr.plot(kind='barh', ax=ax,
                    color=colors, edgecolor='none')

    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(
        'Feature Correlation with Anomaly Score\n'
        '(negative = drives anomaly detection)',
        fontweight='bold'
    )
    ax.set_xlabel('Pearson Correlation with Anomaly Score')

    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png")
    plt.show()

    print('\nTop 5 features driving anomalies (most negative correlation):')
    print(score_corr.head(5))

    # Anomaly Breakdown
    model_only = model_flags & ~rule_flags
    rules_only = rule_flags & ~model_flags
    both = model_flags & rule_flags

    print('\nAnomaly trigger breakdown:')
    print(f'  Model only (subtle patterns) : {model_only.sum():>7,}  ({model_only.mean():.2%})')
    print(f'  Rules only (known faults)    : {rules_only.sum():>7,}  ({rules_only.mean():.2%})')
    print(f'  Both model + rules           : {both.sum():>7,}  ({both.mean():.2%})')
    print(f'  Total anomalous              : {is_anomaly.sum():>7,}  ({is_anomaly.mean():.2%})')

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ['Model only', 'Rules only', 'Both']
    sizes = [model_only.sum(), rules_only.sum(), both.sum()]
    colors = ['#4C72B0', '#DD8452', '#C44E52']

    ax.pie(sizes, labels=labels,
           colors=colors,
           autopct='%1.1f%%',
           startangle=90)
    ax.set_title('Anomaly Source Breakdown', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / "anomaly_source_breakdown.png")
    plt.show()


    # Explainability
    known_fault_mask = (
        (df['has_error'] == 1) |
        (df['power_discrepancy'] > 2.0) |
        (df['power_kw'] < 0)
    )

    total_anomalies = df['is_anomaly'].sum()
    known_fault_anomalies = df.loc[known_fault_mask, 'is_anomaly'].sum()

    print("\n=== Anomaly Explainability ===")
    print(f"Total anomalies detected        : {total_anomalies:,}")
    print(f"Anomalies matching known faults : {known_fault_anomalies:,}")
    print(f"Fraction explained by rules     : {known_fault_anomalies / total_anomalies:.2%}")

    error_mask = df['has_error'] == 1
    disc_mask = df['power_discrepancy'] > 2.0

    total_error_events = error_mask.sum()
    total_disc_events = disc_mask.sum()

    model_error_detected = (model_flags & error_mask).sum()
    model_disc_detected = (model_flags & disc_mask).sum()

    print("\n=== Model-Only Detection (No Rules) ===")
    print(f"Firmware errors detected by model only : {model_error_detected / total_error_events:.2%}")
    print(f"Power discrepancy >2 detected by model : {model_disc_detected / total_disc_events:.2%}")

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()