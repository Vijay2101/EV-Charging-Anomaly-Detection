import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
from pathlib import Path


def load_and_preprocess(path: str) -> pd.DataFrame:
    """
    Load CSV, parse timestamps, sort, and add derived columns:
    - power_calc_kw   : V * I / 1000
    - power_discrepancy : |reported - calculated| power
    - has_error       : 1 if error_code != 0
    - energy_rate     : kWh / duration_sec
    """
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df = df.sort_values(['station_id', 'timestamp']).reset_index(drop=True)

    df['power_calc_kw'] = (df['voltage'] * df['current']) / 1000.0
    df['power_discrepancy'] = (df['power_kw'] - df['power_calc_kw']).abs()
    df['has_error'] = (df['error_code'] != 0).astype(int)
    df['energy_rate'] = df['energy_kwh'] / df['duration_sec'].clip(lower=1)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="eda_outputs")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    df = load_and_preprocess(args.input)

    print(f'Shape: {df.shape}')
    print('\nDtypes:')
    print(df.dtypes)
    print('\nMissing values:')
    print(df.isnull().sum())
    print('\nDescribe:')
    print(df.describe())
    print('\nError events describe:')
    print(df[df['error_code'] != 0].describe())

    # 1. Distributions
    numeric_cols = [
        'voltage', 'current', 'power_kw',
        'temperature_c', 'duration_sec', 'energy_kwh'
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, col in zip(axes.flatten(), numeric_cols):
        ax.hist(df[col].dropna(), bins=80, color='#4C72B0', alpha=0.85)
        ax.set_title(col, fontsize=12, fontweight='bold')

        q1, q99 = df[col].quantile([0.01, 0.99])
        ax.axvline(q1, color='red', linestyle='--', linewidth=1)
        ax.axvline(q99, color='orange', linestyle='--', linewidth=1)

    fig.suptitle(
        'Feature Distributions (red=1st pct, orange=99th pct)',
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(output_dir / "feature_distributions.png")
    plt.show()

    # 2. Correlation Heatmap
    corr_cols = numeric_cols + [
        'power_calc_kw', 'power_discrepancy',
        'energy_rate', 'has_error'
    ]

    corr = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        linewidths=0.5,
        vmin=-1,
        vmax=1,
        mask=mask,
        ax=ax
    )
    ax.set_title('Pearson Correlation Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png")
    plt.show()

    # 3. Temporal Patterns
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    df.groupby('hour')['power_kw'].mean().plot(
        ax=axes[0], color='#DD8452', marker='o', linewidth=2
    )
    axes[0].set_title('Avg Power by Hour of Day', fontweight='bold')
    axes[0].set_xlabel('Hour (UTC)')
    axes[0].set_ylabel('kW')

    day_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    df.groupby('dayofweek')['power_kw'].mean().rename(
        index=dict(enumerate(day_names))
    ).plot(ax=axes[1], color='#55A868', marker='o', linewidth=2)

    axes[1].set_title('Avg Power by Day of Week', fontweight='bold')
    axes[1].set_xlabel('Day')

    df.groupby('hour')['has_error'].mean().plot(
        ax=axes[2], color='#C44E52', marker='o', linewidth=2
    )
    axes[2].set_title('Error Rate by Hour of Day', fontweight='bold')
    axes[2].set_xlabel('Hour (UTC)')
    axes[2].set_ylabel('Fraction with error')
    axes[2].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    plt.tight_layout()
    plt.savefig(output_dir / "temporal_patterns.png")
    plt.show()

    # 4. Error Code Analysis
    errors = df[df['error_code'] != 0]

    print(f'\nTotal events with errors: {len(errors):,} ({len(errors)/len(df):.2%})')
    print(f'Unique error codes: {errors["error_code"].nunique()}')
    print('\nTop error codes:')
    print(errors['error_code'].value_counts().head(10))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    errors['error_code'].value_counts().head(15).plot(
        kind='bar',
        ax=axes[0],
        color='#C44E52'
    )
    axes[0].set_title('Top 15 Error Codes', fontweight='bold')

    station_err = df.groupby('station_id')['has_error'].mean() \
                     .sort_values(ascending=False).head(20)

    station_err.plot(
        kind='bar',
        ax=axes[1],
        color='#8172B2'
    )
    axes[1].set_title('Error Rate by Station (Top 20)', fontweight='bold')
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    plt.tight_layout()
    plt.savefig(output_dir / "error_analysis.png")
    plt.show()

    # 5. Power vs Temperature + Discrepancy
    sample = df.sample(min(15000, len(df)), random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sc = axes[0].scatter(
        sample['temperature_c'],
        sample['power_kw'],
        c=sample['has_error'],
        cmap='RdYlGn_r',
        alpha=0.4,
        s=6
    )

    plt.colorbar(sc, ax=axes[0], label='has_error')
    axes[0].set_xlabel('Temperature (°C)')
    axes[0].set_ylabel('Power (kW)')
    axes[0].set_title('Power vs Temperature (green=OK, red=fault)',
                      fontweight='bold')

    axes[1].hist(
        df['power_discrepancy'].clip(0, 5),
        bins=100,
        color='#4C72B0',
        alpha=0.85
    )
    axes[1].axvline(2.0, color='red', linestyle='--', linewidth=2)
    axes[1].set_title('Power Discrepancy Distribution', fontweight='bold')
    axes[1].set_xlabel('|Reported − Calculated| Power (kW)')

    plt.tight_layout()
    plt.savefig(output_dir / "power_vs_temp_and_discrepancy.png")
    plt.show()

    pct = (df['power_discrepancy'] > 2.0).mean()
    print(f'\nEvents with power discrepancy > 2 kW: {pct:.2%}')


if __name__ == "__main__":
    main()