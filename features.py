import pandas as pd
import numpy as np


def basic_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic feature creation:
    - power_calc_kw
    - power_discrepancy
    - has_error
    - energy_rate
    - hour, dayofweek, is_weekend
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['station_id', 'timestamp']).reset_index(drop=True)

    df['power_calc_kw'] = (df['voltage'] * df['current']) / 1000.0
    df['power_discrepancy'] = (df['power_kw'] - df['power_calc_kw']).abs()
    df['has_error'] = (df['error_code'] != 0).astype(int)
    df['energy_rate'] = df['energy_kwh'] / df['duration_sec'].clip(lower=1)

    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    return df


def add_rolling_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.copy()

    for col in ['voltage', 'current', 'power_kw', 'temperature_c']:
        grp = df.groupby('station_id')[col]

        df[f'{col}_roll_mean'] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f'{col}_roll_std'] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0)
        )
        df[f'{col}_delta'] = df[col] - df[f'{col}_roll_mean']

    return df


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    sess = df.groupby('session_id').agg(
        session_error_rate=('has_error', 'mean'),
        session_event_count=('session_id', 'count'),
        session_total_energy=('energy_kwh', 'sum'),
        session_avg_power=('power_kw', 'mean'),
        session_max_temp=('temperature_c', 'max'),
    ).reset_index()

    return df.merge(sess, on='session_id', how='left')


def add_station_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    baseline_cols = ['voltage', 'current', 'power_kw', 'temperature_c']

    station_stats = df.groupby('station_id')[baseline_cols].agg(['mean', 'std'])
    station_stats.columns = ['_'.join(c) for c in station_stats.columns]
    station_stats = station_stats.reset_index()

    df = df.merge(station_stats, on='station_id', how='left')

    for col in baseline_cols:
        std = df[f'{col}_std'].fillna(1e-6).replace(0, 1e-6)
        df[f'{col}_zscore'] = (df[col] - df[f'{col}_mean'].fillna(0)) / std

    return df


def build_feature_matrix(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Select model features and apply median imputation.
    """
    X = df[feature_cols].fillna(df[feature_cols].median())
    return X