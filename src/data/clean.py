import os
from typing import Optional, List

import numpy as np
import pandas as pd


def harmonize_dates(df: pd.DataFrame, date_col: str = "date", fmt: Optional[str] = None) -> pd.DataFrame:
    """Convert `date_col` to pd.Timestamp, store as timezone-naive UTC, and sort.

    Non-parseable dates become NaT.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], format=fmt, errors="coerce")
    # Drop timezone info to keep things consistent
    if df[date_col].dt.tz is not None:
        df[date_col] = df[date_col].dt.tz_convert(None)
    df = df.sort_values(by=date_col).reset_index(drop=True)
    return df


def fill_missing(df: pd.DataFrame, strategy: str = "ffill", numeric_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Fill missing values using a simple strategy.

    strategy: 'ffill', 'bfill', 'mean' or 'zero'
    """
    df = df.copy()
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if strategy == "ffill":
        df[numeric_cols] = df[numeric_cols].ffill()
    elif strategy == "bfill":
        df[numeric_cols] = df[numeric_cols].bfill()
    elif strategy == "mean":
        for c in numeric_cols:
            df[c] = df[c].fillna(df[c].mean())
    elif strategy == "zero":
        df[numeric_cols] = df[numeric_cols].fillna(0)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # For non-numeric, do a forward fill as a reasonable default
    non_numeric = [c for c in df.columns if c not in numeric_cols]
    if non_numeric:
        df[non_numeric] = df[non_numeric].ffill().bfill()

    return df


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Return boolean mask where values are considered outliers by z-score."""
    s = series.dropna()
    if s.empty:
        return pd.Series([False] * len(series), index=series.index)
    mean = s.mean()
    std = s.std()
    if std == 0 or np.isnan(std):
        return pd.Series([False] * len(series), index=series.index)
    z = (series - mean) / std
    return z.abs() > threshold


def remove_outliers(df: pd.DataFrame, col: str, threshold: float = 3.0) -> pd.DataFrame:
    """Remove rows where `col` is an outlier according to z-score."""
    mask = detect_outliers_zscore(df[col], threshold=threshold)
    return df.loc[~mask].reset_index(drop=True)


def clean_dataframe(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "value",
    fill_strategy: str = "ffill",
    outlier_threshold: Optional[float] = 3.0,
) -> pd.DataFrame:
    """Run a small cleaning pipeline: harmonize dates, fill missing, remove outliers.

    Returns a cleaned DataFrame ready for feature engineering.
    """
    df = df.copy()
    df = harmonize_dates(df, date_col=date_col)

    # Ensure value column is numeric
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    df = fill_missing(df, strategy=fill_strategy, numeric_cols=[value_col])

    if outlier_threshold is not None:
        df = remove_outliers(df, col=value_col, threshold=outlier_threshold)

    return df.reset_index(drop=True)


def save_interim(df: pd.DataFrame, out_path: str) -> str:
    """Save dataframe to CSV, creating directories as needed. Returns path."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
