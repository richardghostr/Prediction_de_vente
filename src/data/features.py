from typing import List, Optional

import pandas as pd


def add_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["dayofweek"] = df[date_col].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    return df


def add_lags(
    df: pd.DataFrame,
    value_col: str = "value",
    lags: Optional[List[int]] = None,
    date_col: str = "date",
) -> pd.DataFrame:
    df = df.copy()
    if lags is None:
        lags = [1]
    df = df.sort_values(by=date_col)
    for l in lags:
        df[f"lag_{l}"] = df[value_col].shift(l)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    value_col: str = "value",
    windows: Optional[List[int]] = None,
    date_col: str = "date",
) -> pd.DataFrame:
    df = df.copy()
    if windows is None:
        windows = [3]
    df = df.sort_values(by=date_col)
    for w in windows:
        df[f"roll_mean_{w}"] = df[value_col].rolling(window=w, min_periods=1).mean()
        df[f"roll_std_{w}"] = df[value_col].rolling(window=w, min_periods=1).std().fillna(0)
    return df


def encode_categorical(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    df = df.copy()
    if cols is None:
        cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cols:
        df[c] = pd.Categorical(df[c]).codes
    return df


def build_feature_pipeline(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "value",
    lags: Optional[List[int]] = None,
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    df = df.copy()
    df = add_time_features(df, date_col=date_col)
    df = add_lags(df, value_col=value_col, lags=lags, date_col=date_col)
    df = add_rolling_features(df, value_col=value_col, windows=windows, date_col=date_col)
    df = encode_categorical(df)
    return df
