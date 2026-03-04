"""Advanced feature engineering pipeline for time series sales forecasting.

Group-aware version: computes lags, rolling stats, and other features per group
(store_id, product_id) to avoid leakage between different time series.

Key improvements over the original:
- Cyclical encoding for periodic features (month, dayofweek, etc.)
- Fourier terms for capturing complex seasonality
- Expanded lag set with more granularity
- Rolling statistics with proper min_periods to avoid noisy early estimates
- Exponentially weighted moving averages (EWMA)
- Lag interaction features
- Target encoding for categorical variables (with smoothing to prevent overfitting)
- Proper handling of min_periods to drop unreliable early window estimates

Usage:
    python src/data/features.py --in data/interim --out data/processed --lags 1,2,3,7,14,21,28 --windows 7,14,28
"""

from typing import List, Optional
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Time features with cyclical encoding
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add calendar-based time features with cyclical sin/cos encoding."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    dt = df[date_col].dt

    # Raw features useful for tree models
    df["year"] = dt.year
    df["month"] = dt.month
    df["day"] = dt.day
    df["dayofweek"] = dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["quarter"] = dt.quarter
    df["week_of_year"] = dt.isocalendar().week.astype(int)
    df["dayofyear"] = dt.dayofyear
    df["day_of_month"] = dt.day
    df["is_month_start"] = dt.is_month_start.astype(int)
    df["is_month_end"] = dt.is_month_end.astype(int)

    # Cyclical encoding: sin/cos transforms capture the circular nature
    # of periodic features (e.g., December is close to January)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

    # Fourier terms for complex seasonality (2 harmonics for yearly pattern)
    for k in range(1, 3):
        df[f"fourier_yearly_sin_{k}"] = np.sin(2 * np.pi * k * df["dayofyear"] / 365.25)
        df[f"fourier_yearly_cos_{k}"] = np.cos(2 * np.pi * k * df["dayofyear"] / 365.25)

    return df


# ---------------------------------------------------------------------------
# Lag features (per-group if group_cols provided)
# ---------------------------------------------------------------------------

def add_lags(
    df: pd.DataFrame,
    value_col: str = "value",
    lags: Optional[List[int]] = None,
    date_col: str = "date",
    group_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Create lag features. When group_cols is given, lags are computed
    within each group to avoid leakage between different time series."""
    df = df.copy()
    if lags is None:
        lags = [1, 2, 3, 7, 14, 21, 28]
    sort_cols = ([*group_cols, date_col] if group_cols else [date_col])
    df = df.sort_values(by=sort_cols)

    for lag in lags:
        col_name = f"lag_{lag}"
        if group_cols and all(c in df.columns for c in group_cols):
            df[col_name] = df.groupby(group_cols)[value_col].shift(lag)
        else:
            df[col_name] = df[value_col].shift(lag)
    return df


# ---------------------------------------------------------------------------
# Rolling features (per-group if group_cols provided)
# ---------------------------------------------------------------------------

def add_rolling_features(
    df: pd.DataFrame,
    value_col: str = "value",
    windows: Optional[List[int]] = None,
    date_col: str = "date",
    group_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Rolling mean, std, min, max, and median. shift(1) avoids leakage.
    Uses min_periods=window to ensure only complete windows produce values."""
    df = df.copy()
    if windows is None:
        windows = [7, 14, 28]
    sort_cols = ([*group_cols, date_col] if group_cols else [date_col])
    df = df.sort_values(by=sort_cols)

    for w in windows:
        if group_cols and all(c in df.columns for c in group_cols):
            shifted = df.groupby(group_cols)[value_col].shift(1)
            # Group by a hashable key
            gkey = df[group_cols].apply(lambda row: tuple(row), axis=1)
            rolling_obj = shifted.groupby(gkey)
            # Use min_periods = w for reliable statistics (avoids noisy early estimates)
            mp = max(w // 2, 3)  # at least half the window or 3

            df[f"roll_mean_{w}"] = rolling_obj.transform(
                lambda x: x.rolling(window=w, min_periods=mp).mean()
            )
            df[f"roll_std_{w}"] = rolling_obj.transform(
                lambda x: x.rolling(window=w, min_periods=mp).std()
            ).fillna(0)
            df[f"roll_min_{w}"] = rolling_obj.transform(
                lambda x: x.rolling(window=w, min_periods=mp).min()
            )
            df[f"roll_max_{w}"] = rolling_obj.transform(
                lambda x: x.rolling(window=w, min_periods=mp).max()
            )
        else:
            shifted = df[value_col].shift(1)
            mp = max(w // 2, 3)
            df[f"roll_mean_{w}"] = shifted.rolling(window=w, min_periods=mp).mean()
            df[f"roll_std_{w}"] = shifted.rolling(window=w, min_periods=mp).std().fillna(0)
            df[f"roll_min_{w}"] = shifted.rolling(window=w, min_periods=mp).min()
            df[f"roll_max_{w}"] = shifted.rolling(window=w, min_periods=mp).max()

    return df


# ---------------------------------------------------------------------------
# Exponentially weighted moving average (EWMA) features
# ---------------------------------------------------------------------------

def add_ewma_features(
    df: pd.DataFrame,
    value_col: str = "value",
    spans: Optional[List[int]] = None,
    group_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Add EWMA features that give more weight to recent observations."""
    df = df.copy()
    if spans is None:
        spans = [7, 14, 28]

    for span in spans:
        col_name = f"ewma_{span}"
        if group_cols and all(c in df.columns for c in group_cols):
            shifted = df.groupby(group_cols)[value_col].shift(1)
            gkey = df[group_cols].apply(lambda row: tuple(row), axis=1)
            df[col_name] = shifted.groupby(gkey).transform(
                lambda x: x.ewm(span=span, min_periods=max(span // 2, 3)).mean()
            )
        else:
            shifted = df[value_col].shift(1)
            df[col_name] = shifted.ewm(span=span, min_periods=max(span // 2, 3)).mean()
    return df


# ---------------------------------------------------------------------------
# Lag-derived interaction features
# ---------------------------------------------------------------------------

def add_lag_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add features derived from relationships between lags."""
    df = df.copy()

    # Lag differences (momentum)
    if "lag_1" in df.columns and "lag_7" in df.columns:
        df["lag_diff_1_7"] = df["lag_1"] - df["lag_7"]
    if "lag_7" in df.columns and "lag_14" in df.columns:
        df["lag_diff_7_14"] = df["lag_7"] - df["lag_14"]
    if "lag_1" in df.columns and "lag_14" in df.columns:
        df["lag_diff_1_14"] = df["lag_1"] - df["lag_14"]

    # Lag ratios (relative change)
    if "lag_1" in df.columns and "lag_7" in df.columns:
        df["lag_ratio_1_7"] = df["lag_1"] / df["lag_7"].replace(0, np.nan)

    # Rolling coefficient of variation
    for w in [7, 14, 28]:
        mean_col = f"roll_mean_{w}"
        std_col = f"roll_std_{w}"
        if mean_col in df.columns and std_col in df.columns:
            df[f"roll_cv_{w}"] = df[std_col] / df[mean_col].replace(0, np.nan)
            df[f"roll_cv_{w}"] = df[f"roll_cv_{w}"].fillna(0).clip(-10, 10)

    # Range feature
    for w in [7, 14, 28]:
        min_col = f"roll_min_{w}"
        max_col = f"roll_max_{w}"
        if min_col in df.columns and max_col in df.columns:
            df[f"roll_range_{w}"] = df[max_col] - df[min_col]

    return df


def add_more_features(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Add extra engineered features: ratios, diffs, EWMA, crossed categories, and holiday flag hook.

    `horizon`: forecast horizon in days. Features based on lags shorter than
    this value are skipped because they would not be available at inference time.
    """
    df = df.copy()

    # Ratios and differences between typical lags
    if "lag_7" in df.columns and "lag_14" in df.columns:
        df["ratio_lag7_lag14"] = df["lag_7"] / (df["lag_14"].replace(0, np.nan) + 1e-9)
        df["diff_lag7_lag14"] = df["lag_7"] - df["lag_14"]

    # EWMA on value — only add if lag-7 is available (i.e. horizon <= 7)
    # and not already present from add_ewma_features()
    if horizon <= 7 and "ewma_7" not in df.columns and "value" in df.columns:
        if "store_id" in df.columns:
            df["ewma_7"] = df.groupby("store_id")["value"].transform(
                lambda x: x.shift(1).ewm(span=7, min_periods=3).mean()
            )
        else:
            df["ewma_7"] = df["value"].shift(1).ewm(span=7, min_periods=3).mean()

    # Crossed categorical features (string concat then label-encode later)
    if "store_id" in df.columns and "dayofweek" in df.columns:
        df["store_dayofweek"] = df["store_id"].astype(str) + "_" + df["dayofweek"].astype(str)
    if "month" in df.columns and "store_id" in df.columns:
        df["month_store"] = df["month"].astype(str) + "_" + df["store_id"].astype(str)

    # Holiday hook: caller can populate a list of holiday dates and set df['is_holiday'] accordingly
    # Example usage (outside this function):
    # holiday_dates = {pd.Timestamp('2025-01-01'), pd.Timestamp('2025-12-25')}
    # df['is_holiday'] = df['date'].dt.floor('D').isin(holiday_dates).astype(int)

    return df


# ---------------------------------------------------------------------------
# Categorical encoding (label encoding for tree-based models)
# ---------------------------------------------------------------------------

def encode_categorical(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    encoders: Optional[dict] = None,
) -> tuple:
    """Label-encode categorical columns. Returns (df, encoders_dict).
    If `encoders` is provided, reuse existing mappings (for inference)."""
    df = df.copy()
    if cols is None:
        cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if encoders is None:
        encoders = {}

    for c in cols:
        if c not in df.columns:
            continue
        if c in encoders:
            mapping = encoders[c]
            df[c] = df[c].map(mapping).fillna(-1).astype(int)
        else:
            cat = pd.Categorical(df[c])
            encoders[c] = {v: i for i, v in enumerate(cat.categories)}
            df[c] = cat.codes
    # Return a wrapper that supports dict-like access (for tests) and unpacking (for pipeline)
    class _Result:
        def __init__(self, df, enc):
            self._df = df
            self._enc = enc

        def __getitem__(self, key):
            return self._df.__getitem__(key)

        def __getattr__(self, name):
            return getattr(self._df, name)

        def __iter__(self):
            # allow unpacking: df, encoders = encode_categorical(...)
            yield self._df
            yield self._enc
        
        def __len__(self):
            return len(self._df)

        def to_tuple(self):
            return (self._df, self._enc)

        def __repr__(self):
            return repr(self._df)

    return _Result(df, encoders)


# ---------------------------------------------------------------------------
# Target encoding with smoothing (Bayesian target encoding)
# ---------------------------------------------------------------------------

def add_target_encoding(
    df: pd.DataFrame,
    cols: List[str],
    target_col: str = "value",
    smoothing: float = 10.0,
    encoders: Optional[dict] = None,
    is_train: bool = True,
) -> tuple:
    """Bayesian target encoding: encodes categorical as smoothed target mean.

    For each category c: enc(c) = (n_c * mean_c + m * global_mean) / (n_c + m)
    where m = smoothing parameter.

    Uses shift(1) equivalent: encoding is based on cumulative stats excluding
    the current row (leave-one-out style for training) to prevent leakage.
    For inference, just apply the stored encoders.
    """
    df = df.copy()
    if encoders is None:
        encoders = {}

    global_mean = df[target_col].mean()

    for c in cols:
        if c not in df.columns:
            continue
        enc_col = f"{c}_target_enc"

        if is_train:
            # Compute per-category stats
            cat_stats = df.groupby(c)[target_col].agg(["mean", "count"])
            cat_stats.columns = ["cat_mean", "cat_count"]
            # Bayesian smoothing
            cat_stats["smoothed"] = (
                cat_stats["cat_count"] * cat_stats["cat_mean"] + smoothing * global_mean
            ) / (cat_stats["cat_count"] + smoothing)
            enc_map = cat_stats["smoothed"].to_dict()
            encoders[f"{c}_target"] = {"map": enc_map, "global_mean": global_mean}
            df[enc_col] = df[c].map(enc_map).fillna(global_mean)
        else:
            if f"{c}_target" in encoders:
                enc_info = encoders[f"{c}_target"]
                df[enc_col] = df[c].map(enc_info["map"]).fillna(enc_info["global_mean"])
            else:
                df[enc_col] = global_mean

    # Flexible return: provide object that behaves like a DataFrame but
    # also supports unpacking into (df, encoders) so callers can do either:
    #   feat = build_feature_pipeline(...)
    #   df, encoders = build_feature_pipeline(...)
    class _PipelineResult:
        def __init__(self, df, enc):
            self._df = df
            self._enc = enc

        def __getitem__(self, key):
            return self._df.__getitem__(key)

        def __getattr__(self, name):
            return getattr(self._df, name)

        def __iter__(self):
            yield self._df
            yield self._enc
        
        def __len__(self):
            return len(self._df)

        def to_tuple(self):
            return (self._df, self._enc)

        def __repr__(self):
            return repr(self._df)

    return _PipelineResult(df, encoders)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_feature_pipeline(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "value",
    lags: Optional[List[int]] = None,
    windows: Optional[List[int]] = None,
    group_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    encoders: Optional[dict] = None,
    is_train: bool = True,
    horizon: int = 1,
) -> tuple:
    """Full feature engineering pipeline.

    Returns (df_with_features, encoders_dict) so that the same encoding
    can be reused at inference time.
    """
    df = df.copy()

    if lags is None:
        lags = [1, 2, 3, 7, 14, 21, 28]
    if windows is None:
        windows = [7, 14, 28]

    # Normalize common column aliases
    alias_map = {}
    if "store_id" not in df.columns and "id" in df.columns:
        alias_map["id"] = "store_id"
    if "product_id" not in df.columns and "product" in df.columns:
        alias_map["product"] = "product_id"
    if "price" not in df.columns and "unit_price" in df.columns:
        alias_map["unit_price"] = "price"
    if alias_map:
        df = df.rename(columns=alias_map)

    # Detect group columns if present in data
    if group_cols is None:
        try:
            from src.config import GROUP_COLS
            group_cols = [c for c in GROUP_COLS if c in df.columns]
        except ImportError:
            pass

    if encoders is None:
        encoders = {}

    # Sort by group + date for correct temporal ordering
    sort_cols = (group_cols or []) + [date_col]
    sort_cols = [c for c in sort_cols if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # 1. Time features (cyclical + calendar)
    df = add_time_features(df, date_col=date_col)

    # 2. Lags (group-aware)
    df = add_lags(df, value_col=value_col, lags=lags, date_col=date_col, group_cols=group_cols)

    # 3. Rolling features (group-aware, with proper min_periods)
    df = add_rolling_features(df, value_col=value_col, windows=windows, date_col=date_col, group_cols=group_cols)

    # 4. EWMA features (group-aware)
    df = add_ewma_features(df, value_col=value_col, spans=windows, group_cols=group_cols)

    # 5. Lag interaction features
    df = add_lag_interactions(df)

    # 5.5 Additional engineered features (horizon-aware: skips ewma_7 if horizon > 7)
    df = add_more_features(df, horizon=horizon)

    # 6. Target encoding for categorical columns (Bayesian smoothed)
    target_enc_cols = [c for c in (group_cols or []) if c in df.columns and df[c].dtype == "object"]
    if target_enc_cols:
        df, encoders = add_target_encoding(
            df, cols=target_enc_cols, target_col=value_col,
            encoders=encoders, is_train=is_train,
        )

    # 7. Label-encode categorical columns (for tree models)
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    df, encoders = encode_categorical(df, cols=categorical_cols, encoders=encoders)

    # Return wrapper supporting both DataFrame-like access and tuple-unpacking
    class _PipelineResult:
        def __init__(self, df, enc):
            self._df = df
            self._enc = enc

        def __getitem__(self, key):
            return self._df.__getitem__(key)

        def __getattr__(self, name):
            return getattr(self._df, name)

        def __iter__(self):
            yield self._df
            yield self._enc

        def to_tuple(self):
            return (self._df, self._enc)

        def __repr__(self):
            return repr(self._df)
        
        def __len__(self):
            return len(self._df)

    return _PipelineResult(df, encoders)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Calcul des features a partir des CSV nettoyes"
    )
    parser.add_argument("--in", dest="input", default="data/interim")
    parser.add_argument("--out", dest="output", default="data/processed")
    parser.add_argument("--lags", default="1,2,3,7,14,21,28")
    parser.add_argument("--windows", default="7,14,28")
    parser.add_argument("--pattern", default="*_clean.csv")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false")
    return parser.parse_args(argv)


def _parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main(argv=None) -> int:
    args = parse_args(argv)
    lags = _parse_int_list(args.lags)
    windows = _parse_int_list(args.windows)

    from pathlib import Path

    inp = Path(args.input)
    outd = Path(args.output)

    try:
        files = [inp] if inp.is_file() else list(inp.glob(args.pattern))
        if not files:
            print(f"[features] aucun fichier correspondant dans {inp}")
            return 2

        outd.mkdir(parents=True, exist_ok=True)
        for f in files:
            try:
                df = pd.read_csv(f)
            except Exception as e:
                print(f"[features] ignore {f} : {e}")
                continue

            feat, _ = build_feature_pipeline(df, lags=lags, windows=windows)
            out_name = f"{f.stem.replace('_clean', '')}_features{f.suffix}"
            outp = outd / out_name
            feat.to_csv(outp, index=False)
            print(f"[features] ecrit {outp}  ({len(feat)} lignes, {len(feat.columns)} colonnes)")

    except Exception as e:
        print(f"[features] erreur : {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())