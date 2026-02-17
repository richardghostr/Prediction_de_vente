"""Features temporelles et agrégations -- version group-aware.

Le pipeline construit des features par groupe (store_id, product_id) pour éviter
de mélanger les séries temporelles différentes.

Entrées / sorties (contrat):
- Entrée attendue : CSV nettoyé produit par `clean.py`, colonnes minimales `date`, `value`.
  Si `store_id` et `product_id` sont présents, les lags/rolling sont calculés par groupe.
- Sortie : DataFrame enrichi avec colonnes de features.

Exemple d'utilisation :
    python src/data/features.py --in data/interim --out data/processed --lags 1,7,14 --windows 7,14
"""

from typing import List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Time features (applied globally -- same for all groups)
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add calendar-based time features."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["dayofweek"] = df[date_col].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["quarter"] = df[date_col].dt.quarter
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
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
    """Create lag features.  When group_cols is given, lags are computed
    within each group to avoid leakage between different time series."""
    df = df.copy()
    if lags is None:
        lags = [1]
    df = df.sort_values(by=([*group_cols, date_col] if group_cols else [date_col]))

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
    """Rolling mean and std.  shift(1) avoids data leakage.
    When group_cols is given, rolling is computed within each group."""
    df = df.copy()
    if windows is None:
        windows = [7]
    df = df.sort_values(by=([*group_cols, date_col] if group_cols else [date_col]))

    for w in windows:
        if group_cols and all(c in df.columns for c in group_cols):
            shifted = df.groupby(group_cols)[value_col].shift(1)
            rolling = shifted.groupby(df[group_cols].apply(tuple, axis=1))
            df[f"roll_mean_{w}"] = rolling.rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f"roll_std_{w}"] = rolling.rolling(window=w, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
        else:
            shifted = df[value_col].shift(1)
            df[f"roll_mean_{w}"] = shifted.rolling(window=w, min_periods=1).mean()
            df[f"roll_std_{w}"] = shifted.rolling(window=w, min_periods=1).std().fillna(0)
    return df


# ---------------------------------------------------------------------------
# Categorical encoding (label encoding for tree-based models)
# ---------------------------------------------------------------------------

def encode_categorical(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    encoders: Optional[dict] = None,
) -> tuple:
    """Label-encode categorical columns.  Returns (df, encoders_dict).
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
            # Reuse existing mapping; unseen categories get -1
            mapping = encoders[c]
            df[c] = df[c].map(mapping).fillna(-1).astype(int)
        else:
            cat = pd.Categorical(df[c])
            encoders[c] = {v: i for i, v in enumerate(cat.categories)}
            df[c] = cat.codes
    return df, encoders


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
) -> tuple:
    """Full feature engineering pipeline.

    Returns (df_with_features, encoders_dict) so that the same encoding
    can be reused at inference time.

    Parameters
    ----------
    group_cols : list of str, optional
        Columns that identify separate time series (e.g. ["store_id", "product_id"]).
        When provided, lags and rolling stats are computed *within* each group.
    categorical_cols : list of str, optional
        Columns to label-encode. Defaults to auto-detecting object/category cols.
    encoders : dict, optional
        Previously fitted encoders to reuse at inference time.
    """
    df = df.copy()

    # Detect group columns if present in data
    if group_cols is None:
        # Try default from config
        try:
            from src.config import GROUP_COLS
            if all(c in df.columns for c in GROUP_COLS):
                group_cols = GROUP_COLS
        except ImportError:
            pass

    # 1. Time features
    df = add_time_features(df, date_col=date_col)

    # 2. Lags (group-aware)
    df = add_lags(df, value_col=value_col, lags=lags, date_col=date_col, group_cols=group_cols)

    # 3. Rolling features (group-aware)
    df = add_rolling_features(df, value_col=value_col, windows=windows, date_col=date_col, group_cols=group_cols)

    # 4. Encode categoricals
    if categorical_cols is None:
        # Auto-detect + include group cols if they are strings
        auto_cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
        categorical_cols = list(set(auto_cats))
    df, encoders = encode_categorical(df, cols=categorical_cols, encoders=encoders)

    return df, encoders


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
    parser.add_argument("--lags", default="1,7,14")
    parser.add_argument("--windows", default="7,14")
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
