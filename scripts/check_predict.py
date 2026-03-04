#!/usr/bin/env python3
"""Quick offline prediction checker using `data/interim/train.csv`.

Run from project root:
    python scripts/check_predict.py
"""
import traceback
from pathlib import Path
import sys
import pandas as pd

# Ensure project root is on sys.path so `src` imports work when script
# is executed from varying working directories or terminals.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_series_from_df(df: pd.DataFrame):
    # detect date col
    date_col = None
    for c in df.columns:
        if "date" in c.lower() or c.lower() in ("ds", "timestamp", "datetime"):
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    # detect value col
    val_candidates = [c for c in df.columns if c.lower() in ("value", "revenue", "amount", "sales", "y", "qty", "quantity", "unit_price", "price")]
    if val_candidates:
        value_col = val_candidates[0]
    else:
        num_cols = df.select_dtypes(include=["number"]).columns
        value_col = num_cols[0] if len(num_cols) > 0 else df.columns[1] if len(df.columns) > 1 else df.columns[0]

    df2 = df.copy()
    df2[date_col] = pd.to_datetime(df2[date_col], errors="coerce")
    grouped = df2.groupby(df2[date_col].dt.floor("D")).agg({
        **{c: "sum" for c in df2.select_dtypes(include=["number"]).columns if c != date_col},
        **{c: "first" for c in df2.select_dtypes(exclude=["number"]).columns if c != date_col},
    }).reset_index()

    series = []
    for _, row in grouped.iterrows():
        item = {"date": pd.to_datetime(row[date_col]).date().isoformat()}
        if value_col in grouped.columns:
            v = row[value_col]
            item["value"] = None if pd.isna(v) else float(v)
        else:
            item["value"] = None
        series.append(item)
    return series


def main():
    ROOT = Path(__file__).resolve().parents[1]
    sample = ROOT / "data" / "interim" / "train.csv"
    if sample.exists():
        try:
            df = pd.read_csv(sample, low_memory=False)
        except Exception:
            print("Failed to read train.csv")
            traceback.print_exc()
            return
    else:
        print("train.csv not found at", sample, "— falling back to synthetic series")
        # create a small synthetic dataframe
        df = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=10, freq="D"),
            "value": [100 + i * 5 for i in range(10)],
        })

    series = build_series_from_df(df)
    print("Built series length:", len(series))

    try:
        from src.models.predict import predict_series, get_model_path
    except Exception:
        print("Failed to import prediction module")
        traceback.print_exc()
        return

    print("Model path:", get_model_path())

    try:
        res = predict_series(series, horizon=7)
        fc = res.get("forecast", [])
        print("Forecast points:", len(fc))
        for r in fc[:10]:
            print(r)
    except Exception:
        print("predict_series raised an exception")
        traceback.print_exc()


if __name__ == "__main__":
    main()
