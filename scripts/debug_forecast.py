#!/usr/bin/env python3
"""Debug forecast: load CSV from Downloads and call predict_series()."""
from pathlib import Path
import pandas as pd
import json
import traceback


def main():
    csv_path = Path(r"C:\Users\richa\Downloads\combined_forecast (2).csv")
    if not csv_path.exists():
        print("CSV not found:", csv_path)
        return

    df = pd.read_csv(csv_path)
    # Build series: keep ds as date, y as value
    series = []
    for _, r in df.iterrows():
        val = None if pd.isna(r.get("y")) else float(r.get("y"))
        series.append({"date": r.get("ds"), "value": val})

    # Use last 90 days of history if available
    hist = series
    if len(series) > 90:
        hist = series[-90:]

    # Ensure project root is on sys.path for `src` imports
    import sys
    proj_root = Path(__file__).resolve().parents[1]
    if str(proj_root) not in sys.path:
        sys.path.insert(0, str(proj_root))

    try:
        from src.models.predict import predict_series, get_model_path
    except Exception:
        print("Failed to import predict module")
        traceback.print_exc()
        return

    print("Model path:", get_model_path())

    try:
        # First try high-level call
        res = predict_series(hist, horizon=14)
        print("===== predict_series result =====")
        print(json.dumps(res, indent=2, ensure_ascii=False))

        # Now low-level: call feature builder to inspect features and model predict
        from src.data.features import build_feature_pipeline
        from src.models.predict import get_model
        feats, enc = build_feature_pipeline(
            pd.DataFrame(hist), date_col="date", value_col="value",
            lags=None, windows=None, group_cols=None, categorical_cols=None, encoders=None, is_train=False,
        )
        print("features shape:", None if feats is None else feats.shape)
        if feats is not None and not feats.dropna().empty:
            X = feats.drop(columns=[c for c in feats.columns if c in ("value","date")], errors="ignore")
            X_last = X.dropna().iloc[[-1]]
            print("X_last columns:", list(X_last.columns)[:30])
            model = get_model()
            X_aligned = None
            try:
                from src.models.predict import _align_features_to_model
                X_aligned = _align_features_to_model(X_last, model)
            except Exception:
                X_aligned = X_last
            print("X_aligned shape:", X_aligned.shape)
            try:
                pred = float(model.predict(X_aligned)[0])
                print("Model raw predict on last row:", pred)
            except Exception:
                print("Model predict failed:")
                traceback.print_exc()
        else:
            print("No usable features produced (feats is None or empty)")

    except Exception:
        print("predict_series raised:")
        traceback.print_exc()


if __name__ == '__main__':
    main()
