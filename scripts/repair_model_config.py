"""Recompute bias_correction for the latest model artifact and update its config.

Usage:
    python scripts/repair_model_config.py

This script will:
- find the latest model in `models/artifacts`
- load its `_config.json` (or create one)
- load the training features CSV (from config or reasonable fallbacks)
- rebuild features, align to model, compute test-set bias and update `bias_correction`
"""
from pathlib import Path
import json
import sys
import math

import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
ARTIFACTS = PROJECT_ROOT / "models" / "artifacts"


def find_latest_model():
    files = sorted(ARTIFACTS.glob("model_*.pkl"))
    return files[-1] if files else None


def load_config(model_path: Path):
    cfgp = model_path.parent / (model_path.stem + "_config.json")
    if cfgp.exists():
        try:
            return json.loads(cfgp.read_text()), cfgp
        except Exception:
            return {}, cfgp
    return {}, cfgp


def save_config(cfg, cfgp: Path):
    cfgp.write_text(json.dumps(cfg, indent=2))
    print(f"Wrote config: {cfgp}")


def coerce_object_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    df_out = df_in.copy()
    for c in df_out.columns:
        if pd.api.types.is_object_dtype(df_out[c].dtype) or isinstance(df_out[c].dtype, pd.CategoricalDtype):
            try:
                df_out[c] = pd.Categorical(df_out[c]).codes
            except Exception:
                df_out[c] = pd.to_numeric(df_out[c], errors="coerce").fillna(0).astype(float)
    return df_out


def main():
    mp = find_latest_model()
    if mp is None:
        print("No model artifact found in models/artifacts")
        sys.exit(1)
    print(f"Model path: {mp}")

    model = joblib.load(mp)
    cfg, cfgp = load_config(mp)

    # Determine features source
    feat_path = None
    if cfg.get("training_features_path"):
        cand = Path(cfg.get("training_features_path"))
        if cand.exists():
            feat_path = cand

    if feat_path is None:
        # fallback locations
        candidates = [PROJECT_ROOT / "data" / "processed" / "train_features.csv",
                      PROJECT_ROOT / "data" / "interim" / "train.csv",
                      PROJECT_ROOT / "data" / "processed" / "uploaded_generated_training_10950_features.csv"]
        for c in candidates:
            if c.exists():
                feat_path = c
                break

    if feat_path is None:
        print("No training features file found. Provide training_features_path in model config or place train_features.csv in data/processed.")
        sys.exit(1)

    print(f"Using features file: {feat_path}")
    df = pd.read_csv(feat_path, parse_dates=["date"]) if feat_path.suffix.lower() == ".csv" else pd.read_csv(feat_path, parse_dates=["date"]) 

    # Import pipeline and config defaults
    try:
        from src.data.features import build_feature_pipeline
        from src.config import PREDICT_LAGS, PREDICT_WINDOWS, FEATURE_EXCLUDE
    except Exception as e:
        print("Failed to import project modules:", e)
        sys.exit(1)

    lags = cfg.get("lags", PREDICT_LAGS)
    windows = cfg.get("windows", PREDICT_WINDOWS)
    encoders = cfg.get("encoders")

    print(f"Rebuilding features with lags={lags} windows={windows}")
    res = build_feature_pipeline(df, lags=lags, windows=windows, encoders=encoders, is_train=False)
    if isinstance(res, tuple):
        df_feat, _ = res
    else:
        df_feat = res

    df_feat = df_feat.dropna().reset_index(drop=True)

    # Determine horizon (default 14)
    horizon = int(cfg.get("horizon", 14))
    unique_dates = sorted(df_feat["date"].unique())
    if horizon >= len(unique_dates):
        horizon = max(1, len(unique_dates) // 5)

    cutoff = unique_dates[-horizon]
    test_mask = df_feat["date"] >= cutoff

    exclude = set(FEATURE_EXCLUDE) | {"value", "date"}
    feature_cols = [c for c in df_feat.columns if c not in exclude]

    X_test = df_feat.loc[test_mask, feature_cols].copy()
    y_test = df_feat.loc[test_mask, "value"].copy()

    # Align features to model
    try:
        from src.models.predict import _align_features_to_model
        X_test = _align_features_to_model(X_test, model)
    except Exception:
        # best-effort: drop extras and add zeros for missing
        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
            for col in expected:
                if col not in X_test.columns:
                    X_test[col] = 0
            extra = [c for c in X_test.columns if c not in expected]
            if extra:
                X_test = X_test.drop(columns=extra)
            X_test = X_test[expected]

    X_test = coerce_object_columns(X_test)

    y_pred = model.predict(X_test)

    # Handle possible transform
    target_transform = cfg.get("target_transform") or ("log1p" if cfg.get("target_log") else None)
    if target_transform == "log1p":
        try:
            y_pred = np.expm1(y_pred)
        except Exception:
            pass

    y_test = pd.to_numeric(y_test, errors="coerce").fillna(0).astype(float)

    if len(y_test) == 0:
        print("No test rows to compute bias. Aborting.")
        sys.exit(1)

    bias_test = float(np.mean(y_pred - y_test))
    bias_correction = -bias_test

    print(f"Computed Bias_test={bias_test:.6f}, bias_correction={bias_correction:.6f}")

    cfg["bias_correction"] = float(bias_correction)
    cfg["bias_space"] = "original"
    if target_transform:
        cfg["target_transform"] = target_transform

    save_config(cfg, cfgp)


if __name__ == "__main__":
    main()
