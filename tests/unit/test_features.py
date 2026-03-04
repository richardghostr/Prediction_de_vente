import os
import sys
from pathlib import Path

import pandas as pd

# Ensure the repository `src/` directory is importable as a top-level package named `src`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data import clean, features


def _make_train_df():
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=10, freq="D"),
            "value": [1, 2, None, 4, 100, 6, 7, None, 9, 10],
            "category": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
        }
    )


def test_clean_and_time_features():
    df = _make_train_df()
    # use a slightly lower threshold to reliably remove the clear outlier in the sample
    cleaned = clean.clean_dataframe(df, date_col="date", value_col="value", fill_strategy="mean", outlier_threshold=2.0)
    assert "date" in cleaned.columns
    assert cleaned["value"].isna().sum() == 0
    # After removing the outlier (100) it should no longer appear in the values
    assert not (cleaned["value"] == 100).any()


def test_lags_and_rolling():
    df = _make_train_df()
    cleaned = clean.clean_dataframe(df, date_col="date", value_col="value", fill_strategy="ffill", outlier_threshold=None)
    feat = features.build_feature_pipeline(cleaned, date_col="date", value_col="value", lags=[1, 2], windows=[3])
    # lags created
    assert "lag_1" in feat.columns and "lag_2" in feat.columns
    # rolling mean created
    assert "roll_mean_3" in feat.columns
    # time features
    assert "dayofweek" in feat.columns and "is_weekend" in feat.columns


def test_encode_categorical():
    df = _make_train_df()
    enc = features.encode_categorical(df, cols=["category"])
    assert enc["category"].dtype.kind in "iu"

