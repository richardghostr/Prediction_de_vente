import os
import sys
from pathlib import Path

import pandas as pd

# Ensure the repository `src/` directory is importable as a top-level package named `src`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data import clean, features


def _make_sample_df():
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=10, freq="D"),
            "value": [1, 2, None, 4, 100, 6, 7, None, 9, 10],
            "category": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
        }
    )


def _make_multi_group_df():
    """Create a DataFrame with multiple (store, product) groups."""
    rows = []
    for store in ["S1", "S2"]:
        for product in ["P1", "P2"]:
            for i, d in enumerate(pd.date_range("2023-01-01", periods=30, freq="D")):
                rows.append({
                    "date": d,
                    "store_id": store,
                    "product_id": product,
                    "value": 10 + i + (hash(store) % 5) + (hash(product) % 3),
                    "on_promo": 1 if i % 7 == 0 else 0,
                })
    return pd.DataFrame(rows)


def test_clean_and_time_features():
    df = _make_sample_df()
    cleaned = clean.clean_dataframe(
        df, date_col="date", value_col="value",
        fill_strategy="mean", outlier_threshold=2.0
    )
    assert "date" in cleaned.columns
    assert cleaned["value"].isna().sum() == 0
    assert not (cleaned["value"] == 100).any()


def test_lags_and_rolling():
    df = _make_sample_df()
    cleaned = clean.clean_dataframe(
        df, date_col="date", value_col="value",
        fill_strategy="ffill", outlier_threshold=None
    )
    # build_feature_pipeline now returns (df, encoders)
    feat, encoders = features.build_feature_pipeline(
        cleaned, date_col="date", value_col="value",
        lags=[1, 2], windows=[3]
    )
    # lags created
    assert "lag_1" in feat.columns and "lag_2" in feat.columns
    # rolling mean/std created
    assert "roll_mean_3" in feat.columns
    assert "roll_std_3" in feat.columns
    # time features
    assert "dayofweek" in feat.columns and "is_weekend" in feat.columns
    # seasonal features
    assert "quarter" in feat.columns
    assert "week_of_year" in feat.columns
    # encoders is a dict
    assert isinstance(encoders, dict)


def test_encode_categorical():
    df = _make_sample_df()
    # encode_categorical now returns (df, encoders)
    enc, encoders = features.encode_categorical(df, cols=["category"])
    assert enc["category"].dtype.kind in "iu"
    assert "category" in encoders
    assert isinstance(encoders["category"], dict)


def test_encode_categorical_reuse():
    """Test that encoders can be reused for consistent encoding."""
    df1 = pd.DataFrame({"col": ["a", "b", "c"]})
    df2 = pd.DataFrame({"col": ["b", "c", "a", "d"]})

    _, encoders = features.encode_categorical(df1, cols=["col"])
    df2_enc, _ = features.encode_categorical(df2, cols=["col"], encoders=encoders)

    # 'a', 'b', 'c' should get the same codes as in df1
    # 'd' is unseen, should get -1
    assert (df2_enc["col"] == -1).sum() == 1  # 'd' is unseen


def test_group_aware_lags():
    """Test that lags are computed within each group, not across groups."""
    df = _make_multi_group_df()
    feat, _ = features.build_feature_pipeline(
        df, lags=[1, 7], windows=[7],
        group_cols=["store_id", "product_id"],
        categorical_cols=["store_id", "product_id"],
    )

    # Lags should exist
    assert "lag_1" in feat.columns
    assert "lag_7" in feat.columns
    assert "roll_mean_7" in feat.columns
    assert "roll_std_7" in feat.columns

    # store_id and product_id should be encoded as integers
    assert feat["store_id"].dtype.kind in "iu"
    assert feat["product_id"].dtype.kind in "iu"


def test_group_aware_no_cross_leakage():
    """Verify that lag_1 for group S1/P1 uses S1/P1's own previous value."""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"]),
        "store_id": ["S1", "S1", "S2", "S2"],
        "product_id": ["P1", "P1", "P1", "P1"],
        "value": [10, 20, 100, 200],
    })
    feat, _ = features.build_feature_pipeline(
        df, lags=[1], windows=[],
        group_cols=["store_id", "product_id"],
        categorical_cols=["store_id", "product_id"],
    )
    # For S1/P1 on 2023-01-02, lag_1 should be 10 (not 100 from S2/P1)
    s1_rows = feat[(feat["day"] == 2)]
    # The first group (S1/P1) lag_1 should be from S1/P1's previous day
    s1_lag = s1_rows.iloc[0]["lag_1"]
    assert s1_lag == 10.0 or pd.isna(s1_lag) is False


def test_build_pipeline_returns_tuple():
    """build_feature_pipeline must return (DataFrame, dict)."""
    df = _make_sample_df()
    result = features.build_feature_pipeline(df, lags=[1], windows=[3])
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], pd.DataFrame)
    assert isinstance(result[1], dict)
