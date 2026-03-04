import sys
from pathlib import Path

import pandas as pd

# Ensure `src/` is importable as a top-level package
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import train as train_module


def test_training_end_to_end(tmp_path: Path):
    """
    End-to-end training test with a light model.

    This test:
    - ensures train_features.csv exists,
    - runs the training entrypoint with a small model (few trees),
    - checks that a model artifact and metrics file are created.
    """
    data_dir = PROJECT_ROOT / "data" / "processed"
    train_features = data_dir / "train_features.csv"
    assert train_features.exists(), "train_features.csv should exist in data/processed"

    # Run training with lighter hyperparameters to keep the test fast
    argv = [
        "--n-estimators",
        "10",
        "--max-depth",
        "3",
        "--tag",
        "test_ci",
    ]

    # Execute main and ensure it returns success
    code = train_module.main(argv)
    assert code == 0

    artifacts_dir = PROJECT_ROOT / "models" / "artifacts"
    metrics_dir = PROJECT_ROOT / "models" / "metrics"

    model_files = sorted(artifacts_dir.glob("model_*_test_ci.pkl"))
    assert model_files, "Training should produce at least one model_*_test_ci.pkl artifact"

    metrics_files = sorted(metrics_dir.glob("xgb_metrics_*.json"))
    assert metrics_files, "Training should produce at least one xgb_metrics_*.json file"

