
"""Configuration partagée du projet.

Contient chemins, constantes et utilitaire minimal pour créer les dossiers attendus.
Ce fichier doit rester léger et stable : les changements importants doivent faire l'objet
d'une petite PR coordonnée entre Personne A et Personne B.
"""
from pathlib import Path
from datetime import datetime
import os


ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# Models and artifacts
MODELS_DIR = ROOT / "models"
MODELS_ARTIFACTS_DIR = MODELS_DIR / "artifacts"
MODELS_METRICS_DIR = MODELS_DIR / "metrics"
ARCHIVE_DIR = ROOT / "archive"

# Reproducibility
RANDOM_SEED = 42

# Feature config (must match between train and predict)
PREDICT_LAGS = [1, 7, 14]
PREDICT_WINDOWS = [7, 14]

# Group columns: each unique (store, product) combination is a separate time series.
# Lags and rolling features MUST be computed within each group to avoid data leakage.
GROUP_COLS = ["store_id", "product_id"]

# Columns that are label-encoded and used as model features
CATEGORICAL_FEATURES = ["store_id", "product_id"]

# Binary features kept as-is
BINARY_FEATURES = ["on_promo"]

# Columns to exclude from model features (identifiers / targets that would cause leakage)
# NOTE: store_id and product_id are ENCODED separately, their raw text is excluded here.
FEATURE_EXCLUDE = {"value", "date", "id", "category", "revenue", "unit_price"}

# Date formatting
DATE_FORMAT = "%Y-%m-%d"

# Naming templates
def model_filename(date: str, tag: str) -> str:
	return f"model_{date}_{tag}.pkl"

def metadata_filename(date: str, tag: str) -> str:
	return f"model_{date}_{tag}.json"


def ensure_dirs():
	"""Create required directories if they do not exist."""
	for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_ARTIFACTS_DIR, MODELS_METRICS_DIR, ARCHIVE_DIR):
		os.makedirs(d, exist_ok=True)


if __name__ == "__main__":
	ensure_dirs()
	now = datetime.now().strftime(DATE_FORMAT)
	print(f"Project directories ensured. DATE={now}")


