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