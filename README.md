# Prévision des ventes - Projet

Copyright (c) 2026 richa

Ce dépôt contient un pipeline local de bout en bout pour la prévision des ventes : ingestion, nettoyage, génération de features, entraînement et service d'inférence.

Le projet est organisé pour être léger, reproductible et exécutable en local (ou dans des conteneurs Docker). Les scripts sont écrits en Python et exposent des CLIs simples pour chaque étape.

**Principaux objectifs**

- Fournir un pipeline reproductible pour expérimenter des modèles de séries temporelles.
- Séparer clairement les étapes : `ingest` -> `clean` -> `features` -> `train` -> `serve`.
- Faciliter l'intégration dans des CI/CD et des conteneurs.

**Contenu principal**

- `src/data/ingest.py` : ingestion de CSV bruts depuis `data/raw`, calcule checksum et écrit `<stem>_ingest.csv` (option `--write-meta` pour la métadonnée JSON).
- `src/data/clean.py` : nettoyage et heuristiques simples (normalisation des dates, remplissage, détection/suppression d'outliers), écrit `<stem>_clean.csv`.
- `src/data/features.py` : création de variables temporelles, lags, roulings et encodages, écrit `<stem>_features.csv`.
- `src/models/train.py` : script d'entraînement (ex. XGBoost), sauvegarde d'artefacts dans `models/artifacts/`.
- `src/models/predict.py` et `src/serve/api.py` : prédiction / point d'entrée API pour servir un modèle entraîné.
- `notebooks/` : notebooks d'EDA et d'expérimentation.

Structure de dossier (extrait)

- `data/` : jeux de données organisés (`raw/`, `interim/`, `processed/`, `external/`)
- `src/` : sources Python organisés par responsabilité (`data/`, `models/`, `serve/`, `ui/`, `utils/`)
- `models/` : artefacts, métriques et modèles sérialisés
- `tests/` : tests unitaires et d'intégration
- `docs/` : documentation d'utilisation et design

Prérequis

- Python 3.10+ (installer via `pyenv`, `venv` ou conda)
- Installer les dépendances :

```bash
pip install -r requirements.txt
```

Utilisation rapide (local)

1) Ingestion (depuis `data/raw` ou un fichier CSV)

```bash
python src/data/ingest.py --in data/raw --out data/raw
# ou pour un fichier unique
python src/data/ingest.py --in data/raw/sample.csv --out data/raw
```

Sortie : `data/raw/<stem>_ingest.csv` (et optionnellement `<stem>_ingest.csv.metadata.json` si `--write-meta`).

2) Nettoyage

```bash
python src/data/clean.py --in data/raw --out data/interim --pattern "*_ingest.csv"
# ou fichier unique
python src/data/clean.py --in data/raw/sample_ingest.csv --out data/interim
```

Sortie : `data/interim/<stem>_clean.csv`.

3) Génération de features

```bash
python src/data/features.py --in data/interim --out data/processed --pattern "*_clean.csv"
```

Sortie : `data/processed/<stem>_features.csv`.

4) Entraînement et évaluation

```bash
python src/models/train.py --in data/processed --out models/artifacts
python src/models/evaluate.py --pred models/artifacts/pred.csv --truth data/processed/...
```

5) Servir le modèle (exemple: API)

```bash
python src/serve/api.py --model models/artifacts/model.pkl --port 8000
```

Tests

```bash
pytest -q
```

Docker & déploiement

- `Dockerfile` et `docker-compose.yml` sont fournis pour construire et exécuter le service. Voir `docs/run.md` pour les instructions détaillées.

Conventions & contrats de fichiers

- Les runners écrivent des fichiers avec les suffixes `_ingest`, `_clean`, `_features` pour garantir la compatibilité entre étapes.
- Les scripts CLI acceptent un chemin fichier ou dossier. Par défaut : `data/raw -> data/interim -> data/processed`.

Bonnes pratiques

- Versionner les données critiques ou stocker les checksums (gestion déjà présente dans `ingest.py`).
- Garder les transformations idempotentes et documentées dans `src/data/*`.

Contribution

1. Ouvrir une issue pour discuter du changement.
2. Créer une branche dédiée et un PR contenant des tests si possible.

Contacts / documentation

- Documentation d'exécution : [docs/run.md](docs/run.md)
- Conception : [docs/design.md](docs/design.md)

---

Si vous voulez, je peux :

- Exécuter les commandes `--help` pour chaque CLI et ajouter des exemples concrets en sortie.
- Ajouter une section « Quick start » plus courte pour faire tourner le pipeline en 3 commandes.
- Traduire ce README en anglais.

Dites-moi quelle option vous préférez.
