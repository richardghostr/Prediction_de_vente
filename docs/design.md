## Architecture générale

Le projet **Prediction de Vente** implémente un pipeline de prévision de ventes de bout en bout, inspiré des structures de projets data type *cookiecutter-data-science*.

- **Couche données (`src/data/`)**
  - `ingest.py` : ingestion de CSV bruts vers des fichiers normalisés `*_ingest.csv` en s’appuyant sur :
    - `configs/mapping_config.yml` (cibles, alias, règles de validation),
    - `src/utils/mapping_optimized.py` (détection intelligente des colonnes et nettoyage numérique).
  - `clean.py` : nettoyage des séries temporelles (harmonisation des dates, remplissage de valeurs manquantes, suppression d’outliers) vers `*_clean.csv`.
  - `features.py` : *feature engineering* temporel et *group‑aware* (par `store_id`, `product_id`) produisant `*_features.csv`.

- **Couche modèle (`src/models/`)**
  - `train.py` : entraînement XGBoost sur données multi‑séries avec :
    - pipeline de features commun à `features.py`,
    - découpage temporel train/test par dates uniques,
    - métriques train/test + CV temporel,
    - sauvegarde d’artefacts (`model_*.pkl`) et d’un fichier de configuration associé (`*_config.json`) contenant lags/windows, encoders, noms de features et chemin des features d’entraînement.
  - `predict.py` : prévision auto‑régressive :
    - chargement paresseux du dernier modèle + config,
    - reconstruction de features cohérentes avec l’entraînement,
    - support de multiples séries via colonnes de groupe (`store_id`, `product_id`),
    - fallback naïf (répétition de la dernière valeur) en absence de modèle.
  - `evaluate.py` : évaluation group‑aware :
    - reconstitution du pipeline de features (y compris à partir d’un `*_clean.csv` si besoin),
    - séparation train/test temporelle alignée avec `train.py`,
    - métriques robustes (MAE, RMSE, R², MAPE/sMAPE, biais, ratio d’overfitting),
    - diagnostics de distribution (shift train/test, proportion de valeurs quasi nulles).

- **Service API (`src/serve/`)**
  - `api.py` (FastAPI) expose :
    - `GET /health`, `GET /model-info`, `GET /metrics`,
    - `POST /predict`, `POST /predict/batch`.
  - `schemas.py` (Pydantic v2) définit les schémas d’entrée/sortie, avec :
    - parsing de dates robuste,
    - capture d’extras (régressseurs) dans `DataPoint.extras`,
    - validations fortes (taille des séries, présence d’au moins une valeur non nulle, etc.).

- **UI (`src/ui/`)**
  - `streamlit_app.py` : dashboard “data‑science” complet (upload CSV, EDA, séries temporelles, prévision, inspection du modèle).
  - `dashboard.py` : dashboard “pipeline” orchestrant les étapes *Ingestion → Nettoyage → Features → Prédiction → Visualisation* via des sous‑modules (`ingest_ui`, `clean_ui`, `features_ui`, `predict_ui`, `visualize_ui`).

- **Utilitaires & config**
  - `src/config.py` : configuration centralisée (chemins, paramètres de features et de grouping, constantes).
  - `src/utils/logging.py` : configuration commune du logging (format homogène et niveau contrôlé par `LOG_LEVEL`).
  - `src/utils/mapping_optimized.py` : moteurs de mapping de colonnes pour ingestion (similarité de noms + heuristiques de type).

- **Infra**
  - `Dockerfile` : image unique Python 3.11 pour l’API et les dashboards Streamlit.
  - `docker-compose.yml` :
    - service `api` : démarre l’API, entraîne un modèle de base si nécessaire,
    - services `streamlit` et `streamlit_admin` : dashboards pipeline et data‑science, avec pré‑entraînement éventuel du modèle.

## Pipeline de données

1. **Ingestion** (`ingest.py`)
   - Entrée : fichiers CSV bruts potentiellement hétérogènes (noms et formats de colonnes variés).
   - Étapes :
     - lecture du CSV brut,
     - chargement de la configuration de mapping `mapping_config.yml`,
     - détection des colonnes cibles (`id`, `date`, `value`) via similarité de noms + heuristiques de type,
     - normalisation des types (`date` en datetime, `value` en numérique, etc.),
     - validation (seuils de valeurs manquantes, contraintes sur `value`).
   - Sortie : `data/raw/<stem>_ingest.csv`, + un fichier de métadonnées JSON optionnel (`*.metadata.json`).

2. **Nettoyage** (`clean.py`)
   - Entrée : `*_ingest.csv`.
   - Étapes :
     - harmonisation de la colonne `date` (`pd.to_datetime`, suppression de timezone, tri),
     - conversion de `value` en numérique (`to_numeric`),
     - remplissage des valeurs manquantes (stratégies `ffill`, `bfill`, `mean`, `zero`),
     - détection d’outliers par z‑score et suppression.
   - Sortie : `data/interim/<stem>_clean.csv`.

3. **Feature engineering** (`features.py`)
   - Entrée : `*_clean.csv`.
   - Étapes :
     - ajout de features calendaires (année, mois, jour, jour de semaine, week‑of‑year, etc.),
     - calcul des lags et statistiques roulantes (moyenne, écart‑type) **par groupe** (`store_id`, `product_id`) pour éviter la fuite d’information entre séries,
     - encodage label‑encoding des colonnes catégorielles, avec dictionnaires d’encodage sérialisables.
   - Sortie : `data/processed/<stem>_features.csv`.

4. **Entraînement** (`models/train.py`)
  - Entrée : un CSV de features (généralement `uploaded_generated_training_10950_features.csv` ou `train_features.csv`).
   - Étapes :
     - normalisation des noms de colonnes (remap `id`→`store_id` si nécessaire),
     - application du pipeline de features group‑aware (cohérent avec `features.py`),
     - découpe temporelle en train/test par dates uniques (horizon configurable),
     - entraînement d’un `XGBRegressor` avec hyperparamètres modérés et validation croisée temporelle,
     - calcul et sauvegarde des métriques détaillées,
     - sérialisation du modèle (`model_YYYYMMDD_<tag>.pkl`) et d’un fichier `*_config.json` contenant :
       - la configuration de features (lags, windows),
       - les encoders catégoriels,
       - les noms de features,
       - le chemin précis du fichier de features d’entraînement.

5. **Inférence & service** (`models/predict.py`, `serve/api.py`)
   - `predict.py` :
     - charge le modèle le plus récent et la config associée,
     - reconstruit les features pour une série historique donnée en réutilisant les mêmes encoders et paramètres de lags/windows,
     - effectue une prévision auto‑régressive jour par jour sur un horizon donné, en injectant les prédictions successives dans l’historique,
     - applique des garde‑fous (clip à 0, gestion des NaN/inf).
   - `api.py` :
     - encapsule `predict_series` derrière une API HTTP avec schémas Pydantic,
     - journalise les requêtes de prédiction et les informations de modèle dans `models/metrics`.

## Choix de modélisation

- **Modèle principal** : XGBoost régressif (`XGBRegressor`) :
  - robuste, non linéaire, adapté à des combinaisons de features calendaires, lags et agrégations,
  - fonctionne bien sans normalisation stricte des features et gère nativement des encodages entiers pour les catégorielles.

- **Structure des séries** :
  - multi‑séries : chaque couple `(store_id, product_id)` représente une série indépendante,
  - les lags et rolling windows sont calculés **intra‑groupe** pour respecter la causalité.

- **Stratégie de validation** :
  - split temporel par dates uniques (les dernières dates vont en test), ce qui permet de conserver toutes les séries dans les deux ensembles tout en respectant l’ordre temporel,
  - validation croisée de type `TimeSeriesSplit` sur l’ensemble des données pour obtenir une estimation plus stable de la performance.

- **Métriques** :
  - MAE / RMSE / R² pour la performance globale,
  - MAPE et sMAPE calculés de façon robuste (exclusion ou traitement spécial des valeurs quasi nulles),
  - biais moyen (sur‑ ou sous‑prévision) pour détecter des dérives systématiques,
  - ratio d’overfitting (RMSE_test / RMSE_train).

- **Baselines** :
  - modèle `DummyRegressor` (moyenne) généré automatiquement si aucun artefact n’est présent,
  - prévision naïve (répétition de la dernière valeur) utilisée comme repli dans l’UI et l’API en cas d’indisponibilité du modèle entraîné.

## Logging et observabilité

- `src/utils/logging.py` fournit une configuration commune :
  - format : `timestamp | level | logger_name | message`,
  - niveau contrôlé via la variable d’environnement `LOG_LEVEL` (par défaut `INFO`).
- Les modules critiques (`models/train.py`, `models/evaluate.py`, `serve/api.py`) utilisent `get_logger(__name__)` pour :
  - tracer les chemins de données utilisés (features d’entraînement, artefacts modèle),
  - enregistrer les paramètres d’entraînement/évaluation et les avertissements (séries manquantes, features manquantes, horizon ajusté, etc.),
  - faciliter le débogage et le suivi en production.
