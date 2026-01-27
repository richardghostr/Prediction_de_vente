# Plan chronologique détaillé des tâches (Personne A & Personne B)

Ce document décrit, dans l'ordre chronologique recommandé, toutes les tâches à effectuer pour mener à bien le projet de prévision des ventes. Pour chaque tâche : rôle principal (A ou B), fichiers à modifier/créer, dépendances, livrables et notes de coordination.

---

## Préambule
- Règle générale : petites PRs, revues croisées (A <-> B).
- Branches : `feat/<who>/<desc>` (ex: `feat/B/api-predict`).
- Les chemins indiqués sont relatifs à la racine du dépôt.

---

## Phase 0 — Initialisation et configuration (Journée 0 — 1)
Objectif : préparer l'environnement et conventions partagées.

- Tâche 0.1 — Définir config partagée (A+B)
  - Rôle : A+B (coordination)
  - Fichiers : `src/config.py`, `docs/run.md`
  - Description : définir variables (paths: `data/`, `models/artifacts/`, seed, formats de date). Petite PR coordonnée.
  - Livrable : `src/config.py` stable, `docs/run.md` avec commande d'exécution dev.

- Tâche 0.2 — Remplir `requirements.txt` (B propose, A valide)
  - Rôle : B (proposition) / A (validation)
  - Fichiers : `requirements.txt`
  - Description : lister versions minimales (`pandas`, `numpy`, `scikit-learn`, `xgboost`, `prophet?`, `pydantic`, `fastapi`, `uvicorn`, `streamlit`, `joblib`, `pytest`).
  - Livrable : fichier `requirements.txt` utilisable pour `pip install -r`.

---

## Phase 1 — Ingestion & structure des données (Jours 1–3)
Objectif : garantir un pipeline simple pour entrer les données brutes.

- Tâche 1.1 — Ingestion minimale (Personne A)
  - Rôle : A
  - Fichiers : `src/data/ingest.py`, `data/raw/README.md` (ajout)
  - Description : script léger pour lire CSVs d'exemple, valider schéma minimal et écrire dans `data/raw/`.
  - Dépendances : `pandas`.
  - Livrable : `src/data/ingest.py` + instructions d'utilisation.

- Tâche 1.2 — Politique de données (Personne B rédige, A approuve)
  - Rôle : B rédige, A approuve
  - Fichiers : `data/raw/README.md`, `docs/run.md`
  - Description : indiquer emplacement attendu, règle: ne pas commit gros fichiers.

---

## Phase 2 — Nettoyage & features de base (Jours 2–6)
Objectif : produire jeux `data/interim/` et fonctions de feature engineering testables.

- Tâche 2.1 — Nettoyage (Personne A)
  - Rôle : A
  - Fichiers : `src/data/clean.py`, `data/interim/` (outputs)
  - Description : fonctions pour harmoniser dates, traiter NaN, détecter outliers basiques.
  - Livrable : scripts et exemples `data/interim/*.csv`.

- Tâche 2.2 — Feature engineering (Personne A)
  - Rôle : A
  - Fichiers : `src/data/features.py`, `tests/unit/test_features.py`
  - Description : fonctions pures produisant lags, rolling, features temporelles (dayofweek, month), encodage catégoriel.
  - Tests unitaires : écrire tests couvrant chaque transformation.
  - Livrable : fonctions testées importables.

- Note coordination : B ne doit pas changer le format des outputs sans prévenir A — contract-first.

---

## Phase 3 — Baselines & notebooks (Jours 4–10)
Objectif : itérer rapidement sur modèles simples pour fournir artefacts d'exemple.

- Tâche 3.1 — Baselines exploratoires (Personne A)
  - Rôle : A
  - Fichiers : `notebooks/00_eda.ipynb`, `notebooks/01_baselines_prophet.ipynb`
  - Description : EDA et une baseline Prophet ou ARIMA pour avoir référence.
  - Livrable : notebook avec résultats, premières métriques sauvegardées dans `models/metrics/`.

- Tâche 3.2 — Modèle XGBoost simple (Personne A)
  - Rôle : A
  - Fichiers : `src/models/train.py`, `src/models/predict.py`, `models/artifacts/` (ex: `model_YYYYMMDD_baseline.pkl`), `models/metrics/*.json`
  - Description : pipeline d'entraînement (préprocessing minimal via `src/data/features.py`), sauvegarde model + métriques.
  - Livrable : artifact modèle et métriques JSON.

- Dépendance pour B : A doit fournir un modèle d'exemple (ex: `models/artifacts/sample_model.pkl`) et la fonction `predict(sample_df, horizon)` pour que B implémente l'API.

---

## Phase 4 — Contract-first & schémas API (Jours 8–11)
Objectif : définir contract d'API (A fournit signature, B rédige schémas Pydantic).

- Tâche 4.1 — Fournir `predict()` contract (Personne A)
  - Rôle : A
  - Fichiers : `src/models/predict.py`, `docs/examples/predict_example.json`
  - Description : documenter la signature `predict(sample_df, horizon) -> DataFrame/JSON`, types attendus, colonnes obligatoires.
  - Livrable : script `src/models/predict.py` ou notebook snippet + petit dataset exemple.

- Tâche 4.2 — Schémas Pydantic & exemples (Personne B)
  - Rôle : B
  - Fichiers : `src/serve/schemas.py`, `docs/examples/predict_example.json` (déjà créé)
  - Description : valider les exemples fournis par A, affiner les schémas d'entrée/sortie.

---

## Phase 5 — API minimal & tests unitaires (Jours 10–14)
Objectif : exposer l'endpoint `/predict` localement et valider son comportement.

- Tâche 5.1 — Implémenter API minimal (Personne B)
  - Rôle : B
  - Fichiers : `src/serve/api.py`, `src/serve/schemas.py`, `src/config.py`
  - Description : endpoints `/predict` (POST), `/health` (GET), `/metrics` (GET). Loader modèle en lecture seule depuis `models/artifacts/` ; appeler `predict()` fourni par A ou wrapper local.
  - Livrable : API démarrable via `uvicorn src.serve.api:app --reload`.

- Tâche 5.2 — Tests API (Personne B)
  - Rôle : B
  - Fichiers : `tests/unit/test_api.py`, `tests/integration/test_end_to_end.py`
  - Description : tests de validation d'input, cas d'erreur, fonctionnement `/health`.
  - Livrable : suite de tests exécutables via `pytest`.

- Coordination : A vérifie que réponses respectent le format convenu.

---

## Phase 6 — UI Streamlit (Jours 12–16)
Objectif : fournir une interface simple pour charger jeux et afficher prédictions.

- Tâche 6.1 — Streamlit minimal (Personne B)
  - Rôle : B
  - Fichiers : `src/ui/streamlit_app.py`, `docs/run.md`
  - Description : upload CSV, sélection horizon, bouton `Predict` qui appelle l'API `/predict` (ou charge modèle local si mode offline).
  - Livrable : UI utilisable localement (`streamlit run src/ui/streamlit_app.py`).

- Tâche 6.2 — Démo & screenshots (B)
  - Rôle : B
  - Fichiers : `reports/figures/`, `README.md` (section demo)
  - Description : captures, petites instructions pour demo.

---

## Phase 7 — Conteneurisation & run (Jours 14–18)
Objectif : packager API + UI en image Docker et faciliter démarrage local via `docker-compose`.

- Tâche 7.1 — Dockerfile & Compose (Personne B)
  - Rôle : B
  - Fichiers : `Dockerfile`, `docker-compose.yml`, `infra/*` (scripts)
  - Description : préparer images pour API et UI, variables d'environnement pour emplacement des artefacts.
  - Livrable : image Docker testée localement.

- Tâche 7.2 — Documentation run (Personne B)
  - Rôle : B
  - Fichiers : `docs/run.md`, `README.md`
  - Description : instructions `docker-compose up`, build image, où déposer modèle si non inclus.

---

## Phase 8 — Tests d'intégration, CI/CD et qualité (Jours 16–22)
Objectif : valider bout-en-bout et automatiser checks.

- Tâche 8.1 — Tests d'intégration (A+B)
  - Rôle : A+B
  - Fichiers : `tests/integration/test_end_to_end.py`, `tests/unit/*`
  - Description : test qui fait pipeline complet : ingestion -> features -> train (ou modèle fourni) -> API `/predict` -> assertions sur shape/valeurs.

- Tâche 8.2 — Pipeline CI (B lead)
  - Rôle : B
  - Fichiers : `.github/workflows/ci.yml` (ou équivalent)
  - Description : lint, `pip install -r requirements.txt`, tests unitaires + integration, build Docker image.

---

## Phase 9 — Observabilité, monitoring et production hardening (Jours 22–30)
Objectif : garantir fonctionnement durable et capacités de diagnostic.

- Tâche 9.1 — Logs & metrics (Personne B)
  - Rôle : B
  - Fichiers : `src/utils/logging.py`, `src/serve/api.py` (instrumentation), `docs/run.md`
  - Description : logs structurés (JSON), endpoint `/metrics` simple, exporter métriques basiques.

- Tâche 9.2 — Politique d'artifacts (A lead, B surveille)
  - Rôle : A+B
  - Fichiers : `models/artifacts/` (naming), `archive/` (historique), `models/metrics/`
  - Description : convention `model_<YYYYMMDD>_<tag>.pkl`, metadata JSON accompagnant chaque modèle.

---

## Phase 10 — Handover, documentation finale et formation (Jours 30+)
Objectif : stabiliser, documenter et préparer production ou démonstration.

- Tâche 10.1 — Document final & formation (B)
  - Rôle : B
  - Fichiers : `README.md`, `docs/run.md`, `docs/features_and_roles.md`, `docs/roles_tasks.md`
  - Description : mise à jour finale pour onboarding d'un 3ème contributeur, checklist avant merge vers `main`.

- Tâche 10.2 — Archivage modèle (A)
  - Rôle : A
  - Fichiers : `archive/`, `models/metrics/`
  - Description : ajouter modèle stable + metadata, snapshot des notebooks.

---

## Remarques de coordination importantes
- Toujours créer une issue pour tout changement majeur de contrat (ex: nouvelles colonnes d'entrée).
- Les PRs doivent être petites ; chaque PR modifiant `src/serve/*` doit inclure un test API et un exemple dans `docs/examples/`.
- Personne A fournit au minimum un modèle d'exemple et la fonction `predict()` avant que B implémente l'API.
- Respecter la règle lecture seule pour `models/artifacts/` côté B : B lit uniquement les artefacts fournis par A.

---

## Fichiers-clés récapitulatif par rôle

- Personne A (Data / Modélisation) :
  - `src/data/ingest.py`, `src/data/clean.py`, `src/data/features.py`
  - `src/models/train.py`, `src/models/predict.py`, `src/models/evaluate.py`
  - `notebooks/*`, `models/artifacts/*`, `models/metrics/*`

- Personne B (Produit / Déploiement) :
  - `src/serve/api.py`, `src/serve/schemas.py`, `src/ui/streamlit_app.py`
  - `Dockerfile`, `docker-compose.yml`, `infra/*`, `docs/run.md`, `src/utils/logging.py`

---

Ce plan est prêt à être utilisé comme checklist partagée. Si vous voulez, je peux :
- générer automatiquement les issues GitHub (si repo connecté),
- créer des templates de PR et de tests unitaires de base, ou
- commencer l'implémentation d'une des tâches (précisez laquelle).
