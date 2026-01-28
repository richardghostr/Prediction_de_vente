Scénario utilisateur — "Le Rush du Lundi Matin"
=================================================

Résumé
------
Utilisateur: Marc, Responsable Régional (boulangerie)
Objectif: Ajuster commandes et planning pour la semaine afin d'éviter le gaspillage.

Flux (vue métier → fichier/étape technique)
-------------------------------------------------

1. Accès et Connexion (Phase 7 — Docker & Run)
   - Marc ouvre l'interface Streamlit déployée dans un conteneur.
   - Fichiers clés: `Dockerfile`, `docker-compose.yml`, `src/ui/streamlit_app.py`.
   - Responsable: Personne B (packaging, déploiement).

2. Ingestion des données fraîches (Phase 1 — Ingestion)
   - Marc téléverse le CSV des ventes via l'UI.
   - Le backend appelle `src/data/ingest.py` qui valide, normalise et écrit dans `data/raw/`.
   - Produit annexe: `data/raw/<timestamp>_sales.csv` + `data/raw/<timestamp>_sales.metadata.json`.
   - Responsable: Personne A (logique d'ingest), Personne B (upload UI + orchestration).

3. Nettoyage et Préparation (Phase 2 — Features)
   - `src/data/clean.py` applique corrections (jours fériés, outliers).
   - `src/data/features.py` crée lags, rolling, ajoute variables externes (météo, événements locaux).
   - Responsable: Personne A.

4. Demande de Prédiction (Phase 4 & 5 — API & Schémas)
   - UI envoie POST `/predict` au serveur FastAPI (`src/serve/api.py`).
   - Requête validée avec `src/serve/schemas.py` (Pydantic).
   - L'API charge le modèle depuis `models/artifacts/` (ex. `model_v1.pkl`) et exécute `predict()`.
   - Responsable: Personne B (API), Personne A (modèle et logique predict).

5. Visualisation et Décision (Phase 6 — UI)
   - Streamlit affiche courbes des ventes réelles et prévisions (7 jours), et alerte sur anomalies.
   - L'utilisateur peut télécharger un CSV récapitulatif par boutique.
   - Fichiers/UI: `src/ui/streamlit_app.py`, `docs/examples/predict_example.json`.

6. Monitoring et Amélioration (Phase 8 & 9 — Qualité & Observabilité)
   - Chaque prédiction est journalisée (logs + métriques JSON dans `models/metrics/`).
   - Un job de monitoring alerte Personne A si les métriques chutent (RMSE augmente, dérive détectée).

Critères d'acceptation (AC)
---------------------------
- AC1 — Accessibilité: L'UI Streamlit est disponible via `docker compose up` et accessible en interne.
- AC2 — Ingest: Téléversement CSV via UI génère un fichier CSV validé dans `data/raw/` et une metadata JSON.
- AC3 — Préparation: `clean.py` et `features.py` transforment les données sans erreurs pour l'échantillon fourni.
- AC4 — API: `/predict` accepte la payload validée et retourne un JSON de prévisions avec code 200.
- AC5 — Visualisation: L'UI affiche graphes et propose un bouton pour télécharger le récapitulatif.
- AC6 — Observabilité: Chaque appel de prédiction est enregistré, et une métrique sommaire est écrite dans `models/metrics/`.

Tests d'acceptation proposés
---------------------------
- Test E2E minimal (automatisable):
  1. Démarrer services (api minimal + db si nécessaire) en local.
  2. Simuler upload d'un petit `sample_sales.csv` via l'endpoint d'upload ou en appelant `ingest.py` directement.
  3. Lancer les scripts `clean.py` et `features.py` (ou appeler un wrapper qui exécute les deux).
  4. Appeler `/predict` avec la payload produite.
  5. Vérifier: réponse 200, JSON de longueur 7 (7 jours), fichier metrics écrit, et artefact model chargé sans erreur.

Trace de responsabilité (qui fait quoi)
-------------------------------------
- Personne A: construire `ingest.py` robuste, `clean.py`, `features.py`, entraîner et versionner modèle (`models/artifacts/`).
- Personne B: intégrer l'upload à l'UI, fournir `src/serve/api.py`, packager en Docker, mettre en place monitoring/CI.

Prochaines tâches recommandées
------------------------------
1. Ajouter ce scénario aux documents utilisateur (fait).
2. Implémenter le test E2E automatisé (CI) qui suit les étapes ci‑dessus.
3. Standardiser le format metadata JSON écrit lors de l'ingest (checksum, timestamp, source, row_count).

Fichiers de référence dans le repo
---------------------------------
- Ingest: `src/data/ingest.py`
- Cleaning: `src/data/clean.py`
- Features: `src/data/features.py`
- API: `src/serve/api.py`, `src/serve/schemas.py`
- UI: `src/ui/streamlit_app.py`
- Models: `models/artifacts/`
- Metrics: `models/metrics/`

---
Rédigé pour faciliter l'implémentation technique et la validation métier — dites si vous voulez que je génère le test E2E automatisé maintenant.
