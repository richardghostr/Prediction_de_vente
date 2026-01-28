# Politique et emplacement des données brutes

Ce dossier `data/raw/` est l'emplacement local prévu pour stocker les jeux de données bruts utilisés par le projet.

Règles générales
- Ne pas committer les jeux de données volumineux dans le dépôt Git.
- Conserver uniquement de petits exemples ou des fichiers d'exemple (sample) si nécessaire pour les tests.
- Pour chaque jeu déposé, ajouter un fichier de métadonnées associé (ex: `metadata_<nom_fichier>.json`) contenant : source, date_récupération, checksums, rows, colonnes.

Procédure d'ingestion (résumé)
1. Placer temporairement le/les fichiers sources dans un emplacement local (ex: dossier temporaire hors du repo).
2. Lancer le script d'ingestion pour valider et copier dans `data/raw/` :
   ```powershell
   python src/data/ingest.py <chemin_vers_csv_ou_dossier> data/raw
   ```
3. Le script va valider les colonnes minimales, normaliser le format et écrire le(s) fichier(s) dans `data/raw/`.

Bonnes pratiques
- Ajouter une entrée dans `.gitignore` pour éviter d'ajouter des CSV volumineux accidentellement.
- Versionner les artefacts de modèle dans `models/artifacts/` (nommage `model_<YYYYMMDD>_<tag>.pkl`) et conserver les métadonnées dans `models/metrics/`.
- Documenter la provenance des données dans `data/raw/README.md` et `docs/`.

Contact
- En cas de doute sur la sensibilité ou la taille des données, contacter Personne A (Data) avant tout commit.
