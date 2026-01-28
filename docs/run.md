## Exécution locale (développement)

Pré-requis : Python 3.8+ et `pip`.

1) Créer et activer un environnement virtuel (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Installer les dépendances

```powershell
pip install -r requirements.txt
```

3) Préparer les dossiers (optionnel)

```powershell
python -c "from src.config import ensure_dirs; ensure_dirs()"
```

4) Lancer l'API (développement)

```powershell
uvicorn src.serve.api:app --reload --port 8000
```

5) Lancer l'UI (Streamlit)

```powershell
streamlit run src/ui/streamlit_app.py
```

Notes importantes

```powershell
pytest -q
```

## Politique de données brutes

- Ne committez PAS de jeux de données volumineux dans le dépôt.
- Utilisez `data/raw/` comme dossier local pour les données validées par `src/data/ingest.py`.
- Procédure courte : déposer le(s) fichier(s) source localement puis exécuter :

```powershell
python src/data/ingest.py <chemin_vers_csv_ou_dossier> data/raw
```

- Ajoutez des métadonnées pour chaque ingestion (voir `data/raw/README.md`).
