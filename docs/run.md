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
- Placer un modèle d'exemple dans `models/artifacts/` (ex: `model_20260127_v1.pkl`) pour tester l'API et l'UI.
- Les chemins et conventions sont définis dans `src/config.py`.
- Pour exécuter les tests :

```powershell
pytest -q
```
