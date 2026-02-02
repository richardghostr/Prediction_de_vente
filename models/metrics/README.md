
# Metrics — modèles de prévision

Ce dossier contient les fichiers de métriques produits par les runs de modèles (Prophet, XGBoost). Ce README décrit les fichiers présents, les clés JSON courantes et un petit exemple d'utilisation.

## Fichiers présents

- `prophet_metrics_final.json` : métriques finales issues du pipeline Prophet.
- `xgb_metrics_final.json` : métriques finales issues du pipeline XGBoost.
- `xgb_metrics_20260201.json` : métriques d'un run XGBoost horodaté (ex. debug / comparaison).

## Clés observées et signification

Les fichiers sont au format JSON. Voici les clés principales rencontrées :

- `MAE_test` : erreur moyenne absolue sur l'ensemble de test (plus bas = mieux).
- `RMSE_test` : racine de l'erreur quadratique sur test (pèse plus lourd les grosses erreurs).
- `MAE_train`, `RMSE_train` : mêmes métriques sur l'ensemble d'entraînement.
- `horizon_days` / `num_test_samples` / `num_train_samples` / `num_samples` : tailles des jeux (utile pour contextualiser les métriques).
- `CV_mae_mean` : moyenne des MAE calculés sur les folds de cross‑validation temporelle.
- `CV_rmse_mean` : moyenne des RMSE sur les folds (lorsqu'elle est fournie).
- `CV_mae_folds` : liste des valeurs MAE par fold (permet d'examiner la variance entre folds).
- `CV_folds` : nombre de folds utilisés dans la CV (si présent).

> Remarque : tous les fichiers n'ont pas exactement les mêmes clés (ex. certains runs incluent seulement `num_samples`), adaptez l'analyse en conséquence.

## Exemple d'utilisation (Python)

Voici un petit exemple pour charger et afficher les métriques de manière lisible :

```python
import json
from pathlib import Path

METRICS_DIR = Path(__file__).parent
for p in METRICS_DIR.glob('*.json'):
	with open(p, 'r', encoding='utf-8') as f:
		m = json.load(f)
	print(p.name)
	for k, v in m.items():
		print(f'  {k}: {v}')
	print()
```

## Interprétation rapide

- Comparez `MAE_test` et `MAE_train` : un écart important peut indiquer du sur‑apprentissage.
- Regardez `CV_mae_mean` et la variance dans `CV_mae_folds` : si la variance est élevée, le modèle est instable selon les périodes.
- Utilisez `num_test_samples` / `horizon_days` pour savoir si l'évaluation porte sur une fenêtre représentative.

## Bonnes pratiques

- Conserver l'historique des fichiers (noms horodatés) pour pouvoir comparer runs successifs.
- Normaliser l'interprétation en convertissant les métriques en unités métier compréhensibles (ex. € ou unités vendues).
- Sauvegarder également les fichiers de `forecast` correspondants (dossier `models/artifacts`) pour analyser cas d'erreur point par point.

Si vous voulez, je peux :
- ajouter un script `compare_metrics.py` pour résumer automatiquement les différences entre runs, ou
- ajouter des badges/une table synthétique dans ce README listant les derniers runs et leurs `MAE_test`.

