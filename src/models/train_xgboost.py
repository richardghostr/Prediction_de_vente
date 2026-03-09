"""Wrapper: exécute le notebook associé à l'entraînement XGBoost.

Ce fichier exécute `notebooks/02_modeling_xgboost.ipynb` via nbconvert
et renvoie l'artefact généré dans `models/artifacts` si présent.
"""
from pathlib import Path
import json
# nbformat and nbconvert are optional in the runtime; import lazily in _run_notebook


NOTEBOOK = Path(__file__).resolve().parents[2] / 'notebooks' / '02_modeling_xgboost.ipynb'


def _run_notebook(nb_path: Path, timeout: int = 600) -> Path:
    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
    except ImportError as e:
        raise RuntimeError("Missing dependency 'nbformat' or 'nbconvert'. Install with: pip install nbformat nbconvert") from e

    nb = nbformat.read(str(nb_path), as_version=4)
    ep = ExecutePreprocessor(timeout=timeout, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': str(nb_path.parent)}})
    # Do not write the executed notebook to disk; we only need to execute it.
    return None


def train_xgboost(timestamp: str = None) -> dict:
    if not NOTEBOOK.exists():
        raise FileNotFoundError(f"Notebook not found: {NOTEBOOK}")
    _run_notebook(NOTEBOOK)

    artifacts = Path('models') / 'artifacts'
    # Accept multiple naming conventions: look for any pkl with 'xgb' or 'xg' in name
    candidates = sorted([p for p in artifacts.glob('*.pkl') if 'xgb' in p.name.lower() or 'xg' in p.name.lower() or 'xgboost' in p.name.lower()])
    if not candidates:
        return {'model_path': None, 'metrics': None}
    latest = candidates[-1]

    # Find companion JSON (config/metrics) for the model using heuristics.
    cfg = None
    def _find_config(artifacts_dir: Path, model_path: Path):
        metrics_dir = Path('models') / 'metrics'
        jfiles = []
        jfiles.extend(sorted(artifacts_dir.glob('*.json')))
        if metrics_dir.exists():
            jfiles.extend(sorted(metrics_dir.glob('*.json')))
        if not jfiles:
            return None
        stem = model_path.stem.lower()
        for j in reversed(jfiles):
            name = j.name.lower()
            if stem in name and ('config' in name or 'metric' in name or 'metrics' in name):
                return j
        keywords = ['xgb', 'xg', 'xgboost']
        for j in reversed(jfiles):
            for kw in keywords:
                if kw in j.name.lower():
                    return j
        return jfiles[-1]

    cfg_path = _find_config(artifacts, latest)
    if cfg_path and cfg_path.exists():
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        except Exception:
            cfg = None

    metrics = None
    if cfg:
        if isinstance(cfg, dict) and 'metrics' in cfg:
            metrics = cfg.get('metrics')
        else:
            metrics = cfg

    return {'model_path': str(latest), 'metrics': metrics}

