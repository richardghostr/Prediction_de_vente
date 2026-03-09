"""Wrapper: exécute le notebook associé à l'entraînement LightGBM.

Ce fichier exécute `notebooks/03_modeling_lightgbm.ipynb` via nbconvert
et renvoie l'artefact généré dans `models/artifacts` si présent.
"""
from pathlib import Path
import json
import sys
# nbformat and nbconvert are optional in the runtime; import lazily in _run_notebook


NOTEBOOK = Path(__file__).resolve().parents[2] / 'notebooks' / '03_modeling_lightgbm.ipynb'


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


def train_lightgbm(timestamp: str = None) -> dict:
    if not NOTEBOOK.exists():
        raise FileNotFoundError(f"Notebook not found: {NOTEBOOK}")
    _run_notebook(NOTEBOOK)

    # After execution, try to find latest LightGBM artifact
    artifacts = Path('models') / 'artifacts'
    # Accept multiple naming conventions produced by notebooks: look for any pkl
    # that contains 'lgb' or 'lgbm' in the filename.
    candidates = sorted([p for p in artifacts.glob('*.pkl') if 'lgb' in p.name.lower() or 'lgbm' in p.name.lower() or 'lightgbm' in p.name.lower()])
    if not candidates:
        return {'model_path': None, 'metrics': None}
    latest = candidates[-1]

    # Find companion JSON (config/metrics) for the model. Notebooks may
    # produce varied filenames, so attempt several heuristics.
    cfg = None
    def _find_config(artifacts_dir: Path, model_path: Path):
        # Look in both artifacts and a shared metrics directory
        metrics_dir = Path('models') / 'metrics'
        jfiles = []
        jfiles.extend(sorted(artifacts_dir.glob('*.json')))
        if metrics_dir.exists():
            jfiles.extend(sorted(metrics_dir.glob('*.json')))
        if not jfiles:
            return None
        stem = model_path.stem.lower()
        # Prefer json files that include the model stem and 'config' or 'metrics'
        for j in reversed(jfiles):
            name = j.name.lower()
            if stem in name and ('config' in name or 'metric' in name or 'metrics' in name):
                return j
        # Fallback to any json containing model keyword
        keywords = ['lgb', 'lgbm', 'lightgbm']
        for j in reversed(jfiles):
            for kw in keywords:
                if kw in j.name.lower():
                    return j
        # Last resort: return newest json
        return jfiles[-1]

    cfg_path = _find_config(artifacts, latest)
    if cfg_path and cfg_path.exists():
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        except Exception:
            cfg = None

    # Some notebooks dump metrics at the top level of the JSON (not under 'metrics').
    metrics = None
    if cfg:
        if isinstance(cfg, dict) and 'metrics' in cfg:
            metrics = cfg.get('metrics')
        else:
            metrics = cfg

    return {'model_path': str(latest), 'metrics': metrics}

