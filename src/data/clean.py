"""Nettoyage des donnees brutes pour series temporelles de vente.

Pipeline: harmonisation des dates, remplissage des valeurs manquantes,
detection/suppression des outliers.

Entrees / sorties (contrat):
- Entree attendue : CSV brut produit par `ingest.py`, nomme `*_ingest.csv`
  (colonnes minimales `date`, `value`).
- Sortie : CSV nettoye ecrit dans `data/interim` sous le nom `*_clean.csv`.

Exemple d'utilisation :
    python src/data/clean.py --in data/raw --out data/interim --fill ffill --outlier 3.0
"""

import os
from typing import Optional, List

import numpy as np
import pandas as pd


def harmonize_dates(df: pd.DataFrame, date_col: str = "date", fmt: Optional[str] = None) -> pd.DataFrame:
    """Convertit la colonne de date en `pd.Timestamp`, supprime le fuseau et trie.

    Les dates non analysables deviennent `NaT`.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], format=fmt, errors="coerce")
    try:
        if df[date_col].dt.tz is not None:
            df[date_col] = df[date_col].dt.tz_convert(None)
    except Exception:
        pass
    df = df.sort_values(by=date_col).reset_index(drop=True)
    return df


def fill_missing(df: pd.DataFrame, strategy: str = "ffill", numeric_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Remplit les valeurs manquantes selon une strategie simple.

    strategy: 'ffill', 'bfill', 'mean' ou 'zero'
    """
    df = df.copy()
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if strategy == "ffill":
        df[numeric_cols] = df[numeric_cols].ffill()
    elif strategy == "bfill":
        df[numeric_cols] = df[numeric_cols].bfill()
    elif strategy == "mean":
        for c in numeric_cols:
            df[c] = df[c].fillna(df[c].mean())
    elif strategy == "zero":
        df[numeric_cols] = df[numeric_cols].fillna(0)
    else:
        raise ValueError(f"Strategie inconnue : {strategy}")

    non_numeric = [c for c in df.columns if c not in numeric_cols]
    if non_numeric:
        df[non_numeric] = df[non_numeric].ffill().bfill()

    return df


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Renvoie un masque booleen indiquant les outliers selon le z-score."""
    s = series.dropna()
    if s.empty:
        return pd.Series([False] * len(series), index=series.index)
    mean = s.mean()
    std = s.std()
    if std == 0 or np.isnan(std):
        return pd.Series([False] * len(series), index=series.index)
    z = (series - mean) / std
    return z.abs() > threshold


def remove_outliers(df: pd.DataFrame, col: str, threshold: float = 3.0) -> pd.DataFrame:
    """Supprime les lignes ou `col` est un outlier selon le z-score."""
    mask = detect_outliers_zscore(df[col], threshold=threshold)
    return df.loc[~mask].reset_index(drop=True)


def clean_dataframe(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "value",
    fill_strategy: str = "ffill",
    outlier_threshold: Optional[float] = 3.0,
) -> pd.DataFrame:
    """Pipeline de nettoyage minimal : dates, remplissage, suppression d'outliers."""
    df = df.copy()
    if date_col in df.columns:
        df = harmonize_dates(df, date_col=date_col)

    if value_col in df.columns:
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    df = fill_missing(df, strategy=fill_strategy, numeric_cols=[value_col] if value_col in df.columns else None)

    if outlier_threshold is not None and outlier_threshold > 0 and value_col in df.columns:
        df = remove_outliers(df, col=value_col, threshold=outlier_threshold)

    return df.reset_index(drop=True)


def save_interim(df: pd.DataFrame, out_path: str) -> str:
    """Enregistre le DataFrame en CSV, cree les dossiers si necessaire. Retourne le chemin."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Execute le nettoyage sur des CSV (defaut : data/raw -> data/interim)")
    parser.add_argument("--in", dest="input", default="data/raw", help="Fichier ou dossier d'entree (defaut : data/raw)")
    parser.add_argument("--out", dest="output", default="data/interim", help="Dossier de sortie (defaut : data/interim)")
    parser.add_argument("--fill", dest="fill", default="ffill", choices=["ffill", "bfill", "mean", "zero"], help="Strategie de remplissage des valeurs manquantes")
    parser.add_argument("--outlier", dest="outlier", type=float, default=3.0, help="Seuil z-score pour retirer les outliers (0 pour desactiver)")
    parser.add_argument("--pattern", dest="pattern", default="*_ingest.csv", help="Pattern glob pour selectionner les fichiers d'entree")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false", help="Desactiver les messages")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    input_path = args.input
    output_dir = args.output
    fill_strategy = args.fill
    outlier_threshold = args.outlier
    pattern = args.pattern
    verbose = args.verbose

    from pathlib import Path

    inp = Path(input_path)
    outd = Path(output_dir)

    try:
        if inp.is_file():
            df = pd.read_csv(inp)
            cleaned = clean_dataframe(
                df,
                date_col="date",
                value_col="value",
                fill_strategy=fill_strategy,
                outlier_threshold=(None if outlier_threshold == 0 else outlier_threshold),
            )
            out_name = f"{inp.stem.replace('_ingest', '')}_clean{inp.suffix}"
            outp = outd / out_name
            save_interim(cleaned, str(outp))
            if verbose:
                print(f"[clean] ecrit {outp}")
        elif inp.is_dir():
            files = list(inp.glob(pattern))
            if not files:
                print(f"[clean] aucun fichier correspondant a {pattern} dans {inp}")
                return 2
            outd.mkdir(parents=True, exist_ok=True)
            for f in files:
                try:
                    df = pd.read_csv(f)
                except Exception as e:
                    print(f"[clean] ignore {f} : {e}")
                    continue
                cleaned = clean_dataframe(
                    df,
                    date_col="date",
                    value_col="value",
                    fill_strategy=fill_strategy,
                    outlier_threshold=(None if outlier_threshold == 0 else outlier_threshold),
                )
                out_name = f"{f.stem.replace('_ingest', '')}_clean{f.suffix}"
                outp = outd / out_name
                save_interim(cleaned, str(outp))
                if verbose:
                    print(f"[clean] ecrit {outp}")
        else:
            print(f"[clean] entree invalide : {inp}")
            return 2
    except Exception as e:
        print(f"[clean] erreur : {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
