"""Features temporelles et agrégations.

Entrées / sorties (contrat):
- Entrée attendue : CSV nettoyé produit par `clean.py`, nommé `*_clean.csv` (colonnes minimales `date`, `value`).
- Sortie : DataFrame enrichi avec colonnes de features. Le runner écrit par défaut `data/processed/*_features.csv`.

Exemple d'utilisation :
    python src/data/features.py --in data/interim --out data/processed --lags 1,7 --windows 3,7
"""

from typing import List, Optional

import pandas as pd


def add_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["dayofweek"] = df[date_col].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    return df


def add_lags(
    df: pd.DataFrame,
    value_col: str = "value",
    lags: Optional[List[int]] = None,
    date_col: str = "date",
) -> pd.DataFrame:
    df = df.copy()
    if lags is None:
        lags = [1]
    df = df.sort_values(by=date_col).reset_index(drop=True)
    for lag in lags:
        df[f"lag_{lag}"] = df[value_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    value_col: str = "value",
    windows: Optional[List[int]] = None,
    date_col: str = "date",
) -> pd.DataFrame:
    df = df.copy()
    if windows is None:
        windows = [3]
    df = df.sort_values(by=date_col)
    # IMPORTANT: shift(1) to avoid data leakage - rolling stats must use only past values
    shifted = df[value_col].shift(1)
    for w in windows:
        df[f"roll_mean_{w}"] = shifted.rolling(window=w, min_periods=w).mean()
        df[f"roll_std_{w}"] = shifted.rolling(window=w, min_periods=w).std().fillna(0)
    return df


def encode_categorical(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    df = df.copy()
    if cols is None:
        cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cols:
        df[c] = pd.Categorical(df[c]).codes
    return df


def build_feature_pipeline(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "value",
    lags: Optional[List[int]] = None,
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    df = df.copy()
    df = add_time_features(df, date_col=date_col)
    df = add_lags(df, value_col=value_col, lags=lags, date_col=date_col)
    df = add_rolling_features(df, value_col=value_col, windows=windows, date_col=date_col)
    df = encode_categorical(df)
    return df


def parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Calcul des features à partir des CSV nettoyés (par défaut : data/interim -> data/processed)")
    parser.add_argument("--in", dest="input", default="data/interim", help="Fichier ou dossier d'entrée (défaut : data/interim)")
    parser.add_argument("--out", dest="output", default="data/processed", help="Dossier de sortie pour les CSV de features (défaut : data/processed)")
    parser.add_argument("--lags", dest="lags", default="1", help="Lags séparés par virgule, ex. '1,7' (défaut : 1)")
    parser.add_argument("--windows", dest="windows", default="3", help="Fenêtres rolling séparées par virgule, ex. '3,7' (défaut : 3)")
    parser.add_argument("--pattern", dest="pattern", default="*_clean.csv", help="Pattern glob pour sélectionner les fichiers d'entrée (défaut : '*_clean.csv')")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false", help="Désactiver les messages")
    return parser.parse_args(argv)


def _parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main(argv=None) -> int:
    args = parse_args(argv)
    input_path = args.input
    output_dir = args.output
    lags = _parse_int_list(args.lags)
    windows = _parse_int_list(args.windows)
    pattern = args.pattern
    verbose = args.verbose

    from pathlib import Path

    inp = Path(input_path)
    outd = Path(output_dir)

    try:
        if inp.is_file():
            df = pd.read_csv(inp)
            feat = build_feature_pipeline(df, lags=lags, windows=windows)
            outd.mkdir(parents=True, exist_ok=True)
            out_name = f"{inp.stem.replace('_clean','')}_features{inp.suffix}"
            outp = outd / out_name
            feat.to_csv(outp, index=False)
            if verbose:
                print(f"[features] écrit {outp}")
        elif inp.is_dir():
            files = list(inp.glob(pattern))
            if not files:
                print(f"[features] aucun fichier correspondant à {pattern} dans {inp}")
                return 2
            outd.mkdir(parents=True, exist_ok=True)
            for f in files:
                try:
                    df = pd.read_csv(f)
                except Exception as e:
                    print(f"[features] ignoré {f} : {e}")
                    continue
                feat = build_feature_pipeline(df, lags=lags, windows=windows)
                out_name = f"{f.stem.replace('_clean','')}_features{f.suffix}"
                outp = outd / out_name
                feat.to_csv(outp, index=False)
                if verbose:
                    print(f"[features] écrit {outp}")
        else:
            print(f"[features] entrée invalide : {inp}")
            return 2
    except Exception as e:
        print(f"[features] erreur : {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
