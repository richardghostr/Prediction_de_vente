import os
from typing import Optional, List

import numpy as np
import pandas as pd


def harmonize_dates(df: pd.DataFrame, date_col: str = "date", fmt: Optional[str] = None) -> pd.DataFrame:
    """Normalize la colonne de date.

    - Convertit `date_col` en datetime via `pd.to_datetime` (les erreurs deviennent NaT).
    - Supprime les informations de fuseau si présentes (timezone-aware -> naive).
    - Trie les lignes par date et réindexe.

    Arguments:
    - df: DataFrame d'entrée
    - date_col: nom de la colonne contenant la date
    - fmt: format explicite de date (optionnel)

    Retour: DataFrame copié et trié.
    """
    df = df.copy()
    # Conversion robuste des dates (les valeurs non-parsables deviennent NaT)
    df[date_col] = pd.to_datetime(df[date_col], format=fmt, errors="coerce")

    # Si la série a des tz-aware timestamps, on retire le fuseau pour conserver la cohérence
    try:
        if df[date_col].dt.tz is not None:
            df[date_col] = df[date_col].dt.tz_convert(None)
    except Exception:
        # catch général : si la colonne n'est pas datetime, on laisse tel quel
        pass

    # Tri et réindexation pour faciliter les opérations de séries temporelles ultérieures
    df = df.sort_values(by=date_col).reset_index(drop=True)
    return df


def fill_missing(df: pd.DataFrame, strategy: str = "ffill", numeric_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Remplit les valeurs manquantes selon une stratégie simple.

    - `ffill`: propagation vers l'avant
    - `bfill`: propagation vers l'arrière
    - `mean`: remplissage par la moyenne (numérique)
    - `zero`: remplacer par 0 (numérique)

    Non-numériques: on applique `ffill` puis `bfill` pour tenter de restaurer une valeur.
    """
    df = df.copy()
    if numeric_cols is None:
        # Détecte automatiquement les colonnes numériques si non fournies
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
        raise ValueError(f"Stratégie inconnue : {strategy}")

    # Pour les colonnes non numériques, forward/backward fill comme heuristique
    non_numeric = [c for c in df.columns if c not in numeric_cols]
    if non_numeric:
        df[non_numeric] = df[non_numeric].ffill().bfill()

    return df


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Renvoie un masque (bool) indiquant les outliers par z-score.

    - Si l'écart-type est nul ou la série vide, aucun outlier n'est signalé.
    - Les NaN sont ignorés dans le calcul du z-score mais le masque a la même longueur que `series`.
    """
    s = series.dropna()
    if s.empty:
        return pd.Series([False] * len(series), index=series.index)

    mean = s.mean()
    std = s.std()
    if std == 0 or np.isnan(std):
        return pd.Series([False] * len(series), index=series.index)

    z = (series - mean) / std
    # Retourne True pour les indices où la valeur dépasse le seuil absolu
    return z.abs() > threshold


def remove_outliers(df: pd.DataFrame, col: str, threshold: float = 3.0) -> pd.DataFrame:
    """Supprime les lignes où la colonne `col` est identifiée comme outlier.

    - Utilise `detect_outliers_zscore` pour construire le masque.
    - Réindexe le DataFrame résultant.
    """
    mask = detect_outliers_zscore(df[col], threshold=threshold)
    return df.loc[~mask].reset_index(drop=True)


def clean_dataframe(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "value",
    fill_strategy: str = "ffill",
    outlier_threshold: Optional[float] = 3.0,
) -> pd.DataFrame:
    """Pipeline de nettoyage léger pour séries temporelles simples.

    Étapes réalisées:
    1. Normalisation de la colonne de date (`harmonize_dates`).
    2. Conversion de `value_col` en numérique (coerce -> NaN).
    3. Remplissage des valeurs manquantes selon `fill_strategy`.
    4. Suppression optionnelle des outliers si `outlier_threshold` > 0.

    Retourne un DataFrame réindexé.
    """
    df = df.copy()

    # 1) Dates
    df = harmonize_dates(df, date_col=date_col)

    # 2) Valeurs numériques
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # 3) Remplissage
    df = fill_missing(df, strategy=fill_strategy, numeric_cols=[value_col])

    # 4) Outliers (optionnel)
    if outlier_threshold is not None and outlier_threshold > 0:
        df = remove_outliers(df, col=value_col, threshold=outlier_threshold)

    return df.reset_index(drop=True)


def save_interim(df: pd.DataFrame, out_path: str) -> str:
    """Enregistre le DataFrame en CSV; crée le répertoire parent si nécessaire.

    Retourne le chemin du fichier écrit pour usage en sortie CLI.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def parse_args(argv=None):
    """Analyse des arguments CLI pour le script de nettoyage.

    Arguments pris en charge (exemples):
    --in    : fichier ou dossier d'entrée (par défaut `data/raw`)
    --out   : dossier de sortie (par défaut `data/interim`)
    --fill  : stratégie de remplissage (`ffill`, `bfill`, `mean`, `zero`)
    --outlier: seuil z-score pour retirer les outliers (0 pour désactiver)
    --pattern: glob pattern pour sélectionner les fichiers (par défaut `*_ingest.csv`)
    """
    import argparse

    parser = argparse.ArgumentParser(description="Exécute le nettoyage sur des CSV (par défaut : data/raw -> data/interim)")
    parser.add_argument("--in", dest="input", default="data/raw", help="Fichier ou dossier d'entrée (défaut : data/raw)")
    parser.add_argument("--out", dest="output", default="data/interim", help="Dossier de sortie (défaut : data/interim)")
    parser.add_argument("--fill", dest="fill", default="ffill", choices=["ffill", "bfill", "mean", "zero"], help="Stratégie de remplissage des valeurs manquantes")
    parser.add_argument("--outlier", dest="outlier", type=float, default=3.0, help="Seuil z-score pour retirer les outliers (mettre 0 pour désactiver)")
    parser.add_argument("--pattern", dest="pattern", default="*_ingest.csv", help="Pattern glob pour sélectionner les fichiers d'entrée")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false", help="Désactiver les messages")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Point d'entrée CLI. Gère input file/dir et écrit les fichiers nettoyés.

    Comportement:
    - Si `--in` est un fichier, nettoie ce fichier et écrit `<stem>_clean.csv` dans `--out`.
    - Si `--in` est un dossier, cherche les fichiers correspondant à `--pattern` et les traite en boucle.
    """
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
        # Cas fichier unique
        if inp.is_file():
            df = pd.read_csv(inp)
            cleaned = clean_dataframe(
                df,
                date_col="date",
                value_col="value",
                fill_strategy=fill_strategy,
                outlier_threshold=(None if outlier_threshold == 0 else outlier_threshold),
            )
            # nom de sortie : <stem>_clean<suffix>
            out_name = f"{inp.stem.replace('_ingest','')}_clean{inp.suffix}"
            outp = outd / out_name
            save_interim(cleaned, str(outp))
            if verbose:
                print(f"[clean] écrit {outp}")

        # Cas dossier: itère sur les fichiers matching
        elif inp.is_dir():
            files = list(inp.glob(pattern))
            if not files:
                print(f"[clean] aucun fichier correspondant à {pattern} dans {inp}")
                return 2
            outd.mkdir(parents=True, exist_ok=True)
            for f in files:
                try:
                    df = pd.read_csv(f)
                except Exception as e:
                    print(f"[clean] ignoré {f} : {e}")
                    continue
                cleaned = clean_dataframe(
                    df,
                    date_col="date",
                    value_col="value",
                    fill_strategy=fill_strategy,
                    outlier_threshold=(None if outlier_threshold == 0 else outlier_threshold),
                )
                out_name = f"{f.stem.replace('_ingest','')}_clean{f.suffix}"
                outp = outd / out_name
                save_interim(cleaned, str(outp))
                if verbose:
                    print(f"[clean] écrit {outp}")
        else:
            print(f"[clean] entrée invalide : {inp}")
            return 2
    except Exception as e:
        print(f"[clean] erreur : {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
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
	"""Remplit les valeurs manquantes selon une stratégie simple.

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
		raise ValueError(f"Stratégie inconnue : {strategy}")

	non_numeric = [c for c in df.columns if c not in numeric_cols]
	if non_numeric:
		df[non_numeric] = df[non_numeric].ffill().bfill()

	return df


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
	"""Renvoie un masque booléen indiquant les outliers selon le z-score."""
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
	"""Supprime les lignes où `col` est un outlier selon le z-score."""
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
	"""Enregistre le DataFrame en CSV, crée les dossiers si nécessaire. Retourne le chemin."""
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	df.to_csv(out_path, index=False)
	return out_path


def parse_args(argv=None):
	import argparse

	parser = argparse.ArgumentParser(description="Exécute le nettoyage sur des CSV (défaut : data/raw -> data/interim)")
	parser.add_argument("--in", dest="input", default="data/raw", help="Fichier ou dossier d'entrée (défaut : data/raw)")
	parser.add_argument("--out", dest="output", default="data/interim", help="Dossier de sortie (défaut : data/interim)")
	parser.add_argument("--fill", dest="fill", default="ffill", choices=["ffill", "bfill", "mean", "zero"], help="Stratégie de remplissage des valeurs manquantes")
	parser.add_argument("--outlier", dest="outlier", type=float, default=3.0, help="Seuil z-score pour retirer les outliers (0 pour désactiver)")
	parser.add_argument("--pattern", dest="pattern", default="*_ingest.csv", help="Pattern glob pour sélectionner les fichiers d'entrée")
	parser.add_argument("--no-verbose", dest="verbose", action="store_false", help="Désactiver les messages")
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
			out_name = f"{inp.stem.replace('_ingest','')}_clean{inp.suffix}"
			outp = outd / out_name
			save_interim(cleaned, str(outp))
			if verbose:
				print(f"[clean] écrit {outp}")
		elif inp.is_dir():
			files = list(inp.glob(pattern))
			if not files:
				print(f"[clean] aucun fichier correspondant à {pattern} dans {inp}")
				return 2
			outd.mkdir(parents=True, exist_ok=True)
			for f in files:
				try:
					df = pd.read_csv(f)
				except Exception as e:
					print(f"[clean] ignoré {f} : {e}")
					continue
				cleaned = clean_dataframe(
					df,
					date_col="date",
					value_col="value",
					fill_strategy=fill_strategy,
					outlier_threshold=(None if outlier_threshold == 0 else outlier_threshold),
				)
				out_name = f"{f.stem.replace('_ingest','')}_clean{f.suffix}"
				outp = outd / out_name
				save_interim(cleaned, str(outp))
				if verbose:
					print(f"[clean] écrit {outp}")
		else:
			print(f"[clean] entrée invalide : {inp}")
			return 2
	except Exception as e:
		print(f"[clean] erreur : {e}")
		return 1

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

