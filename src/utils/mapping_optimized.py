"""
Utilitaires optimisés pour le mapping et la transformation de données.
Inclut nettoyage robuste des colonnes numériques pour éviter yhat=None.
"""
import re
from collections import defaultdict
from typing import Dict, List, Any

import numpy as np
import pandas as pd

try:
    from rapidfuzz.fuzz import ratio
except ImportError:
    from difflib import SequenceMatcher
    def ratio(s1: str, s2: str) -> float:
        return SequenceMatcher(None, s1, s2).ratio() * 100


def normalize_name(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r'[\s\-_]+', '_', s)
    s = re.sub(r'[^a-z0-9_]', '', s)
    return s.strip('_')


def _is_date_like(series: pd.Series, threshold: float = 0.7) -> bool:
    if series.empty:
        return False
    sample = series.dropna().sample(min(len(series.dropna()), 100))
    parsed = pd.to_datetime(sample.astype(str), errors='coerce')
    return parsed.notna().mean() >= threshold


def _is_numeric_like(series: pd.Series, threshold: float = 0.8) -> bool:
    if series.dtype.kind in 'ifc':
        return True
    if series.empty:
        return False
    sample = series.dropna().sample(min(len(series.dropna()), 100))
    coerced = pd.to_numeric(sample.astype(str).str.replace(',', '.', regex=False), errors='coerce')
    return coerced.notna().mean() >= threshold


def _uniqueness_score(series: pd.Series) -> float:
    if series.empty: return 0.0
    return series.nunique() / len(series)


def clean_numeric_column(col: pd.Series) -> pd.Series:
    """Nettoie et convertit une colonne en float/int si possible."""
    col_clean = col.astype(str)
    col_clean = col_clean.str.replace(r"[ €$]", "", regex=True)
    col_clean = col_clean.str.replace(",", ".", regex=False)
    col_clean = col_clean.replace(['nan','NaN','None','-',''], np.nan)
    return pd.to_numeric(col_clean, errors='coerce')


def find_best_mapping_optimize(
    df: pd.DataFrame,
    targets: List[str],
    aliases: Dict[str, List[str]],
    name_threshold: float = 50.0  # plus permissif
) -> Dict[str, str]:
    """Mapping optimisé qui renvoie toutes les colonnes et détecte correctement 'value'."""
    source_cols = list(df.columns)
    norm_sources = {col: normalize_name(col) for col in source_cols}
    scores = defaultdict(list)

    for target in targets:
        for source_col in source_cols:
            norm_source = norm_sources[source_col]
            alias_matches = (normalize_name(alias) for alias in aliases.get(target, [target]))
            name_score = max((ratio(norm_source, alias) for alias in alias_matches), default=0)

            type_bonus = 0
            if target == 'date' and _is_date_like(df[source_col]):
                type_bonus = 35
            elif target == 'value':
                # Nettoyage pour identifier correctement les nombres
                cleaned_col = clean_numeric_column(df[source_col])
                if _is_numeric_like(cleaned_col):
                    type_bonus = 30
            elif target == 'id':
                type_bonus = int(_uniqueness_score(df[source_col]) * 25)

            total_score = name_score + type_bonus
            scores[target].append((source_col, total_score))

    mapping = {}
    used_sources = set()
    # Trie par nombre de candidats pour privilégier les cibles rares
    sorted_targets = sorted(scores.keys(), key=lambda k: len(scores[k]))

    for target in sorted_targets:
        candidates = sorted(scores[target], key=lambda item: item[1], reverse=True)
        for source_col, score in candidates:
            if source_col not in used_sources:
                mapping[target] = source_col
                used_sources.add(source_col)
                break

    # Inclure toutes les autres colonnes restantes
    for col in df.columns:
        if col not in mapping.values():
            mapping[f"extra_{col}"] = col

    return mapping


def apply_mapping_and_transform(
    df: pd.DataFrame,
    mapping: Dict[str, str],
    data_types: Dict[str, str]
) -> pd.DataFrame:
    """Applique le mapping et transforme les types avec nettoyage des numériques."""
    rename_dict = {v: k for k, v in mapping.items()}
    transformed_df = df[list(mapping.values())].rename(columns=rename_dict)

    for col, dtype in data_types.items():
        if col in transformed_df.columns:
            try:
                dtype_str = str(dtype)

                if col == 'value':
                    transformed_df[col] = clean_numeric_column(transformed_df[col])
                elif col == 'id':
                    extracted = transformed_df[col].astype(str).str.extract(r"(\d+)")[0]
                    transformed_df[col] = pd.to_numeric(extracted, errors='coerce', downcast='integer')
                elif 'datetime' in dtype_str:
                    transformed_df[col] = pd.to_datetime(transformed_df[col], errors='coerce', dayfirst=True, infer_datetime_format=True)
                elif dtype_str == 'category':
                    transformed_df[col] = transformed_df[col].astype('category')
                else:
                    transformed_df[col] = pd.to_numeric(transformed_df[col], errors='coerce', downcast='float')

            except Exception as e:
                print(f"Avertissement: impossible de convertir '{col}' en '{dtype}': {e}")
    return transformed_df
