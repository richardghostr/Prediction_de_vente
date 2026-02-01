"""
Utilitaires optimisés pour le mapping et la transformation de données.
Utilise PyYAML pour la configuration et des types de données optimisés.
"""
import re
from collections import defaultdict
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import yaml

try:
    from rapidfuzz.fuzz import ratio
except ImportError:
    from difflib import SequenceMatcher
    def ratio(s1: str, s2: str) -> float:
        return SequenceMatcher(None, s1, s2).ratio() * 100

def load_config(config_path: str) -> Dict[str, Any]:
    """Charge la configuration depuis un fichier YAML."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def normalize_name(name: str) -> str:
    """Normalise un nom de colonne pour une comparaison robuste."""
    s = name.lower().strip()
    s = re.sub(r'[\s\-_]+', '_', s)
    s = re.sub(r'[^a-z0-9_]', '', s)
    return s.strip('_')

def _is_date_like(series: pd.Series, threshold: float = 0.7) -> bool:
    """Vérifie si une série semble contenir des dates de manière performante."""
    if series.empty: return False
    sample = series.dropna().sample(min(len(series.dropna()), 100))
    if sample.empty: return False
    # La conversion en chaîne est coûteuse, on la fait sur un échantillon
    parsed = pd.to_datetime(sample.astype(str), errors='coerce')
    return parsed.notna().mean() >= threshold

def _is_numeric_like(series: pd.Series, threshold: float = 0.8) -> bool:
    """Vérifie si une série est majoritairement numérique."""
    if series.dtype.kind in 'ifc': return True # Déjà numérique
    if series.empty: return False
    sample = series.dropna().sample(min(len(series.dropna()), 100))
    if sample.empty: return False
    coerced = pd.to_numeric(sample, errors='coerce')
    return coerced.notna().mean() >= threshold

def _uniqueness_score(series: pd.Series) -> float:
    """Calcule le ratio de valeurs uniques, indicateur pour les IDs."""
    if series.empty: return 0.0
    # Utilise .nunique() qui est optimisé
    return series.nunique() / len(series)

def find_best_mapping(
    df: pd.DataFrame,
    targets: List[str],
    aliases: Dict[str, List[str]],
    name_threshold: float = 70.0
) -> Dict[str, str]:
    """Trouve le meilleur mapping en combinant nom et type de données."""
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
            elif target == 'value' and _is_numeric_like(df[source_col]):
                type_bonus = 30
            elif target == 'id':
                type_bonus = int(_uniqueness_score(df[source_col]) * 25)

            total_score = name_score + type_bonus
            if total_score >= name_threshold:
                scores[target].append((source_col, total_score))

    mapping = {}
    used_sources = set()
    sorted_targets = sorted(scores.keys(), key=lambda k: len(scores[k]))

    for target in sorted_targets:
        candidates = sorted(scores[target], key=lambda item: item[1], reverse=True)
        for source_col, score in candidates:
            if source_col not in used_sources:
                mapping[target] = source_col
                used_sources.add(source_col)
                break
    
    return mapping

def apply_mapping_and_transform(
    df: pd.DataFrame,
    mapping: Dict[str, str],
    data_types: Dict[str, str]
) -> pd.DataFrame:
    """Applique le mapping et transforme les types de données pour optimiser la mémoire."""
    rename_dict = {v: k for k, v in mapping.items()}
    # Sélectionne et renomme en une seule opération
    transformed_df = df[list(mapping.values())].rename(columns=rename_dict)

    for col, dtype in data_types.items():
        if col in transformed_df.columns:
            try:
                dtype_str = str(dtype)

                # IDs : extraire uniquement les chiffres avant tout casting
                if col == 'id':
                    extracted = transformed_df[col].astype(str).str.extract(r"(\d+)")[0]
                    transformed_df[col] = pd.to_numeric(extracted, errors='coerce', downcast='integer')

                if 'datetime' in dtype_str:
                    # Essaye d'inférer plusieurs formats avec priorité au dayfirst
                    transformed_df[col] = pd.to_datetime(
                        transformed_df[col], errors='coerce', dayfirst=True, infer_datetime_format=True
                    )
                elif dtype_str == 'category':
                    transformed_df[col] = transformed_df[col].astype('category')
                else:
                    # Nettoie les nombres avec virgule décimale (ex: "1,5") avant conversion
                    if col == 'value':
                        transformed_df[col] = (
                            transformed_df[col]
                            .astype(str)
                            .str.replace(',', '.', regex=False)
                        )
                        # Convertit des nombres écrits en lettres simples (english) si présents
                        word_map = {
                            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
                            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
                        }
                        transformed_df[col] = transformed_df[col].apply(
                            lambda x: word_map.get(str(x).strip().lower(), x)
                        )
                    # Utilise downcast pour les types numériques pour économiser de la mémoire
                    transformed_df[col] = pd.to_numeric(transformed_df[col], errors='coerce', downcast='float')
                    try:
                        target_dtype = np.dtype(dtype)
                        if target_dtype.kind in ('i', 'u'):
                            transformed_df[col] = transformed_df[col].astype(target_dtype)
                        elif target_dtype.kind == 'f':
                            transformed_df[col] = transformed_df[col].astype(target_dtype)
                    except TypeError:
                        pass

            except (ValueError, TypeError) as e:
                print(f"Avertissement: Impossible de convertir la colonne '{col}' en '{dtype}'. Erreur: {e}")
                continue
    return transformed_df

def validate_transformed_df(
    df: pd.DataFrame,
    rules: Dict[str, Any]
) -> List[str]:
    """Valide le DataFrame transformé en utilisant des règles configurables."""
    issues = []
    
    # Validation des valeurs manquantes
    thresholds = rules.get('missing_thresholds', {})
    for col, threshold in thresholds.items():
        if col in df.columns:
            missing_ratio = df[col].isna().mean()
            if missing_ratio > threshold:
                issues.append(f"'{col}': Taux de valeurs manquantes ({missing_ratio:.2%}) dépasse le seuil ({threshold:.2%}).")

    # Validation des plages de valeurs
    if 'value' in df.columns and 'value' in rules:
        rule = rules['value']
        if 'min' in rule and (df['value'] < rule['min']).any():
            issues.append(f"'value': Des valeurs sont inférieures au minimum autorisé ({rule['min']}).")
        if 'max' in rule and (df['value'] > rule['max']).any():
            issues.append(f"'value': Des valeurs sont supérieures au maximum autorisé ({rule['max']}).")
            
    return issues
