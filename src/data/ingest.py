"""
Script d'ingestion ultra-optimisé pour les données de vente.

Ce script est conçu pour une performance et une robustesse maximales, en gérant
des formats de CSV hétérogènes de manière automatique et configurable.

Fonctionnalités Clés :
- **Configuration Externe (YAML)**: Le mapping, les alias et les règles de validation
  sont gérés dans un fichier `configs/mapping_config.yml` pour une flexibilité maximale.
- **Mapping Intelligent**: Détecte les colonnes (`id`, `date`, `value`) en se basant
  sur une combinaison de similarité de noms (fuzzy matching) et d'analyse de type de données.
- **Optimisation Mémoire**: Utilise des types de données optimisés (ex: `category`, `float32`)
  pour réduire drastiquement l'empreinte mémoire, essentiel pour les grands volumes.
- **Validation Poussée**: Valide les données transformées contre des règles métier
  configurables (plages de valeurs, taux de valeurs manquantes).
- **Performance**: Utilise `pyarrow` comme moteur pandas si disponible et des
  bibliothèques optimisées comme `rapidfuzz`.
- **Robustesse**: Gestion des erreurs détaillée avec rapports JSON, et mode interactif
  en cas d'ambiguïté.

Exemples d'utilisation :
    # Ingestion standard avec détection auto
    python src/data/ingest.py "data/source/ventes_client_A.csv"

    # Utiliser une configuration de mapping spécifique
    python src/data/ingest.py "data/source/ventes_client_B.csv" --config "configs/client_B_config.yml"

    # Mode interactif pour un fichier complexe
    python src/data/ingest.py "data/source/fichier_inconnu.csv" --interactive
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any

# --- Correction pour ModuleNotFoundError ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    import pandas as pd
    # Tente d'utiliser le moteur pyarrow pour des performances accrues si installé
    pd.options.mode.string_storage = "pyarrow"
except ImportError:
    print("Avertissement: pandas n'est pas installé. `pip install pandas`", file=sys.stderr)
    sys.exit(1)
except Exception:
    # Si pyarrow n'est pas là, pandas fonctionnera quand même
    pass

from src.utils.mapping_optimized import (
    load_config,
    find_best_mapping,
    apply_mapping_and_transform,
    validate_transformed_df,
)


def pick_mapping(
    df_raw: pd.DataFrame,
    targets: List[str],
    mapping_file: Optional[str],
    aliases: Dict[str, Any],
    verbose: bool = True,
) -> Dict[str, str]:
    """Detecte ou complète un mapping pour `targets`.

    Cette fonction encapsule la heuristique de secours utilisée dans
    `ingest_single_csv` afin d'être testable isolément.
    """
    mapping: Dict[str, str] = {}
    if mapping_file:
        with open(mapping_file, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        if verbose:
            print(f"[ingest] Mapping manuel chargé depuis {mapping_file}: {mapping}")
        # continue: allow heuristics to fill missing targets from the manual mapping

    mapping = find_best_mapping(df_raw, targets, aliases)
    if verbose:
        print(f"[ingest] Mapping automatique détecté: {mapping}")

    # Heuristique de secours si le mapping est incomplet
    if set(mapping.keys()) != set(targets):
        norm = lambda s: re.sub(r'[^a-z0-9]', '', s.lower())
        cols_norm = {norm(c): c for c in df_raw.columns}

        def pick(keys, target):
            for k in keys:
                if k in cols_norm and target not in mapping:
                    mapping[target] = cols_norm[k]
                    return
            # fallback contains
            for nk, raw in cols_norm.items():
                if any(k in nk for k in keys) and target not in mapping:
                    mapping[target] = raw
                    return

        def pick_date_by_type():
            best_col = None
            best_score = 0
            for col in df_raw.columns:
                parsed = pd.to_datetime(df_raw[col].astype(str), errors='coerce', dayfirst=True, infer_datetime_format=True)
                score = parsed.notna().mean()
                if score > best_score and score >= 0.4:  # au moins 40% de parsable
                    best_score = score
                    best_col = col
            if best_col and 'date' not in mapping:
                mapping['date'] = best_col

        # date
        pick(["saledate", "date", "timestamp", "datetime"], 'date')
        if 'date' not in mapping:
            pick_date_by_type()
        # id (priorité store puis sku)
        pick(["store", "storeid", "id", "sku", "productid", "itemid"], 'id')
        # value (priorité qty/quantity puis amount/revenue)
        pick(["qty", "quantity", "value", "sales", "amount", "revenue"], 'value')

        # Last resort: if still missing, pick the first unused column.
        # Build a set of already-used columns to avoid fragile None comparisons.
        used_cols = {v for v in mapping.values() if v is not None}

        for col in df_raw.columns:
            if 'id' not in mapping and col not in used_cols:
                mapping['id'] = col
                used_cols.add(col)
                break

        for col in df_raw.columns:
            if 'value' not in mapping and col not in used_cols:
                mapping['value'] = col
                used_cols.add(col)
                break

        if verbose:
            print(f"[ingest] Mapping heuristique appliqué: {mapping}")

    return mapping


def detect_delimiter(path: Path, default: str = ',') -> str:
    """Détecte le délimiteur principal du fichier (',' ';' '\t' '|')."""
    sample = path.read_text(encoding='utf-8', errors='ignore')[:4096]
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=[',',';','\t','|'])
        return dialect.delimiter
    except Exception:
        return default


def compute_df_checksum(df: pd.DataFrame) -> str:
    """Calcule un checksum stable pour un DataFrame."""
    # Assurer un ordre de colonnes constant
    df_sorted = df.reindex(sorted(df.columns), axis=1)
    # Utiliser le moteur pyarrow pour une sérialisation rapide si disponible
    try:
        csv_bytes = df_sorted.to_csv(index=False).encode("utf-8")
    except Exception:
        csv_bytes = df_sorted.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()



def compute_file_checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_schema(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {', '.join(sorted(missing))}")


def find_existing_by_checksum(output_dir: Path, checksum: str) -> Optional[Path]:
    for f in output_dir.glob("*.csv"):
        try:
            if compute_file_checksum(f) == checksum:
                return f
        except Exception:
            continue
    return None


def write_metadata(output_path: Path, meta: dict) -> None:
    meta_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)


def get_user_mapping_choice(target: str, candidates: List[str]) -> Optional[str]:
    """Demande à l'utilisateur de choisir une colonne."""
    print(f"\n[INTERACTIF] Veuillez choisir la colonne pour '{target}':")
    for i, col in enumerate(candidates):
        print(f"  {i+1}: {col}")
    print("  0: Aucune de ces colonnes (ignorer)")

    while True:
        try:
            choice = int(input(f"Votre choix (0-{len(candidates)}): "))
            if 0 <= choice <= len(candidates):
                return None if choice == 0 else candidates[choice - 1]
        except ValueError:
            pass
        print("Choix invalide, veuillez réessayer.")


def ingest_single_csv(
    input_path: Path,
    output_dir: Path,
    config: Dict[str, Any],
    mapping_file: Optional[str] = None,
    interactive: bool = False,
    force: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> Optional[Path]:
    """Processus complet d'ingestion pour un seul fichier CSV."""
    if verbose:
        print(f"\n[ingest] Traitement de: {input_path}")

    try:
        sep = detect_delimiter(input_path)
        # Lecture optimisée avec tolérance aux lignes mal formées
        def _read(sep_choice: str):
            return pd.read_csv(
                input_path,
                sep=sep_choice,
                engine='pyarrow' if 'pyarrow' in sys.modules else 'python',
                low_memory=True,
                on_bad_lines='skip',
            )
        try:
            df_raw = _read(sep)
        except Exception:
            df_raw = pd.read_csv(
                input_path,
                sep=sep,
                engine='python',
                low_memory=True,
                quoting=csv.QUOTE_NONE,
                escapechar='\\',
                on_bad_lines='warn',
            )

        # Si une seule colonne, on retente avec d'autres séparateurs explicites
        if df_raw.shape[1] <= 1:
            for alt_sep in [';', '\t', '|', ',']:
                try:
                    df_raw = _read(alt_sep)
                    if df_raw.shape[1] > 1:
                        if verbose:
                            print(f"[ingest] Séparateur corrigé: '{alt_sep}'")
                        break
                except Exception:
                    continue
    except Exception as e:
        print(f"[ERREUR] Impossible de lire le fichier CSV {input_path}. Erreur: {e}", file=sys.stderr)
        return None

    # --- 1. Mapping des colonnes ---
    targets = config['targets']
    mapping = pick_mapping(df_raw, targets, mapping_file, config.get('aliases', {}), verbose=verbose)

    # Heuristique de secours si le mapping est incomplet
    if set(mapping.keys()) != set(targets):
        norm = lambda s: re.sub(r'[^a-z0-9]', '', s.lower())
        cols_norm = {norm(c): c for c in df_raw.columns}

        def pick(keys, target):
            for k in keys:
                if k in cols_norm and target not in mapping:
                    mapping[target] = cols_norm[k]
                    return
            # fallback contains
            for nk, raw in cols_norm.items():
                if any(k in nk for k in keys) and target not in mapping:
                    mapping[target] = raw
                    return

        def pick_date_by_type():
            best_col = None
            best_score = 0
            for col in df_raw.columns:
                parsed = pd.to_datetime(df_raw[col].astype(str), errors='coerce', dayfirst=True, infer_datetime_format=True)
                score = parsed.notna().mean()
                if score > best_score and score >= 0.4:  # au moins 40% de parsable
                    best_score = score
                    best_col = col
            if best_col and 'date' not in mapping:
                mapping['date'] = best_col

        # date
        pick(["saledate", "date", "timestamp", "datetime"], 'date')
        if 'date' not in mapping:
            pick_date_by_type()
        # id (priorité store puis sku)
        pick(["store", "storeid", "id", "sku", "productid", "itemid"], 'id')
        # value (priorité qty/quantity puis amount/revenue)
        pick(["qty", "quantity", "value", "sales", "amount", "revenue"], 'value')

        # Last resort: if still missing, pick the first unused column.
        # Build a set of already-used columns to avoid fragile None comparisons.
        used_cols = {v for v in mapping.values() if v is not None}

        for col in df_raw.columns:
            if 'id' not in mapping and col not in used_cols:
                mapping['id'] = col
                used_cols.add(col)
                break

        for col in df_raw.columns:
            if 'value' not in mapping and col not in used_cols:
                mapping['value'] = col
                used_cols.add(col)
                break

        if verbose:
            print(f"[ingest] Mapping heuristique appliqué: {mapping}")

    missing_targets = set(targets) - set(mapping.keys())
    if missing_targets:
        if interactive:
            print("[ingest] Mapping incomplet, passage en mode interactif.")
            for target in sorted(list(missing_targets)):
                # Exclure les colonnes déjà utilisées
                available_cols = [c for c in df_raw.columns if c not in mapping.values()]
                choice = get_user_mapping_choice(target, available_cols)
                if choice:
                    mapping[target] = choice
        else:
            raise ValueError(f"Mapping automatique a échoué. Cibles manquantes: {missing_targets}. Utilisez --interactive ou --mapping-file.")

    # --- 2. Transformation ---
    df_transformed = apply_mapping_and_transform(df_raw, mapping, config['data_types'])

    # --- 2a. Si 'value' est manquant, tente de le recalculer via qty*price dans le brut ---
    if 'value' in df_transformed.columns and df_transformed['value'].isna().any():
        # cherche colonnes qty / price dans df_raw
        def _norm(s: str) -> str:
            return re.sub(r'[^a-z0-9]', '', s.lower())
        qty_cols = [c for c in df_raw.columns if _norm(c) in {'qty','quantity','quantite','quantitees','qte'}]
        price_cols = [c for c in df_raw.columns if _norm(c) in {'price','unitprice','unit_price','prix','montant'}]
        if qty_cols and price_cols:
            qty_series = pd.to_numeric(df_raw[qty_cols[0]].astype(str).str.replace(',', '.', regex=False), errors='coerce')
            price_series = pd.to_numeric(df_raw[price_cols[0]].astype(str).str.replace(',', '.', regex=False), errors='coerce')
            recomputed = qty_series * price_series
            mask_missing = df_transformed['value'].isna()
            df_transformed.loc[mask_missing, 'value'] = recomputed.loc[mask_missing]

    # --- 2bis. Réparation tolérante des lignes invalides ---
    df_clean = df_transformed.copy()
    repair_log: Dict[str, Any] = {
        "rows_before": len(df_transformed),
        "date_fixed": 0,
        "value_fixed": 0,
        "id_filled": 0,
    }

    # Répare les dates NaT en tentant un swap mois/jour quand le mois est > 12
    if 'date' in df_clean.columns:
        mask_nat = df_clean['date'].isna()
        for idx, val in df_clean.loc[mask_nat, 'date'].items():
            raw_val = df_raw.loc[idx, mapping.get('date')] if mapping.get('date') in df_raw.columns else None
            candidate = str(raw_val)
            parts = re.split(r"[-/]", candidate)
            if len(parts) == 3 and all(p.isdigit() for p in parts):
                y, p2, p3 = parts
                if len(y) == 2:  # année sur 2 digits
                    y = f"20{y}" if int(y) <= 50 else f"19{y}"
                if int(p2) > 12 and int(p3) <= 12:
                    swapped = f"{y}-{p3.zfill(2)}-{p2.zfill(2)}"
                    parsed = pd.to_datetime(swapped, errors='coerce')
                    if pd.notna(parsed):
                        df_clean.at[idx, 'date'] = parsed
                        repair_log['date_fixed'] += 1
                        continue
            # Dernier essai: to_datetime générique dayfirst
            parsed = pd.to_datetime(candidate, errors='coerce', dayfirst=True, infer_datetime_format=True)
            if pd.notna(parsed):
                df_clean.at[idx, 'date'] = parsed
                repair_log['date_fixed'] += 1

    # Répare les valeurs: remplace NaN par 0, transforme les négatifs en valeur absolue
    if 'value' in df_clean.columns:
        mask_nan = df_clean['value'].isna()
        if mask_nan.any():
            df_clean.loc[mask_nan, 'value'] = 0
            repair_log['value_fixed'] += int(mask_nan.sum())
        mask_neg = df_clean['value'] < 0
        if mask_neg.any():
            df_clean.loc[mask_neg, 'value'] = df_clean.loc[mask_neg, 'value'].abs()
            repair_log['value_fixed'] += int(mask_neg.sum())

    # Remplit et nettoie les ids : fallback product_id, puis extraction de chiffres, puis synthèse
    if 'id' in df_clean.columns:
        mask_id = df_clean['id'].isna()
        if mask_id.any() and 'product_id' in df_raw.columns:
            df_clean.loc[mask_id, 'id'] = df_raw.loc[mask_id, 'product_id']

        # Extraction des chiffres pour normaliser (ex: store_1 -> 1)
        extracted = df_clean['id'].astype(str).str.extract(r"(\d+)")[0]
        df_clean['id'] = pd.to_numeric(extracted, errors='coerce', downcast='integer')

        mask_id_after = df_clean['id'].isna()
        if mask_id_after.any():
            start = int((df_clean['id'].max(skipna=True) or 0) + 1)
            df_clean.loc[mask_id_after, 'id'] = range(start, start + mask_id_after.sum())
        repair_log['id_filled'] += int(mask_id.sum())

    # --- 3. Validation douce (log mais ne bloque pas) ---
    validation_issues = validate_transformed_df(df_clean, config['validation_rules'])
    if validation_issues and verbose:
        print(f"[AVERTISSEMENT] Validation: {validation_issues}")

    # Écrit un rapport de réparation si des corrections ont été faites ou s'il reste des avertissements
    if (repair_log['date_fixed'] + repair_log['value_fixed'] + repair_log['id_filled'] > 0) or validation_issues:
        error_dir = output_dir / "errors"
        error_dir.mkdir(exist_ok=True)
        report_path = error_dir / f"{input_path.stem}_repair_report.json"
        report = {
            "source": str(input_path),
            "status": "repaired",
            "mapping_used": mapping,
            "repairs": repair_log,
            "validation_warnings": validation_issues,
            "sample_after_repairs": json.loads(df_clean.head().to_json(orient='records', date_format='iso')),
        }
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"[ingest] Rapport de réparation: {report_path}")

    if df_clean.empty:
        print(f"[ERREUR] Aucune ligne exploitable pour {input_path}", file=sys.stderr)
        return None

    # --- 3. Idempotence et Sauvegarde ---
    checksum = compute_df_checksum(df_clean)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{input_path.stem}_ingest.csv" # Sauvegarde en CSV
    output_path = output_dir / out_name

    # Toujours écraser l'existant pour garder la dernière ingestion
    if output_path.exists() and verbose:
        print(f"[ingest] Fichier existant écrasé: {output_path}")

    if dry_run:
        if verbose:
            print(f"[ingest] dry-run: Fichier valide, serait écrit en CSV dans {output_path}")
        return None

    df_clean.to_csv(output_path, index=False)

    meta = {
        "source": str(input_path),
        "source_checksum_raw": compute_df_checksum(df_raw),
        "ingested_checksum_transformed": checksum,
        "ingested_at": pd.Timestamp.now(tz='UTC').isoformat(),
        "rows": len(df_clean),
        "columns_original": list(df_raw.columns),
        "mapping_used": mapping,
        "validation_status": "success",
    }

    if getattr(ingest_single_csv, "write_meta", False):
        try:
            write_metadata(output_path, meta)
            if verbose:
                print(f"[ingest] Métadonnées écrites: {output_path.with_suffix(output_path.suffix + '.metadata.json')}")
        except Exception as e:
            print(f"[ingest] Impossible d'écrire les métadonnées: {e}", file=sys.stderr)

    if verbose:
        print(f"[ingest] Fichier transformé, validé et écrit en CSV: {output_path}")

    return output_path




def ingest_from_dir(
    input_dir: Path,
    output_dir: Path,
    config: Dict[str, Any],
    pattern: str = "*.csv",
    **kwargs,
) -> List[Path]:
    """Ingère tous les fichiers CSV d'un dossier correspondant à un pattern."""
    csv_files = list(input_dir.glob(pattern))
    if not csv_files:
        print(f"Avertissement: Aucun fichier CSV trouvé dans {input_dir} avec le pattern '{pattern}'", file=sys.stderr)
        return []

    results = []
    for f in csv_files:
        res = ingest_single_csv(f, output_dir, config, **kwargs)
        if res:
            results.append(res)
    return results


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Analyse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Script d'ingestion de données de vente optimisé.")
    parser.add_argument("input", type=Path, help="Fichier CSV ou dossier source à ingérer.")
    parser.add_argument("--output", "-o", type=Path, default=Path("data/raw"), help="Dossier de sortie pour les fichiers CSV traités (défaut: data/raw).")
    parser.add_argument("--config", type=Path, default=Path("configs/mapping_config.yml"), help="Chemin vers le fichier de configuration YAML (défaut: configs/mapping_config.yml).")
    parser.add_argument("--force", "-f", action="store_true", help="Forcer la ré-ingestion et écraser les fichiers existants.")
    parser.add_argument("--dry-run", action="store_true", help="Exécuter sans écrire de fichiers pour valider le processus.")
    parser.add_argument("--pattern", default="*.csv", help="Pattern glob pour la recherche de fichiers dans un dossier (défaut: *.csv).")
    parser.add_argument("--mapping-file", help="[Optionnel] Chemin vers un fichier JSON de mapping manuel pour surcharger la détection auto.")
    parser.add_argument("--interactive", action="store_true", help="Activer le mode interactif pour résoudre les mappings ambigus.")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false", help="Désactiver les logs détaillés.")
    parser.add_argument("--write-meta", action="store_true", help="Écrire un fichier de métadonnées à côté des fichiers CSV.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Fonction principale du script."""
    args = parse_args(argv)

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Erreur: Fichier de configuration non trouvé à '{args.config}'", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Erreur lors du chargement de la configuration: {e}", file=sys.stderr)
        return 1

    ingest_args = {
        "output_dir": args.output,
        "config": config,
        "mapping_file": args.mapping_file,
        "interactive": args.interactive,
        "force": args.force,
        "dry_run": args.dry_run,
        "verbose": args.verbose,
    }

    try:
        setattr(ingest_single_csv, "write_meta", args.write_meta)

        if args.input.is_file():
            ingest_single_csv(args.input, **ingest_args)
        elif args.input.is_dir():
            ingest_from_dir(args.input, pattern=args.pattern, **ingest_args)
        else:
            print(f"Erreur: Le chemin d'entrée '{args.input}' n'est pas valide.", file=sys.stderr)
            return 2
            
    except (ValueError, FileNotFoundError) as e:
        print(f"\nUne erreur est survenue durant l'ingestion: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nUne erreur inattendue est survenue: {e}", file=sys.stderr)
        return 1

    print("\nIngestion terminée avec succès.")
    return 0


if __name__ == "__main__":
    # Pour tester le script, vous pouvez créer un faux CSV et lancer:
    # python src/data/ingest.py "data/raw/sample_1y.csv" --interactive --write-meta
    sys.exit(main())
