"""Script d'ingestion minimal.

Fonctionnalités :
- lecture d'un fichier CSV ou d'un dossier de CSVs
- validation d'un schéma minimal (colonnes requises)
- idempotence via checksum (évite doublons)
- options CLI : --output, --force, --dry-run, --pattern, --required-cols
- écriture d'un fichier metadata JSON à côté du CSV écrit

Usage exemple :
    python src/data/ingest.py sample.csv --output data/raw

"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


DEFAULT_REQUIRED_COLUMNS = {"id", "date", "value"}


def compute_df_checksum(df: pd.DataFrame) -> str:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
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


def ingest_single_csv(
    input_path: Path,
    output_dir: Path,
    required_columns: Iterable[str],
    force: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> Optional[Path]:
    """Ingest a single CSV file: validate, idempotence, write and metadata.

    Returns the output path (or None if dry_run).
    """
    if verbose:
        print(f"[ingest] Lecture du fichier: {input_path}")

    df = pd.read_csv(input_path)
    validate_schema(df, required_columns)
    checksum = compute_df_checksum(df)

    output_dir.mkdir(parents=True, exist_ok=True)

    # output file name: <stem>_ingest<suffix>
    out_name = f"{input_path.stem}_ingest{input_path.suffix}"
    output_path = output_dir / out_name

    existing = find_existing_by_checksum(output_dir, checksum)
    if existing and not force:
        # If an identical file exists but with a different name, create a canonical ingest-named copy.
        if existing.name == out_name:
            if verbose:
                print(f"[ingest] Fichier identique déjà présent: {existing} (skip)")
            return existing
        else:
            # copy existing to the desired output name and copy metadata if present
            import shutil

            try:
                shutil.copy2(existing, output_path)
                # copy metadata if exists
                existing_meta = existing.with_suffix(existing.suffix + ".metadata.json")
                if existing_meta.exists():
                    shutil.copy2(existing_meta, output_path.with_suffix(output_path.suffix + ".metadata.json"))
                if verbose:
                    print(f"[ingest] Fichier identique trouvé {existing}; copié vers {output_path}")
                return output_path
            except Exception:
                if verbose:
                    print(f"[ingest] Fichier identique trouvé {existing} (skip)")
                return existing

    # output file name: <stem>_ingest<suffix>
    out_name = f"{input_path.stem}_ingest{input_path.suffix}"
    output_path = output_dir / out_name

    if dry_run:
        if verbose:
            print(f"[ingest] dry-run: fichier valide, would write to {output_path}")
        return None

    df.to_csv(output_path, index=False)

    meta = {
        "source": str(input_path),
        "fetched_at": pd.Timestamp.now().isoformat(),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "checksum": checksum,
    }

    # write metadata only if caller requested it via attribute
    write_meta_flag = getattr(ingest_single_csv, "write_meta", False)
    if write_meta_flag:
        write_metadata(output_path, meta)

    if verbose:
        print(f"[ingest] fichier validé et écrit dans: {output_path}")
        if write_meta_flag:
            print(f"[ingest] métadonnées écrites: {output_path.with_suffix(output_path.suffix + '.metadata.json')}")

    return output_path


def ingest_from_dir(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*.csv",
    required_columns: Iterable[str] = DEFAULT_REQUIRED_COLUMNS,
    force: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> List[Path]:
    csv_files = list(input_dir.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f"Aucun fichier CSV trouvé dans {input_dir} (pattern={pattern})")

    results: List[Path] = []
    for f in csv_files:
        res = ingest_single_csv(f, output_dir, required_columns, force=force, dry_run=dry_run, verbose=verbose)
        if res:
            results.append(res)
    return results


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingestion CSV vers data/raw avec validation et metadata")
    parser.add_argument("input", help="Fichier CSV ou dossier contenant CSVs")
    parser.add_argument("--output", "-o", default="data/raw", help="Dossier de sortie (default=data/raw)")
    parser.add_argument("--force", "-f", action="store_true", help="Forcer l'écrasement/duplication même si checksum identique")
    parser.add_argument("--dry-run", action="store_true", help="Valide sans écrire les fichiers")
    parser.add_argument("--pattern", default="*.csv", help="Pattern glob pour sélectionner les fichiers dans un dossier (default=*.csv)")
    parser.add_argument(
        "--required-cols",
        default=",".join(sorted(DEFAULT_REQUIRED_COLUMNS)),
        help="Liste séparée par des virgules des colonnes requises (default=id,date,value)",
    )
    parser.add_argument("--no-verbose", dest="verbose", action="store_false", help="Réduire la verbosité")
    parser.add_argument("--write-meta", dest="write_meta", action="store_true", help="Ecrire le fichier metadata JSON alongside the CSV (default: false)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)
    output_dir = Path(args.output)
    force = bool(args.force)
    dry_run = bool(args.dry_run)
    pattern = args.pattern
    write_meta = bool(getattr(args, "write_meta", False))
    required_cols = [c.strip() for c in args.required_cols.split(",") if c.strip()]
    verbose = bool(args.verbose)

    try:
        if input_path.is_file():
            # pass write_meta flag to the single ingest function via attribute
            ingest_single_csv.write_meta = write_meta
            ingest_single_csv(input_path, output_dir, required_cols, force=force, dry_run=dry_run, verbose=verbose)
        elif input_path.is_dir():
            ingest_single_csv.write_meta = write_meta
            ingest_from_dir(input_path, output_dir, pattern=pattern, required_columns=required_cols, force=force, dry_run=dry_run, verbose=verbose)
        else:
            print(f"Chemin invalide : {input_path}")
            return 2
    except Exception as e:
        print(f"Erreur lors de l'ingestion: {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
