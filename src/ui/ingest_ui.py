import streamlit as st
from pathlib import Path
import pandas as pd
import shutil
import tempfile
from typing import Optional

try:
    from src.data.ingest import ingest_single_csv
    from src.utils.mapping_optimized import load_config
except Exception:
    ingest_single_csv = None
    load_config = None


def render_ingest_ui():
    st.header("Ingestion des données")
    st.write("Utilise `src/data/ingest.py` pour effectuer l'ingestion sans réimplémenter la logique.")

    uploaded = st.file_uploader("Uploader un fichier CSV pour ingestion", type=["csv"])
    mapping_file = st.text_input("Chemin du mapping (optionnel)", value="")
    out_dir = st.text_input("Dossier de sortie pour l'ingestion", value="data/raw")
    write_meta = st.checkbox("Écrire métadonnées (si supporté)", value=False)

    if uploaded is not None:
        tmp = Path(tempfile.gettempdir()) / f"uploaded_{uploaded.name}"
        with open(tmp, "wb") as f:
            shutil.copyfileobj(uploaded, f)
        st.success(f"Fichier temporaire écrit: {tmp}")

        # load config if available
        if load_config is not None:
            try:
                cfg = load_config(str(Path("configs") / "mapping_config.yml"))
            except Exception:
                cfg = {"targets": ["date", "value", "id"], "aliases": {}, "data_types": {}, "validation_rules": {}}
        else:
            cfg = {"targets": ["date", "value", "id"], "aliases": {}, "data_types": {}, "validation_rules": {}}

        if st.button("Lancer l'ingestion"):
            if ingest_single_csv is None:
                st.error("La fonction `ingest_single_csv` n'est pas disponible dans l'environnement.")
                return
            try:
                outp = Path(out_dir)
                # `ingest_single_csv` uses an attribute to control write_meta; set it here
                try:
                    setattr(ingest_single_csv, "write_meta", bool(write_meta))
                except Exception:
                    pass
                res = ingest_single_csv(Path(tmp), outp, cfg, mapping_file=mapping_file or None)
                if res:
                    st.success(f"Ingestion terminée, fichier écrit: {res}")
                    try:
                        df = pd.read_csv(res)
                        st.subheader("Aperçu du fichier ingéré")
                        st.dataframe(df.head())
                    except Exception as e:
                        st.warning(f"Impossible d'ouvrir le fichier ingéré pour aperçu: {e}")
                else:
                    st.warning("Aucun fichier produit par l'ingestion (dry-run ou erreur silencieuse).")
            except Exception as e:
                st.error(f"Erreur lors de l'ingestion: {e}")

    # show existing ingested files
    st.markdown("---")
    st.subheader("Fichiers ingérés existants")
    raw_dir = Path("data/raw")
    if raw_dir.exists():
        files = sorted(raw_dir.glob("*_ingest.csv"))
        if files:
            for f in files:
                st.write(f.name)
        else:
            st.info("Aucun fichier *_ingest.csv trouvé dans data/raw")
    else:
        st.info("Dossier data/raw introuvable")
