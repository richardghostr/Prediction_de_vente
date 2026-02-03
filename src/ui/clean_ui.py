import streamlit as st
from pathlib import Path
import pandas as pd

try:
    from src.data.clean import clean_dataframe, save_interim
except Exception:
    clean_dataframe = None
    save_interim = None


def render_clean_ui():
    st.header("Traitement des données (Nettoyage)")
    st.write("Utilise `src/data/clean.py` pour harmoniser et nettoyer les séries.")

    raw_dir = Path("data/raw")
    files = []
    if raw_dir.exists():
        files = sorted(raw_dir.glob("*.csv"))

    selected = st.selectbox("Choisir un fichier ingéré", options=["-- none --"] + [str(p) for p in files])
    if selected and selected != "-- none --":
        try:
            df = pd.read_csv(selected)
            st.subheader("Aperçu du fichier sélectionné")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Impossible de lire le fichier sélectionné: {e}")
            return

        fill_strategy = st.selectbox("Fill strategy", options=["ffill", "bfill", "mean", "zero"], index=0)
        outlier = st.number_input("Outlier z-score threshold (0 pour désactiver)", value=3.0)

        if st.button("Lancer le nettoyage"):
            if clean_dataframe is None:
                st.error("`clean_dataframe` non disponible dans l'environnement")
                return
            try:
                cleaned = clean_dataframe(df, date_col="date", value_col="value", fill_strategy=fill_strategy, outlier_threshold=(None if outlier == 0 else outlier))
                st.success("Nettoyage effectué")
                st.subheader("Aperçu après nettoyage")
                st.dataframe(cleaned.head())
                if save_interim is not None and st.checkbox("Enregistrer le résultat dans data/interim", value=True):
                    outp = Path("data/interim")
                    outp.mkdir(parents=True, exist_ok=True)
                    fname = outp / f"{Path(selected).stem.replace('_ingest','')}_clean.csv"
                    save_interim(cleaned, str(fname))
                    st.info(f"Fichier nettoyé enregistré: {fname}")
            except Exception as e:
                st.error(f"Erreur pendant le nettoyage: {e}")

    else:
        st.info("Aucun fichier ingéré à traiter. Lancez d'abord l'ingestion ou placez un fichier dans data/raw.")
