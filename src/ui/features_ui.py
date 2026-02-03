import streamlit as st
from pathlib import Path
import pandas as pd

try:
    from src.data.features import build_feature_pipeline
except Exception:
    build_feature_pipeline = None


def render_features_ui():
    st.header("Feature Engineering")
    st.write("Utilise `src/data/features.py` pour générer les features à partir des fichiers nettoyés.")

    interim_dir = Path("data/interim")
    files = []
    if interim_dir.exists():
        files = sorted(interim_dir.glob("*_clean.csv"))

    if not files:
        st.info("Aucun fichier nettoyé trouvé dans data/interim. Exécutez d'abord le nettoyage.")
        return

    sel = st.selectbox("Choisir un fichier clean -> features", options=[str(p) for p in files])
    try:
        df = pd.read_csv(sel)
    except Exception as e:
        st.error(f"Impossible de lire le fichier sélectionné: {e}")
        return

    st.subheader("Aperçu du fichier clean sélectionné")
    st.dataframe(df.head())

    lags_input = st.text_input("Lags (comma-separated)", value="1,7")
    windows_input = st.text_input("Rolling windows (comma-separated)", value="3,7")

    def parse_list(s: str):
        try:
            return [int(x.strip()) for x in s.split(",") if x.strip()]
        except Exception:
            return None

    if st.button("Générer les features"):
        if build_feature_pipeline is None:
            st.error("`build_feature_pipeline` non disponible dans l'environnement")
            return
        lags = parse_list(lags_input)
        windows = parse_list(windows_input)
        try:
            feat = build_feature_pipeline(df, lags=lags, windows=windows)
            st.success("Features générées")
            st.subheader("Aperçu des features")
            st.dataframe(feat.head())
            if st.checkbox("Enregistrer les features dans data/processed", value=True):
                outd = Path("data/processed")
                outd.mkdir(parents=True, exist_ok=True)
                out_name = outd / f"{Path(sel).stem.replace('_clean','')}_features{Path(sel).suffix}"
                feat.to_csv(out_name, index=False)
                st.info(f"Features enregistrées: {out_name}")
        except Exception as e:
            st.error(f"Erreur lors de la génération des features: {e}")
