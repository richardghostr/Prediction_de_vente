import streamlit as st
from pathlib import Path

from src.ui import ingest_ui, clean_ui, predict_ui, visualize_ui, features_ui

st.set_page_config(page_title="Pipeline Dashboard", layout="wide")

st.title("Pipeline: Ingestion → Nettoyage → Feature Engineering → Prédiction → Visualisation")

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Phase",
        options=[
            "Ingestion des données",
            "Traitement des données",
            "Analyse et prédiction",
            "Visualisation des résultats",
        ],
    )

st.write("Sélectionnez une phase dans la barre latérale pour afficher l'interface correspondante.")

if page == "Ingestion des données":
    ingest_ui.render_ingest_ui()
elif page == "Traitement des données":
    tab1, tab2 = st.tabs(["Nettoyage", "Feature Engineering"])
    with tab1:
        clean_ui.render_clean_ui()
    with tab2:
        features_ui.render_features_ui()
elif page == "Analyse et prédiction":
    predict_ui.render_predict_ui()
elif page == "Visualisation des résultats":
    visualize_ui.render_visualize_ui()
else:
    st.info("Choisissez une phase.")
