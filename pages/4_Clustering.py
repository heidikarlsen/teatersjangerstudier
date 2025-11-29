import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import umap
import plotly.express as px


st.set_page_config(page_title="Teatersjangerstudier - Clustering", page_icon="🎭", layout="wide")

# === Last metadata ===
meta_df = pd.read_excel("Sjangre_kategorisert_010625.xlsx")


st.header("Clustering av tekster (TF–IDF → UMAP → KMeans)")

st.markdown(
    """
Denne funksjonen grupperer tekster på bakgrunn av mønstre gjennom 
en trestegs-prosess:

1. **TF–IDF**: vektlegger ord som er særpregede for hver tekst.
2. **UMAP**: reduserer den høydimensjonale TF–IDF-matrisen til 2D på en måte som bevarer struktur.
3. **KMeans-clustering**: unsupervised metode som finner grupper av tekster basert på likhet.

Dette kan gi innsikt i om sjangre ligner på hverandre, om tekster danner undergrupper på tvers av sjangre, 
og hvilke ord som er typiske for hver klynge.
"""
)

# --------------------------------------------------
# DATA: fulltekstene også brukt til TF-IDF på frekvenssiden
# --------------------------------------------------

df_full = meta_df[
    meta_df["urn"].notna() & meta_df["urn"].str.startswith("URN")
].copy()

# Sjekk lokale filer
available_files = set(os.listdir("filer"))

def urn_to_filename(urn):
    return urn.replace("URN:NBN:", "")

df_full["file_exists"] = df_full["urn"].apply(lambda u: urn_to_filename(u) in available_files)
df_available = df_full[df_full["file_exists"]].copy()

def load_local_text(urn):
    filename = urn_to_filename(urn)
    path = os.path.join("filer", filename)
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except:
        return ""

# --------------------------------------------------
# Velg sjangre
# --------------------------------------------------

all_genres = sorted(df_available["genre"].dropna().unique().tolist())

selected_genres = st.multiselect(
    "Velg én eller flere sjangre for clustering:",
    all_genres,
)

if not selected_genres:
    st.stop()

sub = df_available[df_available["genre"].isin(selected_genres)].copy()

if sub.empty:
    st.error("Ingen tekster for de valgte sjangrene.")
    st.stop()

# Last tekster
texts = []
labels = []
urns = []

for _, row in sub.iterrows():
    txt = load_local_text(row["urn"])
    if txt.strip():
        texts.append(txt)
        label = f"{row['year']} — {row['title']} — {row['author']}"
        labels.append(label)
        urns.append(row["urn"])

if len(texts) < 3:
    st.error("Trenger minst tre tekster for clustering.")
    st.stop()

# --------------------------------------------------
# KMeans parametere
# --------------------------------------------------

k = st.slider("Antall klynger (K)", 2, 12, 4)

# --------------------------------------------------
# Beregning
# --------------------------------------------------

if st.button("Kjør clustering"):

    # ---- TF-IDF ----
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"[A-Za-zÆØÅæøå]+",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )

    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # ---- UMAP ----
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine")
    embedding = reducer.fit_transform(X)

    # ---- KMeans ----
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(embedding)

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    df_plot = pd.DataFrame({
        "x": embedding[:, 0],
        "y": embedding[:, 1],
        "Cluster": clusters,
        "Verk": labels,
        "URN": urns,
        "Sjanger": sub["genre"].tolist(),
        "År": sub["year"].tolist(),
    })

    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        color="Cluster",
        hover_data=["Verk", "Sjanger", "År"],
        title="Clustering av tekster (UMAP + KMeans)",
    )

    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------
    # Ord som karakteriserer hver klynge
    # --------------------------------------------------
    st.subheader("Karakteristiske ord for hver klynge")

    X_dense = X.toarray()
    cluster_df = pd.DataFrame(X_dense)
    cluster_df["cluster"] = clusters

    for cluster_id in sorted(set(clusters)):
        st.write(f"### Cluster {cluster_id}")

        # gjennomsnittlig tf-idf for ord i denne klyngen
        mean_vector = cluster_df[cluster_df["cluster"] == cluster_id].iloc[:, :-1].mean(axis=0)

        top_idx = mean_vector.argsort()[::-1][:25]
        top_words = [(feature_names[i], mean_vector[i]) for i in top_idx]

        df_words = pd.DataFrame(top_words, columns=["Ord", "TF-IDF"])
        st.dataframe(df_words)

    # --------------------------------------------------
    # Oversikt over hvilke verk som havnet i hvilke klynger
    # --------------------------------------------------
    st.subheader("Verk per klynge")

    for cluster_id in sorted(set(clusters)):
        st.write(f"### Cluster {cluster_id}")
        group = df_plot[df_plot["Cluster"] == cluster_id]
        for _, r in group.iterrows():
            st.write(f"- **{r['Verk']}** ({r['Sjanger']}, {r['År']})")

