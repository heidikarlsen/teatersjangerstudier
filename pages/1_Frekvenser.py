import streamlit as st
import dhlab as dh 
import pandas as pd
from dhlab import Counts
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Teatersjangerstudier - Frekvenser", page_icon="🎭", layout="wide")

# ================================================================
# === 1) Sammnenligning av frekvenser for nøkkelord i sjangre ===
# ================================================================


st.header("Sammenlign nøkkelord på tvers av sjangre")

st.markdown(
    """
Skriv inn et **søkeord** (f.eks. `norsk*`, `patriot*`, `nation*`).  
Analysen viser hvor ofte ordet forekommer i hver sjanger, både som **antall treff** og **relativ frekvens**.
"""
)

keyword = st.text_input("Nøkkelord (wildcard støttes)", "")

if keyword:

    # ----------------------------
    # 1. Hent alle sjangre med minst én URN
    # ----------------------------
    full_df = meta_df[
        meta_df["urn"].notna() & meta_df["urn"].str.startswith("URN")
    ].copy()

    genres = sorted(full_df["genre"].dropna().unique())

    # ----------------------------
    # 2. Funksjon: frekvens i én sjanger
    # ----------------------------
    def keyword_stats_for_genre(genre, keyword):
        """Returnerer total forekomster + relativ frekvens for sjanger."""
        gdf = full_df[full_df["genre"].str.casefold() == genre.casefold()]
        urns = gdf["urn"].tolist()

        if not urns:
            return 0, 0.0

        corpus = dh.Corpus()
        corpus.extend_from_identifiers(urns)

        counts = Counts(corpus)
        total_tokens = counts.frame.sum().sum()

        # pattern matching for wildcard
        key = keyword.casefold()
        keys = [w for w in counts.frame.index if w.casefold().startswith(key.replace("*", ""))]

        absolute = counts.frame.loc[keys].sum().sum() if keys else 0
        relative = absolute / total_tokens if total_tokens > 0 else 0

        return int(absolute), float(relative)

    # ----------------------------
    # 3. Beregn for alle sjangre
    # ----------------------------
    rows = []
    for g in genres:
        abs_count, rel_freq = keyword_stats_for_genre(g, keyword)
        n_works = full_df[full_df["genre"] == g]["urn"].nunique()
        rows.append((g, n_works, abs_count, rel_freq))

    result_df = pd.DataFrame(rows, columns=["Genre", "Works", "Hits", "RelativeFreq"])
    result_df = result_df.sort_values("Hits", ascending=False)

    st.subheader(f"Resultater for nøkkelord: **{keyword}**")
    st.write("Sortert etter antall forekomster:")

    # Vis tabellen
    st.dataframe(result_df, use_container_width=True)

    # ----------------------------
    # 4. HEATMAP: sjanger × tiår
    # ----------------------------

    st.subheader("Fordeling over tid (tiår) og sjanger")

    # Lag tiårs-kolonne
    full_df["decade"] = (full_df["year"] // 10) * 10

    heat_data = []

    for g in genres:
        gdf = full_df[full_df["genre"] == g]
        for decade in sorted(gdf["decade"].unique()):
            urns = gdf[gdf["decade"] == decade]["urn"].tolist()
            if not urns:
                hits = 0
            else:
                corpus = dh.Corpus()
                corpus.extend_from_identifiers(urns)
                counts = Counts(corpus)

                key = keyword.casefold()
                keys = [w for w in counts.frame.index if w.casefold().startswith(key.replace("*", ""))]

                hits = counts.frame.loc[keys].sum().sum() if keys else 0

            heat_data.append([g, decade, hits])

    heat_df = pd.DataFrame(heat_data, columns=["Genre", "Decade", "Hits"])

    # Plotly heatmap
    fig = px.imshow(
        heat_df.pivot(index="Genre", columns="Decade", values="Hits").fillna(0),
        labels=dict(x="Decade", y="Genre", color="Hits"),
        aspect="auto",
        color_continuous_scale="Reds"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Merk: Relative frekvenser beregnes for hele sjangeren, mens heatmap viser rene forekomster per tiår."
    )


# ======================================================================
# === 2) Sammnenligning av frekvenser i sjangerdefinerte delkorpora ===
# ======================================================================

st.title("Frekvenser i delkorpora definert ved sjangerbenevnelser")

st.markdown(
    """
- Gå til menyen i sidebaren og velg en **sjanger** for å se frekvente ord for denne – eller velg to **sjangre** for sammenligning. (Klikk utenfor sjangerboksen for å få bort nedtrekksmenyen etter du har valgt sjanger/e.)
- Avgrens tidsperioden hvis ønskelig eller la det stå uendret for å hente ut frekvenser for alle tekstene innen sjangeren på 1800-tallet.
- Angi hvilke av de mest frekvente ordene du ønsker å se. Merk at de aller mest frekvente gjerne vil være funksjonsord.  
  Default er satt til spennet fra det 50. til det 70. mest frekvente ordet.
- Når du er klar til at frekvenslistene hentes ut, klikk på **Vis frekvenser** (og vær litt tålmodig 😊)
    """
)

# === Last metadata ===
meta_df = pd.read_excel("Sjangre_kategorisert_010625.xlsx")

# === Funksjon for å hente URN-er ===
def get_urns_for_genre(df, genre, year_from, year_to):
    genre_filtered = df[
        (df["genre"].str.strip().str.casefold() == genre.casefold()) &
        (df["urn"].notna()) &
        (df["urn"].str.startswith("URN")) &
        (df["year"].between(year_from, year_to))
    ]
    return genre_filtered["urn"].tolist()

# === Funksjon for å hente frekvensliste ===
def get_top_words_with_freq(urns, start=50, end=70):
    corpus = dh.Corpus()
    corpus.extend_from_identifiers(identifiers=urns)
    counts = Counts(corpus)
    freq_series = counts.frame.sum(axis=1).sort_values(ascending=False)
    selected = freq_series.iloc[start:end]
    return [f"{word} ({count})" for word, count in selected.items()]

# === Tell sjangre som har minst én fulltekst-URN ===
genre_counts = (
    meta_df[meta_df["urn"].notna() & meta_df["urn"].str.startswith("URN")]
    .groupby("genre")["urn"]
    .count()
    .to_dict()
)

# Lag meny med "Sjanger (antall)"
all_genres = sorted([f"{genre} ({count})" for genre, count in genre_counts.items()])

# === SIDEBAR ===
with st.sidebar.form(key="freq_form"):
    selected_genres = st.multiselect(
        "Velg én sjanger eller to sjangre for sammenligning", all_genres, default=[all_genres[0]]
    )

    year_from = st.number_input("Fra år", min_value=1800, max_value=1899, value=1800)
    year_to = st.number_input("Til år", min_value=1800, max_value=1899, value=1899)

    start_rank = st.slider("Start (Frekvensrangering)", min_value=1, max_value=500, value=50)
    end_rank = st.slider("Slutt (Frekvensrangering)", min_value=1, max_value=500, value=70)

    submitted = st.form_submit_button("Vis frekvenser")

# === Kjør kun hvis en eller to sjangre er valgt ===
if submitted:
    if 1 <= len(selected_genres) <= 2:
        genre_names = [g.split(" (")[0] for g in selected_genres]
        urn_lists = [get_urns_for_genre(meta_df, genre, year_from, year_to) for genre in genre_names]
        top_lists = [get_top_words_with_freq(urns, start_rank, end_rank) for urns in urn_lists]

        # Bygg resultat-tabell
        data = {"Frekvensrangering": list(range(start_rank, start_rank + len(top_lists[0])))}
        for name, words in zip(selected_genres, top_lists):
            data[name] = words

        df = pd.DataFrame(data)
        st.markdown(f"### De {start_rank}–{end_rank} mest frekvente ordene i korpuset/ene")
        df.set_index("Frekvensrangering", inplace=True)
        st.dataframe(df, use_container_width=True)

    else:
        st.warning("Velg én eller to sjangre.")
