import streamlit as st
import dhlab as dh 
import pandas as pd
from dhlab import Counts
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Teatersjangerstudier - Frekvenser", page_icon="🎭", layout="wide")

# === Last metadata ===
meta_df = pd.read_excel("Sjangre_kategorisert_010625.xlsx")

# ================================================================
# === 1) Sammnenligning av frekvenser per sjanger for nøkkelord ===
# ================================================================


st.header("1.Sammenlign nøkkelord på tvers av sjangre")

st.markdown(
    """
Skriv inn et **søkeord** (f.eks. `norsk*`, `patriot*`, `nation*`).  
Analysen viser hvor ofte ordet forekommer i hver sjanger, både som **antall treff** og **relativ frekvens**.
Du får: 

- antall **verk hvor ordet forekommer**
- totalt antall **treff**
- **relativ frekvens** (per sjanger)
- **heatmap** som viser utvikling per tiår

"""
)


# --- Velg tiårsintervall for heatmap ---
min_decade = int(meta_df["year"].min() // 10 * 10)
max_decade = int(meta_df["year"].max() // 10 * 10)

heat_range = st.slider(
    "Velg tidsintervall for heatmap (tiår)",
    min_value=min_decade,
    max_value=max_decade,
    value=(min_decade, max_decade),
    step=10
)


keyword = st.text_input("Nøkkelord (wildcard støttes)", "")

if keyword:

    # ----------------------------
    # 1. Forbered data
    # ----------------------------
    full_df = meta_df[
        meta_df["urn"].notna() & meta_df["urn"].str.startswith("URN")
    ].copy()

    full_df["decade"] = (full_df["year"] // 10) * 10

    genres = sorted(full_df["genre"].dropna().unique())
    key_base = keyword.replace("*", "").casefold()

    # ----------------------------
    # 2. Analysefunksjon
    # ----------------------------
    def stats_for_genre(genre, keyword):
        """Returnerer: hits, relative freq, works_with_hits, works_total."""
        gdf = full_df[full_df["genre"].str.casefold() == genre.casefold()]
        urns = gdf["urn"].tolist()

        if not urns:
            return 0, 0.0, 0, 0

        corpus = dh.Corpus()
        corpus.extend_from_identifiers(urns)
        counts = Counts(corpus)

        # total tokens
        total_tokens = counts.frame.sum().sum()

        # wildcard matching
        match_words = [w for w in counts.frame.index if w.casefold().startswith(key_base)]

        if not match_words:
            return 0, 0.0, 0, len(urns)

        # total hits
        hits = int(counts.frame.loc[match_words].sum().sum())

        # count how many distinct works contain the word(s)
        per_work = counts.frame.loc[match_words].sum(axis=0)
        works_with_hits = int((per_work > 0).sum())

        relative = hits / total_tokens if total_tokens > 0 else 0

        return hits, relative, works_with_hits, len(urns)

    # ----------------------------
    # 3. Beregn sjangerstatistikk
    # ----------------------------
    rows = []
    for g in genres:
        hits, rel, works_hit, n_total = stats_for_genre(g, keyword)
        if hits > 0:   # ta kun med sjangre der ordet faktisk forekommer
            rows.append((g, hits, rel, works_hit, n_total))

    result_df = pd.DataFrame(rows, columns=["Genre", "Hits", "RelativeFreq", "WorksWithHits", "WorksTotal"])
    result_df = result_df.sort_values("Hits", ascending=False)

    st.subheader(f"Resultater for nøkkelord: **{keyword}**")
    st.dataframe(result_df, use_container_width=True)

    # ----------------------------
    # 4. HEATMAP
    # ----------------------------
    st.subheader("Temporal distribution (hits per decade per genre)")

    heat_rows = []

    for g in genres:
        gdf = full_df[full_df["genre"] == g]
        decades = sorted(gdf["decade"].unique())

        for d in decades:
            if not (heat_range[0] <= d <= heat_range[1]):
                continue

            sub = gdf[gdf["decade"] == d]
            urns = sub["urn"].tolist()

            if not urns:
                heat_rows.append([g, d, 0])
                continue

            corpus = dh.Corpus()
            corpus.extend_from_identifiers(urns)
            counts = Counts(corpus)
            match_words = [w for w in counts.frame.index if w.casefold().startswith(key_base)]

            hits = int(counts.frame.loc[match_words].sum().sum()) if match_words else 0
            heat_rows.append([g, d, hits])

    heat_df = pd.DataFrame(heat_rows, columns=["Genre", "Decade", "Hits"])
    heat_pivot = heat_df.pivot(index="Genre", columns="Decade", values="Hits").fillna(0)

    fig = px.imshow(
        heat_pivot,
        labels=dict(x="Decade", y="Genre", color="Hits"),
        aspect="auto",
        color_continuous_scale="Reds"
    )

    fig.update_yaxes(tickmode="array", tickvals=list(range(len(heat_pivot.index))), ticktext=list(heat_pivot.index))

    st.plotly_chart(fig, use_container_width=True)

# ======================================================================
# === 2) Sammnenligning av frekvenser i sjangerdefinerte delkorpora ===
# ======================================================================

st.header("2.Frekvenser i delkorpora definert ved sjangerbenevnelser")

st.markdown(
    """
- Gå til menyen i sidebaren og velg en **sjanger** for å se frekvente ord for denne – eller velg to **sjangre** for sammenligning. (Klikk utenfor sjangerboksen for å få bort nedtrekksmenyen etter du har valgt sjanger/e.)
- Avgrens tidsperioden hvis ønskelig eller la det stå uendret for å hente ut frekvenser for alle tekstene innen sjangeren på 1800-tallet.
- Angi hvilke av de mest frekvente ordene du ønsker å se. Merk at de aller mest frekvente gjerne vil være funksjonsord.  
  Default er satt til spennet fra det 50. til det 70. mest frekvente ordet.
- Når du er klar til at frekvenslistene hentes ut, klikk på **Vis frekvenser** (og vær litt tålmodig 😊)
    """
)



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
