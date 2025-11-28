import streamlit as st
import dhlab as dh 
import pandas as pd
from dhlab import Counts
import plotly.express as px
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
import os

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

Du får først en tabell med: 

- **relativ frekvens** (per sjanger) (som tabellen er sortert etter)
- antall **verk hvor ordet forekommer**
- totalt antall **treff**

deretter:

- **heatmap** som viser utvikling (relativ frekvens) per tiår

NB. Bregningen tar noe tid (følg med på det "arbeidende" ikonet øverst til høyre). Først kommer tabellen, så jobber programmet videre og viser etter hvert heatmap 

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
    def stats_for_genre(genre):
        """Returnerer: hits, relative freq, works_with_hits."""
        gdf = full_df[full_df["genre"].str.casefold() == genre.casefold()] #casefold() istedenfor lower() unødvendig i denne sammenheng, men greit å vite om
        urns = gdf["urn"].tolist()

        if not urns:
            return 0, 0.0, 0

        corpus = dh.Corpus()
        corpus.extend_from_identifiers(urns)
        counts = Counts(corpus)

        total_tokens = counts.frame.sum().sum()
        match_words = [w for w in counts.frame.index if w.casefold().startswith(key_base)]

        if not match_words:
            return 0, 0.0, 0

        hits = int(counts.frame.loc[match_words].sum().sum())

        per_work = counts.frame.loc[match_words].sum(axis=0)
        works_with_hits = int((per_work > 0).sum())

        relative = hits / total_tokens if total_tokens else 0

        return hits, relative, works_with_hits

    # ----------------------------
    # 3. Sjangerstatistikk
    # ----------------------------
    rows = []
    for g in genres:
        hits, rel, works_hit = stats_for_genre(g)
        if hits > 0:
            rows.append((g, hits, rel, works_hit))

    result_df = pd.DataFrame(rows, columns=["Genre", "Hits", "RelativeFreq", "WorksWithHits"])

    # Sorter etter relativ frekvens, ikke rå hits
    result_df = result_df.sort_values("RelativeFreq", ascending=False)

    st.subheader(f"Resultater for nøkkelord: **{keyword}**")
    st.dataframe(result_df, use_container_width=True)

    # ----------------------------
    # 4. HEATMAP — relativ frekvens per tiår
    # ----------------------------
    st.subheader("Fordeling over tid (relativ frekvens per tiår per sjanger)")

    active_genres = result_df["Genre"].tolist()  # bare sjangre med treff

    heat_rows = []

    for g in active_genres:
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

            total_tokens = counts.frame.sum().sum()
            match_words = [w for w in counts.frame.index if w.casefold().startswith(key_base)]
            hits = int(counts.frame.loc[match_words].sum().sum()) if match_words else 0

            rel = hits / total_tokens if total_tokens else 0
            heat_rows.append([g, d, rel])

    heat_df = pd.DataFrame(heat_rows, columns=["Genre", "Decade", "RelFreq"])
    heat_pivot = heat_df.pivot(index="Genre", columns="Decade", values="RelFreq").fillna(0)

    fig = px.imshow(
        heat_pivot,
        labels=dict(x="Decade", y="Genre", color="Relativ frekvens"),
        aspect="auto",
        color_continuous_scale="Reds"
    )

    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(heat_pivot.index))),
        ticktext=list(heat_pivot.index)
    )

    st.plotly_chart(fig, use_container_width=True)


# ================================================================
# === 2) TF–IDF-ANALYSER =========
# ================================================================

st.header("2. TF–IDF-analyser")

st.markdown(
    """
Denne funksjonen viser hvilke ord som er **mest distinktive** i ett verk eller en sjanger, 
basert på *Term Frequency – Inverse Document Frequency (TF–IDF)*. TF–IDF måler hvor karakteristisk et ord er for et dokument sammenlignet med resten av korpuset.
"""
)

with st.expander("Se mer inngående forklaring"):
    st.markdown(
        """
Vi bruker scikit-learns implementasjon av TF–IDF, som beregner termvekter som normalisert termfrekvens multiplisert med den log-skalerte inverse dokumentfrekvensen.

Metoden kombinerer:

- **Term Frequency (TF):** hvor ofte ordet forekommer i dokumentet.  
- **Inverse Document Frequency (IDF):** en logaritmisk nedvekting av ord som forekommer i mange dokumenter i korpuset.  
  Logaritmisk skalering gjør at forskjeller i dokumentfrekvens håndteres på en måte som demper effekten av svært høye eller svært lave verdier, slik at vektingen reflekterer relative forskjeller heller enn rene absolutte frekvenser.

Ord som forekommer ofte i ett dokument, men sjelden i andre dokumenter, får **høy TF–IDF-score**.  
Ord som finnes i mange dokumenter i korpuset (som funksjonsord) får **lav score**.
"""
    )

st.markdown(
    """
Vi bruker **lokale fulltekstfiler** hentet fra Nasjonalbiblioteket.

- **Verk vs. korpus** sammenligner ett verk med hele korpuset (248 fulltekster).  
- **Sjanger vs. korpus** sammenligner en sjanger med hele korpuset (248 fulltekster).  
- **Verk vs. verk** viser hvilke ord som skiller to tekster fra hverandre  
  (TF–IDF er egentlig ikke designet for dette, men det kan likevel gi interessante indikasjoner dersom man blar forbi egennavn og funksjonsord).
"""
)


# -------------------------------
# 1. Hent alle verk som faktisk har fulltekst
# -------------------------------
df_fulltext = meta_df[
    meta_df["urn"].notna()
    & meta_df["urn"].str.startswith("URN")
].copy()

# Sjekk hvilke filer som faktisk finnes
available_files = set(os.listdir("filer"))

def urn_to_filename(urn):
    """URN → filnavn basert på NB-format."""
    return urn.replace("URN:NBN:", "")

df_fulltext["file_exists"] = df_fulltext["urn"].apply(
    lambda u: urn_to_filename(u) in available_files
)

df_available = df_fulltext[df_fulltext["file_exists"]].copy()

# Format visningsnavn
def format_work_row(row):
    return f"{row['year']} — {row['title']} — {row['author']}"

df_available["label"] = df_available.apply(format_work_row, axis=1)

# -------------------------------
# Funksjon: Les lokal tekstfil
# -------------------------------
def load_local_text(urn):
    """Leser lokal fulltekst basert på filnavn."""
    filename = urn_to_filename(urn)
    path = os.path.join("filer", filename)

    if not os.path.exists(path):
        return ""

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except:
        return ""

# -------------------------------
# Bruker velger hvilken TF–IDF-analyse
# -------------------------------
mode = st.radio(
    "Velg analysemetode:",
    ["Verk vs. korpus", "Sjanger vs. korpus", "Verk vs. verk"],
)


top_n = st.slider("Hvor mange distinktive ord skal vises?", 20, 300, 100)


# ------------------------------------------------
# === A) TF–IDF: VERK VS HELE KORPUSET ===========
# ------------------------------------------------
if mode == "Verk vs. korpus":

    st.subheader("TF–IDF: verk sammenlignet med hele korpuset")

    # --- Velg SJANGER først ---
    genre_choice = st.selectbox(
        "Velg sjanger",
        sorted(df_available["genre"].unique()),
        key="tfidf_genre_single"
    )

    # Filtrer verk etter valgt sjanger
    df_subset = df_available[df_available["genre"] == genre_choice]

    # Velg VERK innen sjanger
    work_choice = st.selectbox(
        "Velg verk",
        df_subset["label"].tolist(),
        key="tfidf_work_single"
    )

    selected_urn = df_subset.loc[df_subset["label"] == work_choice, "urn"].iloc[0]

    # Hent tekst
    text_target = load_local_text(selected_urn)

    if st.button("Beregn TF–IDF for valgt verk"):
        if not text_target:
            st.error("Kunne ikke hente fulltekst for valgt verk.")
        else:
            # --- Lag korpuset: alle andre verk med tekst ---
            corpus_texts = []
            corpus_labels = []

            for _, row in df_available.iterrows():
                if row["urn"] != selected_urn:
                    txt = load_local_text(row["urn"])
                    if txt:
                        corpus_texts.append(txt)
                        corpus_labels.append(row["label"])

            full_corpus = [text_target] + corpus_texts

            # --- TF–IDF ---
            vectorizer = TfidfVectorizer(
                lowercase=True,
                token_pattern=r"[A-Za-zÆØÅæøå]+",
                min_df=2,
            )

            X = vectorizer.fit_transform(full_corpus)
            feature_names = vectorizer.get_feature_names_out()

            tfidf_target = X.toarray()[0]
            tfidf_rest = X.toarray()[1:].mean(axis=0)

            diff = tfidf_target - tfidf_rest
            top_idx = diff.argsort()[::-1][:top_n]

            rows = []
            for idx in top_idx:
                rows.append((feature_names[idx], tfidf_target[idx], tfidf_rest[idx], diff[idx]))

            df_tfidf = pd.DataFrame(rows, columns=["Ord", "TF-IDF (verk)", "TF-IDF (korpus)", "Forskjell"])

            st.write("### Mest særpregede ord i verket")
            st.dataframe(df_tfidf, use_container_width=True)

            fig = px.bar(
                df_tfidf,
                x="Ord",
                y="Forskjell",
                title="Særpregede ord (TF–IDF forskjell: verk minus korpus)",
            )
            st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------
# === B) TF–IDF: SJANGER VS HELE KORPUSET ========
# ------------------------------------------------
elif mode == "Sjanger vs. korpus":

    st.subheader("TF–IDF: sjanger sammenlignet med hele korpuset")

    # Velg sjanger (bare sjangre som har tilgjengelige fulltekster)
    genre_options = sorted(df_available["genre"].dropna().unique())
    selected_genre = st.selectbox("Velg sjanger", genre_options)

    # Finn alle verk i denne sjangeren
    gdf = df_available[df_available["genre"] == selected_genre]

    if len(gdf) == 0:
        st.warning("Ingen fulltekstfiler tilgjengelig for denne sjangeren.")
    else:
        if st.button("Beregn TF–IDF for sjanger"):

            # Slå sammen alle tekster i sjangeren
            genre_texts = []
            for _, row in gdf.iterrows():
                txt = load_local_text(row["urn"])
                if txt:
                    genre_texts.append(txt)

            if not genre_texts:
                st.error("Kunne ikke hente fulltekstene for valgt sjanger.")
            else:
                genre_concat = "\n\n".join(genre_texts)

                # Lag korpuset (alle andre verk)
                corpus_texts = []
                for _, row in df_available.iterrows():
                    if row["genre"] != selected_genre:   # ekskluder sjangerteksten
                        txt = load_local_text(row["urn"])
                        if txt:
                            corpus_texts.append(txt)

                full_corpus = [genre_concat] + corpus_texts

                # --- TF–IDF ---
                vectorizer = TfidfVectorizer(
                    lowercase=True,
                    token_pattern=r"[A-Za-zÆØÅæøå]+",
                    min_df=2,
                )

                X = vectorizer.fit_transform(full_corpus)
                feature_names = vectorizer.get_feature_names_out()

                tfidf_genre = X.toarray()[0]
                tfidf_rest = X.toarray()[1:].mean(axis=0)

                diff = tfidf_genre - tfidf_rest
                top_idx = diff.argsort()[::-1][:top_n]

                rows = []
                for idx in top_idx:
                    rows.append((feature_names[idx], tfidf_genre[idx], tfidf_rest[idx], diff[idx]))

                df_tfidf = pd.DataFrame(
                    rows,
                    columns=["Ord", "TF-IDF (sjanger)", "TF-IDF (korpus)", "Forskjell"]
                )

                st.write(f"### Mest særpregede ord i sjangeren *{selected_genre}*")
                st.dataframe(df_tfidf, use_container_width=True)

                fig = px.bar(
                    df_tfidf,
                    x="Ord",
                    y="Forskjell",
                    title=f"Særpregede ord (TF–IDF forskjell: {selected_genre} minus korpus)",
                )
                st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# === C) TF–IDF: VERK VS VERK ====================
# ------------------------------------------------
else:

    st.subheader("TF–IDF: sammenlign to verk")

    col1, col2 = st.columns(2)

    # -------------------------
    #  Verkkolonne 1
    # -------------------------
    with col1:
        genre1 = st.selectbox(
            "Velg sjanger for verk 1",
            sorted(df_available["genre"].unique()),
            key="tfidf_genre_1"
        )

        df_subset1 = df_available[df_available["genre"] == genre1]

        work1 = st.selectbox(
            "Velg verk 1",
            df_subset1["label"].tolist(),
            key="tfidf_work_1"
        )

    # -------------------------
    #  Verkkolonne 2
    # -------------------------
    with col2:
        genre2 = st.selectbox(
            "Velg sjanger for verk 2",
            sorted(df_available["genre"].unique()),
            key="tfidf_genre_2"
        )

        df_subset2 = df_available[df_available["genre"] == genre2]

        work2 = st.selectbox(
            "Velg verk 2",
            df_subset2["label"].tolist(),
            key="tfidf_work_2"
        )

    # -----------------------
    # Hent tekst
    # -----------------------
    urn1 = df_subset1.loc[df_subset1["label"] == work1, "urn"].iloc[0]
    urn2 = df_subset2.loc[df_subset2["label"] == work2, "urn"].iloc[0]

    if st.button("Beregn TF–IDF mellom verkene"):
        text1 = load_local_text(urn1)
        text2 = load_local_text(urn2)

        if not text1 or not text2:
            st.error("Kunne ikke hente fulltekst for ett eller begge verk.")
        else:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                token_pattern=r"[A-Za-zÆØÅæøå]+",
            )
            X = vectorizer.fit_transform([text1, text2])
            feature_names = vectorizer.get_feature_names_out()

            tfidf1 = X.toarray()[0]
            tfidf2 = X.toarray()[1]

            diff = tfidf1 - tfidf2
            abs_diff = abs(diff)

            top_idx = abs_diff.argsort()[::-1][:top_n]

            rows = []
            for idx in top_idx:
                rows.append((feature_names[idx], tfidf1[idx], tfidf2[idx], diff[idx]))

            df_tfidf = pd.DataFrame(
                rows, columns=["Ord", "TF-IDF (verk 1)", "TF-IDF (verk 2)", "Forskjell"]
            )

            df_tfidf = df_tfidf.sort_values("Forskjell", ascending=False)

            st.write("### Distinktive ord mellom verkene")
            st.dataframe(df_tfidf, use_container_width=True)

            fig = px.bar(
                df_tfidf,
                x="Ord",
                y="Forskjell",
                title="Forskjell i TF–IDF (verk 1 minus verk 2)",
            )
            st.plotly_chart(fig, use_container_width=True)



# ======================================================================
# === 3) Sammnenligning av frekvenser i sjangerdefinerte delkorpora ===
# ======================================================================

st.header("3.Frekvenser i delkorpora definert ved sjangerbenevnelser")

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
