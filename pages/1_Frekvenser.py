import streamlit as st
import dhlab as dh 
import pandas as pd
from dhlab import Counts

st.set_page_config(page_title="Teatersjangerstudier - Frekvenser", page_icon="🎭", layout="wide")

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
