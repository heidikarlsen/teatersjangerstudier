import streamlit as st
import dhlab as dh 
import pandas as pd
from dhlab.text.conc_coll import Concordance


st.set_page_config(page_title = "Teatersjangerstudier - Konkordanser", page_icon ="🎭", layout = "wide")
st.title("Konkordanser i delkorpora definert ved sjangerbenevnelser")
st.markdown("Velg sjanger i sidebaren eller la eventuelt **'Alle'** stå om du ønsker å søke etter konkordanser på tvers av alle dokumentene. Definer eventuelt andre kritierer eller la det stå urørt for å søke i alle tekster eller alle innen valgt sjanger.")


query = st.text_input("Søk", "", placeholder="Skriv inn søkeuttrykk her")


# === Last metadata ===
meta_df = pd.read_excel("Sjangre_kategorisert_010625.xlsx")


# --- Sidebar ---
st.sidebar.header("Filtrering av korpus")

# Sjangervalg -  med minst én URN med fulltekst
genres_with_urns = (
    meta_df[meta_df["urn"].notna() & meta_df["urn"].str.startswith("URN")]
    ["genre"]
    .dropna()
    .unique()
)
all_genres = ["Alle"] + sorted(genres_with_urns)

selected_genre = st.sidebar.selectbox("Sjanger", all_genres)


# Filtrer metadata basert på valgt sjanger (for å hente forfatterne)
if selected_genre == "Alle":
    genre_filtered_df = meta_df[meta_df["author"].notna()]
else:
    genre_filtered_df = meta_df[
        (meta_df["genre"].str.strip().str.casefold() == selected_genre.casefold()) &
        (meta_df["author"].notna())
    ]



# Hent forfattere som har skrevet i valgt sjanger
genre_authors = sorted(genre_filtered_df["author"].unique())
genre_authors.insert(0, "Alle")
selected_author = st.sidebar.selectbox("Velg forfatter", genre_authors)
all_authors = sorted(meta_df["author"].dropna().unique())
all_authors.insert(0, "Alle")  # Legg til 'Alle' som førstevalg


#Andre filtre
year_from = st.sidebar.number_input("Fra år", min_value=1800, max_value=1899, value=1800)
year_to = st.sidebar.number_input("Til år", min_value=1800, max_value=1899, value=1899)
window_size = st.sidebar.slider("Konkordansevindu (antall ord før/etter)", 5, 25, 10)
max_hits = st.sidebar.slider("Maks antall treff", 10, 1000, 100)




# --- Filtrér metadata ---
filtered = meta_df[
    (meta_df["urn"].notna()) &
    (meta_df["urn"].str.startswith("URN")) &
    (meta_df["year"].between(year_from, year_to))
]

if selected_genre != "Alle":
    filtered = filtered[filtered["genre"].str.strip().str.casefold() == selected_genre.casefold()]


if selected_author != "Alle":
    filtered = filtered[filtered["author"] == selected_author]


urns = filtered["urn"].tolist()
st.markdown(f"**Antall verk i valgt korpus:** {len(urns)}")

# --- Vis konkordanser ---
if query and urns:
    corpus = dh.Corpus()
    corpus.extend_from_identifiers(urns)
    concordance = Concordance(corpus=corpus, query=query, window=window_size, limit=max_hits)

    st.markdown(f"### Treff ({concordance.size})")
    for _, row in concordance.show(n=min(max_hits, concordance.size), style=False).iterrows():
        urn = row["urn"]
        context = row["concordance"].replace("<b>", "**").replace("</b>", "**")

        match_meta = filtered[filtered["urn"] == urn]
        if not match_meta.empty:
            meta = match_meta.iloc[0]
            title = meta.get("title", "")
            author = meta.get("author", "")
            year = meta.get("year", "")
            url = f"https://urn.nb.no/{urn}"
            header = f"[{title} – {author} – {year}]({url})"
            st.markdown(f"{header}<br>{context}", unsafe_allow_html=True)
else:
    st.info("Skriv inn et søkeord og velg en sjanger for å vise konkordanser.")


