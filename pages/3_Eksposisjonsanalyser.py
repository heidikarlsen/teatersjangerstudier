import streamlit as st
import dhlab as dh 
import pandas as pd
import openai


st.set_page_config(page_title = "Teatersjangerstudier - AI-analyser", page_icon ="🎭", layout = "wide")
st.title("Chat med korpusene ved hjelp av utdrag fra stykkene (ideelt eksposisjonen) via API-et til Open AI.")
st.markdown("")
st.markdown(
    """
Denne funksjonen er under utvikling. Målet er å bruke utdrag fra eksposisjonene i fulltekster fra Nasjonalbiblioteket som grunnlag for analyser og samtaler med en språkmodell.  

Her kan du teste grensesnittet – men AI-en er foreløpig ikke koblet til. 

Når tjenesten er på plass, vil du kunne Velge sjanger i sidebaren eller la eventuelt **'Alle'** stå om du ønsker å analysere eksposisjonene på tvers av sjanger. Definer eventuelt andre kritierer eller la det stå urørt for å søke i tilfeldig utdrag/alle tekster eller alle/tilfeldig utdrag innen valgt sjanger.
"""
)


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


# --- Dummy-ekstrakt-funksjon ---
def get_text_excerpt(urn, n=300):
    corpus = dh.Corpus()
    corpus.extend_from_identifiers(urns)
    #fulltext = corpus.get_text()
    #return fulltext[:n]


# --- Eksempel på uttrekk-skisse ---
#excerpts = [get_text_excerpt(urn, n=300) for urn in urns[:3]]
#combined_excerpt = "\n\n".join(excerpts)


#Chat-grensesnitt 

#openai.api_key = st.secrets["openai"]["api_key"]

if prompt := st.chat_input("Still et spørsmål til valgt/e korpus/er"):
    #response = openai.ChatCompletion.create(
        #model="gpt-4",
        #messages=[
           # {"role": "system", "content": "Du er en teater- og litteraturforsker som analyserer norske teaterstykker fra 1800-tallet."},
           # {"role": "user", "content": prompt}
        #]
    #)
    #st.chat_message("assistant").markdown(response["choices"][0]["message"]["content"])
    st.chat_message("user").markdown(prompt)
    
    st.chat_message("assistant").markdown(
        "**Smør deg med litt tålmodighet 🧈⏳**\n\n"
        "Tjenesten er fortsatt under utvikling, og AI-svaret kommer snart! "
        "Foreløpig brukes ingen språkmodell. Når dette er aktivert, vil spørsmålet ditt bli sendt sammen med utdrag fra korpuset."
    )