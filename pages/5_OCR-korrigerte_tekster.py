import streamlit as st
import pandas as pd
import dhlab as dh
from dhlab import Counts 
import numpy as np
import os
import re
import html
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px

st.set_page_config(page_title="Teatersjangerstudier - Diverse analyser av OCR-korrigerte tekster", page_icon="🎭", layout="wide")


# ================================================================
# === 1) Frekvenser ===
# ================================================================

st.header("1.Frekvenser")

st.markdown(
    """
Velg et verk - eller to verk for sammenligning - i menyen under og se de mest frekvente termene i verket/verkene. 

På grunn av forskjellige språk og dialekter i tesktene er ikke et stoppordfilter lagt inn. Du må derfor scrolle en del ned 
før du ser frekvente ord med en viss semantisk tyngde og kommer forbi egennavn. 

Under tabellen kan du søke etter frekvenser for et spesifikt ord. 

"""
)

# --------------------------------------------------
# 1. Hent kun filer som finnes i ocr_kor-mappen
# --------------------------------------------------

Path = "filer_ocr_kor"

available_files = [
    f for f in os.listdir(Path)
    if not f.startswith(".")  # ignorer skjulte filer
]

# Filene er nå URN
available_urns = [f.replace(".txt", "") for f in available_files]

# --------------------------------------------------
# 2. Koble på metadata
# --------------------------------------------------

meta_df = pd.read_excel("1800-1849_020326.xlsx")

meta_df["urn"] = (
    meta_df["urn"]
    .str.replace("URN:NBN:", "", regex=False)
    .str.strip()
)

df_available = meta_df[
    meta_df["urn"].isin(available_urns)
].copy()

def format_label(row):
    return f"{row['title']} — {row['author']} — {row['year']} — {row['genre']}"

df_available["label"] = df_available.apply(format_label, axis=1)

# Sorter
df_available = df_available.sort_values(["year", "author"])



# --------------------------------------------------
# 3. Velg verk 
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    label1 = st.selectbox(
        "Velg verk",
        df_available["label"].tolist(),
        key="verk1"
    )

with col2:
    label2 = st.selectbox(
        "Velg eventuelt verk 2",
        ["—"] + df_available["label"].tolist(),
        key="verk2"
    )

# --------------------------------------------------
# 4. Funksjon for å lese tekst
# --------------------------------------------------

def load_text_from_urn(urn):
    filename = f"{urn}.txt"
    path = os.path.join(Path, filename)

    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# --------------------------------------------------
# 5. Frekvensberegning
# --------------------------------------------------

def compute_frequencies(text, top_n=500):
    tokens = text.split()
    total = len(tokens)

    freq = pd.Series(tokens).value_counts()
    rel = freq / total

    df = pd.DataFrame({
        "Token": freq.index,
        "Antall": freq.values,
        "Relativ frekvens": rel.values
    })

    return df.head(top_n)



def show_freq_table(df):
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Token": st.column_config.TextColumn(
                "Token",
                width="medium"
            ),
            "Antall": st.column_config.NumberColumn(
                "Antall",
                width="small"
            ),
            "Relativ frekvens": st.column_config.NumberColumn(
                "Relativ frekvens",
                format="%.4f",
                width="small"
            ),
        }
    )
# --------------------------------------------------
# 6. Vis frekvenser
# --------------------------------------------------

if label1:
    urn1 = df_available[df_available["label"] == label1]["urn"].iloc[0]
    text1 = load_text_from_urn(urn1)

    freq1 = compute_frequencies(text1)

    if label2 != "—":
        urn2 = df_available[df_available["label"] == label2]["urn"].iloc[0]
        text2 = load_text_from_urn(urn2)
        freq2 = compute_frequencies(text2)

        colA, colB = st.columns(2)

        with colA:
            show_freq_table(freq1)

        with colB:
            show_freq_table(freq2)

    else:
        st.dataframe(freq1)



st.divider()


st.markdown(
    """
Skriv inn et **søkeord** (f.eks. `norsk*`, `patriot*`, `nation*`).  
Du ser hvor ofte ordet forekommer i hvert verk, både som **antall treff** og **relativ frekvens** (som tabellen er sortert etter).

"""
)

# --------------------------------------------------
# Søkeord-input
# --------------------------------------------------

keyword = st.text_input("Nøkkelord (wildcard støttes)", "")

if keyword:

    key_base = keyword.replace("*", "").lower()

    rows = []

    for _, row in df_available.iterrows():

        urn = row["urn"]
        text = load_text_from_urn(urn)

        tokens = text.split()
        total_tokens = len(tokens)

        if total_tokens == 0:
            continue

        # tell treff
        hits = sum(
            1 for t in tokens
            if t.lower().startswith(key_base)
        )

        if hits > 0:
            rel = hits / total_tokens

            rows.append({
                "Tittel": row["title"],
                "Forfatter": row["author"],
                "År": row["year"],
                "Sjanger": row["genre"],
                "Treff": hits,
                "Relativ frekvens": rel
            })

    if rows:

        result_df = pd.DataFrame(rows)

        result_df = result_df.sort_values(
            "Relativ frekvens",
            ascending=False
        )

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Relativ frekvens": st.column_config.NumberColumn(format="%.5f")
            }
        )

        st.markdown(
            f"**Antall verk med treff:** {len(result_df)}  \n"
            f"**Totalt antall treff:** {result_df['Treff'].sum()}"
        )

    else:
        st.info("Ingen treff.")



st.divider()


# ================================================================
# === 2) TF–IDF-ANALYSER =========
# ================================================================

st.header("2. TF–IDF-analyser")

st.markdown(
    """
Denne funksjonen viser hvilke ord som er **mest distinktive** i ett verk, 
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

Ord som forekommer ofte i ett dokument, men sjelden i andre dokumenter, får gjerne **høy TF–IDF-score**, mens ord som finnes i mange tekster får lavere score.  

I denne visningen rangeres ordene etter hvor mye høyere TF–IDF-vekt de har i det valgte verket enn i de øvrige tekstene i utvalget. Resultatet viser derfor **relativ særpreg**, ikke en absolutt liste over de viktigste ordene i verket.

"""
    )

st.markdown(
    """
Her bruker vi de **OCR-korrigerte filene** basert på fulltekstene fra Nasjonalbiblioteket.

Analysen sammenligner **ett valgt verk** med **de øvrige verkene i dette utvalget**. 
Resultatet viser ord som får høyere TF–IDF-vekt i det valgte verket enn i de andre tekstene, og kan derfor leses som en oversikt over ord som er **relativt særpregede i akkurat dette materialet**.

OBS: Utvalget er foreløpig lite (9 dramaer totalt per 16.04.2026), så resultatene bør brukes **eksplorativt** og tolkes med varsomhet. 
Høye scorer betyr ikke nødvendigvis at et ord er tematisk viktig; de kan også skyldes navn, sjeldne former, språkvariasjon eller andre trekk som skiller verket fra de øvrige tekstene.

"""
)

label_tf_idf = st.selectbox(
    "Velg verk",
        df_available["label"].tolist(),
        key="verk"

)


top_n = st.slider("Hvor mange distinktive ord skal vises?", 20, 300, 100)


# Hent tekst

urn = df_available[df_available["label"] == label_tf_idf]["urn"].iloc[0]
text_target = load_text_from_urn(urn)


if st.button("Beregn TF–IDF for valgt verk"):
    if not text_target:
        st.error("Kunne ikke hente fulltekst for valgt verk.")
    else:
        # --- Lag korpuset: alle andre verk med tekst ---
        corpus_texts = []
        corpus_labels = []

        for _, row in df_available.iterrows():
            if row["urn"] != urn:
                txt = load_text_from_urn(row["urn"])
                if txt:
                    corpus_texts.append(txt)
                    corpus_labels.append(row["label"])

        full_corpus = [text_target] + corpus_texts

        # --- TF–IDF ---
        vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"[A-Za-zÆØÅæøå]+",
            min_df=1,
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



# ================================================================
# === 3) Konkordanser ===
# ================================================================

st.header("3.Konkordanser")

st.markdown(
    """
Skriv inn et ord (wildcard støttes) og se det i et vindu av kontekst.
Du får en liste med: 

- Tittelen på stykket, navnet på forfatter, årstall og sjangerbenevnelse
- konkret forekomst av søkeordet
- kontekst på et antall ord du selv bestemmer, før og etter søkeordet

"""
)


query = st.text_input("Søk", "", placeholder="Skriv inn søkeuttrykk her")

query = query.strip().lower()
wildcard = query.endswith("*")
query_base = query[:-1] if wildcard else query

window_size = st.slider("Konkordansevindu (antall ord før/etter)", 5, 30, 20)
max_hits = st.slider("Maks antall treff", 10, 1000, 100)

if query and not df_available.empty:

    rows = []
    hit_count = 0

    for _, row in df_available.iterrows():
        urn = row["urn"]
        text = load_text_from_urn(urn)
        tokens = text.split()

        for i, token in enumerate(tokens):
            token_lower = token.lower()

            if (wildcard and token_lower.startswith(query_base)) or (not wildcard and token_lower == query_base):

                start = max(0, i - window_size)
                end = min(len(tokens), i + window_size + 1)

                left = " ".join(tokens[start:i])
                keyword = tokens[i]
                right = " ".join(tokens[i+1:end])

                # full kontekst (fast 100 ord før/etter)
                left_full = " ".join(tokens[max(0, i-100):i])
                right_full = " ".join(tokens[i+1:i+101])

                # gjør treffet tydelig i full kontekst
                full_context = f"{left_full} **{keyword}** {right_full}"

                rows.append({
                    "verk": format_label(row),
                    "left": left,
                    "treff": keyword,
                    "right": right,
                    "full": full_context
                })

                hit_count += 1
                if hit_count >= max_hits:
                    break

        if hit_count >= max_hits:
            break

    if not rows:
        st.info("Ingen treff funnet.")
    else:
        st.caption(f"Viser {len(rows)} treff")

        # Vis “rad” + expander under (enkelt, lesbart)
        for idx, r in enumerate(rows, start=1):

            # Linja: venstre … treff … høyre
            st.markdown(
                f"{r['left']} … **{r['treff']}** … {r['right']}",
            )

            # Klikk for mer kontekst (med verkinfo bare her)
            with st.expander("Se mer kontekst - og info om verket"):
                st.caption(r["verk"])
                st.markdown(r["full"])
