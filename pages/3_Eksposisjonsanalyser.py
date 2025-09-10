import streamlit as st
import pandas as pd
import openai
import pyarrow
import altair as alt

st.markdown(
    """
    <script>
        window.parent.document.querySelector('section.main').scrollTo(0,0);
    </script>
    """,
    unsafe_allow_html=True
)

st.set_page_config(page_title = "Teatersjangerstudier - AI-analyser", page_icon ="🎭", layout = "wide")
st.title("Analyser av dramaenes 2000 første tokens (ideelt eksposisjonen) via API-et til Open AI")
st.markdown("")
st.markdown(
    """
De to tusen første tegn er hentet ut fra hvert teaterstykke med tilgjengelig fulltekst fra Nasjonalbiblioteket. Disse er sendt til en stor språkmodell (LLM), som har returnert det som ideelt skal være en eksposisjonsanalysene.   

Du kan se resultater av AI-analysene - form, tone og konflikttype - fordelt på sjanger, og du kan inspisere enkeltverk og se flere detaljer fra den enkelte automatiske analysen. 

Snart vil du også kunne sammenligne deler av de automatiske analysene per sjanger ved hjelp av samme språkmodell. Scroller du ned, kan du du teste grensesnittet – men AI-en er foreløpig ikke koblet til. Se mer info i sidebaren.

"""
)


# === Last metadata ===
meta_df = pd.read_parquet("expositions_flat.parquet")


conflicts_long = pd.read_parquet("conflicts_long.parquet")
themes_long = pd.read_parquet("themes_long.parquet")
staging_cues_long = pd.read_parquet("staging_cues_long.parquet")
style_notes_long = pd.read_parquet("style_notes_long.parquet")
notable_motifs_long = pd.read_parquet("notable_motifs_long.parquet")
characters_long = pd.read_parquet("characters_long.parquet")
evidence_long = pd.read_parquet("evidence_long.parquet")
uncertainties_long = pd.read_parquet("uncertainties_long.parquet")



# --- Sidebar ---
st.sidebar.header("Mer om denne siden")

st.sidebar.markdown(
    """
    De to tusen første tokens er hentet ut fra hvert teaterstykke med tilgjengelig fulltekst fra Nasjonalbiblioteket.  

    Disse er sendt til OpenAI sitt API (modell: GPT-5), som har returnert et JSON-objekt per verk (ideelt eksposisjonsanalysene).

    👉 Se prompten brukt i analysen nederst på siden.

    """
)

st.sidebar.header("Filtrering av eksposisjoner for AI-analyser")

with st.sidebar.expander("Mer info"):
    st.markdown(
        """
        Snart vil du kunne få gjennomført AI-analyser av AI-analysene.  

        Velg da sjanger i sidebaren eller la eventuelt **'Alle'** stå velg da sjanger i sidebaren eller la eventuelt **'Alle'** stå om du ønsker å analysere eksposisjonene på tvers av sjanger. 

        Definer eventuelt andre kritierer eller la det stå urørt for å søke i tilfeldig utdrag/alle tekster eller alle/tilfeldig utdrag innen valgt sjanger.
        """
    )


# Sjangervalg -  med minst én URN med fulltekst

# Tell sjangre med minst én URN med fulltekst
genre_counts = (
    meta_df[meta_df["urn"].notna() & meta_df["urn"].str.startswith("URN")]
    .groupby("genre")["urn"]
    .count()
    .to_dict()
)

# Lag meny med "Sjanger (antall)"
all_genres = ["Alle"] + sorted([f"{genre} ({count})" for genre, count in genre_counts.items()])

# Multiselect (brukeren kan velge én, flere eller "Alle")
selected_genres = st.sidebar.multiselect(
    "Velg én eller flere sjangre",
    options=all_genres,
    default=["Alle"]
)


#Andre filtre
year_from = st.sidebar.number_input("Fra år", min_value=1800, max_value=1899, value=1800)
year_to = st.sidebar.number_input("Til år", min_value=1800, max_value=1899, value=1899)



# --- Filtrér metadata ---
filtered = meta_df[
    (meta_df["urn"].notna()) &
    (meta_df["urn"].str.startswith("URN")) &
    (meta_df["year"].between(year_from, year_to))
]

# Sjangre (multiselect)

if "Alle" not in selected_genres:
    genre_names = [g.split(" (")[0] for g in selected_genres]  # fjern antall i parentes
    filtered = filtered[filtered["genre"].str.strip().isin(genre_names)]


# Først filtrer på sjanger (men uten år og forfatter ennå)
if "Alle" not in selected_genres:
    genre_names = [g.split(" (")[0] for g in selected_genres]
    genre_filtered_df = meta_df[meta_df["genre"].str.strip().isin(genre_names)]
else:
    genre_filtered_df = meta_df


# Så hent bare forfattere som finnes i det sjangerfiltrerte datasettet

genre_authors = sorted(genre_filtered_df["author"].dropna().unique())
genre_authors.insert(0, "Alle")
selected_author = st.sidebar.selectbox("Velg forfatter", genre_authors)


#Selve analysene

st.markdown("<h2 style='text-align: center;'>Tone og form</h2>", unsafe_allow_html=True)
st.write('')

if len(filtered) > 0:
    # --- Tone ---
    tone_counts = (
        filtered.groupby(["genre", "tone"])["urn"]
        .count()
        .reset_index()
        .rename(columns={"urn": "count"})
    )
    tone_counts["percent"] = tone_counts.groupby("genre")["count"].transform(lambda x: x / x.sum() * 100)

    # Totalt antall verk per sjanger
    genre_totals = (
        tone_counts.groupby("genre")["count"]
        .sum()
        .reset_index()
        .rename(columns={"count": "total"})
    )
    genre_totals["genre_label"] = "(" + genre_totals["total"].astype(str) + ") " + genre_totals["genre"]

    # Slå sammen totalsum og etikett inn i tone_counts
    tone_counts = tone_counts.merge(genre_totals, on="genre")

    # Sortering basert på totalsum
    genre_order = genre_totals.sort_values("total", ascending=False)["genre_label"].tolist()

    tone_chart = (
        alt.Chart(tone_counts)
        .mark_bar()
        .encode(
            y=alt.Y("genre_label:N", sort=genre_order, title=""),
            x=alt.X("percent:Q", stack="normalize", title="Andel (%)"),
            color=alt.Color("tone:N", title="Tone"),
            tooltip=["genre", "tone", "count", alt.Tooltip("percent:Q", format=".1f"),alt.Tooltip("total:Q", title="Totalt verk i sjanger")]
        )
        .properties(width=350, height=1000, title="Tone (fordeling per sjanger)")
    )

    # --- Form ---
    form_counts = (
        filtered.groupby(["genre", "form"])["urn"]
        .count()
        .reset_index()
        .rename(columns={"urn": "count"})
    )
    form_counts["percent"] = form_counts.groupby("genre")["count"].transform(lambda x: x / x.sum() * 100)

    # Legg til totalsum og ny etikett
    form_counts = form_counts.merge(genre_totals, on="genre")
    form_counts["genre_label"] = "(" + form_counts["total"].astype(str) + ") " + form_counts["genre"]

    form_chart = (
        alt.Chart(form_counts)
        .mark_bar()
        .encode(
            y=alt.Y("genre_label:N", sort=genre_order, title=""),
            x=alt.X("percent:Q", stack="normalize", title="Andel (%)"),
            color=alt.Color("form:N", title="Form"),
            tooltip=["genre", "form", "count", alt.Tooltip("percent:Q", format=".1f"),alt.Tooltip("total:Q", title="Totalt verk i sjanger")]
        )
        .properties(width=350, height=1000, title="Form (fordeling per sjanger)")
    )

    # Legg i to kolonner
    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(tone_chart, use_container_width=True)
    with col2:
        st.altair_chart(form_chart, use_container_width=True)

else:
    st.info("Ingen verk funnet for det valgte utvalget.")



st.markdown("<h2 style='text-align: center;'>Konflikttyper og utforsking av enkeltverk</h2>", unsafe_allow_html=True)
st.write('')

if len(filtered) > 0:
    selected_urns = filtered["urn"].unique().tolist()

    # --- Konflikttyper ---
    conflicts_with_genre = conflicts_long.merge(meta_df[["urn", "genre"]], on="urn")
    conflict_counts = (
        conflicts_with_genre[conflicts_with_genre["urn"].isin(selected_urns)]
        .groupby(["genre", "conflict_type"])["urn"]
        .nunique()
        .reset_index()
        .rename(columns={"urn": "count"})
    )

    # Andeler per sjanger
    conflict_counts["percent"] = conflict_counts.groupby("genre")["count"].transform(
        lambda x: x / x.sum() * 100
    )

    # Totalt antall verk per sjanger
    genre_totals = (
        conflict_counts.groupby("genre")["count"]
        .sum()
        .reset_index()
        .rename(columns={"count": "total"})
    )
    genre_totals["genre_label"] = "(" + genre_totals["total"].astype(str) + ") " + genre_totals["genre"]

    conflict_counts = conflict_counts.merge(genre_totals, on="genre")
    genre_order = genre_totals.sort_values("total", ascending=False)["genre_label"].tolist()

    conflict_chart = (
        alt.Chart(conflict_counts)
        .mark_bar()
        .encode(
            y=alt.Y("genre_label:N", sort=genre_order, title=""),
            x=alt.X("percent:Q", stack="normalize", title="Andel (%)"),
            color=alt.Color("conflict_type:N", title="Konflikttype"),
            tooltip=[
                "genre",
                "conflict_type",
                "count",
                alt.Tooltip("percent:Q", format=".1f"),
                alt.Tooltip("total:Q", title="Totalt verk i sjanger"),
            ],
        )
        .properties(width=350, height=1000, title="Konflikttyper (fordeling per sjanger)")
    )


    # To kolonner
    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(conflict_chart, use_container_width=True)


    with col2:

        # Velg sjanger først (fra de filtrerte verkene i utvalget)
        available_genres = filtered["genre"].dropna().unique().tolist()
        if available_genres:
            genre_choice = st.selectbox("Velg en sjanger", ["Velg en sjanger"] + sorted(available_genres), key="detail_genre", label_visibility="collapsed")

            if genre_choice != "Velg en sjanger":

                # Filtrer verk i valgt sjanger
                works_in_genre = filtered[filtered["genre"] == genre_choice].copy()
                if not works_in_genre.empty:
                    works_in_genre["label"] = (
                        works_in_genre["title"].fillna("Uten tittel") + " – " +
                        works_in_genre["author"].fillna("Ukjent forfatter") + " (" +
                        works_in_genre["year"].astype(str) + ")"
                    )

                    

                    # Velg enkeltverk
                    work_choice = st.selectbox("Velg et verk", ["Velg et verk"] + works_in_genre["label"].tolist(),key="detail_work", label_visibility="collapsed")

                    if work_choice != "Velg et verk":
                

                        work_row = works_in_genre[works_in_genre["label"] == work_choice].iloc[0]

                        # Vis detaljprofil
                        with st.expander("Vis eksposisjonsanalyse", expanded=False):
                            st.markdown(f"**Brief:** {work_row['brief']}")
                            st.markdown(f"**Tone:** {work_row['tone']}  \n**Form:** {work_row['form']}")
                            st.markdown(f"**Conflict type:** {work_row['conflict_types']}  \n**Conflict incipient:** {work_row.get('conflict_incip', '–')}")
                            st.markdown(f"**Setting:** {work_row['setting_time']}, {work_row['setting_place']} ({work_row['social_milieu']})")

                            if pd.notna(work_row["themes"]) and work_row["themes"]:
                                st.markdown("**Themes:**")
                                st.markdown("\n".join([f"- {t}" for t in str(work_row['themes']).split('; ')]))

                            if pd.notna(work_row["notable_motifs"]) and work_row["notable_motifs"]:
                                st.markdown("**Notable motifs:**")
                                st.markdown("\n".join([f"- {m}" for m in str(work_row['notable_motifs']).split('; ')]))

                            if pd.notna(work_row["style_notes"]) and work_row["style_notes"]:
                                st.markdown("**Style notes:**")
                                st.markdown("\n".join([f"- {s}" for s in str(work_row['style_notes']).split('; ')]))
                    else:
                        st.info("Velg et verk for å se detaljert eksposisjonsanalyse")
                else:
                    st.info("Ingen sjangre tilgjengelige for valgt utvalg.")
            else: 
                st.info("Velg en sjanger for å bla i verkene.")

else:
    st.info("Ingen sjangre funnet for det valgte utvalget.")


# Chat-grensesnitt i expander
with st.expander("💬 Still et spørsmål til valgt/e korpus/er"):
    #openai.api_key = st.secrets["openai"]["api_key"]

    if prompt := st.chat_input("Skriv spørsmålet ditt her..."):
        #response = openai.ChatCompletion.create(
        #    model="gpt-4",
        #    messages=[
        #        {"role": "system", "content": "Du er en teater- og litteraturforsker som analyserer norske teaterstykker fra 1800-tallet."},
        #        {"role": "user", "content": prompt}
        #    ]
        #)
        #st.chat_message("assistant").markdown(response["choices"][0]["message"]["content"])
        st.chat_message("user").markdown(prompt)

        st.chat_message("assistant").markdown(
            "**Smør deg med litt tålmodighet 🧈⏳**\n\n"
            "Tjenesten er fortsatt under utvikling, og AI-svaret kommer snart! "
            "Foreløpig brukes ingen språkmodell. Når dette er aktivert, vil spørsmålet ditt bli sendt sammen med utdrag fra korpuset."
        )


