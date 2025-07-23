import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 


st.set_page_config(page_title = "Teatersjangerstudier", page_icon ="🎭", layout = "wide")

st.title("Teatersjangerstudier")


st.markdown("### :blue[Dette er en app for å utforske teatersjangre på 1800-tallet. Vi benytter et metadata-excel-ark vi har laget med sjangerspesifikasjoner og kobler dette med tekstutvinningsmetoder på de tekstene i korpuset vi har tilgjengelig i fulltekst.]")

st.markdown("#### Fulltekstversjonene er fra Nasjonalbiblioteket. Vi bruker deres ressurser for korpusbygging og digital tekstanalyse.")

st.markdown("#### [ImaginaNation-korpuset](https://doi.org/10.18261/edda.111.3.3) (Skare-Malvik et al. 2024) har vært til stor hjelp for å bygge korpuset i metadata-excel-arket.")


st.divider()


df = pd.read_excel("Sjangre_kategorisert_010625.xlsx")


# Les data
df = pd.read_excel("Sjangre_kategorisert_010625.xlsx")


# Antall totalt og med fulltekst
total_verk = df.shape[0]
verk_med_urn = df["urn"].notna().sum()
unike_forfattere = df["author"].nunique()
unike_sjangre = df["genre"].nunique()


st.markdown("## Informasjon om korpuset")
st.markdown(f"Korpuset vårt består av {total_verk} verk, "
            f"{unike_forfattere} unike forfattere og "
            f"{unike_sjangre} unike sjangerbenevnelser.")
st.markdown(f"Av disse er {verk_med_urn} digitalt tilgjengelige i fulltekst. "
            "I grafen under er andelen som **ikke har fulltekst** markert i lysere farge.")



# Beregner for hver sjanger: total og med fulltekst
sjanger_total = df["genre"].value_counts()
sjanger_med_urn = df[df["urn"].notna()]["genre"].value_counts()


# Setter sammen i dataramme
sjanger_df = pd.DataFrame({
    "Total": sjanger_total,
    "Med fulltekst": sjanger_med_urn
}).fillna(0).astype(int).sort_values("Total", ascending=True)



# Plot
fig, ax = plt.subplots(figsize=(10, 8))

# Lysere bakgrunn: total
bars_total = ax.barh(sjanger_df.index, sjanger_df["Total"], color="#a6bddb", label="Totalt (inkl. uten URN)")

# Mørkere foran: kun med fulltekst
bars_fulltext = ax.barh(sjanger_df.index, sjanger_df["Med fulltekst"], color="#0570b0", label="Digitalt tilgjengelig")

# Tall bak hver stolpe
for i, (total, digital) in enumerate(zip(sjanger_df["Total"], sjanger_df["Med fulltekst"])):
    ax.text(total + 1, i, f"{digital}/{total}", va="center")


ax.set_title("Sjangerbetegnelser (1800–1899)")
ax.set_xlabel("Antall verk")
ax.set_ylabel("Sjanger")
ax.grid(axis='x')
ax.legend()

plt.tight_layout()
st.pyplot(fig)




st.markdown("## Om denne appen")
st.markdown("Per nå er det mulig å sammenlikne frekvenser mellom ulike sjangre og søke etter konkordanser.")
st.markdown("Velg enten **frekvenser** eller **konkordanser** i sidebaren til venstre")