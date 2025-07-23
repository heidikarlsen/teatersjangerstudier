# 🎭 Teatersjangerstudier: A Streamlit multipage app for analyzing 19th-century Norwegian drama

This is a Streamlit web application for exploring Norwegian drama from 1800 to 1899. The app links **genre metadata** to full-text drama documents made available by the **National Library of Norway**.

It allows users to:

- Select one or two genres and view and compare frequent words (frequency analysis). Filter based on publication year available. 
- Search for terms in the full-text corpus using concordance analysis (KWIC). Filter based on author and publication year available. 

The data combines manually curated metadata by professor Ellen Rees and postdoctoral fellow Heidi Leclaire-Karlsen, both at the University of Oslo, with document identifiers (`URN`s) that point to full-text materials in the digital collections of the Norwegian National Library.


The app and associated data are shared under a CC BY 4.0 license.
Citation suggestion: Teatersjangerstudier (2025). A Streamlit app for analyzing 19th-century Norwegian drama.


### Link to the app: 

The app is hosted on Streamlit Community Cloud, where it can be accessed via this public link: 
https://teatersjangerstudier.streamlit.app/


### How to run locally

If you want to run this app locally: 

```bash
pip install -r requirements.txt
streamlit run Hjem.py
```

### License 
This app is shared under a [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).  
Please credit the creator if you use or publish results based on this app.