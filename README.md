# About
This project is a `Streamlit` application of a local RAG system, whose task is to answer questions about attached `.pdf` documents. 

![interface photo](https://raw.githubusercontent.com/haritonn/pdf_rag/main/assets/interface.png)

Core features:
- `Streamlit` simple & cool interface;
- Ability to switch backbone LLM, db name & even embedder model directly through interface;
- Ability to select Top-K best sources for model;
- Easily scalable, everything is implemented through abstract classes;
- Working with popular `Ollama` framework;
- Ability to check sources and concrete contexts which LLM found helpful for answering;
- Simple launching.

## Installation
```sh
git clone git@github.com:haritonn/pdf_rag.git
cd pdf_rag/
```

## Launching
Since it local make sure that previous copy of `qdrant_db/` folder doesn't exist. Then:
```sh
uv run streamlit run app.py
```
