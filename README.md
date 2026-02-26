# NEAR metadata chatbot

Streamlit-based Excel metadata chatbot with a semantic RAG flow in
[semantic_rag_chatbot.py](semantic_rag_chatbot.py).

## Background

This chatbot makes it easier for users to look up NEAR metadata by providing an intuitive interface to search and retrieve information from complex metadata datasets. Instead of navigating through [NEAR Maelstrom catalogue](https://www.maelstrom-research.org/search#lists?type=studies&query=network(in(Mica_network.id,near)),variable(limit(0,20)),study(in(Mica_study.className,Study),limit(0,20))), users can ask natural language questions and get relevant answers powered by semantic search and retrieval-augmented generation.

## Setup (uv)

- Create the environment and install deps from the lockfile:
  - `uv sync`

## Run

- Main RAG app:
  - `uv run streamlit run semantic_rag_chatbot.py`

## Demo

- Try the demo at: [near-chatbot.streamlit.app/](https://near-chatbot.streamlit.app/)

## Configuration

- Set `GROQ_api_key` in `.env` or Streamlit secrets.
