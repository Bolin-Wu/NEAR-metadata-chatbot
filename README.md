# NEAR metadata chatbot

Streamlit-based Excel metadata chatbot with a semantic RAG flow in
[semantic_rag_chatbot.py](semantic_rag_chatbot.py).

## Setup (uv)

- Create the environment and install deps from the lockfile:
  - `uv sync`

## Run

- Main RAG app:
  - `uv run streamlit run semantic_rag_chatbot.py`

## Configuration

- Set `GROQ_api_key` in `.env` or Streamlit secrets.
