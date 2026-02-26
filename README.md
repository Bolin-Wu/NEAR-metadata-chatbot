# NEAR metadata chatbot

A Streamlit-based metadata chatbot powered by semantic search and retrieval-augmented generation (RAG) for easy NEAR metadata lookup.

## About

This chatbot makes it easier for users to look up NEAR metadata by providing an intuitive interface to search and retrieve information from complex metadata datasets. Instead of navigating through the [NEAR Maelstrom catalogue](<https://www.maelstrom-research.org/search#lists?type=studies&query=network(in(Mica_network.id,near)),variable(limit(0,20)),study(in(Mica_study.className,Study),limit(0,20))>), users can ask natural language questions and get relevant answers powered by semantic search and retrieval-augmented generation.

## Quick Start

**Try the app:** [near-chatbot.streamlit.app/](https://near-chatbot.streamlit.app/)

Simply visit the link and start asking questions about NEAR metadata. No installation required.

## Local Setup (for developers)

To run the app locally:

1. Create the environment and install dependencies:

   ```bash
   uv sync
   ```

2. Start the app:
   ```bash
   uv run streamlit run semantic_rag_chatbot.py
   ```

## Configuration

To use the app, set your `GROQ_api_key` in `.env` or Streamlit secrets.

For implementation details, see [semantic_rag_chatbot.py](semantic_rag_chatbot.py).
