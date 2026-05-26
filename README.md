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

## Compare Embedding Models

Use the semantic comparison script to test whether switching from
`sentence-transformers/all-MiniLM-L6-v2` to `text-embedding-3-small` is worth it
for aging/cohort metadata retrieval.

### Aging-focused benchmark (recommended)

```bash
uv run python scripts/evaluation/compare_embedding_models.py \
   --queries-file scripts/evaluation/aging_semantic_queries.json \
   --k 3
```

This benchmark uses aging-research queries in
`scripts/evaluation/aging_semantic_queries.json` and prints side-by-side
search results from both vector stores (top 3 per model).

### One-query quick check

```bash
uv run python scripts/evaluation/compare_embedding_models.py \
   --query "What variables measure depressive symptoms in older adults?" \
   --database SNAC-K \
   --k 3
```

Defaults:

- Model A: `sentence-transformers/all-MiniLM-L6-v2`
- Model B: `text-embedding-3-small`
