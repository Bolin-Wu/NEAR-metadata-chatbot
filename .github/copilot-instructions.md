# Copilot instructions

## Project overview

- Streamlit-based Excel metadata chatbot with three entrypoints: [app.py](app.py), [demo_chatbot.py](demo_chatbot.py), and the main RAG app [semantic_rag_chatbot.py](semantic_rag_chatbot.py).
- Core flow in [semantic_rag_chatbot.py](semantic_rag_chatbot.py): upload Excel → read all sheets with pandas → stringify sheets into `Document`s → chunk with `RecursiveCharacterTextSplitter` → embed with `HuggingFaceEmbeddings` → persist to Chroma at [chroma_excel_db/](chroma_excel_db/) → retrieve top-k → prompt `ChatGroq` → streamlit chat UI.
- The rule-based demo in [demo_chatbot.py](demo_chatbot.py) uses `st.session_state.df` and a simple keyword router (no LLM).

## Key dependencies & integration points

- LLM: Groq via `ChatGroq` (langchain-groq). API key expected as `GROQ_api_key` in Streamlit secrets or `.env` (loaded via `dotenv`).
- Vector store: Chroma persisted under [chroma_excel_db/](chroma_excel_db/). Embedding model is `sentence-transformers/all-MiniLM-L6-v2`.

## Runtime patterns & conventions

- Session state is the primary state container: `vectorstore`, `file_hash`, and `messages` live in `st.session_state` for the RAG app; the demo app uses `st.session_state.df`.
- File reprocessing is avoided by hashing the uploaded file (`compute_file_hash`) and resetting `vectorstore` only when the hash changes.
- Use `st.cache_resource` for model and vectorstore setup (see `get_embeddings()` and `process_excel_to_vectorstore()`).
- Source transparency: the RAG app shows retrieved chunks in a “Sources used” expander and sanitizes `Document.metadata.source` to a filename (see `safe_source` in [semantic_rag_chatbot.py](semantic_rag_chatbot.py)).
- The file uploader uses a fixed key (`FILE_UPLOADER_KEY`) so reset actions can clear the upload state (see [semantic_rag_chatbot.py](semantic_rag_chatbot.py)).

## Developer workflows

- Use uv for dependency management: `uv sync` (see [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock)).
- Run apps with Streamlit via uv: `uv run streamlit run semantic_rag_chatbot.py` (or `demo_chatbot.py` / `app.py`).

## When editing

- Preserve the Streamlit chat UX: messages are appended to `st.session_state.messages` and rendered with `st.chat_message`.
- Keep Chroma persistence path stable unless explicitly changing storage layout.
