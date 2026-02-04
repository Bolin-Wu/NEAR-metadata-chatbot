import os
import shutil
import glob
import time

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq  
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

from scripts.xml_parser import parse_xml_to_text
load_dotenv()


# ── Configuration ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Semantic RAG XML Metadata Chatbot", layout="wide")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # fast & good quality
CHROMA_DIR = "./chroma_xml_db"  # persistent folder
DATA_ROOT = "./data"  # Root data directory

# Grok API key (store securely in .env or Streamlit secrets)
try:
    GROQ_API_KEY = st.secrets["GROQ_api_key"]
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_api_key")

# ── Functions ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Preparing embeddings model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def load_vectorstore():
    """Load existing vector store from disk."""
    if os.path.exists(CHROMA_DIR):
        try:
            embeddings = get_embeddings()
            vectorstore = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embeddings,
                collection_name="xml_metadata"
            )
            if vectorstore._collection.count() > 0:
                return vectorstore
        except Exception as e:
            st.error(f"Error loading vector store: {e}")
    
    return None

# ── Main App ──────────────────────────────────────────────────────────────────

st.title("🧠 Semantic RAG Chatbot with XML Metadata")
st.markdown("Ask natural questions about NEAR metadata!")

# Try to load existing vector store
vectorstore = load_vectorstore()

if vectorstore is None:
    st.warning("⚠️ No knowledge database found!")
    st.info("""
    To train a vector store, run the training script:
    
    ```bash
    python scripts/train_vectorstore.py
    ```
    
    This will allow you to select which database to index.
    """)
    st.session_state.vectorstore = None
else:
    st.session_state.vectorstore = vectorstore
    st.success("✓ Vector store loaded! Ready to chat.")

# ── Chat ──────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your XML metadata..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        vectorstore = st.session_state.get("vectorstore")
        if vectorstore is None:
            st.warning("Vector store not available. Please run the training script first!")
            response = ""
        else:
            with st.spinner("Thinking..."):
                llm = ChatGroq(
                    groq_api_key=GROQ_API_KEY,
                    model_name="llama-3.1-8b-instant",
                    temperature=0.1,
                )
                prompt_template = """You are an expert in epidemiology and aging research, specializing in cohort study metadata. Use only the provided XML context from cohort studies to answer questions accurately and precisely. Cite sources (e.g., file names, variable names, or specific data points) when possible. Provide insights relevant to epidemiological research, such as study designs, variable definitions, or potential confounders. If the context is insufficient for a complete answer, suggest refining the query or uploading additional data.

                Context: {context}

                Question: {question}

                Answer:"""

                PROMPT = PromptTemplate(
                    template=prompt_template, input_variables=["context", "question"]
                )

                retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

                rag_chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | PROMPT
                    | llm
                    | StrOutputParser()
                )

                response = rag_chain.invoke(prompt)

                with st.expander("Sources used"):
                    for doc in retriever.invoke(prompt):
                        st.caption(f"File: {doc.metadata.get('file', 'unknown')} (from {doc.metadata.get('source', 'unknown')})")
                        st.write(doc.page_content[:300] + "...")

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# Cleanup buttons
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()