import os
import shutil
import glob
import time

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq  
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

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

st.title("🧠 Semantic RAG Chatbot with Metadata")
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

if prompt := st.chat_input("Ask about your metadata..."):
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
                prompt_template = """You are an expert in epidemiology and aging research, specializing in cohort study metadata.

                Your task is to answer questions about variables and metadata from cohort studies using ONLY the provided context.

                CRITICAL INSTRUCTIONS - FOLLOW THESE STRICTLY:
                1. NEVER copy or paste raw variable definitions, metadata blocks, or XML data into your answer
                2. ALWAYS extract the key information and rewrite it in your own narrative sentences
                3. When listing related variables, present them in a clear table format (see below)
                4. Include variable names and their labels
                5. Include categories when available
                6. Group related variables by theme or category
                7. Cite cohort/wave/table information when relevant
                8. If context is insufficient, clearly state that and suggest refining the query

                START YOUR ANSWER DIRECTLY WITH YOUR NARRATIVE - DO NOT INCLUDE ANY RAW DATA AT THE BEGINNING.

                TABLE FORMAT FOR VARIABLES:
                If there are multiple related variables, use this markdown table format:

                | Variable Name | Label | Categories |
                |---|---|---|
                | SN3B15_5 | Choice of transportation to/from work | Motorcycle/Moped, Car, Bus, Train, Bicycle, Walking |
                | SN3B15_6 | Frequency of using transportation | Daily, Several times per week, Weekly |

                Example of CORRECT full answer:
                "In the SNAC-K cohort, several variables relate to mobility and transportation. These variables assess different aspects of how participants move and travel.

                | Variable Name | Label | Categories |
                |---|---|---|
                | SN3B15_5 | Choice of transportation to/from work | Motorcycle/Moped, Car, Bus, Train, Bicycle, Walking |
                | SN3B15_6 | Frequency of using transportation | Daily, Several times per week, Weekly |

                These variables are part of the Lifestyle Behaviours domain and are found in the SNAC-K_wave4_2010_c3 table. They help researchers understand..."

                Context from database:
                {context}

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
                        st.caption(f"File: {doc.metadata.get('file', 'unknown')}")
                        st.write(doc.page_content[:300] + "...")

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# Cleanup buttons
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()