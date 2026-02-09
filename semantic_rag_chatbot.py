import os
import sys
from pathlib import Path

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq  
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Semantic RAG XML Metadata Chatbot", layout="wide")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DEMO_DB = "./chroma_demo_db"      # Small demo (in GitHub)
CHROMA_PROD_DB = "./chroma_prod_db"      # Production (cloud storage)
DATA_ROOT = "./data"

# Safe way to get deployment environment (default to "demo")
try:
    DEPLOYMENT_ENV = st.secrets["DEPLOYMENT_ENV"]
except (FileNotFoundError, KeyError, AttributeError):
    DEPLOYMENT_ENV = "demo"

# Select which database to use
if DEPLOYMENT_ENV.lower() == "production":
    ACTIVE_CHROMA_DIR = CHROMA_PROD_DB
else:
    ACTIVE_CHROMA_DIR = CHROMA_DEMO_DB

# Safe way to get API key
try:
    GROQ_API_KEY = st.secrets["GROQ_api_key"]
except (FileNotFoundError, KeyError, AttributeError):
    GROQ_API_KEY = os.getenv("GROQ_api_key")

# ── Functions ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Preparing embeddings model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def load_vectorstore():
    """Load vector store from disk (demo or production)."""
    if os.path.exists(ACTIVE_CHROMA_DIR):
        try:
            embeddings = get_embeddings()
            vectorstore = Chroma(
                persist_directory=ACTIVE_CHROMA_DIR,
                embedding_function=embeddings,
                collection_name="xml_metadata"
            )
            if vectorstore._collection.count() > 0:
                return vectorstore
        except Exception as e:
            st.error(f"Error loading vector store: {e}")
    
    return None

# ── Main App ──────────────────────────────────────────────────────────────────

st.title("💬 Semantic RAG Chatbot with Metadata")
st.markdown(f"Ask natural questions about NEAR metadata! (Environment: **{DEPLOYMENT_ENV.upper()}**)")

# Display which database is being used
if DEPLOYMENT_ENV.lower() == "production":
    st.info("🛠️ Using **Production** vector database")
else:
    st.info("🖥️ Using **Demo** vector database")

if "vectorstore_initialized" not in st.session_state:
    vectorstore = load_vectorstore()
    st.session_state.vectorstore = vectorstore
    st.session_state.vectorstore_initialized = True
else:
    vectorstore = st.session_state.vectorstore

if vectorstore is None:
    st.error(f"❌ Vector database not found at {ACTIVE_CHROMA_DIR}")
    st.stop()
else:
    st.success(f"✓ Vector store loaded with {vectorstore._collection.count()} items. Ready to chat!")

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
            st.warning("Vector store not available!")
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

                retriever = vectorstore.as_retriever(search_kwargs={"k": 10})  

                rag_chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | PROMPT
                    | llm
                    | StrOutputParser()
                )

                try:
                    response = rag_chain.invoke(prompt)
                except Exception as e:
                    if "rate_limit" in str(e).lower() or "429" in str(e):
                        st.error("⚠️ Rate limit reached. Please try again in a few moments.")
                        response = "I'm temporarily unavailable due to high usage. Please try again shortly."
                    else:
                        st.error(f"Error: {str(e)}")
                        response = ""

                with st.expander("Sources used"):
                    for doc in retriever.invoke(prompt):
                        st.caption(f"File: {doc.metadata.get('source', 'unknown')}")
                        st.write(doc.page_content[:300] + "...")

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()