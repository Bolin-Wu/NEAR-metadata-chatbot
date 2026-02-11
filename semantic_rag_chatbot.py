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

def get_database_description_once(vectorstore):
    """Retrieve and cache database description from vector store."""
    if vectorstore is None:
        return ""
    
    try:
        # Query for database description documents
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        docs = retriever.invoke("cohort study design overview background")
        
        for doc in docs:
            if doc.metadata.get("type") == "database_description":
                return doc.page_content
    except Exception as e:
        st.warning(f"Could not retrieve database description: {e}")
    
    return ""

def filter_and_organize_context(query, vectorstore):
    """Retrieve and organize context - only variable definitions for efficiency."""
    if vectorstore is None:
        return ""
    
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
        docs = retriever.invoke(query)
        
        var_defs = []
        
        for doc in docs:
            if doc.metadata.get("type") == "variable_definitions":
                var_defs.append(doc)
        
        # Return only variable definitions (description cached separately)
        context_text = "\n\n---\n\n".join([doc.page_content for doc in var_defs])
        return context_text
    except Exception as e:
        st.error(f"Error retrieving context: {e}")
        return ""

# ── Main App ──────────────────────────────────────────────────────────────────

st.title("💬 Chatbot with NEAR Metadata")
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

# Cache database description (fetch once)
if "db_description" not in st.session_state:
    st.session_state.db_description = get_database_description_once(vectorstore)

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

                COHORT BACKGROUND:
                {cohort_background}

                ---

                Your task is to answer questions about variables and metadata from this cohort.

                ### VARIABLE DATA:
                {context}

                ### Question from user:
                {question}

                ### Your Response Instructions:
                - Group related variables by theme and by their original tables/sources
                - Start with a clear, natural explanation of the topic based on the related cohort background
                - Use your own words to describe what the variables measure
                - Then, present the variable information in TWO separate markdown tables as specified below:

                TABLE 1 - Variables by Content (what they measure):
                | Raw Variable Name | Label | Categories |
                |---|---|---|
                | variable_name | What it measures in plain English | Category values if applicable |

                TABLE 2 - Variable Availability Across Tables/Sources:
                | Raw Variable Name | Original Table | Source file |
                |---|---|---|
                | variable_name | Table Name | File name |
                
                CRITICAL RULES FOR TABLE 1:
                - Table 1, Column 1: MUST show the raw variable NAME/ID (e.g., löpnr, kön)
                - Table 1, Column 2: MUST show what the variable measures in plain English
                - Table 1, Column 3: MUST show category values (e.g., "1=man, 2=woman" or "N/A" if no categories)
                
                CRITICAL RULES FOR TABLE 2:
                - Table 2, Column 1: Raw Variable NAME/ID
                - Table 2, Column 2: Original Table name (e.g., SNAC-K_c1)
                - Table 2, Column 3: Source file name - EXTRACT from "File: xxx.xml" at the TOP of each chunk
                - For EACH variable, list EVERY source/table/file combination where it appears
                - If "walk" appears in SNAC-K_c1 AND SNAC-K_c2, show BOTH rows:
                | walk | SNAC-K_c1 | SNAC-K_Cohort1_Baseline.xml | 
                | walk | SNAC-K_c2 | SNAC-K_Cohort2_Baseline.xml | 
                - Do NOT consolidate - show each source separately
                - NEVER invent file names - only use files mentioned in the context
                - NEVER invent raw variable names - only use variable names mentioned in the context
                - If source file cannot be determined, use "Unknown" instead of guessing
                
                HOW TO FIND SOURCE FILE:
                - Look for "File: xxx.xml" at the BEGINNING of each chunk
                - This is the ONLY reliable source of file information
                - Extract the filename exactly as shown (e.g., SNAC-K_Cohort1_Baseline.xml)
                
                EXAMPLE OF CORRECT FORMAT:
                "In SNAC-K, several variables measure basic demographics. Participants are identified by a unique proband number (löpnr). The cohort includes both men and women, tracked through a sex variable. Birth dates are recorded to calculate age.


                | Raw Variable Name | Label | Categories |
                |---|---|---|
                | löpnr | Unique participant identifier | N/A (unique ID) |
                | kön | Participant's biological sex | 1=man, 2=woman |
                | Birthday | Date of birth for age calculation | Date format |

                | Raw Variable Name | Original Table | Source File |
                |---|---|---|
                | löpnr | SNAC-K_c1 | SNAC-K_Cohort1_Baseline.xml | 
                | löpnr | SNAC-K_c2 | SNAC-K_Cohort2_Baseline.xml | 
                | kön | SNAC-K_c1 | SNAC-K_Cohort1_Baseline.xml | 
                | kön | SNAC-K_c2 | SNAC-K_Cohort2_Baseline.xml | 

                Answer:"""

                PROMPT = PromptTemplate(
                    template=prompt_template, 
                    input_variables=["cohort_background", "context", "question"]
                )

                # Get context (only variable definitions for efficiency)
                context = filter_and_organize_context(prompt, vectorstore)
                cohort_background = st.session_state.get("db_description", "")

                rag_chain = (
                    {
                        "cohort_background": lambda _: cohort_background,
                        "context": lambda _: context,
                        "question": RunnablePassthrough()
                    }
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

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()