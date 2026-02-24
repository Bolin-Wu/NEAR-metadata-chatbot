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

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from database_utils import get_available_databases

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NEAR Metadata Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# Add logo to sidebar
with st.sidebar:
    logo_path = Path("logo/NEAR-chatbot.jpg")
    if logo_path.exists():
        st.image(str(logo_path), width=150)
    st.markdown("---")
    
    # Display available databases for future filtering
    st.subheader("💡 Available Databases")
    available_dbs = get_available_databases()
    if available_dbs:
        st.caption("(Filtering by database coming soon)")
        for db in available_dbs:
            st.caption(f"✓ {db}")
    else:
        st.warning("No databases found in ./data")
    
    st.markdown("---")

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

def get_database_description(query, vectorstore):
    """Retrieve database description from vector store based on user query."""
    if vectorstore is None:
        return ""
    
    try:
        # Query for database description documents relevant to user's question
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        docs = retriever.invoke(query)
        
        for doc in docs:
            if doc.metadata.get("type") == "database_description":
                return doc.page_content
    except Exception as e:
        st.warning(f"Could not retrieve database description: {e}")
    
    return ""

def is_variable_question(question):
    """Determine if the question is about specific variables or general cohort info."""
    variable_keywords = [
        "variable", "field", "column", "measure", "data", "table", 
        "available in", "which variables", "what variables", "what data",
        "categories", "values", "definition", "how to find",
        "named", "called", "list of", "all variables", "different variables",
        "raw variable name", "labels", "source"
    ]
    
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in variable_keywords)

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

st.title("💬 NEAR Metadata Chatbot")
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
                
                # Determine question type
                is_var_question = is_variable_question(prompt)
                # Retrieve relevant cohort background based on user's prompt
                cohort_background = get_database_description(prompt, vectorstore)
                
                if is_var_question:
                    # Variable-specific prompt with tables
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
                    - Group related variables by theme
                    - Start with a clear, natural explanation of the topic based on the related cohort background
                    - Use your own words to describe what the variables measure
                    - Then, present the variable information in a markdown table as specified below:

                    TABLE - Variables by Content (what they measure):
                    | Raw Variable Name | Label | Categories |
                    |---|---|---|
                    | variable_name | What it measures in plain English | Category values if applicable |
                    
                    CRITICAL RULES:
                    - Column 1: MUST show the raw variable NAME/ID (e.g., löpnr, kön)
                    - Column 2: MUST show what the variable measures in plain English
                    - Column 3: MUST show category values (e.g., "1=man, 2=woman" or "N/A" if no categories)
                    - NEVER invent variable names - only use variable names mentioned in the context
                    
                    IMPORTANT NOTE ON VARIABLE AVAILABILITY:
                    For information about which cohorts and tables contain these variables, please refer to the Maelstrom catalogue at: https://www.maelstrom-research.org/
                    
                    EXAMPLE OF CORRECT FORMAT:
                    "In SNAC-K, several variables measure basic demographics. Participants are identified by a unique proband number (löpnr). The cohort includes both men and women, tracked through a sex variable. Birth dates are recorded to calculate age.

                    | Raw Variable Name | Label | Categories |
                    |---|---|---|
                    | löpnr | Unique participant identifier | N/A (unique ID) |
                    | kön | Participant's biological sex | 1=man, 2=woman |
                    | Birthday | Date of birth for age calculation | Date format |
                    
                    For detailed information on where these variables are available across different cohort studies, please check the Maelstrom catalogue."

                    Answer:"""

                    PROMPT = PromptTemplate(
                        template=prompt_template, 
                        input_variables=["cohort_background", "context", "question"]
                    )

                    # Get context (only variable definitions for efficiency)
                    context = filter_and_organize_context(prompt, vectorstore)

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
                else:
                    # General cohort question - no variables table
                    prompt_template = """You are an expert in epidemiology and aging research, specializing in cohort study metadata.

                    ### COHORT BACKGROUND:
                    {cohort_background}

                    ### Question from user:
                    {question}

                    ### Your Response Instructions:
                    - Answer the user's question about the cohort using the background information provided
                    - Be clear and concise, using plain language
                    - Reference specific cohort names and details when relevant
                    - Do NOT include variable tables - only provide narrative explanations

                    Answer:"""

                    PROMPT = PromptTemplate(
                        template=prompt_template, 
                        input_variables=["cohort_background", "question"]
                    )

                    rag_chain = (
                        {
                            "cohort_background": lambda _: cohort_background,
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