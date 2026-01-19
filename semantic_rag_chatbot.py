import os
import hashlib
import shutil


import streamlit as st
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq  
from langchain_core.prompts import PromptTemplate
import tempfile
from dotenv import load_dotenv
load_dotenv()


# ── Configuration ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Semantic RAG Excel Chatbot", layout="wide")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # fast & good quality
CHROMA_DIR = "./chroma_excel_db"  # persistent folder

# Grok API key (store securely in .env or Streamlit secrets)
try:
    GROQ_API_KEY = st.secrets["GROQ_api_key"]
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_api_key")

# ── Functions ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Preparing embeddings model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@st.cache_resource(show_spinner="Building vector store...")
def process_excel_to_vectorstore(file_path: str):
    # Read all sheets
    excel_data = pd.read_excel(file_path, sheet_name=None)
    
    documents = []
    
    safe_source = "uploaded_excel.xlsx"  # Generic, non-sensitive source name

    for sheet_name, df in excel_data.items():
        # Convert dataframe to text representation (you can customize this)
        text = f"Sheet: {sheet_name}\n\n" + df.to_string(index=False)
        
        # Create LangChain Document
        doc = Document(
            page_content=text,
            metadata={"source": safe_source, "sheet": sheet_name}
        )
        documents.append(doc)
    
    # Split into smaller chunks (important for better retrieval)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings and store
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="excel_metadata"
    )
    
    return vectorstore

# ── Main App ──────────────────────────────────────────────────────────────────
FILE_UPLOADER_KEY = "excel_uploader"

if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0

st.title("🧠 Semantic RAG Chatbot with Excel")
st.markdown("Upload your metadata Excel → ask natural questions!")

uploaded_file = st.file_uploader(
    "Upload Excel file (.xlsx)",
    type=["xlsx"],
    key=f"{FILE_UPLOADER_KEY}_{st.session_state.reset_counter}",
)

## check file hash to avoid reprocessing same file
def compute_file_hash(uploaded_file):
    if uploaded_file is None:
        return None
    hash_md5 = hashlib.md5()
    uploaded_file.seek(0)
    for chunk in iter(lambda: uploaded_file.read(4096), b""):
        hash_md5.update(chunk)
    uploaded_file.seek(0)
    return hash_md5.hexdigest()

file_hash = compute_file_hash(uploaded_file)

if file_hash is not None and (
    "file_hash" not in st.session_state or st.session_state.file_hash != file_hash
):
    st.session_state.vectorstore = None  # Reset vector store
    st.session_state.file_hash = file_hash

if uploaded_file is not None:
    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        if st.session_state.get("vectorstore") is None:
            with st.spinner("Processing Excel → creating semantic index..."):
                st.session_state.vectorstore = process_excel_to_vectorstore(tmp_path)
            st.success("Semantic index ready! You can now ask questions.")
    finally:
        os.unlink(tmp_path)  # Delete temp file after processing
    
    # Optional: show preview
    if st.checkbox("Show data preview"):
        df_preview = pd.read_excel(uploaded_file, nrows=10)
        st.dataframe(df_preview)

# ── Chat ──────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your Excel metadata..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        vectorstore = st.session_state.get("vectorstore")
        if vectorstore is None:
            st.warning("Please upload an Excel file first!")
            response = ""
        else:
            with st.spinner("Thinking..."):
                llm = ChatGroq(
                    groq_api_key=GROQ_API_KEY,
                    model_name="llama-3.1-8b-instant",
                    temperature=0.1,
                )
                prompt_template = """You are an expert in epidemiology and aging research, specializing in cohort study metadata. Use only the provided Excel context from cohort studies to answer questions accurately and precisely. Cite sources (e.g., sheet names, variable names, or specific data points) when possible. Provide insights relevant to epidemiological research, such as study designs, variable definitions, or potential confounders. If the context is insufficient for a complete answer, suggest refining the query or uploading additional data.

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
                        st.caption(f"Sheet: {doc.metadata.get('sheet', 'unknown')} (from uploaded Excel)")
                        st.write(doc.page_content[:300] + "...")

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# Cleanup buttons
col_clear, col_reset = st.columns(2)
with col_clear:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

with col_reset:
    if st.button("Reset Index & Chats"):
        st.session_state.messages = []
        if "vectorstore" in st.session_state:
            del st.session_state.vectorstore
        if "file_hash" in st.session_state:
            del st.session_state.file_hash
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)  # Clear persistent vectorstore
        st.session_state.reset_counter += 1
        st.rerun()