import os
import shutil

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
DATA_DIR = "./data/SNAC-K"  # Folder containing XML files (now includes all subfolders)

# Grok API key (store securely in .env or Streamlit secrets)
try:
    GROQ_API_KEY = st.secrets["GROQ_api_key"]
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_api_key")

# ── Functions ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Preparing embeddings model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@st.cache_resource(show_spinner="Loading or building vector store...")
def process_xmls_to_vectorstore(data_dir: str):
    import glob
    import hashlib
    
    def get_files_hash(data_dir):
        files = []
        json_file = os.path.join(data_dir, "snac-k.json")
        if os.path.exists(json_file):
            files.append(json_file)
        xml_files = glob.glob(os.path.join(data_dir, "*.xml"))  # Remove recursive=True
        files.extend(xml_files)
        
        hasher = hashlib.md5()
        for f in sorted(files):
            with open(f, 'rb') as file:
                hasher.update(file.read())
        return hasher.hexdigest()
    
    current_hash = get_files_hash(data_dir)
    hash_file = os.path.join(CHROMA_DIR, "files_hash.txt")
    
    embeddings = get_embeddings()
    
    # Check if hash matches and try to load existing
    if os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            stored_hash = f.read().strip()
        if stored_hash == current_hash:
            try:
                vectorstore = Chroma(
                    persist_directory=CHROMA_DIR,
                    embedding_function=embeddings,
                    collection_name="xml_metadata"
                )
                if vectorstore._collection.count() > 0:
                    return vectorstore
            except Exception:
                pass  # Proceed to rebuild if loading fails
    
    # Build from scratch if hash changed or loading failed
    xml_files = glob.glob(os.path.join(data_dir, "*.xml"))  # Remove recursive=True
    
    documents = []
    
    # Add the JSON description file
    json_file = os.path.join(data_dir, "snac-k.json")
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            json_text = f.read()
        doc = Document(
            page_content=f"Database Description: {json_text}",
            metadata={"source": json_file, "file": "snac-k.json"}
        )
        documents.append(doc)
    
    if not xml_files and not documents:
        raise FileNotFoundError(f"No XML or JSON files found in {data_dir}")
    
    for file_path in xml_files:
        file_name = os.path.basename(file_path)
        
        text = parse_xml_to_text(file_path)
        
        doc = Document(
            page_content=text,
            metadata={"source": file_path, "file": file_name}
        )
        documents.append(doc)
    
    # Split into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,  # Smaller, since each variable is self-contained
        chunk_overlap=0,  # Small overlap to maintain context
        separators=["Variable:", "\n\n", "\n"]  # Split on Variable first
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings and store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="xml_metadata"
    )
    
    # Save the new hash
    with open(hash_file, 'w') as f:
        f.write(current_hash)
    
    return vectorstore

# ── Main App ──────────────────────────────────────────────────────────────────

st.title("🧠 Semantic RAG Chatbot with XML Metadata")
st.markdown("Ask natural questions about your local XML metadata!")

# Load vector store on startup (pre-built from ./data/)
try:
    vectorstore = process_xmls_to_vectorstore(DATA_DIR)
    st.session_state.vectorstore = vectorstore
    st.success("Vector store loaded from local XMLs! Ready to chat.")
except Exception as e:
    st.error(f"Error loading vector store: {e}. Check your {DATA_DIR} folder and XML files.")
    vectorstore = None
    st.session_state.vectorstore = None

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
            st.warning("Vector store not loaded. Check your data folder!")
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
col_clear, col_reset = st.columns(2)
with col_clear:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

with col_reset:
    if st.button("Reset Index & Chats"):
        st.session_state.messages = []
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)  # Clear persistent vectorstore
        st.rerun()