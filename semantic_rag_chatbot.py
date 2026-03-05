import os
import logging
from pathlib import Path
import tempfile
import re
from io import BytesIO

import streamlit as st
import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq  
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import shutil
import zipfile

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NEAR Metadata Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB = "./chroma_prod_db"          # Production database (cloud storage)
DATA_ROOT = "./data"

# Known databases (hardcoded as fallback when data/ folder not available)
KNOWN_DATABASES = [
    "Betula", "GAS_SNAC_S", "GENDER", "H70", "KP", 
    "OCTO-Twin", "SATSA", "SNAC-B", "SNAC-K", "SNAC-N", "SWEOLD"
]

# Safe way to get API key
try:
    GROQ_API_KEY = st.secrets["GROQ_api_key"]
except (FileNotFoundError, KeyError, AttributeError):
    GROQ_API_KEY = os.getenv("GROQ_api_key")

# Cloud storage URL for production vector database (optional)
# Use HuggingFace Hub: huggingface_hub.download() with repo_id
try:
    HUGGINGFACE_REPO_ID = st.secrets.get("HUGGINGFACE_REPO_ID")
except:
    HUGGINGFACE_REPO_ID = os.getenv("HUGGINGFACE_REPO_ID")

# ── Functions ─────────────────────────────────────────────────────────────────

def extract_markdown_tables(text):
    """Extract markdown tables from text, handling multiple tables with headers.
    
    Args:
        text: String containing markdown tables
    
    Returns:
        List of tuples: (table_header_text, DataFrame) for each table found
    """
    tables_with_headers = []
    
    # Split by table blocks - look for patterns of | ... |
    # Markdown tables are identified by lines starting with |
    lines = text.split('\n')
    current_table = []
    last_header = None
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for a header line - text that comes before a table
        if line and not line.startswith('|') and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            # Check if next line starts a table
            if next_line.startswith('|'):
                last_header = line
        
        # Check if this line is part of a table
        if line.startswith('|') and line.endswith('|'):
            current_table.append(line)
        else:
            # End of current table
            if current_table:
                # Parse the accumulated table lines
                df = _parse_markdown_table(current_table)
                if df is not None and not df.empty:
                    tables_with_headers.append((last_header or f"Table {len(tables_with_headers) + 1}", df))
                current_table = []
        
        i += 1
    
    # Don't forget the last table
    if current_table:
        df = _parse_markdown_table(current_table)
        if df is not None and not df.empty:
            tables_with_headers.append((last_header or f"Table {len(tables_with_headers) + 1}", df))
    
    return tables_with_headers


def _parse_markdown_table(table_lines):
    """Parse a list of markdown table lines into a DataFrame.
    
    Args:
        table_lines: List of markdown table lines (| col1 | col2 | ...)
    
    Returns:
        pandas DataFrame or None if invalid
    """
    if len(table_lines) < 2:
        return None
    
    try:
        # Parse header
        header_line = table_lines[0]
        headers = [cell.strip() for cell in header_line.split('|')[1:-1]]
        
        # Skip separator line (contains dashes)
        rows = []
        for line in table_lines[2:]:
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            if len(cells) == len(headers):
                rows.append(cells)
        
        if not rows:
            return None
        
        return pd.DataFrame(rows, columns=headers)
    except Exception as e:
        return None


def export_tables_to_excel(tables_with_headers):
    """Export multiple DataFrames to Excel file with formatted sheets.
    
    Args:
        tables_with_headers: List of (header_text, DataFrame) tuples
    
    Returns:
        BytesIO object containing the Excel file
    """
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for idx, (header, df) in enumerate(tables_with_headers):
            # Create sheet name from header (max 31 chars for Excel)
            sheet_name = header[:31] if header else f"Table {idx + 1}"
            # Sanitize sheet name (remove invalid characters)
            sheet_name = re.sub(r'[\\/:*?"<>|]', '_', sheet_name)
            
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Format the sheet
            worksheet = writer.sheets[sheet_name]
            
            # Header formatting
            header_fill = PatternFill(start_color="FF7BA74F", end_color="FF7BA74F", fill_type="solid")
            header_font = Font(bold=True, color="FFFFFF")
            
            for cell in worksheet[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            
            # Auto-adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
            
            # Apply borders to all cells
            border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )
            for row in worksheet.iter_rows():
                for cell in row:
                    cell.border = border
                    if cell.row > 1:  # Data rows
                        cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
    
    output.seek(0)
    return output


def initialize_production_db():
    """Download production database from HuggingFace Hub if not present locally.
    
    This function:
    1. Checks if the database already exists locally (skip download if so)
    2. Downloads the compressed database from HuggingFace Hub
    3. Extracts the archive to a temporary directory
    4. Verifies the SQLite database was extracted correctly
    5. Moves the database to the final location (CHROMA_DB)
    
    The downloaded database contains pre-built Chroma vector collections
    for all cohort datasets (Betula, SNAC-K, etc.).
    """
    # ─ Check if database already exists locally ───────────────────────────────
    # Skip download if database directory exists and contains files
    # This prevents unnecessary downloads on subsequent app restarts
    if os.path.exists(CHROMA_DB) and os.listdir(CHROMA_DB):
        return
    
    # ─ Validate required configuration ─────────────────────────────────────────
    # Stop immediately if HuggingFace repository ID is not configured
    # This is a required secret in Streamlit Cloud
    if not HUGGINGFACE_REPO_ID:
        st.error("❌ HUGGINGFACE_REPO_ID not configured. Please contact the maintainer.")
        st.stop()
    
    try:
        from huggingface_hub import hf_hub_download
        
        # ─ Show download progress to user ──────────────────────────────────────
        # Create a placeholder that we'll update as the download progresses
        progress_placeholder = st.empty()
        progress_placeholder.info("📥 Downloading production database from HuggingFace Hub...")
        
        # ─ Download the database zip file ──────────────────────────────────────
        # Downloads chroma_prod_db.zip from the configured HuggingFace dataset
        # File is cached in ~/.cache/huggingface to avoid re-downloading
        try:
            zip_path = hf_hub_download(
                repo_id=HUGGINGFACE_REPO_ID,
                filename="chroma_prod_db.zip",
                repo_type="dataset",
                cache_dir=Path.home() / ".cache" / "huggingface"
            )
            logger.info(f"Downloaded database from HuggingFace: {HUGGINGFACE_REPO_ID}")
        except Exception as e:
            # Handle download errors (network issues, missing file, auth errors, etc.)
            error_msg = f"Failed to download from HuggingFace: {e}"
            logger.error(error_msg)
            progress_placeholder.error(f"❌ {error_msg}")
            st.stop()
        
        # ─ Extract the downloaded archive ──────────────────────────────────────
        # Update progress message and create temporary directory for extraction
        progress_placeholder.info("📦 Extracting database...")
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Extract all files from the zip archive
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)
            
            extracted_contents = os.listdir(temp_dir)
            
            # ─ Handle different zip structures ─────────────────────────────────
            # The zip file might contain either:
            # A) Direct contents: chroma.sqlite3, UUID folders, etc. (typical case)
            # B) Wrapped: A single "chroma_prod_db" folder containing everything
            #
            # We need to handle both cases to find the correct source path
            if "chroma_prod_db" in extracted_contents and len(extracted_contents) == 1:
                # Case B: Zip had a top-level chroma_prod_db wrapper folder
                source_path = os.path.join(temp_dir, "chroma_prod_db")
            else:
                # Case A: Zip had direct contents (most common)
                source_path = temp_dir
            
            # ─ Verify database integrity ──────────────────────────────────────
            # The SQLite database file (chroma.sqlite3) must exist
            # If missing, the zip was corrupted or incompatible
            sqlite_path = os.path.join(source_path, "chroma.sqlite3")
            if not os.path.exists(sqlite_path):
                error_files = os.listdir(source_path)[:5]
                progress_placeholder.error(f"❌ Database files corrupted. Found: {error_files}")
                st.stop()
            
            # ─ Clean up and move to final location ─────────────────────────────
            # Remove any old database directory to prevent conflicts
            if os.path.exists(CHROMA_DB):
                shutil.rmtree(CHROMA_DB)
            
            # Move the extracted database to its final location
            # This makes it available to Chroma at CHROMA_DB path
            shutil.move(source_path, CHROMA_DB)
            
            # ─ Final verification ─────────────────────────────────────────────
            # Confirm the move was successful by checking for the SQLite file
            if os.path.exists(os.path.join(CHROMA_DB, "chroma.sqlite3")):
                logger.info("Database extracted and ready")
                progress_placeholder.success("✅ Database downloaded and ready!")
            else:
                # Move succeeded but database file is missing (should never happen)
                progress_placeholder.error("❌ Database extraction failed")
                st.stop()
                
        except Exception as e:
            # Handle zip extraction or file operation errors
            logger.error(f"Extraction failed: {e}")
            progress_placeholder.error(f"❌ Failed to extract: {e}")
            st.stop()
        finally:
            # ─ Cleanup temporary directory ─────────────────────────────────────
            # Always remove temp directory, even if an error occurred
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        
    except ImportError:
        # Handle missing huggingface_hub package
        st.error("❌ Install huggingface_hub: `pip install huggingface_hub`")
        st.stop()
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error: {e}")
        st.error(f"❌ {e}")
        st.stop()

@st.cache_resource(show_spinner="Preparing embeddings model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_available_databases():
    """Get list of available databases by querying Chroma collections."""
    databases = []
    
    try:
        if not os.path.exists(CHROMA_DB):
            st.error(f"❌ Database directory not found: {CHROMA_DB}")
            st.stop()
        
        if not os.listdir(CHROMA_DB):
            st.error(f"❌ Database directory is empty: {CHROMA_DB}")
            st.stop()
        
        embeddings = get_embeddings()
        
        for db_name in KNOWN_DATABASES:
            collection_name = f"{db_name.lower()}_metadata"
            try:
                vectorstore = Chroma(
                    persist_directory=CHROMA_DB,
                    embedding_function=embeddings,
                    collection_name=collection_name
                )
                if vectorstore._collection.count() > 0:
                    databases.append(db_name)
            except Exception as e:
                logger.debug(f"Collection {db_name} not available: {e}")
        
        if not databases:
            logger.warning("No databases could be loaded from Chroma")
        else:
            logger.info(f"Loaded {len(databases)} databases")
        
        return sorted(databases)
    except Exception as e:
        logger.error(f"Error retrieving databases: {e}")
        st.error(f"❌ Could not load databases: {e}")
        return []

# Initialize vectorstore as None (will be set from cache on database selection)
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.selected_database = None

# Initialize vectorstores cache (preload all databases on first access)
if "vectorstores_cache" not in st.session_state:
    st.session_state.vectorstores_cache = {}
    st.session_state.vectorstores_loading = False

# Initialize latest response tables for export
if "latest_tables_with_headers" not in st.session_state:
    st.session_state.latest_tables_with_headers = []

def get_database_description(vectorstore):
    """Retrieve database description from the vector store."""
    if vectorstore is None:
        return ""
    
    try:
        all_docs = vectorstore._collection.get()
        
        if all_docs and all_docs.get('metadatas'):
            for i, metadata in enumerate(all_docs['metadatas']):
                if metadata.get("type") == "database_description":
                    if all_docs.get('documents') and i < len(all_docs['documents']):
                        return all_docs['documents'][i]
        return ""
    except Exception as e:
        logger.error(f"Could not retrieve database description: {e}")
        return ""

def get_relevant_background(query, vectorstore):
    """Retrieve cohort background information relevant to the user's query."""
    if vectorstore is None:
        return ""
    
    try:
        # Perform semantic search on the entire collection
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(query)
        
        background_docs = [
            doc for doc in docs 
            if doc.metadata.get("type") == "database_description"
        ]
        
        if not background_docs:
            return get_database_description(vectorstore)
        
        return "\n\n---\n\n".join(
            [doc.page_content for doc in background_docs[:2]]
        )
    except Exception as e:
        logger.error(f"Error retrieving background: {e}")
        return get_database_description(vectorstore)

def filter_and_organize_context(query, vectorstore):
    """Retrieve variable definitions relevant to the query.
    
    Returns context with source information for each variable.
    """
    if vectorstore is None:
        return "", []
    
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
        docs = retriever.invoke(query)
        
        var_defs = [
            doc for doc in docs 
            if doc.metadata.get("type") == "variable_definitions"
        ]
        
        # Include source information in context
        context_parts = []
        for doc in var_defs:
            source = doc.metadata.get("source", "Unknown")
            # Remove .xml suffix if present
            if source.endswith(".xml"):
                source = source[:-4]
            context_parts.append(f"[Source: {source}]\n{doc.page_content}")
        
        context_text = "\n\n---\n\n".join(context_parts)
        return context_text, var_defs
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        st.error(f"Error retrieving context: {e}")
        return "", []



# ── Main App ──────────────────────────────────────────────────────────────────

st.title("💬 NEAR Metadata Chatbot")

# Verify HUGGINGFACE_REPO_ID is available before initializing database
if not HUGGINGFACE_REPO_ID:
    st.error("❌ HUGGINGFACE_REPO_ID not configured in Streamlit secrets.")
    st.stop()

# Initialize production database from cloud if needed (MUST BE FIRST!)
initialize_production_db()

# Initialize available databases (after database is ready)
if "available_databases_loaded" not in st.session_state:
    with st.spinner("Discovering available databases..."):
        st.session_state.available_databases = get_available_databases()
        st.session_state.available_databases_loaded = True
    
    if not st.session_state.available_databases:
        st.error("""
        ❌ **No databases were discovered!**
        
        Please try:
        - Click "Refresh" in the sidebar to retry
        - Restart the app
        - Contact the maintainer
        """)

# Preload all vectorstores in the background (cache them)
if not st.session_state.vectorstores_loading and st.session_state.available_databases:
    if len(st.session_state.vectorstores_cache) == 0:
        st.session_state.vectorstores_loading = True
        with st.spinner("Preloading vector stores..."):
            embeddings = get_embeddings()
            for db in st.session_state.available_databases:
                try:
                    collection_name = f"{db.lower()}_metadata"
                    vectorstore = Chroma(
                        persist_directory=CHROMA_DB,
                        embedding_function=embeddings,
                        collection_name=collection_name
                    )
                    if vectorstore._collection.count() > 0:
                        st.session_state.vectorstores_cache[db] = vectorstore
                except Exception as e:
                    st.warning(f"Could not preload {db}: {e}")
        st.session_state.vectorstores_loading = False

# Add logo and controls to sidebar
with st.sidebar:
    logo_path = Path("logo/NEAR-chatbot.jpg")
    if logo_path.exists():
        col1, col2, col3 = st.columns([0.5, 2.5, 0.5])
        with col2:
            st.image(str(logo_path), use_container_width=True)
    st.markdown("---")
    
    # Database selection
    st.subheader("Select Database")
    
    if st.session_state.available_databases:
        selected_database = st.radio(
            "Choose a database to query:",
            options=st.session_state.available_databases,
            index=0,
            key="database_radio"
        )
        
        # Switch to selected database (from cache - instant!)
        if selected_database != st.session_state.selected_database:
            if selected_database in st.session_state.vectorstores_cache:
                st.session_state.vectorstore = st.session_state.vectorstores_cache[selected_database]
                st.session_state.selected_database = selected_database
            else:
                st.error(f"Vector store for {selected_database} not available")
        
    else:
        st.warning("No databases available")
    
    st.markdown("---")
    
    # Contact & Support
    st.subheader("Contact & Support")
    st.markdown("""
    **Maintainer:** Bolin Wu (NEAR)
    
    [📧 bolin.wu@ki.se](mailto:bolin.wu@ki.se)
    """)
    
    st.markdown("---")
    
    # Coming Soon / Roadmap
    with st.expander("🚀 Feature Coming Soon"):
        st.markdown("""
        - **Multi-Database Search**: Query across multiple databases
        - **Search History**: Track previous searches
        """)

# Add disclaimer about reference information
st.info("""
**ℹ️ Disclaimer:** The information provided here is for a quick overview only. 
For more accurate and comprehensive metadata, please check with [Maelstrom catalogue](https://www.maelstrom-research.org/search#lists?type=studies&query=network(in(Mica_network.id,near)),variable(limit(0,20)),study(in(Mica_study.className,Study),limit(0,20))) 
or the NEAR team.
""")

# Display search suggestions
st.markdown("### 💡 Example prompts:")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("- What cognitive tests are included?")
    st.caption("- How is sleep measured?")
    st.caption("- What social engagement data do you have?")
with col2:
    st.caption("- Tell me about this cohort")
    st.caption("- What physical activity data is there?")
    st.caption("- Are there medication records?")
with col3:
    st.caption("- How do you assess mental health?")
    st.caption("- What nutrition data is available?")
    st.caption("- What biomarkers were measured?")

# ── Chat ──────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

placeholder_text = f"Ask about {st.session_state.selected_database} metadata..."
if prompt := st.chat_input(placeholder_text):
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
                
                # Get context (already filtered to selected database via collection)
                # Returns both context text and document list
                context, context_docs = filter_and_organize_context(prompt, vectorstore)

                # Use unified prompt template for metadata queries with table format
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
                | Variable Name | Label | Categories | Source |
                |---|---|---|---|
                | variable_name | What it measures in plain English | Category values if applicable | source_file_name |
                
                CRITICAL RULES FOR EXTRACTING VARIABLE NAMES:
                - Column 1: EXTRACT EXACTLY the text that appears after "Variable: " in the source data
                - Do NOT use any other field names, labels, or descriptions
                - Do NOT use text from fields like "Description:" or "Sociodemographic Economic Characteristics:"
                - EXAMPLE: If you see "Variable: age_HT_rounded", MUST write "age_HT_rounded" in Column 1
                - EXAMPLE: If you see "Variable: löpnr", MUST write "löpnr" in Column 1
                - NEVER modify, shorten, or translate the variable name
                - NEVER invent variable names - only extract exactly what follows "Variable: "

                CRITICAL RULES FOR SOURCE COLUMN:
                - Column 4: Extract the source from "[Source: filename]" at the start of each variable's data block
                - Every variable MUST have its source - do not leave this column empty
                
                IMPORTANT NOTE ON VARIABLE AVAILABILITY:
                For information about which cohorts and tables contain these variables, please refer to the Maelstrom catalogue at: https://www.maelstrom-research.org/
                
                EXAMPLE OF CORRECT FORMAT:
                "In SNAC-K, several variables measure basic demographics. Participants are identified by a unique proband number (löpnr). The cohort includes both men and women, tracked through a sex variable. Birth dates are recorded to calculate age.

                | Variable Name | Label | Categories | Source |
                |---|---|---|---|
                | löpnr | Unique participant identifier | N/A (unique ID) | SNAC_K_Baseline |
                | kön | Participant's biological sex | 1=man, 2=woman | SNAC_K_Baseline |
                | Birthday | Date of birth for age calculation | Date format | SNAC_K_Baseline |
                

                Answer:"""

                PROMPT = PromptTemplate(
                    template=prompt_template, 
                    input_variables=["cohort_background", "context", "question"]
                )

                rag_chain = (
                    {
                        "cohort_background": lambda user_query: get_relevant_background(user_query, vectorstore),
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

        # Prepend database hint to response
        selected_db = st.session_state.get("selected_database")
        if selected_db and response:
            response_with_hint = f"📍 **{selected_db}**\n\n{response}"
        else:
            response_with_hint = response
        
        st.markdown(response_with_hint)
        
        # Extract tables from response and enable download if found
        tables_with_headers = extract_markdown_tables(response)
        st.session_state.latest_tables_with_headers = tables_with_headers
        
        # Add download button if tables were found
        if tables_with_headers:
            excel_file = export_tables_to_excel(tables_with_headers)
            st.download_button(
                label=f"📥 Download as Excel ({len(tables_with_headers)} table{'s' if len(tables_with_headers) > 1 else ''})",
                data=excel_file,
                file_name=f"near_metadata_{st.session_state.selected_database}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_tables"
            )

    st.session_state.messages.append({"role": "assistant", "content": response_with_hint})

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()