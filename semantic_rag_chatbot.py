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
from langchain_openai import ChatOpenAI  # For xAI Grok
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

# Safe way to get API keys
try:
    GROQ_API_KEY = st.secrets["GROQ_api_key"]
except (FileNotFoundError, KeyError, AttributeError):
    GROQ_API_KEY = os.getenv("GROQ_api_key")

try:
    XAI_api_key = st.secrets["XAI_api_key"]
except (FileNotFoundError, KeyError, AttributeError):
    XAI_api_key = os.getenv("XAI_api_key")

# Cloud storage URL for production vector database (optional)
# Use HuggingFace Hub: huggingface_hub.download() with repo_id
try:
    HUGGINGFACE_REPO_ID = st.secrets.get("HUGGINGFACE_REPO_ID")
except:
    HUGGINGFACE_REPO_ID = os.getenv("HUGGINGFACE_REPO_ID")

# ── Functions ─────────────────────────────────────────────────────────────────

def _find_table_blocks(text):
    """Find all table blocks in text and return their components.
    
    Args:
        text: String potentially containing markdown tables
    
    Returns:
        List of dicts with keys:
        - 'before_lines': Text lines before this table
        - 'header': Header line (text directly before table)
        - 'table_lines': The markdown table lines
    """
    lines = text.split('\n')
    blocks = []
    current_table = []
    last_header = None
    before_lines = []
    
    for i, original_line in enumerate(lines):
        line = original_line.strip()
        is_header = False
        
        # Look for header (non-table text followed by table)
        if line and not line.startswith('|') and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line.startswith('|'):
                last_header = original_line
                is_header = True
        
        # Check if this is a table line
        if line.startswith('|') and line.endswith('|'):
            current_table.append(line)
        else:
            # End of table - save it
            if current_table:
                blocks.append({
                    'before_lines': before_lines,
                    'header': last_header,
                    'table_lines': current_table,
                    'after_line': original_line
                })
                current_table = []
                last_header = None
                before_lines = []
            elif not is_header:
                # Don't add to before_lines if this line will be used as header
                before_lines.append(original_line)
    
    # Handle final table if present
    if current_table:
        blocks.append({
            'before_lines': before_lines,
            'header': last_header,
            'table_lines': current_table,
            'after_line': None
        })
    else:
        before_lines.append(None)  # Track remaining lines
    
    return blocks


def _dataframe_to_markdown(df, header=None):
    """Convert a DataFrame to markdown table lines.
    
    Args:
        df: pandas DataFrame
        header: Optional header text to prepend
    
    Returns:
        List of markdown table lines
    """
    if df is None or df.empty:
        return []
    
    lines = []
    if header:
        lines.append(header)
    
    # Header row
    lines.append("|" + "|".join(f" {col} " for col in df.columns) + "|")
    # Separator
    lines.append("|" + "|".join("---" for _ in df.columns) + "|")
    # Data rows
    for _, row in df.iterrows():
        lines.append("|" + "|".join(f" {str(val)} " for val in row) + "|")
    
    return lines


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
            
            # Skip incomplete rows entirely (don't pad with empty strings)
            # With improved LLM instructions, incomplete rows should be rare
            # If a row doesn't have all columns, skip it to maintain data integrity
            if len(cells) != len(headers):
                continue
            
            if cells:  # Only add non-empty rows
                rows.append(cells)
        
        if not rows:
            return None
        
        df = pd.DataFrame(rows, columns=headers)
        
        # Remove rows with empty critical columns (prevent incomplete entries)
        # Keep only rows where Variable Name, Label, and Categories are non-empty
        critical_cols = [col for col in ['Variable Name', 'Label', 'Categories'] if col in df.columns]
        if critical_cols:
            # Filter rows where all critical columns have non-empty values
            df = df[df[critical_cols].apply(lambda row: all(val and str(val).strip() for val in row), axis=1)]
        
        if df.empty:
            return None
        
        # Deduplicate rows with identical Label and Categories
        # Variables with the same label and categories are considered duplicates
        # Keep only the first occurrence (first source file)
        # This makes the export more succinct for reference purposes
        subset_cols = [col for col in ['Label', 'Categories'] if col in df.columns]
        if subset_cols:
            df = df.drop_duplicates(subset=subset_cols, keep='first')
        
        return df
    except Exception as e:
        return None
    
def extract_markdown_tables(text):
    """Extract markdown tables from text, handling multiple tables with headers.
    
    Args:
        text: String containing markdown tables
    
    Returns:
        List of tuples: (table_header_text, DataFrame) for each table found
    """
    tables_with_headers = []
    blocks = _find_table_blocks(text)
    
    for block in blocks:
        df = _parse_markdown_table(block['table_lines'])
        if df is not None and not df.empty:
            header = block['header'] or f"Table {len(tables_with_headers) + 1}"
            tables_with_headers.append((header, df))
    
    return tables_with_headers


def deduplicate_markdown_response(text):
    """Remove duplicate table rows from markdown response and reconstruct.
    
    Extracts tables, deduplicates them, and reconstructs the markdown
    with deduplicated tables. Non-table content is preserved as-is.
    
    Args:
        text: String containing markdown with tables
    
    Returns:
        String with deduplicated tables
    """
    blocks = _find_table_blocks(text)
    result = []
    
    for block in blocks:
        # Add lines before this table
        result.extend(block['before_lines'])
        
        # Parse and deduplicate the table
        df = _parse_markdown_table(block['table_lines'])
        if df is not None and not df.empty:
            # Convert back to markdown
            md_lines = _dataframe_to_markdown(df, block['header'])
            result.extend(md_lines)
        
        # Add line after table (if not end of text)
        if block['after_line'] is not None:
            result.append(block['after_line'])
    
    return "\n".join(result)



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
            # Remove markdown formatting (remove leading ### or ##, etc)
            if header:
                clean_header = re.sub(r'^#+\s*', '', header).strip()
                sheet_name = clean_header[:31] if clean_header else f"Table {idx + 1}"
            else:
                sheet_name = f"Table {idx + 1}"
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

# Initialize selected LLM model
if "selected_llm_model" not in st.session_state:
    st.session_state.selected_llm_model = "Groq (Llama 3.1 8B)"

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

def get_llm(model_name: str):
    """Initialize the selected LLM model.
    
    Args:
        model_name: Name of the model to initialize
    
    Returns:
        Initialized LLM instance
    """
    if model_name == "Grok 4.1 Fast":
        if not XAI_api_key:
            st.error("❌ XAI_api_key not configured")
            st.stop()
        return ChatOpenAI(
            api_key=XAI_api_key,
            model="grok-4-1-fast-non-reasoning",
            base_url="https://api.x.ai/v1",
            temperature=0.1,
        )
    else:  # Default to Groq Llama 3.1 8B
        if not GROQ_API_KEY:
            st.error("❌ GROQ_api_key not configured")
            st.stop()
        return ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.1-8b-instant",
            temperature=0.1,
        )




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
            st.image(str(logo_path), width='stretch')
    st.markdown("---")
    
    # Database selection
    st.subheader("Select Database")
    
    if st.session_state.available_databases:
        selected_database = st.radio(
            "Choose a database to query:",
            options=sorted(st.session_state.available_databases),
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
    
    # LLM Model Selection
    st.subheader("LLM Model Selection")
    available_models = ["Groq (Llama 3.1 8B)", "Grok 4.1 Fast"]
    selected_model = st.radio(
        "Choose LLM model:",
        options=available_models,
        index=0,
        key="llm_radio"
    )
    st.session_state.selected_llm_model = selected_model
    
    if selected_model == "Grok 4.1 Fast":
        st.caption("⚡ Testing: Grok 4.1 Fast (XAI)")
    else:
        st.caption("✅ Current: Llama 3.1 8B (Groq)")
    
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

st.info("""
**💡 Tip:** If you're not satisfied with the results, try searching again with different wording. The similar question may yield different results due to the nature of AI-powered responses.
""")

# Display search suggestions
st.markdown("### Example prompts:")
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
                # Get the selected LLM model
                llm = get_llm(st.session_state.selected_llm_model)
                
                # Display which model is being used
                st.caption(f"🔧 Using: {st.session_state.selected_llm_model}")
                
                # Get context (already filtered to selected database via collection)
                # Returns both context text and document list
                context, context_docs = filter_and_organize_context(prompt, vectorstore)

                # Use unified prompt template for metadata queries with table format
                prompt_template = """You are an expert in epidemiology and aging research, specializing in cohort study metadata.
CRITICAL: Do NOT invent or hallucinate data. Only use information explicitly provided in VARIABLE DATA below.

                COHORT BACKGROUND:
                {cohort_background}

                ---

                Your task is to answer questions about variables and metadata from this cohort.

                ### VARIABLE DATA (ONLY SOURCE OF TRUTH):
                {context}

                ### Question from user:
                {question}

                ### Your Response Instructions:
                - Group related variables by theme
                - Start with a clear, natural explanation of the topic based on the related cohort background
                - Use your own words to describe what the variables measure
                - Then, present the variable information in a markdown table with these columns:
                
                | Variable Name | Label | Categories | Source |
                |---|---|---|---|
                | variable_name | What it measures in plain English | Category values if applicable | source_file_name |
                
                ================================================================
                CRITICAL RULES FOR EXTRACTING DATA (READ CAREFULLY):
                ================================================================
                
                1. VARIABLE NAMES (Column 1):
                   - EXTRACT EXACTLY the text that appears after "Variable: " in the source data
                   - Copy-paste the exact variable name - do NOT modify, shorten, or translate
                   - Do NOT use field names like "Description" or "Label" instead
                   - FORBIDDEN: Do NOT invent variable names not in the source
                   - EXAMPLE RIGHT: Source has "Variable: age_HT_rounded" → Write "age_HT_rounded"
                   - EXAMPLE RIGHT: Source has "Variable: löpnr" → Write "löpnr" exactly
                   - EXAMPLE WRONG: Inventing "participant_age" when source only has "löpnr"

                2. LABELS (Column 2):
                   - Extract the description/label EXACTLY as written in the source
                   - Use the "Label:" field value from source data
                   - Keep descriptions concise (1-2 sentences max)
                   - Do NOT shorten, paraphrase, or interpret the label

                3. CATEGORIES (Column 3):
                   - Extract category values EXACTLY as written in source (e.g., "1=man, 2=woman")
                   - If no categories exist, write "N/A (continuous)" or "N/A (unique ID)"
                   - Do NOT invent category mappings not in the source
                   - If values span multiple lines in source, include them all in one cell
                   - Do NOT use HTML tags like <br> or newlines within cells - keep each row as one line

                4. SOURCE (Column 4):
                   - Extract from "[Source: filename]" at the start of each variable block
                   - Must be present for EVERY variable (required field)
                   - Use the exact source filename provided

                5. VALIDATION (CRITICAL - MUST FOLLOW):
                   - EVERY row must have ALL 4 columns completely filled (NO EMPTY CELLS)
                   - If you cannot fill all 4 columns for a row, OMIT that row entirely
                   - NEVER create incomplete rows or truncate table rows
                   - NEVER pad rows with empty cells or partial data
                   - Every variable name must come from the source data
                   - If data is missing from source, DO NOT hallucinate it
                   - Double-check: Each variable in your table should be traceable to the source context
                   - Before you finish, verify: scan the entire table and confirm NO ROWS have empty cells
                   - If the last row is incomplete, DELETE IT and end the table cleanly
                
                DETAILED EXAMPLES OF CORRECT FORMAT:
                
                Example 1 - Demographics in SNAC-K:
                "In SNAC-K, several variables measure basic demographics. Each participant has a unique identifier and recorded biological sex."
                
                | Variable Name | Label | Categories | Source |
                |---|---|---|---|
                | löpnr | Unique participant identifier number | N/A (unique ID) | SNAC_K_Baseline |
                | kön | Participant's biological sex | 1=man, 2=woman | SNAC_K_Baseline |
                
                Example 2 - Physical Measurements:
                "Height and weight are measured at baseline assessment."
                
                | Variable Name | Label | Categories | Source |
                |---|---|---|---|
                | height_cm | Height in centimeters | N/A (continuous) | H70_Baseline_Form |
                | weight_kg | Body weight in kilograms | N/A (continuous) | H70_Baseline_Form |

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
                    logger.info(f"LLM Response (first 500 chars): {response[:500]}")  # Debug log
                except Exception as e:
                    if "rate_limit" in str(e).lower() or "429" in str(e):
                        st.error("⚠️ Rate limit reached. Please try again in a few moments.")
                        response = "I'm temporarily unavailable due to high usage. Please try again shortly."
                    else:
                        st.error(f"Error: {str(e)}")
                        response = ""

        # Prepend database hint to response
        selected_db = st.session_state.get("selected_database")
        
        # Deduplicate response tables
        if response:
            response = deduplicate_markdown_response(response)
        
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
                file_name=f"NEARchatbot_{st.session_state.selected_database}_{pd.Timestamp.now().strftime('%y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_tables"
            )

    st.session_state.messages.append({"role": "assistant", "content": response_with_hint})

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()