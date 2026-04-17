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
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI  # For OpenAI-compatible endpoints
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import shutil
import zipfile

load_dotenv()

# Configure logging (set to WARNING for production, INFO/DEBUG for development)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NEAR Metadata Chatbot (Beta)",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# ── Custom Styling (NEAR Brand) ───────────────────────────────────────────────
# Load external CSS file for clean separation of concerns
css_file = Path("styles/near_brand.css")
if css_file.exists():
    with open(css_file, "r") as f:
        css_content = f.read()
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
else:
    logger.warning(f"CSS file not found at {css_file}. Styling may not be applied.")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB = "./chroma_prod_db"          # Production database (cloud storage)
DATA_ROOT = "./data"

# Known databases (hardcoded as fallback when data/ folder not available)
KNOWN_DATABASES = [
    "Betula", "GAS_SNAC_S", "GENDER", "H70", "KP", 
    "OCTO-Twin", "SALT", "SATSA", "SNAC-B", "SNAC-K", "SNAC-N", "SWEOLD", "TryBo"
]

# LLM Model Names (for consistency across the app)
LLM_MODEL_GPT = "GPT-5.4 Mini (Azure Foundry)"

# LLM Model IDs (technical identifiers for API calls)
GPT_MODEL_ID = "gpt-5.4-mini"
AZURE_FOUNDRY_BASE_URL = "https://llm-chatbot-api.cognitiveservices.azure.com/openai/v1/"

# LLM Hyperparameters
LLM_TEMPERATURE = 0.3           # Balanced: accurate answers with flexibility for general knowledge (0.0=deterministic, 1.0=creative)

# Retrieval Parameters
RETRIEVAL_K_BACKGROUND = 5      # Top-5 docs for cohort background context
RETRIEVAL_K_CONTEXT = 60        # Top-60 docs for variable definitions (increased for better category capture)
EXACT_MATCH_LIMIT = 3          # Exact variable-name hits to merge ahead of semantic results
BACKGROUND_ENRICH_FALLBACK_DOCS = 3  # Fallback docs used when no exact variable match is found
BACKGROUND_ENRICH_MAX_DOCS = 3       # Max docs used to extract enrichment terms
MAX_SELECTED_DATABASES_GPT = 6       # GPT path can handle broader multi-database context
MERGED_CONTEXT_K_MAX_GPT = 48        # Global merged context cap for GPT
BACKGROUND_PER_DB_K = 2              # Small per-database budget for cohort background
BACKGROUND_GLOBAL_MAX_GPT = 10

# Safe way to get API keys
try:
    AZURE_api_key = st.secrets["AZURE_api_key"]
except (FileNotFoundError, KeyError, AttributeError):
    AZURE_api_key = os.getenv("AZURE_api_key")
    

# Cloud storage URL for production vector database (optional)
# Use HuggingFace Hub: huggingface_hub.download() with repo_id
try:
    HUGGINGFACE_REPO_ID = st.secrets.get("HUGGINGFACE_REPO_ID")
except:
    HUGGINGFACE_REPO_ID = os.getenv("HUGGINGFACE_REPO_ID")

# Beta release settings (branch-scoped)
BETA_RELEASE_LABEL = "v1.4 beta"

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
        
        if df.empty:
            return None
        
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



def get_complete_background(vectorstores, llm_model=None):
    """Retrieve database description chunks across selected collections.
    
    Uses semantic search to find and return all chunks tagged as database descriptions,
    providing comprehensive cohort background information as a fallback when query-specific
    background retrieval yields no results.
    """
    if not vectorstores:
        return ""
    
    try:
        background_global_cap = BACKGROUND_GLOBAL_MAX_GPT

        all_groups = []
        for vectorstore in vectorstores:
            retriever = vectorstore.as_retriever(
                search_kwargs={
                    "k": BACKGROUND_PER_DB_K,
                    "filter": {"type": "database_description"}
                }
            )
            docs = retriever.invoke("database description overview cohort study information")
            all_groups.append(docs)

        merged_docs = interleave_document_groups(all_groups)
        merged_docs = deduplicate_documents(merged_docs)[:background_global_cap]
        descriptions = [doc.page_content for doc in merged_docs]

        return "\n\n---\n\n".join(descriptions) if descriptions else ""
    except Exception as e:
        logger.error(f"Could not retrieve database description: {e}")
        return ""

def get_relevant_background(query, vectorstores, llm_model=None, context_docs=None):
    """Retrieve cohort background information relevant to the user's query."""
    if not vectorstores:
        return ""
    
    try:
        # Perform semantic search on the entire collection
        background_global_cap = BACKGROUND_GLOBAL_MAX_GPT
        background_per_db_k = max(1, min(RETRIEVAL_K_BACKGROUND, BACKGROUND_PER_DB_K))
        background_query = query

        # If the user query looks like an exact variable code, enrich background
        # retrieval with matched variable labels/sources to improve relevance.
        candidates = extract_variable_name_candidates(query)
        if candidates and context_docs:
            candidate_set = {c.lower() for c in candidates}
            matched_docs = []

            for doc in context_docs:
                variable_name = (doc.metadata or {}).get("variable_name")
                if variable_name and variable_name.lower() in candidate_set:
                    matched_docs.append(doc)

            if not matched_docs:
                matched_docs = context_docs[:BACKGROUND_ENRICH_FALLBACK_DOCS]

            enrichment_terms = []
            seen_terms = set()
            for doc in matched_docs[:BACKGROUND_ENRICH_MAX_DOCS]:
                source = (doc.metadata or {}).get("source")
                if source:
                    clean_source = source.replace(".xml", "")
                    if clean_source.lower() not in seen_terms:
                        seen_terms.add(clean_source.lower())
                        enrichment_terms.append(clean_source)

                label_match = re.search(r"(?im)^Label:\s*(.+)$", doc.page_content or "")
                if label_match:
                    label = label_match.group(1).strip()
                    if label.lower() not in seen_terms:
                        seen_terms.add(label.lower())
                        enrichment_terms.append(label)

            if enrichment_terms:
                background_query = f"{query} {' '.join(enrichment_terms)}"

        background_groups = []
        for vectorstore in vectorstores:
            retriever = vectorstore.as_retriever(
                search_kwargs={
                    "k": background_per_db_k,
                    "filter": {"type": "database_description"}
                }
            )
            docs = retriever.invoke(background_query)
            background_groups.append(docs)

        background_docs = interleave_document_groups(background_groups)
        background_docs = deduplicate_documents(background_docs)[:background_global_cap]
        
        if not background_docs:
            return get_complete_background(vectorstores, llm_model)
        
        return "\n\n---\n\n".join(
            [doc.page_content for doc in background_docs]
        )
    except Exception as e:
        logger.error(f"Error retrieving background: {e}")
        return get_complete_background(vectorstores, llm_model)


def extract_variable_name_candidates(query: str) -> list[str]:
    """Extract likely variable identifiers from a user query.

    Targets code-like identifiers such as BAS1_H4, v518, or quoted variable names.
    """
    if not query:
        return []

    candidates = []
    seen = set()

    def add_candidate(value: str) -> None:
        cleaned = value.strip().strip("`'\"")
        if not cleaned:
            return
        if cleaned.lower() in seen:
            return
        seen.add(cleaned.lower())
        candidates.append(cleaned)

    for match in re.findall(r"[`\"']([A-Za-z][A-Za-z0-9_:-]*)[`\"']", query):
        add_candidate(match)

    for token in re.findall(r"\b[A-Za-z][A-Za-z0-9_:-]*\b", query):
        if any(char.isdigit() for char in token) or "_" in token or "-" in token or token.isupper():
            add_candidate(token)

    return candidates


def _collection_payload_to_documents(payload: dict | None) -> list[Document]:
    """Convert a Chroma collection.get payload into LangChain Documents."""
    if not payload:
        return []

    documents = payload.get("documents") or []
    metadatas = payload.get("metadatas") or []
    results = []

    for index, page_content in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        results.append(Document(page_content=page_content, metadata=metadata or {}))

    return results


def deduplicate_documents(documents: list[Document]) -> list[Document]:
    """Deduplicate retrieval results while preserving order."""
    unique_docs = []
    seen = set()

    for doc in documents:
        metadata = doc.metadata or {}
        key = (
            metadata.get("source"),
            metadata.get("variable_name"),
            doc.page_content,
        )
        if key in seen:
            continue
        seen.add(key)
        unique_docs.append(doc)

    return unique_docs


def get_document_database_name(doc: Document) -> str:
    """Resolve the database name for a retrieved document."""
    metadata = doc.metadata or {}
    database = metadata.get("database")
    if database:
        return str(database)

    source = metadata.get("source", "")
    if isinstance(source, str) and ":" in source:
        return source.split(":", 1)[0]

    return "Unknown"


def build_database_grouped_context(documents: list[Document]) -> str:
    """Render retrieved documents as database-grouped context for the prompt."""
    if not documents:
        return ""

    grouped_docs: dict[str, list[Document]] = {}
    ordered_databases: list[str] = []

    for doc in documents:
        database = get_document_database_name(doc)
        if database not in grouped_docs:
            grouped_docs[database] = []
            ordered_databases.append(database)
        grouped_docs[database].append(doc)

    database_sections = []
    for database in ordered_databases:
        context_parts = []
        for doc in grouped_docs[database]:
            source = (doc.metadata or {}).get("source", "Unknown")
            variable_name = (doc.metadata or {}).get("variable_name")
            if isinstance(source, str) and source.endswith(".xml"):
                source = source[:-4]

            if not str(source).startswith(f"{database}:"):
                source = f"{database}:{source}"

            if variable_name:
                context_parts.append(f"[Source: {source} | Variable: {variable_name}]\n{doc.page_content}")
            else:
                context_parts.append(f"[Source: {source}]\n{doc.page_content}")

        database_sections.append(
            f"### Database: {database}\n" + "\n\n---\n\n".join(context_parts)
        )

    return "\n\n====================\n\n".join(database_sections)


def interleave_document_groups(document_groups: list[list[Document]]) -> list[Document]:
    """Interleave docs from multiple groups to balance coverage across databases."""
    if not document_groups:
        return []

    merged = []
    max_len = max((len(group) for group in document_groups), default=0)
    for idx in range(max_len):
        for group in document_groups:
            if idx < len(group):
                merged.append(group[idx])

    return merged


def get_retrieval_budget(llm_model: str | None) -> dict:
    """Return model-aware retrieval caps for multi-database querying."""
    max_selected_databases = MAX_SELECTED_DATABASES_GPT
    merged_context_cap = MERGED_CONTEXT_K_MAX_GPT

    return {
        "max_selected_databases": max_selected_databases,
        "merged_context_cap": merged_context_cap,
    }


def safe_query_vectorstores(vectorstores: list, llm_model: str | None) -> list:
    """Cap selected vectorstores based on model budget to avoid token blowups."""
    if not vectorstores:
        return []

    budget = get_retrieval_budget(llm_model)
    max_selected_databases = budget["max_selected_databases"]
    if len(vectorstores) <= max_selected_databases:
        return vectorstores

    return vectorstores[:max_selected_databases]


def exact_variable_name_lookup(query: str, vectorstore) -> list[Document]:
    """Look up variable documents by exact metadata match before semantic search."""
    if vectorstore is None:
        return []

    collection = getattr(vectorstore, "_collection", None)
    if collection is None:
        return []

    candidates = extract_variable_name_candidates(query)
    if not candidates:
        return []

    exact_docs = []

    for candidate in candidates:
        candidate_variants = [candidate]
        for variant in (candidate.upper(), candidate.lower()):
            if variant not in candidate_variants:
                candidate_variants.append(variant)

        for variant in candidate_variants:
            payload = collection.get(
                where={"variable_name": variant},
                include=["documents", "metadatas"],
                limit=EXACT_MATCH_LIMIT,
            )
            exact_docs.extend(_collection_payload_to_documents(payload))

        if exact_docs:
            continue

        payload = collection.get(
            where={"type": "variable_definitions"},
            where_document={"$contains": f"Variable: {candidate}"},
            include=["documents", "metadatas"],
            limit=EXACT_MATCH_LIMIT,
        )
        exact_docs.extend(_collection_payload_to_documents(payload))

    exact_docs = deduplicate_documents(exact_docs)
    if exact_docs:
        logger.debug(
            "Exact variable lookup matched %s docs for candidates %s",
            len(exact_docs),
            candidates,
        )

    return exact_docs

def filter_and_organize_context(query, vectorstores, llm_model=None):
    """Retrieve variable definitions relevant to the query.
    
    Returns context with source information for each variable.
    """
    if not vectorstores:
        return "", []
    
    try:
        # Use model-aware retrieval caps for stable multi-database behavior.
        budget = get_retrieval_budget(llm_model)
        original_count = len(vectorstores)
        vectorstores = safe_query_vectorstores(vectorstores, llm_model)
        if len(vectorstores) < original_count:
            st.toast(
                f"⚠️ Querying {len(vectorstores)} of {original_count} selected databases "
                f"to stay within {llm_model} context limits. Switch to GPT for broader coverage.",
                icon="⚠️",
            )

        k_context_base = RETRIEVAL_K_CONTEXT

        per_db_k = max(4, int(k_context_base / max(1, len(vectorstores))))
        merged_context_cap = budget["merged_context_cap"]
        
        semantic_groups = []
        exact_docs = []

        for vectorstore in vectorstores:
            retriever = vectorstore.as_retriever(
                search_kwargs={
                    "k": per_db_k,
                    "filter": {"type": "variable_definitions"}
                }
            )
            semantic_groups.append(retriever.invoke(query))
            exact_docs.extend(exact_variable_name_lookup(query, vectorstore))

        docs = interleave_document_groups(semantic_groups)
        var_defs = deduplicate_documents(exact_docs + docs)[:merged_context_cap]
        
        # Debug logging: show how many documents were retrieved
        logger.debug(
            f"Query: '{query}' | per_db_k: {per_db_k} "
            f"| DBs: {len(vectorstores)} | Semantic docs: {len(docs)} "
            f"| Exact docs: {len(exact_docs)} | Final docs: {len(var_defs)} "
            f"| merged_cap: {merged_context_cap} | Model: {llm_model}"
        )
        
        context_text = build_database_grouped_context(var_defs)
        logger.debug(f"Final context: {len(var_defs)} variable definitions included ({len(context_text)} chars)")
        return context_text, var_defs
    except Exception as e:
        error_msg = f"Error retrieving context: {e}"
        logger.error(error_msg)
        logger.exception("Full traceback:")
        # Note: Don't show error in UI here since this is called during response generation
        return "", []

def get_llm():
    """Initialize the Azure GPT model."""
    if not AZURE_api_key:
        st.error("❌ AZURE_api_key not configured")
        st.stop()
    return ChatOpenAI(
        api_key=AZURE_api_key,
        model=GPT_MODEL_ID,
        base_url=AZURE_FOUNDRY_BASE_URL,
        temperature=LLM_TEMPERATURE,
    )


# ── Main App ──────────────────────────────────────────────────────────────────

st.title("NEAR Metadata Chatbot (Beta)")
st.caption(f"🧪 Internal Preview • {BETA_RELEASE_LABEL}")

# ── Initialize Session State (MUST BE FIRST) ──────────────────────────────────
# Initialize all session state variables before any code accesses them
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.selected_database = None
if "selected_databases" not in st.session_state:
    st.session_state.selected_databases = []
if "vectorstores_cache" not in st.session_state:
    st.session_state.vectorstores_cache = {}
    st.session_state.vectorstores_loading = False
if "latest_tables_with_headers" not in st.session_state:
    st.session_state.latest_tables_with_headers = []
if "selected_llm_model" not in st.session_state:
    st.session_state.selected_llm_model = LLM_MODEL_GPT
if "available_databases_loaded" not in st.session_state:
    st.session_state.available_databases_loaded = False
    st.session_state.available_databases = []

# Verify HUGGINGFACE_REPO_ID is available before initializing database
if not HUGGINGFACE_REPO_ID:
    st.error("❌ HUGGINGFACE_REPO_ID not configured in Streamlit secrets.")
    st.stop()

# Initialize production database from cloud if needed (MUST BE FIRST!)
initialize_production_db()

# Initialize available databases (after database is ready)
if not st.session_state.available_databases_loaded:
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

# Add controls to sidebar
with st.sidebar:
    logo_path = Path("logo/NEAR-chatbot.jpg")
    if logo_path.exists():
        col1, col2, col3 = st.columns([0.5, 2.5, 0.5])
        with col2:
            st.image(str(logo_path), width='stretch')
    st.markdown("---")
    
    # Database selection
    st.subheader("Select Database(s)")
    
    if st.session_state.available_databases:
        sorted_databases = sorted(st.session_state.available_databases)
        default_databases = st.session_state.selected_databases or sorted_databases[:1]
        default_databases = [db for db in default_databases if db in sorted_databases]
        if not default_databases and sorted_databases:
            default_databases = sorted_databases[:1]

        llm_model_for_caps = st.session_state.get("selected_llm_model", LLM_MODEL_GPT)
        max_selected = get_retrieval_budget(llm_model_for_caps)["max_selected_databases"]

        current_selected_databases = []
        for db_name in sorted_databases:
            checkbox_key = f"database_toggle_{db_name}"
            if checkbox_key in st.session_state and st.session_state[checkbox_key]:
                current_selected_databases.append(db_name)

        trimmed_database_selection = False
        if len(current_selected_databases) > max_selected:
            for db_name in current_selected_databases[max_selected:]:
                st.session_state[f"database_toggle_{db_name}"] = False
            trimmed_database_selection = True

        st.caption("Toggle databases directly below.")
        for db_name in sorted_databases:
            checkbox_key = f"database_toggle_{db_name}"
            if checkbox_key not in st.session_state:
                st.session_state[checkbox_key] = db_name in default_databases

            st.checkbox(db_name, key=checkbox_key)

        selected_databases = [
            db_name for db_name in sorted_databases
            if st.session_state.get(f"database_toggle_{db_name}", False)
        ]

        if trimmed_database_selection:
            st.caption(
                f"You can select up to {max_selected} databases with {llm_model_for_caps}. "
                f"Extra selections were turned off to stay within context limits."
            )

        loaded_databases = []
        selected_vectorstores = []
        for db_name in selected_databases:
            vectorstore = st.session_state.vectorstores_cache.get(db_name)
            if vectorstore is None:
                st.warning(f"Vector store for {db_name} not available")
                continue
            loaded_databases.append(db_name)
            selected_vectorstores.append(vectorstore)
        if len(loaded_databases) > max_selected:
            loaded_databases = loaded_databases[:max_selected]
            selected_vectorstores = selected_vectorstores[:max_selected]
            st.caption(
                f"Using first {max_selected} selected databases for {llm_model_for_caps} to stay within context limits."
            )

        st.session_state.selected_databases = loaded_databases
        st.session_state.selected_database = ", ".join(loaded_databases)
        st.session_state.selected_vectorstores = selected_vectorstores
        st.session_state.vectorstore = selected_vectorstores[0] if selected_vectorstores else None
        
    else:
        st.warning("No databases available")
    
    st.markdown("---")
    
    # Fixed model (GPT-only)
    st.subheader("LLM Model")
    st.session_state.selected_llm_model = LLM_MODEL_GPT
    st.caption("🧠 Azure Foundry GPT-5.4 Mini")
    
    st.markdown("---")
    
    # Contact & Support
    st.subheader("Contact & Support")
    st.markdown("""
    **Maintainer:** [Bolin Wu (NEAR)](https://ki.se/en/people/bolin-wu)
    
    **E-mail:** [📧 bolin.wu@ki.se](mailto:bolin.wu@ki.se)
    """)
    
    st.markdown("---")
    
    # Changelog
    with st.expander("📋 What's New"):
        st.markdown("""
        **v1.3.1** – Exact Variable Lookup
        - Queries like "what is BAS1_H4?" now find variables via direct metadata match
        - Background retrieval enriched with matched variable's source & label
        
        [View all previous releases →](https://github.com/Bolin-Wu/NEAR-metadata-chatbot/releases)
        """)

# Add disclaimer and tip about reference information
st.info("""
**ℹ️ Disclaimer:** This chatbot provides a quick overview based on NEAR metadata submitted to Maelstrom. Some variables may be unavailable, including derived variables, registry data, or recent updates. For complete and validated metadata, refer to the [Maelstrom catalogue](https://www.maelstrom-research.org/search#lists?type=studies&query=network(in(Mica_network.id,near)),variable(limit(0,20)),study(in(Mica_study.className,Study),limit(0,20))) or contact the NEAR team.

**💡 Tip:** This chatbot works best with one-shot queries and does not keep conversational memory. Ask complete standalone questions (avoid follow-up questions that depend on previous turns). If you're not satisfied with the results, try searching again with different wording. The same question may yield different results due to the nature of AI-powered responses.
""")

# Display search suggestions
st.markdown("### Example prompts:")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("- What cognitive tests are included?")
    st.caption("- How is sleep measured?")
    st.caption("- What social engagement data do you have?")
with col2:
    st.caption("- What nutrition data is available?")
    st.caption("- Recommend variables for constructing CIRS.")
    st.caption("- Suggest variables for frailty index.")
with col3:
    st.caption("- Recommend blood pressure variables for hypertension analysis.")
    st.caption("- List ADL variables and map each to core ADL, ADL+IADL.")
    st.caption("- Give all bathing, dressing, toileting, transferring variables with source and categories.")


# ── Chat ──────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

selected_dbs_for_placeholder = st.session_state.get("selected_databases") or []
if len(selected_dbs_for_placeholder) == 1:
    db_scope_label = selected_dbs_for_placeholder[0]
elif len(selected_dbs_for_placeholder) > 1:
    db_scope_label = f"{len(selected_dbs_for_placeholder)} selected databases"
else:
    db_scope_label = "selected databases"

placeholder_text = f"Ask one complete question about {db_scope_label} metadata..."
if prompt := st.chat_input(placeholder_text):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        selected_vectorstores = st.session_state.get("selected_vectorstores") or []
        if not selected_vectorstores:
            st.warning("Vector store not available!")
            response = ""
        else:
            with st.spinner("Thinking..."):
                # Get the GPT model
                llm = get_llm()
                
                # Display which model is being used
                st.caption(f"🔧 Using: {LLM_MODEL_GPT}")
                
                # Get context merged from selected database collections.
                # Returns both context text and document list
                context, context_docs = filter_and_organize_context(
                    prompt,
                    selected_vectorstores,
                    LLM_MODEL_GPT,
                )

                # Avoid blank assistant output when retrieval returns no variable chunks.
                if not context.strip():
                    response = (
                        "I could not find matching variable definitions for that query in the selected database scope. "
                        "Try a more specific term (for example: systolic, diastolic, hypertension, blood pressure)."
                    )
                    logger.warning(
                        "Empty retrieval context for query '%s' in databases '%s' using model '%s'",
                        prompt,
                        st.session_state.get("selected_databases"),
                        LLM_MODEL_GPT,
                    )
                else:

                    # Use unified prompt template for metadata queries with table format
                    prompt_template = """You are an expert in epidemiology and aging research, specializing in cohort study metadata. CRITICAL: Do NOT invent or hallucinate metadata. Only use information explicitly provided in NEAR metadata below.

                COHORT BACKGROUND:
                {cohort_background}

                ---

                Your task is to answer questions about variables and metadata from this cohort.

                ### NEAR metadata (ONLY SOURCE OF TRUTH):
                The metadata below is already organized by database.
                Each database section begins with a heading in this format:
                `### Database: DATABASE_NAME`
                Each variable block under that heading begins with:
                [Source: database_name:source_file_name | Variable: variable_name]
                {context}

                ### Question from user:
                {question}

                ### Number of Selected Databases:
                 {selected_db_count}

                ### Names of Selected Databases:
                {selected_db_names}

                ### Your Response Instructions:
                     - Treat every question as a single-turn request based only on the current question and provided metadata context
                     - DO NOT include follow-up offers such as "If you want, I can...", "Would you like me to...", or "I can also..."
                     - Start with a clear, natural explanation of the topic based on the related cohort background
                     - Use your own words to describe what the variables measure
                
                     **CRITICAL: Organize results by database** (use database name extracted from the Source field):
                     - Create a separate section for each database using the format: `## Database Name`
                     - Under each database section, present variables in a markdown table with these columns:
                
                     | Variable Name | Label | Categories | Source |
                     |---|---|---|---|
                     | variable_name | What it measures in plain English | Category values if applicable | source_file_name |
                
                     - Ensure full variable coverage in each database through grouped rows (do NOT repeat the same concept in multiple rows)
                     - Group related variables by theme WITHIN each database section
                
                     ================================================================
                     CRITICAL RULES FOR EXTRACTING DATA (READ CAREFULLY):
                     ================================================================
                
                     1. VARIABLE NAMES (Column 1):
                         - EXTRACT EXACTLY the text that appears after "Variable: " in the source data
                         - Copy-paste the exact variable name - do NOT modify, shorten, or translate
                         - EXAMPLE RIGHT: Source has "Variable: löpnr". Write "löpnr" exactly

                     2. LABELS (Column 2):
                         - Use the "Label:" field value EXACTLY as written in source data
                         - Do NOT shorten, paraphrase, or interpret the label
                         - NEVER synthesize or invent a representative label for grouped rows
                         - STRICT GROUPING RULE: variables can be grouped only when Label is exactly identical
                         - If label is not verifiable from provided context, use: "N/A (not verifiable from provided context)"

                     3. CATEGORIES (Column 3):
                         - Extract category values EXACTLY as they appear in the source data, typically in a "Categories:" field
                         - FORMAT STRICTLY as: "1=value1, 2=value2, 3=value3" (number=description, comma-separated, NO SEMICOLONS)
                         - If no categories exist, write "N/A"
                         - NEVER infer, normalize, or invent missing category mappings
                         - STRICT GROUPING RULE: variables can be grouped only when Categories are exactly identical (or all are exactly "N/A")
                         - If categories are not verifiable from provided context, use: "N/A (not verifiable from provided context)"

                     4. SOURCE (Column 4):
                         - Extract from the header "[Source: filename | Variable: variable_name]" at the start of each block
                         - Must be present for EVERY variable (required field)
                         - Use the exact source filename provided

                     5. DATABASE ORGANIZATION:
                         - Create separate `## Database Name` headers for each database present in the input context
                         - Use ONLY the variable blocks that appear under the matching `### Database: DATABASE_NAME` input section for that database's output table
                         - NEVER place a variable into a different database's table than the input section where it appears
                         - The Source field still includes the database prefix as a secondary check: "[Source: DATABASE:filename | ...]"

                     6. DEDUPLICATION:
                         - Within the SAME database, merge variables into one row ONLY if BOTH Label and Categories are exactly identical
                         - If either Label or Categories differs, keep them as separate rows (even if conceptually similar)
                         - In merged rows, list all variable names and sources that meet the exact-match rule

                     6A. NO FUZZY MATCHING FOR DEDUP:
                         - Do NOT normalize, relax, or approximate Label/Categories when deciding grouping
                         - Minor wording, spacing, punctuation, or ordering differences mean NOT equal for grouping

                     6B. REQUIRED TWO-PASS WORKFLOW (INTERNAL REASONING, THEN OUTPUT):
                         - PASS 1 (per database): build groups using exact pair key = (Label text, Categories text)
                         - PASS 2 (per database): render the markdown table from concept groups only
                         - HARD CHECK before finalizing each database table: merge rows only when exact pair key is identical
                         - HARD CHECK for completeness: every variable used in PASS 1 must appear in exactly one final grouped row

                     7. VALIDATION (CRITICAL - MUST FOLLOW):
                         - EVERY row must have ALL 4 columns completely filled (NO EMPTY CELLS)
                         - NEVER pad rows with empty cells or partial data
                         - Every variable name must come from the source data
                         - If data is missing from source, DO NOT hallucinate it
                         - For each database, DO NOT output duplicate rows with the same exact (Label, Categories) pair
                         - If a Label or Categories value cannot be quoted exactly from provided context, use the explicit fallback strings above instead of guessing

                     8. HARMONIZATION ACROSS DATABASES (ONLY WHEN MULTIPLE DATABASES ARE SELECTED):
                         - If the number of selected databases is greater than 1, ADD a final section titled: `## Harmonization Suggestions Across Databases`
                         - In this section, include a markdown table with EXACTLY these columns:

                         | Harmonized Concept | Database-Specific Variables | Suggested Harmonized Coding | Notes / Caveats |
                         |---|---|---|---|

                         - STRICT SCOPE: This section is ONLY for cross-database harmonization, never within-database harmonization
                         - Build each row only from variables that appear in the provided context
                         - Only include a harmonization row if the harmonized concept has at least one usable variable from EVERY selected database listed under `Names of Selected Databases`
                         - Exclude concepts that are missing in any selected database
                         - For `Database-Specific Variables`, list database groups in the EXACT same order as the databases listed under `Names of Selected Databases`
                         - Every harmonization row MUST explicitly mention ALL selected databases in `Database-Specific Variables`; if any selected database is missing in that row, DROP the row
                         - Use this strict format for `Database-Specific Variables`: "DB1: var_a, var_b; DB2: var_x; DB3: var_m, var_n"
                         - For `Suggested Harmonized Coding`, propose a practical mapping strategy grounded in observed labels/categories
                         - For `Notes / Caveats`, mention comparability risks such as wording differences, wave differences, missing categories, and instrument differences
                         - If the number of selected databases is 1 or less, DO NOT include any harmonization section
                         - If the number of selected databases is greater than 1 but no concept is available across ALL selected databases, include the section with one row stating: "No robust cross-database harmonization candidates found across all selected databases."
                
                     COMBINED EXAMPLE OF CORRECT FORMAT:

                     Use the same structure whether one or multiple databases are selected: one section per database, and within each database merge rows ONLY when Label and Categories are exactly identical.

                     ## Betula
                     In Betula, several variables measure demographics and cognition.

                     | Variable Name | Label | Categories | Source |
                     |---|---|---|---|
                     | ID | Participant identifier | N/A | Betula_Demographics |
                     | age | Age in years | N/A | Betula_Demographics |
                     | memory_score | Score on memory test | N/A | Betula_Cognition |

                     ## SNAC-K
                     In SNAC-K, demographics are recorded with biological sex information.

                     | Variable Name | Label | Categories | Source |
                     |---|---|---|---|
                     | löpnr | Unique participant identifier number | N/A | SNAC_K_Baseline |
                     | kön | Participant's biological sex | 1=man, 2=woman | SNAC_K_Baseline |

                     ## GAS_SNAC_S
                     Within one database, merge only exact Label+Categories matches.

                     | Variable Name | Label | Categories | Source |
                     |---|---|---|---|
                     | FAS2_D40, FAS3_D40, FAS4_D40 | Difficulty dressing | 0=no, 1=yes | GAS_SNAC_S_Wave2, GAS_SNAC_S_Wave3, GAS_SNAC_S_Wave4 |
                     | BAS1_D40, BAS2_D40, BAS3_D40 | Difficulty dressing (last 30 days) | 0=no, 1=yes | GAS_SNAC_S_Wave2, GAS_SNAC_S_Wave3, GAS_SNAC_S_Wave4 |

                     ## SNAC-B
                     Across another database, the same rule applies: exact matches may merge; different categories must stay separate.

                     | Variable Name | Label | Categories | Source |
                     |---|---|---|---|
                     | FAS1_D40, FAS2_D40, FAS3_D40 | Difficulty dressing | 0=no, 1=yes | SNAC_B_Baseline, SNAC_B_FollowUp |
                     | BAS1_D40, BAS2_D40, BAS3_D40 | Difficulty dressing | 0=no, 1=some difficulty, 2=unable | SNAC_B_Baseline, SNAC_B_FollowUp |

                Answer:"""

                    PROMPT = PromptTemplate(
                        template=prompt_template, 
                        input_variables=["cohort_background", "context", "question", "selected_db_count", "selected_db_names"]
                    )

                    # Capture session state values before passing to chain (LangChain runs lambdas in different context)
                    current_llm_model = LLM_MODEL_GPT
                    selected_db_count = len(st.session_state.get("selected_databases") or [])
                    selected_db_names = ", ".join(st.session_state.get("selected_databases") or [])

                    rag_chain = (
                        {
                            "cohort_background": lambda user_query: get_relevant_background(
                                user_query,
                                selected_vectorstores,
                                current_llm_model,
                                context_docs,
                            ),
                            "context": lambda _: context,
                            "selected_db_count": lambda _: selected_db_count,
                            "selected_db_names": lambda _: selected_db_names,
                            "question": RunnablePassthrough()
                        }
                        | PROMPT
                        | llm
                        | StrOutputParser()
                    )

                    try:
                        logger.debug(f"Starting LLM invocation with query: {prompt[:100]}...")
                        logger.debug(f"Context length: {len(context)} chars, {len(context_docs)} docs")
                        logger.debug(f"Using model: {current_llm_model}")
                        response = rag_chain.invoke(prompt)
                        logger.debug(f"LLM Response received (first 500 chars): {response[:500]}")
                    except Exception as e:
                        error_str = str(e)
                        logger.error(f"LLM Error: {error_str}")
                        logger.exception("Full exception traceback:")
                        
                        # Check for token limit exceeded error
                        if "rate_limit_exceeded" in error_str and "tokens" in error_str.lower():
                            st.error(f"⚠️ Request too large for {current_llm_model}. Try asking about a smaller subset of variables.")
                            response = ""
                        elif "rate_limit" in str(e).lower() or "429" in str(e):
                            st.error(f"⚠️ Rate limit reached for {current_llm_model}. Please try again in a few moments.")
                            response = ""
                        else:
                            st.error(f"Error: {str(e)}")
                            response = (
                                "I ran into an issue while generating the answer. "
                                "Please try again or narrow the query."
                            )

        # Prepend database hint to response
        selected_db = st.session_state.get("selected_database")
        
        selected_db_names = st.session_state.get("selected_databases") or []

        # Trim whitespace while preserving the original LLM response text.
        if response:
            response = response.strip()

        # Some models may return empty/whitespace output for no-hit queries.
        # Ensure users always receive an explicit fallback message.
        if not response:
            response = (
                "I could not find directly related variable definitions for that query in the selected database scope. "
                "Try a different keyword or a broader phrasing."
            )
        
        if selected_db_names and response:
            selected_db_label = ", ".join(selected_db_names)
            response_with_hint = f"📍 **{selected_db_label}**\n\n{response}"
        else:
            response_with_hint = response
        
        st.markdown(response_with_hint)
        
        # Extract tables from response and enable download if found
        tables_with_headers = extract_markdown_tables(response)
        st.session_state.latest_tables_with_headers = tables_with_headers
        
        # Add download button if tables were found
        if tables_with_headers:
            excel_file = export_tables_to_excel(tables_with_headers)
            selected_dbs_for_file = st.session_state.get("selected_databases") or []
            if not selected_dbs_for_file:
                db_slug = "no-db"
            elif len(selected_dbs_for_file) == 1:
                db_slug = selected_dbs_for_file[0]
            else:
                db_slug = f"{len(selected_dbs_for_file)}dbs"
            st.download_button(
                label=f"📥 Download as Excel ({len(tables_with_headers)} table{'s' if len(tables_with_headers) > 1 else ''})",
                data=excel_file,
                file_name=f"NEARchatbot_{db_slug}_{pd.Timestamp.now().strftime('%y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_tables"
            )

    st.session_state.messages.append({"role": "assistant", "content": response_with_hint})

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()