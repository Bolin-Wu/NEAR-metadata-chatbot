import os
import sys
from pathlib import Path
import tempfile
import re
from io import BytesIO

import streamlit as st
import pandas as pd
from openpyxl import Workbook
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
    """Download production database from HuggingFace Hub if not present locally."""
    # Check if database already exists and is populated
    if os.path.exists(CHROMA_DB) and os.listdir(CHROMA_DB):
        return  # Already have local copy
    
    if not HUGGINGFACE_REPO_ID:
        st.error("❌ HUGGINGFACE_REPO_ID not configured. Please contact the maintainer.")
        st.stop()
    
    try:
        from huggingface_hub import hf_hub_download
        
        progress_placeholder = st.empty()
        progress_placeholder.info("📥 Downloading production database from HuggingFace Hub...")
        
        # Download zip from HuggingFace Hub
        zip_path = hf_hub_download(
            repo_id=HUGGINGFACE_REPO_ID,
            filename="chroma_prod_db.zip",
            repo_type="dataset",
            cache_dir=Path.home() / ".cache" / "huggingface"
        )
        
        # Extract to temp directory
        progress_placeholder.info("📦 Extracting database...")
        temp_dir = tempfile.mkdtemp()
        
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # The zip contains the contents of chroma_prod_db directly
            # (not wrapped in a chroma_prod_db folder)
            if os.path.exists(CHROMA_DB):
                shutil.rmtree(CHROMA_DB)
            
            # Move extracted contents to CHROMA_DB
            shutil.move(temp_dir, CHROMA_DB)
            progress_placeholder.success("✅ Database downloaded and ready!")
        except Exception as e:
            progress_placeholder.error(f"❌ Failed to extract: {e}")
            st.error(f"Failed to extract database: {e}")
            st.stop()
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        
    except ImportError:
        st.error("❌ Install huggingface_hub: `pip install huggingface_hub`")
        st.stop()
    except Exception as e:
        st.error(f"❌ Could not download from HuggingFace: {e}")
        st.stop()

@st.cache_resource(show_spinner="Preparing embeddings model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# Known databases (in case data/ folder not available)
def get_available_databases():
    """Get list of available databases by querying Chroma collections.
    
    Works even if ./data folder is not present (e.g., on Streamlit Cloud).
    """
    databases = []
    
    try:
        # Check if the Chroma directory exists
        if not os.path.exists(CHROMA_DB):
            st.error(f"❌ Chroma directory not found: {CHROMA_DB}")
            st.stop()
        
        embeddings = get_embeddings()
        
        # Check each known database to see if its collection exists in Chroma
        for db_name in KNOWN_DATABASES:
            collection_name = f"{db_name.lower()}_metadata"
            try:
                vectorstore = Chroma(
                    persist_directory=CHROMA_DB,
                    embedding_function=embeddings,
                    collection_name=collection_name
                )
                count = vectorstore._collection.count()
                if count > 0:
                    databases.append(db_name)
            except Exception as e:
                pass
        
        return sorted(databases)
    except Exception as e:
        st.warning(f"Could not retrieve available databases: {e}")
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
    """Retrieve database description from the loaded vector store collection.
    
    Since each collection is for a single database, we just need to find the 
    database_description document in the current collection.
    """
    if vectorstore is None:
        return ""
    
    try:
        collection = vectorstore._collection
        all_docs = collection.get()
        
        if all_docs and all_docs.get('metadatas'):
            for i, metadata in enumerate(all_docs['metadatas']):
                if metadata.get("type") == "database_description":
                    # Return the content of this document
                    if all_docs.get('documents') and i < len(all_docs['documents']):
                        return all_docs['documents'][i]
        
        return ""
        
    except Exception as e:
        st.warning(f"Could not retrieve database description: {e}")
        return ""

def get_relevant_background(query, vectorstore):
    """Retrieve cohort background information dynamically based on the user's query.
    
    This function performs semantic search on database_description documents to
    find background information most relevant to the user's question, making the
    background context adaptive rather than static.
    
    Args:
        query: User's question
        vectorstore: Chroma vector store (already filtered to selected database)
    
    Returns:
        str: Relevant cohort background text
    """
    if vectorstore is None:
        return ""
    
    try:
        # Perform semantic search on the entire collection
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(query)
        
        background_docs = []
        
        for doc in docs:
            # Prioritize database_description documents, but include other relevant ones
            if doc.metadata.get("type") == "database_description":
                background_docs.append(doc)
        
        # If no specific database descriptions found, try to get the general description
        if not background_docs:
            general_background = get_database_description(vectorstore)
            if general_background:
                return general_background
            return ""
        
        # Combine multiple relevant background passages for richer context
        # Use top results (limit to 2 to avoid overwhelming the LLM)
        combined_background = "\n\n---\n\n".join(
            [doc.page_content for doc in background_docs[:2]]
        )
        
        return combined_background if combined_background else ""
        
    except Exception as e:
        # Fallback to general description on error
        return get_database_description(vectorstore)

def filter_and_organize_context(query, vectorstore):
    """Retrieve and organize context - only variable definitions.
    
    Since each collection is for a single database, we just need to:
    1. Do semantic search
    2. Filter for variable_definitions type
    3. Return up to 20 results
    
    Args:
        query: User's question
        vectorstore: Chroma vector store (already filtered to selected database)
    
    Returns:
        tuple: (context_text, full_docs_list)
    """
    if vectorstore is None:
        return "", []
    
    try:
        # Perform semantic search on the selected database collection
        retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
        docs = retriever.invoke(query)
        
        var_defs = []
        
        for doc in docs:
            # Only include variable definitions
            if doc.metadata.get("type") == "variable_definitions":
                var_defs.append(doc)
        
        # Return variable definitions and full docs list
        context_text = "\n\n---\n\n".join([doc.page_content for doc in var_defs])
        return context_text, var_defs
    except Exception as e:
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
**ℹ️ Disclaimer:** The information provided here is for reference purposes only. 
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
                | Variable Name | Label | Categories |
                |---|---|---|
                | variable_name | What it measures in plain English | Category values if applicable |
                
                CRITICAL RULES FOR EXTRACTING VARIABLE NAMES:
                - Column 1: EXTRACT EXACTLY the text that appears after "Variable: " in the source data
                - Do NOT use any other field names, labels, or descriptions
                - Do NOT use text from fields like "Description:" or "Sociodemographic Economic Characteristics:"
                - EXAMPLE: If you see "Variable: age_HT_rounded", MUST write "age_HT_rounded" in Column 1
                - EXAMPLE: If you see "Variable: löpnr", MUST write "löpnr" in Column 1
                - NEVER modify, shorten, or translate the variable name
                - NEVER invent variable names - only extract exactly what follows "Variable: "
                
                IMPORTANT NOTE ON VARIABLE AVAILABILITY:
                For information about which cohorts and tables contain these variables, please refer to the Maelstrom catalogue at: https://www.maelstrom-research.org/
                
                EXAMPLE OF CORRECT FORMAT:
                "In SNAC-K, several variables measure basic demographics. Participants are identified by a unique proband number (löpnr). The cohort includes both men and women, tracked through a sex variable. Birth dates are recorded to calculate age.

                | Variable Name | Label | Categories |
                |---|---|---|
                | löpnr | Unique participant identifier | N/A (unique ID) |
                | kön | Participant's biological sex | 1=man, 2=woman |
                | Birthday | Date of birth for age calculation | Date format |
                

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