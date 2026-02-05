import os
import shutil
import glob

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from xml_parser import parse_xml_to_text

# ── Configuration ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "./chroma_xml_db"
DATA_ROOT = "./data"
TRAIN_MODE = "specific"  # "specific" or "all"
SELECTED_DATABASE = "SNAC-K"  # Only used if TRAIN_MODE == "specific"

# ── Functions ─────────────────────────────────────────────────────────────────
def get_embeddings():
    """Initialize embeddings model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def get_available_databases():
    """Get list of available databases in the data folder."""
    if not os.path.exists(DATA_ROOT):
        return []
    
    databases = []
    for item in os.listdir(DATA_ROOT):
        item_path = os.path.join(DATA_ROOT, item)
        if os.path.isdir(item_path):
            # Check if directory contains XML or JSON files
            has_xml = len(glob.glob(os.path.join(item_path, "*.xml"))) > 0
            has_json = len(glob.glob(os.path.join(item_path, "*.json"))) > 0
            if has_xml or has_json:
                databases.append(item)
    
    return sorted(databases)

def process_xmls_to_vectorstore(data_dir: str, database_name: str = None):
    """Process XML files from the specified directory and create vector store."""
    xml_files = glob.glob(os.path.join(data_dir, "*.xml"))
    documents = []
    
    prefix = f"[{database_name}] " if database_name else ""
    print(f"\n{prefix}Processing {data_dir}...")
    
    # Find and add the JSON description file (handle different naming conventions)
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                json_text = f.read()
            doc = Document(
                page_content=f"Database Description: {json_text}",
                metadata={ "source": os.path.basename(json_file), "database": database_name or "unknown"}
            )
            documents.append(doc)
            print(f"  ✓ Loaded {os.path.basename(json_file)}")
        except Exception as e:
            print(f"  ✗ Could not read {json_file}: {e}")
    
    if not xml_files and not documents:
        raise FileNotFoundError(f"No XML or JSON files found in {data_dir}")
    
    # Process XML files
    for file_path in xml_files:
        file_name = os.path.basename(file_path)
        
        try:
            text = parse_xml_to_text(file_path)  # Already cleaned by parser
            
            doc = Document(
                page_content=text,
                metadata={"source": file_name, "database": database_name or "unknown"}
            )
            documents.append(doc)
            print(f"  ✓ Loaded {file_name}")
        except Exception as e:
            print(f"  ✗ Could not parse {file_name}: {e}")
    
    if not documents:
        raise FileNotFoundError(f"No valid documents could be processed from {data_dir}")
    
    print(f"  Total documents: {len(documents)}")
    
    # Split into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,  # Updated from 500
        chunk_overlap=100,  # Updated from 50
        separators=["Variable:", "\n\n", "\n", " "]  # Added space as fallback
    )
    chunks = text_splitter.split_documents(documents)
    print(f"  Total chunks: {len(chunks)}")
    
    return chunks

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("NEAR Metadata Vector Store Training Script")
    print("=" * 60)
    
    # Get available databases
    databases = get_available_databases()
    
    if not databases:
        print(f"\n✗ No databases found in {DATA_ROOT}")
        exit(1)
    
    # Determine which databases to train on
    if TRAIN_MODE == "specific":
        if SELECTED_DATABASE not in databases:
            print(f"\n✗ Database '{SELECTED_DATABASE}' not found in {DATA_ROOT}")
            print(f"Available databases: {', '.join(databases)}")
            exit(1)
        selected_databases = [SELECTED_DATABASE]
        print(f"\nMode: Train on specific database")
        print(f"Database: {SELECTED_DATABASE}")
    elif TRAIN_MODE == "all":
        selected_databases = databases
        print(f"\nMode: Train on all databases")
        print(f"Databases ({len(databases)}): {', '.join(databases)}")
    else:
        print(f"\n✗ Invalid TRAIN_MODE: {TRAIN_MODE}. Use 'specific' or 'all'")
        exit(1)
    
    confirm = input("\nProceed with training? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("Training cancelled.")
        exit(0)
    
    # Clear existing vector store if it exists
    if os.path.exists(CHROMA_DIR):
        confirm_clear = input(f"\n{CHROMA_DIR} already exists. Delete it? (y/n): ").strip().lower()
        if confirm_clear == 'y':
            shutil.rmtree(CHROMA_DIR)
            print(f"Deleted {CHROMA_DIR}")
        else:
            print("Cancelled. Existing vector store will be overwritten.")
    
    # Train vector store
    try:
        all_chunks = []
        
        for db in selected_databases:
            data_dir = os.path.join(DATA_ROOT, db)
            chunks = process_xmls_to_vectorstore(data_dir, database_name=db)
            all_chunks.extend(chunks)
        
        print(f"\nTotal chunks to embed: {len(all_chunks)}")
        
        # Create embeddings and store
        print("Creating embeddings and storing in Chroma...")
        embeddings = get_embeddings()
        vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
            collection_name="xml_metadata"
        )
        
        print(f"\n✓ Training complete!")
        print(f"Vector store created with {vectorstore._collection.count()} items")
        print(f"Vector store saved to: {os.path.abspath(CHROMA_DIR)}")
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        exit(1)