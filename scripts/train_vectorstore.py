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
CHROMA_DEMO_DB = "./chroma_demo_db"      # Small demo database (push to GitHub)
CHROMA_PROD_DB = "./chroma_prod_db"      # Large production database (use cloud storage)
DATA_ROOT = "./data"
TRAIN_MODE = "specific"  # "specific" or "all"
SELECTED_DATABASE = "SNAC-K"  # Only used if TRAIN_MODE == "specific"

# Functions ─────────────────────────────────────────────────────────────────
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
    
    # Find and add the JSON description file
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                json_text = f.read()
            doc = Document(
                page_content=f"Database Description: {json_text}",
                metadata={
                    "source": os.path.basename(json_file), 
                    "database": database_name or "unknown",
                    "type": "database_description"
                }
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
            text = parse_xml_to_text(file_path)
            
            # Extract table name from text if available
            table_name = "unknown"
            for line in text.split('\n'):
                if "Original Table:" in line:
                    table_name = line.split("Original Table:")[-1].strip()
                    break
            
            doc = Document(
                page_content=text,
                metadata={
                    "source": file_name, 
                    "database": database_name or "unknown",
                    "table": table_name,
                    "type": "variable_definitions"
                }
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
        chunk_size=800,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "Variable:",
            "\n",
            " "
        ]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"  Total chunks: {len(chunks)}")
    
    return chunks

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("NEAR Metadata Vector Store Training Script")
    print("=" * 60)
    
    databases = get_available_databases()
    
    if not databases:
        print(f"\n✗ No databases found in {DATA_ROOT}")
        exit(1)
    
    # Choose environment (demo or production)
    print("\nChoose deployment environment:")
    print("1. Demo (to push to GitHub) - will overwrite chroma_demo_db")
    print("2. Production (cloud storage) - will create/overwrite chroma_prod_db")
    env_choice = input("Enter choice (1 or 2): ").strip()
    
    if env_choice == "1":
        target_db = CHROMA_DEMO_DB
        env_name = "DEMO"
    elif env_choice == "2":
        target_db = CHROMA_PROD_DB
        env_name = "PRODUCTION"
    else:
        print("Invalid choice.")
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
    
    confirm = input(f"\nProceed with {env_name} training? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("Training cancelled.")
        exit(0)
    
    # Clear existing vector store if it exists
    if os.path.exists(target_db):
        confirm_clear = input(f"\n{target_db} already exists. Delete it? (y/n): ").strip().lower()
        if confirm_clear == 'y':
            shutil.rmtree(target_db)
            print(f"Deleted {target_db}")
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
        
        print(f"Creating embeddings and storing in {env_name} database...")
        embeddings = get_embeddings()
        vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            persist_directory=target_db,
            collection_name="xml_metadata"
        )
        
        print(f"\n✓ {env_name} training complete!")
        print(f"Vector store created with {vectorstore._collection.count()} items")
        print(f"Vector store saved to: {os.path.abspath(target_db)}")
        
        if env_choice == "1":
            print("\n💡 Next step: Push chroma_demo_db to GitHub")
        else:
            print("\n💡 Next step: Upload chroma_prod_db to cloud storage (S3, GCS, etc.)")
            
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        exit(1)