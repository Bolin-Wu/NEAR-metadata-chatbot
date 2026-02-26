import os
import shutil
import glob

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from xml_parser import parse_xml_to_document
from json_parser import parse_json_to_document

# ── Configuration ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB = "./chroma_prod_db"          # Production database
DATA_ROOT = "./data"

# Functions ─────────────────────────────────────────────────────────────────
def get_embeddings():
    """Initialize embeddings model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def get_available_databases():
    """Get list of available databases by scanning the data directory."""
    if not os.path.exists(DATA_ROOT):
        return []
    
    # Get subdirectories in DATA_ROOT (each subdirectory is a database)
    databases = []
    for item in os.listdir(DATA_ROOT):
        item_path = os.path.join(DATA_ROOT, item)
        if os.path.isdir(item_path):
            databases.append(item)
    
    return sorted(databases)

def process_database_to_vectorstore(data_dir: str, database_name: str = None):
    """Process XML and JSON files from the specified database directory and create vector store."""
    xml_files = glob.glob(os.path.join(data_dir, "*.xml"))
    documents = []
    
    prefix = f"[{database_name}] " if database_name else ""
    print(f"\n{prefix}Processing {data_dir}...")
    
    # Find and add the JSON description file
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    for json_file in json_files:
        try:
            doc = parse_json_to_document(json_file, database_name=database_name)
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
            doc = parse_xml_to_document(file_path, database_name=database_name)
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
    # Train all databases to production
    selected_databases = databases
    target_db = CHROMA_DB
    
    print(f"\nMode: Train all databases")
    print(f"Databases ({len(databases)}): {', '.join(databases)}")
    print(f"Target: {target_db}")
    
    confirm = input(f"\nProceed with training? (y/n): ").strip().lower()
    
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
    
    # Train vector store (separate collection for each database)
    try:
        embeddings = get_embeddings()
        print(f"\nCreating embeddings and storing in production database...")
        print(f"Each database will have its own collection.\n")
        
        total_items = 0
        
        for db in selected_databases:
            data_dir = os.path.join(DATA_ROOT, db)
            chunks = process_database_to_vectorstore(data_dir, database_name=db)
            
            # Create a separate collection for each database
            collection_name = f"{db.lower()}_metadata"
            print(f"\n  Creating collection: {collection_name} with {len(chunks)} chunks...")
            
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=target_db,
                collection_name=collection_name
            )
            
            items_count = vectorstore._collection.count()
            print(f"    ✓ Collection '{collection_name}' created with {items_count} items")
            total_items += items_count
        
        print(f"\n✓ Training complete!")
        print(f"Total items across all collections: {total_items}")
        print(f"Vector store saved to: {os.path.abspath(target_db)}")
        print(f"Collections created: {', '.join([f'{db.lower()}_metadata' for db in selected_databases])}")
        print("\n💡 Next step: Upload chroma_prod_db to HuggingFace Hub")
        print("   huggingface-cli upload bobo200612/near-chroma-prod-db ./chroma_prod_db.zip chroma_prod_db.zip --repo-type=dataset")
            
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        exit(1)