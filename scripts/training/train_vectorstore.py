import os
import shutil
import glob
import argparse
from datetime import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import streamlit as st

from xml_parser import parse_xml_to_documents
from json_parser import parse_json_to_document
import time

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"
AZURE_EMBEDDING_API_VERSION = "2024-02-01"
CHROMA_DB = "./chroma_azure_db"         # Production database
DATA_ROOT = "./data"
BACKUP_ROOT = "./backup"
try:
    AZURE_api_key = st.secrets["AZURE_api_key"]
except (FileNotFoundError, KeyError, AttributeError):
    AZURE_api_key = os.getenv("AZURE_api_key")

try:
    AZURE_openai_endpoint = st.secrets["AZURE_openai_endpoint"]
except (FileNotFoundError, KeyError, AttributeError):
    AZURE_openai_endpoint = os.getenv("AZURE_openai_endpoint")

# Chunk configuration
# XML now uses one variable per document via parse_xml_to_documents()
JSON_CHUNK_SIZE = 1200
JSON_CHUNK_OVERLAP = 200

# Functions ─────────────────────────────────────────────────────────────────
def get_embeddings():
    return AzureOpenAIEmbeddings(
        azure_deployment=EMBEDDING_MODEL,
        azure_endpoint=AZURE_openai_endpoint,
        api_key=AZURE_api_key,
        api_version=AZURE_EMBEDDING_API_VERSION,
    )

def parse_args():
    """Parse command line arguments for retraining."""
    parser = argparse.ArgumentParser(
        description="Train Chroma vector store from NEAR metadata folders."
    )
    parser.add_argument(
        "--data-root",
        default=DATA_ROOT,
        help="Root folder containing one subfolder per database (default: ./data)",
    )
    parser.add_argument(
        "--target-db",
        default=CHROMA_DB,
        help="Chroma persist directory (default: ./chroma_azure_db)",
    )
    parser.add_argument(
        "--embedding-model",
        default=EMBEDDING_MODEL,
        help="Azure embedding deployment name (default: text-embedding-3-small)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmations",
    )
    parser.add_argument(
        "--backup-existing",
        action="store_true",
        help="Backup target DB before deleting when retraining",
    )
    parser.add_argument(
        "--backup-root",
        default=BACKUP_ROOT,
        help="Directory where backups are stored (default: ./backup)",
    )
    return parser.parse_args()

def get_available_databases(data_root: str):
    """Get list of available databases by scanning the data directory."""
    if not os.path.exists(data_root):
        return []
    
    # Get subdirectories in data_root (each subdirectory is a database)
    databases = []
    for item in os.listdir(data_root):
        item_path = os.path.join(data_root, item)
        if os.path.isdir(item_path):
            databases.append(item)
    
    return sorted(databases)

def maybe_backup_directory(path: str, backup_root: str):
    """Create a timestamped backup of an existing directory."""
    os.makedirs(backup_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = os.path.basename(os.path.normpath(path))
    backup_path = os.path.join(backup_root, f"{folder_name}_backup_{timestamp}")
    shutil.copytree(path, backup_path)
    print(f"Backed up existing DB to {backup_path}")

def process_database_to_vectorstore(data_dir: str, database_name: str = None):
    """Process XML and JSON files from the specified database directory and create vector store."""
    xml_files = glob.glob(os.path.join(data_dir, "*.xml"))
    documents = []
    
    prefix = f"[{database_name}] " if database_name else ""
    print(f"\n{prefix}Processing {data_dir}...")
    
    # Initialize text splitter for JSON documents only
    # JSON (database description) uses larger chunks and focuses on paragraph breaks
    json_splitter = RecursiveCharacterTextSplitter(
        chunk_size=JSON_CHUNK_SIZE,  # Larger chunks for coherent descriptive text
        chunk_overlap=JSON_CHUNK_OVERLAP,
        separators=[
            "\n\n",  # Paragraph breaks
            "Data Collection Events:",  # Match actual output from json_parser
            "Summary of Shared Protocols",
            "Population:",
            "\n",
            " "
        ]
    )
    
    # Find and add the JSON description file
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    
    if not json_files:
        print(f"  ⚠ No JSON files found in {data_dir}")
    
    for json_file in json_files:
        try:
            doc = parse_json_to_document(json_file, database_name=database_name)
            # Split JSON documents with the JSON-optimized splitter
            json_chunks = json_splitter.split_documents([doc])
            documents.extend(json_chunks)
            print(f"  ✓ Loaded {os.path.basename(json_file)} ({len(json_chunks)} chunks)")
        except Exception as e:
            print(f"  ✗ Could not read {json_file}: {e}")
    
    if not xml_files and not json_files:
        raise FileNotFoundError(f"No XML or JSON files found in {data_dir}")
    
    if not xml_files:
        print(f"  ⚠ No XML files found in {data_dir}")
    
    # Process XML files
    for file_path in xml_files:
        file_name = os.path.basename(file_path)
        
        try:
            variable_docs = parse_xml_to_documents(file_path, database_name=database_name)
            documents.extend(variable_docs)
            print(f"  ✓ Loaded {file_name} ({len(variable_docs)} variable-docs)")
        except Exception as e:
            print(f"  ✗ Could not parse {file_name}: {e}")
    
    if not documents:
        raise FileNotFoundError(f"No valid documents could be processed from {data_dir}")
    
    print(f"  Total chunks: {len(documents)}")
    
    return documents

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    data_root = args.data_root
    target_db = args.target_db
    backup_root = args.backup_root

    print("=" * 60)
    print("NEAR Metadata Vector Store Training Script")
    print("=" * 60)
    print(f"Data root: {os.path.abspath(data_root)}")
    print(f"Target DB: {os.path.abspath(target_db)}")
    print(f"Backup root: {os.path.abspath(backup_root)}")
    print(f"Embedding model: {args.embedding_model}")
    
    databases = get_available_databases(data_root)
    
    if not databases:
        print(f"✗ No databases found in {data_root}")
        exit(1)
    
    # Train all databases to production
    selected_databases = databases
    
    print(f"\nMode: Train all databases")
    print(f"Databases ({len(databases)}): {', '.join(databases)}")
    print(f"Target: {target_db}")
    
    confirm = "y" if args.yes else input(f"\nProceed with training? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("Training cancelled.")
        exit(0)
    
    # Clear existing vector store if it exists to avoid duplicate/mixed collections.
    if os.path.exists(target_db):
        if args.backup_existing:
            maybe_backup_directory(target_db, backup_root)

        if args.yes:
            shutil.rmtree(target_db)
            print(f"Deleted existing target DB: {target_db}")
        else:
            confirm_clear = input(
                f"\n{target_db} already exists. Delete it before retraining? (y/n): "
            ).strip().lower()
            if confirm_clear == 'y':
                shutil.rmtree(target_db)
                print(f"Deleted {target_db}")
            else:
                print("Cancelled. Keeping existing DB unchanged.")
                exit(0)
    
    # Train vector store (separate collection for each database)
    try:
        script_start = time.time()
        embeddings = get_embeddings(args.embedding_model)
        print(f"\nCreating embeddings and storing in production database...")
        print(f"Each database will have its own collection.\n")
        
        total_items = 0
        
        for db in selected_databases:
            db_start = time.time()
            data_dir = os.path.join(data_root, db)
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
            db_elapsed = time.time() - db_start
            print(f"    ✓ Collection '{collection_name}' created with {items_count} items ({db_elapsed:.1f}s)")
            total_items += items_count
        
        total_elapsed = time.time() - script_start
        print(f"\n✓ Training complete!")
        print(f"Total items across all collections: {total_items}")
        print(f"Total time: {total_elapsed:.1f}s")
        print(f"Vector store saved to: {os.path.abspath(target_db)}")
        print(f"Collections created: {', '.join([f'{db.lower()}_metadata' for db in selected_databases])}")
        print("\n💡 Next step: Compress and upload chroma_azure_db to HuggingFace Hub")
        archive_name = os.path.basename(os.path.normpath(target_db))
        print(
            f"   python -c \"import shutil; shutil.make_archive('{archive_name}', 'zip', '.', '{archive_name}')\""
        )
        print(
            f"   hf upload bobo200612/near-chroma-prod-db ./{archive_name}.zip {archive_name}.zip --repo-type=dataset --commit-message 'xxxxx'"
        )
            
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        exit(1)