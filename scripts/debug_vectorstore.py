"""Debug script to inspect what's in the vector store."""

import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DEMO_DB = "./chroma_demo_db"
CHROMA_PROD_DB = "./chroma_prod_db"      # Production (cloud storage)

print("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

print("Loading vector store...")
vectorstore = Chroma(
    persist_directory=CHROMA_PROD_DB,
    embedding_function=embeddings,
    collection_name="xml_metadata"
)

total = vectorstore._collection.count()
print(f"Total documents in vector store: {total}\n")

# Query for database descriptions specifically
print("Searching for database_description documents...")
try:
    desc_docs = vectorstore.similarity_search("cohort description background", k=50)
    
    # Filter for actual database descriptions
    db_descriptions = [doc for doc in desc_docs if doc.metadata.get("type") == "database_description"]
    
    print(f"\nFound {len(db_descriptions)} database description documents:\n")
    
    databases_with_desc = {}
    for doc in db_descriptions:
        db = doc.metadata.get("database", "unknown")
        if db not in databases_with_desc:
            databases_with_desc[db] = True
            print(f"  ✓ {db}")
    
    if not db_descriptions:
        print("  (None found - this might be the issue!)")
    
    # Also list all unique database names in the vector store
    print("\n" + "="*60)
    print("All unique database names in vector store:")
    print("="*60)
    
    # Use smaller k to avoid SQL variable limit
    all_docs = vectorstore.similarity_search("data", k=min(1000, total))
    databases = set()
    for doc in all_docs:
        db = doc.metadata.get("database")
        if db:
            databases.add(db)
    
    print(f"\nFound {len(databases)} unique databases:\n")
    for db in sorted(databases):
        has_desc = "✓ (with description)" if db in databases_with_desc else ""
        print(f"  - {db} {has_desc}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
