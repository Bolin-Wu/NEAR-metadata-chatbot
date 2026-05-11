import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── Configuration ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB = "./chroma_prod_db"
DATABASE = "SNAC-N"  # Change this to inspect different collections

# ── Main ──────────────────────────────────────────────────────────────────────

print("=" * 70)
print(f"INSPECTING COLLECTION: {DATABASE}")
print("=" * 70)

try:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    collection_name = f"{DATABASE.lower()}_metadata"
    
    vectorstore = Chroma(
        persist_directory=CHROMA_DB,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    
    collection = vectorstore._collection
    
    # Get all documents
    all_docs = collection.get()
    total_count = collection.count()
    
    print(f"\n✓ Collection loaded: {collection_name}")
    print(f"✓ Total items: {total_count}")
    
    # Count document types
    print(f"\n--- Document Types ---")
    doc_types = {}
    if all_docs and all_docs.get('metadatas'):
        for metadata in all_docs['metadatas']:
            doc_type = metadata.get("type", "unknown")
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        for doc_type, count in sorted(doc_types.items()):
            print(f"  {doc_type}: {count}")
    
    # Show database description
    print(f"\n--- Database Description ---")
    if all_docs and all_docs.get('metadatas'):
        for i, metadata in enumerate(all_docs['metadatas']):
            if metadata.get("type") == "database_description":
                if all_docs.get('documents') and i < len(all_docs['documents']):
                    description = all_docs['documents'][i]
                    # Print first 500 chars
                    print(description[:500] + "..." if len(description) > 500 else description)
                break
    
    # Show sample variables
    print(f"\n--- Sample Variables (first 5) ---")
    if all_docs and all_docs.get('documents') and all_docs.get('metadatas'):
        count = 0
        for i, metadata in enumerate(all_docs['metadatas']):
            if metadata.get("type") == "variable_definitions" and count < 5:
                if i < len(all_docs['documents']):
                    doc = all_docs['documents'][i]
                    print(f"\n[Sample {count+1}]")
                    print(doc[:300] + "..." if len(doc) > 300 else doc)
                    count += 1
    
    # Show metadata fields
    print(f"\n--- Sample Metadata ---")
    if all_docs and all_docs.get('metadatas'):
        for i in range(min(3, len(all_docs['metadatas']))):
            print(f"\n[Item {i+1}]")
            print(all_docs['metadatas'][i])
    
    print(f"\n✓ Inspection complete!")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
