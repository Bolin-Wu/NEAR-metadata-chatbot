import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from xml_parser import parse_xml_to_document
from json_parser import parse_json_to_document

data_dir = "./data/SNAC-K"

# Initialize the text splitters (optimized for each type)
xml_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=[
        "\n\n",  # Paragraph breaks first
        "Variable:",  # Then variable definitions
        "\n",  # Then line breaks
        " "
    ]
)

json_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,  # Larger chunks for coherent descriptive text
    chunk_overlap=200,
    separators=[
        "\n\n",  # Paragraph breaks
        "Data Collection",
        "Description:",
        "\n",
        " "
    ]
)

print("=" * 70)
print("TESTING TEXT SPLITTER FOR XML FILES (variable_definitions)")
print("=" * 70)

# Test XML files
xml_files = [f for f in os.listdir(data_dir) if f.endswith('.xml')][:1]

if xml_files:
    file_path = os.path.join(data_dir, xml_files[0])
    doc = parse_xml_to_document(file_path, database_name="SNAC-K")
    
    chunks = xml_splitter.split_documents([doc])
    
    print(f"\nFile: {xml_files[0]}")
    print(f"Total document size: {len(doc.page_content)} chars")
    print(f"Total chunks: {len(chunks)}")
    print(f"Average chunk size: {len(doc.page_content) // len(chunks) if chunks else 0} chars\n")
    for i, chunk in enumerate(chunks[:5]):
        print(f"--- CHUNK {i+1} ({len(chunk.page_content)} chars) ---")
        print(f"Metadata: {chunk.metadata}")
        print(chunk.page_content[:300] + "..." if len(chunk.page_content) > 300 else chunk.page_content)
        print()

print("\n" + "=" * 70)
print("TESTING TEXT SPLITTER FOR JSON FILES (database_description)")
print("=" * 70)

# Test JSON files
json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')][:1]

if json_files:
    file_path = os.path.join(data_dir, json_files[0])
    doc = parse_json_to_document(file_path, database_name="SNAC-K")
    
    chunks = json_splitter.split_documents([doc])
    
    print(f"\nFile: {json_files[0]}")
    print(f"Total document size: {len(doc.page_content)} chars")
    print(f"Total chunks: {len(chunks)}")
    print(f"Average chunk size: {len(doc.page_content) // len(chunks) if chunks else 0} chars\n")
    for i, chunk in enumerate(chunks[:5]):
        print(f"--- CHUNK {i+1} ({len(chunk.page_content)} chars) ---")
        print(f"Metadata: {chunk.metadata}")
        print(chunk.page_content[:300] + "..." if len(chunk.page_content) > 300 else chunk.page_content)
        print()
else:
    print("\nNo JSON files found in", data_dir)
        