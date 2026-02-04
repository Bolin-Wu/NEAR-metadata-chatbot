import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from xml_parser import parse_xml_to_text

# Get first XML file
data_dir = "./data/SNAC-K"
xml_files = [f for f in os.listdir(data_dir) if f.endswith('.xml')][:1]

if xml_files:
    file_path = os.path.join(data_dir, xml_files[0])
    text = parse_xml_to_text(file_path)
    
    # Apply the same splitter as in the app
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=0,
        separators=["Variable:", "\n\n", "\n"]
    )
    chunks = splitter.split_text(text)
    
    print(f"Total chunks: {len(chunks)}\n")
    for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
        print(f"--- CHUNK {i+1} ({len(chunk)} chars) ---")
        print(chunk)
        print()