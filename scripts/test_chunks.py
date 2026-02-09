import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from xml_parser import parse_xml_to_text

data_dir = "./data/SNAC-K"
xml_files = [f for f in os.listdir(data_dir) if f.endswith('.xml')][:1]

if xml_files:
    file_path = os.path.join(data_dir, xml_files[0])
    cleaned_text = parse_xml_to_text(file_path)  # Already cleaned!
    
    doc = Document(
        page_content=cleaned_text,
        metadata={ "source": os.path.basename(file_path), "database": "SNAC-K"}
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Updated from 500
        chunk_overlap=150,  # Updated from 50
            separators=[
                "\n\n",  # Paragraph breaks first
                "Variable:",  # Then variable definitions
                "\n",  # Then line breaks
                " "
    ]
    )
    chunks = text_splitter.split_documents([doc])
    
    print(f"Total chunks: {len(chunks)}\n")
    for i, chunk in enumerate(chunks[:5]):
        print(f"--- CHUNK {i+1} ({len(chunk.page_content)} chars) ---")
        print(f"Metadata: {chunk.metadata}")
        print(chunk.page_content)
        print()