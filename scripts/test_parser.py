import os
from xml_parser import parse_xml_to_document, parse_xml_to_documents
from json_parser import parse_json_to_document

# Test XML Parser
print("=" * 60)
print("Testing XML Parser")
print("=" * 60)

xml_file_path = "./data/Betula/Betula_T2.xml"

if os.path.exists(xml_file_path):
    variable_docs = parse_xml_to_documents(xml_file_path, database_name="Betula")
    print(f"Variable-level documents: {len(variable_docs)}")

    def print_variable_example(doc, title, max_chars=1200):
        print(f"\n{title}")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Database: {doc.metadata['database']}")
        print(f"  Table: {doc.metadata['table']}")
        print(f"  Type: {doc.metadata['type']}")
        print(f"  Variable Name: {doc.metadata.get('variable_name', 'N/A')}")
        print("  Content:")
        print(doc.page_content[:max_chars])

    if variable_docs:
        print("\n" + "-" * 60)
        print("General Variable Examples (first 3)")
        print("-" * 60)
        for idx, doc in enumerate(variable_docs[:3], start=1):
            print_variable_example(doc, f"Example {idx}")

        docs_with_categories = [doc for doc in variable_docs if "Categories:" in doc.page_content]
        print("\n" + "-" * 60)
        print(f"Variables with Categories: {len(docs_with_categories)}")
        print("Category-rich Examples (first 3)")
        print("-" * 60)
        for idx, doc in enumerate(docs_with_categories[:3], start=1):
            print_variable_example(doc, f"Category Example {idx}")

    # Backward-compatibility check for the legacy combined parser
    combined_doc = parse_xml_to_document(xml_file_path, database_name="Betula")
    print("\nLegacy Combined Document Metadata:")
    print(f"  Source: {combined_doc.metadata['source']}")
    print(f"  Database: {combined_doc.metadata['database']}")
    print(f"  Table: {combined_doc.metadata['table']}")
    print(f"  Type: {combined_doc.metadata['type']}")
    print("\nLegacy Combined Content (first 800 chars):")
    print(combined_doc.page_content[:800])
else:
    print(f"File not found: {xml_file_path}")

# Test JSON Parser
print("\n" + "=" * 60)
print("Testing JSON Parser")
print("=" * 60)

json_file_path = "./data/Betula/betula.json"

if os.path.exists(json_file_path):
    doc = parse_json_to_document(json_file_path, database_name="Betula")
    print("Document Metadata:")
    print(f"  Source: {doc.metadata['source']}")
    print(f"  Database: {doc.metadata['database']}")
    print(f"  Type: {doc.metadata['type']}")
    print("\nDocument Content (relevant descriptions and events):")
    print("-" * 60)
    print(doc.page_content)
else:
    print(f"File not found: {json_file_path}")
    
