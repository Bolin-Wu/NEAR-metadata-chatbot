import os
from xml_parser import parse_xml_to_document
from json_parser import parse_json_to_document

# Test XML Parser
print("=" * 60)
print("Testing XML Parser")
print("=" * 60)

xml_file_path = "./data/SNAC-K/SNAC-K_Cohort1_Baseline.xml"

if os.path.exists(xml_file_path):
    doc = parse_xml_to_document(xml_file_path, database_name="SNAC-K")
    print("Document Metadata:")
    print(f"  Source: {doc.metadata['source']}")
    print(f"  Database: {doc.metadata['database']}")
    print(f"  Table: {doc.metadata['table']}")
    print(f"  Type: {doc.metadata['type']}")
    print("\nDocument Content (first 1500 chars):")
    print(doc.page_content[:1500])
else:
    print(f"File not found: {xml_file_path}")

# Test JSON Parser
print("\n" + "=" * 60)
print("Testing JSON Parser")
print("=" * 60)

json_file_path = "./data/SNAC-K/snac-k.json"

if os.path.exists(json_file_path):
    doc = parse_json_to_document(json_file_path, database_name="SNAC-K")
    print("Document Metadata:")
    print(f"  Source: {doc.metadata['source']}")
    print(f"  Database: {doc.metadata['database']}")
    print(f"  Type: {doc.metadata['type']}")
    print("\nDocument Content (relevant descriptions and events):")
    print("-" * 60)
    print(doc.page_content)
else:
    print(f"File not found: {json_file_path}")
    
