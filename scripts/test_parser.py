import os
from xml_parser import parse_xml_to_text

# Example file path
file_path = "./data/SNAC-K/SNAC-K_Cohort1_Baseline.xml"

if os.path.exists(file_path):
    parsed_text = parse_xml_to_text(file_path)
    print("Parsed Output:")
    print(parsed_text[:4000])  # Print first 1000 chars to see the output
else:
    print(f"File not found: {file_path}")
    print(f"Current working directory: {os.getcwd()}")