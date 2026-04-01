import os
import xml.etree.ElementTree as ET
from langchain_core.documents import Document


def parse_xml_to_text(file_path: str) -> str:
    """Parse XML file and extract text content."""
    with open(file_path, 'r', encoding='utf-8') as f:
        tree = ET.parse(f)
    root = tree.getroot()
    text = f"File: {os.path.basename(file_path)}\n"
    text += "=" * 80 + "\n\n"
    
    # Attributes to skip
    skip_attrs = {'Administrative_information', 'Source', 'Target'}
    
    def extract_element_text(element, indent=0):
        nonlocal text
        prefix = "  " * indent
        
        if element.tag == 'variable':
            name = element.get('name', 'unknown')
            
            text += f"{prefix}Variable: {name}\n"
            
            # Extract attributes directly (label, original_table, Target, Source, etc.)
            attributes_elem = element.find('attributes')
            if attributes_elem is not None:
                for attr in attributes_elem.findall('attribute'):
                    attr_name = attr.get('name', 'unknown')
                    if attr_name not in skip_attrs:
                        attr_value = attr.text or 'N/A'
                        # Capitalize attribute name for display
                        display_name = attr_name.replace('_', ' ').title()
                        text += f"{prefix}  {display_name}: {attr_value}\n"
            
            # Extract categories with their labels and values
            categories_elem = element.find('categories')
            if categories_elem is not None:
                text += f"{prefix}  Categories:\n"
                for cat in categories_elem.findall('category'):
                    cat_name = cat.get('name', 'unknown')
                    cat_attrs = cat.find('attributes')
                    cat_label = 'N/A'
                    if cat_attrs is not None:
                        label_attr = cat_attrs.find("attribute[@name='label']")
                        if label_attr is not None:
                            cat_label = label_attr.text or 'N/A'
                    text += f"{prefix}    - Value {cat_name}: {cat_label}\n"
            
            text += "\n"
        
        
        for child in element:
            extract_element_text(child, indent)
    
    # Extract all variables
    for var in root.findall('variable'):
        extract_element_text(var)
    
    # Clean up formatting - add newlines between variables for better chunking
    lines = text.split('\n')
    cleaned_lines = []
    for i, line in enumerate(lines):
        cleaned_lines.append(line)
        # Add blank line before new Variable entries (but not the first one)
        if line.startswith('Variable:') and i > 0:
            cleaned_lines.insert(-1, '')
    
    cleaned_text = '\n'.join(cleaned_lines).strip()
    return cleaned_text


def _extract_variable_document(variable, file_path: str, database_name: str = None) -> Document:
    """Convert one <variable> XML element into one LangChain Document."""
    skip_attrs = {'Administrative_information', 'Source', 'Target'}

    variable_name = variable.get('name', 'unknown')
    lines = [f"Variable: {variable_name}"]

    table_name = "unknown"

    attributes_elem = variable.find('attributes')
    if attributes_elem is not None:
        for attr in attributes_elem.findall('attribute'):
            attr_name = attr.get('name', 'unknown')
            if attr_name in skip_attrs:
                continue

            attr_value = attr.text or 'N/A'
            display_name = attr_name.replace('_', ' ').title()
            lines.append(f"  {display_name}: {attr_value}")

            if attr_name == 'original_table':
                table_name = attr_value.strip() if attr_value else "unknown"

    categories_elem = variable.find('categories')
    if categories_elem is not None:
        lines.append("  Categories:")
        for cat in categories_elem.findall('category'):
            cat_name = cat.get('name', 'unknown')
            cat_attrs = cat.find('attributes')
            cat_label = 'N/A'
            if cat_attrs is not None:
                label_attr = cat_attrs.find("attribute[@name='label']")
                if label_attr is not None:
                    cat_label = label_attr.text or 'N/A'
            lines.append(f"    - Value {cat_name}: {cat_label}")

    page_content = "\n".join(lines)
    return Document(
        page_content=page_content,
        metadata={
            "source": os.path.basename(file_path),
            "database": database_name or "unknown",
            "table": table_name,
            "type": "variable_definitions",
            "variable_name": variable_name,
        }
    )


def parse_xml_to_documents(file_path: str, database_name: str = None):
    """Parse XML file and return one Document per variable.

    This is the preferred parser for vector indexing because it guarantees
    variable-level chunk boundaries.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        tree = ET.parse(f)

    root = tree.getroot()
    docs = []
    for variable in root.findall('variable'):
        docs.append(_extract_variable_document(variable, file_path, database_name))

    return docs


def extract_table_name(text: str) -> str:
    """Extract table name from parsed XML text."""
    table_name = "unknown"
    for line in text.split('\n'):
        if "Original Table:" in line:
            table_name = line.split("Original Table:")[-1].strip()
            break
    return table_name


def parse_xml_to_document(file_path: str, database_name: str = None) -> Document:
    """Parse XML file and create a single combined LangChain Document.

    Kept for backward compatibility in utility scripts. Training should use
    parse_xml_to_documents() for one-variable-per-document indexing.
    """
    text = parse_xml_to_text(file_path)
    table_name = extract_table_name(text)
    
    doc = Document(
        page_content=text,
        metadata={
            "source": os.path.basename(file_path),
            "database": database_name or "unknown",
            "table": table_name,
            "type": "variable_definitions"
        }
    )
    
    return doc