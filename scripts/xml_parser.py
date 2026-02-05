import os
import xml.etree.ElementTree as ET
# import json

def parse_xml_to_text(file_path: str) -> str:
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
            value_type = element.get('valueType', 'unknown')
            
            text += f"{prefix}Variable: {name}\n"
            text += f"{prefix}  ValueType: {value_type}\n"
            
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
        
        elif element.tag == 'entity':
            name = element.get('name', 'unknown')
            value_type = element.get('valueType', 'unknown')
            
            text += f"{prefix}Entity: {name}\n"
            text += f"{prefix}  ValueType: {value_type}\n"
            
            # Extract attributes directly
            attributes_elem = element.find('attributes')
            if attributes_elem is not None:
                for attr in attributes_elem.findall('attribute'):
                    attr_name = attr.get('name', 'unknown')
                    if attr_name not in skip_attrs:
                        attr_value = attr.text or 'N/A'
                        display_name = attr_name.replace('_', ' ').title()
                        text += f"{prefix}  {display_name}: {attr_value}\n"
            
            text += "\n"
        
        else:
            # Skip non-variable/entity root elements or recurse if needed
            pass
        
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