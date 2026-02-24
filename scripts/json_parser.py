import os
import json
from langchain_core.documents import Document


def extract_relevant_content(json_data: dict) -> str:
    """Extract only relevant descriptions and data collection events from JSON.
    
    Deduplicates identical descriptions to optimize for vectorstore training.
    """
    content_parts = []
    
    # Extract main description if available
    if "objectives" in json_data and json_data["objectives"]:
        for obj in json_data["objectives"]:
            if "value" in obj:
                content_parts.append(f"Objectives: {obj['value']}")
    
    # Extract populations and their descriptions
    if "populations" in json_data and json_data["populations"]:
        # Track unique descriptions to avoid duplication
        seen_descriptions = {}
        
        for pop in json_data["populations"]:
            # Get population name
            pop_name = "Unknown Population"
            if "name" in pop and pop["name"]:
                pop_name = pop["name"][0].get("value", "Unknown Population")
            
            content_parts.append(f"\n\nPopulation: {pop_name}")
            
            # Get population description
            if "description" in pop and pop["description"]:
                description = pop["description"][0].get("value", "")
                if description:
                    content_parts.append(f"Description: {description}")
            
            # Get data collection events with deduplication
            if "dataCollectionEvents" in pop and pop["dataCollectionEvents"]:
                content_parts.append("\nData Collection Events:")
                
                for event in pop["dataCollectionEvents"]:
                    event_name = "Unknown Event"
                    if "name" in event and event["name"]:
                        event_name = event["name"][0].get("value", "Unknown Event")
                    
                    # Get event description
                    event_desc = ""
                    if "description" in event and event["description"]:
                        event_desc = event["description"][0].get("value", "")
                    
                    # Create a normalized key for deduplication (lowercase, strip whitespace)
                    desc_key = event_desc.lower().strip() if event_desc else ""
                    
                    # Get event dates
                    start_year = event.get("startYear", "")
                    end_year = event.get("endYear", "")
                    date_range = f"{start_year}-{end_year}" if (start_year or end_year) else ""
                    
                    # Check if we've seen this description before
                    if desc_key and desc_key in seen_descriptions:
                        # Just add the event reference without repeating the description
                        seen_descriptions[desc_key]["events"].append({
                            "name": event_name,
                            "dates": date_range
                        })
                    else:
                        # First time seeing this description
                        content_parts.append(f"\n  - {event_name}")
                        if event_desc:
                            content_parts.append(f"    Description: {event_desc}")
                        if date_range:
                            content_parts.append(f"    Period: {date_range}")
                        
                        if desc_key:
                            seen_descriptions[desc_key] = {
                                "events": [{
                                    "name": event_name,
                                    "dates": date_range
                                }]
                            }
        
        # Add summary of duplicate descriptions
        duplicates = {k: v for k, v in seen_descriptions.items() if len(v["events"]) > 1}
        if duplicates:
            content_parts.append("\n\n--- Summary of Shared Protocols ---")
            for desc_key, info in duplicates.items():
                event_list = ", ".join([
                    f"{e['name']} ({e['dates']})" if e['dates'] else e['name']
                    for e in info["events"]
                ])
                content_parts.append(f"\nShared Protocol: {event_list}")
    
    return "\n".join(content_parts)


def parse_json_to_document(file_path: str, database_name: str = None) -> Document:
    """Parse a JSON file and extract relevant content for a LangChain Document."""
    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Extract only relevant content
    relevant_content = extract_relevant_content(json_data)
    
    doc = Document(
        page_content=relevant_content,
        metadata={
            "source": os.path.basename(file_path),
            "database": database_name or "unknown",
            "type": "database_description"
        }
    )
    
    return doc
