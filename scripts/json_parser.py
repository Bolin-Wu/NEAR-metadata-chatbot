import os
import json
import re
from langchain_core.documents import Document


def clean_html(text: str) -> str:
    """Remove HTML tags and entities from text."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Decode common HTML entities
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&apos;', "'")
    text = text.replace('&#39;', "'")
    text = text.replace('&nbsp;', ' ')
    text = text.replace('\\r\\n', ' ')
    text = text.replace('\\n', ' ')
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_relevant_content(json_data: dict) -> str:
    """Extract only relevant descriptions and data collection events from JSON.
    
    Deduplicates identical descriptions to optimize for vectorstore training.
    """
    content_parts = []
    
    # Extract main description if available
    if "objectives" in json_data and json_data["objectives"]:
        for obj in json_data["objectives"]:
            if "value" in obj:
                cleaned_value = clean_html(obj['value'])
                content_parts.append(f"Objectives: {cleaned_value}")
    
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
                    cleaned_desc = clean_html(description)
                    content_parts.append(f"Description: {cleaned_desc}")
            
            # Get data collection events with deduplication
            if "dataCollectionEvents" in pop and pop["dataCollectionEvents"]:
                events_content = []
                
                for event in pop["dataCollectionEvents"]:
                    event_name = "Unknown Event"
                    if "name" in event and event["name"]:
                        event_name = event["name"][0].get("value", "Unknown Event")
                    
                    # Get event description
                    event_desc = ""
                    if "description" in event and event["description"]:
                        event_desc = event["description"][0].get("value", "")
                    
                    # Clean HTML from description
                    cleaned_event_desc = clean_html(event_desc) if event_desc else ""
                    
                    # Create a normalized key for deduplication (lowercase, strip whitespace)
                    desc_key = cleaned_event_desc.lower().strip() if cleaned_event_desc else ""
                    
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
                        events_content.append(f"  - {event_name}")
                        if cleaned_event_desc:
                            events_content.append(f"    Description: {cleaned_event_desc}")
                        if date_range:
                            events_content.append(f"    Period: {date_range}")
                        events_content.append("")
                        
                        if desc_key:
                            seen_descriptions[desc_key] = {
                                "events": [{
                                    "name": event_name,
                                    "dates": date_range
                                }]
                            }
                
                # Only add events section if there are events to display
                if events_content:
                    content_parts.append("\nData Collection Events:")
                    content_parts.extend(events_content)
        
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
