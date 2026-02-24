import json
import sys
from pathlib import Path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_json_structure.py <json_file>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        file_size = Path(json_file).stat().st_size
        print(f"📄 File: {json_file}")
        print(f"📊 Size: {file_size:,} bytes")
        print("=" * 80)
        print()
        
        # Pretty print JSON with 2-space indentation
        formatted = json.dumps(data, indent=2, ensure_ascii=False)
        print(formatted)
    
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON - {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"❌ Error: File not found - {json_file}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
