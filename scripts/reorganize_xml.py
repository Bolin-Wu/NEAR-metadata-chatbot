"""
File Structure Reorganization Script (Run Once Only)

This script reorganizes XML files exported from Maelstrom into a cleaner structure.
It processes all databases and removes unnecessary files (entities.xml) while 
renaming variables.xml to match the database name for better organization.

⚠️  IMPORTANT: This script should only be run ONCE after initial data export from Maelstrom.
    Running it multiple times may cause data loss or unexpected behavior.

After running, verify the data/[database]/ structure is organized as expected before
proceeding with vectorstore training.
"""

import os
import shutil

DATA_ROOT = "./data"

# Iterate through all subdirectories in DATA_ROOT
for database in os.listdir(DATA_ROOT):
    database_path = os.path.join(DATA_ROOT, database)
    
    # Skip if it's not a directory
    if not os.path.isdir(database_path):
        continue
    
    print(f"\nProcessing database: {database}")
    
    # List all subdirectories in the database folder
    for subfolder in os.listdir(database_path):
        subfolder_path = os.path.join(database_path, subfolder)
        
        # Skip if it's not a directory
        if not os.path.isdir(subfolder_path):
            continue
        
        # Delete entities.xml if it exists
        entities_path = os.path.join(subfolder_path, "entities.xml")
        if os.path.exists(entities_path):
            os.remove(entities_path)
            print(f"  Deleted {entities_path}")
        
        # Rename and move variables.xml
        variables_path = os.path.join(subfolder_path, "variables.xml")
        if os.path.exists(variables_path):
            new_name = f"{subfolder}.xml"
            new_path = os.path.join(database_path, new_name)
            shutil.move(variables_path, new_path)
            print(f"  Moved and renamed {variables_path} to {new_path}")
        
        # Delete the now-empty folder
        try:
            shutil.rmtree(subfolder_path)
            print(f"  Deleted folder {subfolder_path}")
        except Exception as e:
            print(f"  Error deleting folder {subfolder_path}: {e}")

print("\nReorganization complete for all databases.")