import os
import shutil

DATA_DIR = "./data/SNAC-K"

# List all subdirectories in DATA_DIR
for subfolder in os.listdir(DATA_DIR):
    subfolder_path = os.path.join(DATA_DIR, subfolder)
    if os.path.isdir(subfolder_path):
        # Delete entities.xml if it exists
        entities_path = os.path.join(subfolder_path, "entities.xml")
        if os.path.exists(entities_path):
            os.remove(entities_path)
            print(f"Deleted {entities_path}")
        
        # Rename and move variables.xml
        variables_path = os.path.join(subfolder_path, "variables.xml")
        if os.path.exists(variables_path):
            new_name = f"{subfolder}.xml"
            new_path = os.path.join(DATA_DIR, new_name)
            shutil.move(variables_path, new_path)
            print(f"Moved and renamed {variables_path} to {new_path}")
        
        # Delete the now-empty folder
        shutil.rmtree(subfolder_path)
        print(f"Deleted folder {subfolder_path}")

print("Reorganization complete.")