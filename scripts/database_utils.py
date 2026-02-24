"""Utility functions for database operations."""

import os
import glob
from pathlib import Path


def get_available_databases(data_root: str = None):
    """Get list of available databases in the data folder."""
    if data_root is None:
        # Use absolute path based on this file's location
        data_root = str(Path(__file__).parent.parent / "data")
    
    if not os.path.exists(data_root):
        return []
    
    databases = []
    for item in os.listdir(data_root):
        item_path = os.path.join(data_root, item)
        if os.path.isdir(item_path):
            # Check if directory contains XML or JSON files
            has_xml = len(glob.glob(os.path.join(item_path, "*.xml"))) > 0
            has_json = len(glob.glob(os.path.join(item_path, "*.json"))) > 0
            if has_xml or has_json:
                databases.append(item)
    
    return sorted(databases)
