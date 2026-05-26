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

import argparse
import os
import shutil

DATA_ROOT = "./data_maelstrom_latest"  # Update this to your actual data root directory

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reorganize Maelstrom export folders by removing entities.xml, "
            "moving variables.xml to the database root, and deleting processed subfolders."
        )
    )
    parser.add_argument(
        "--data-root",
        default=DATA_ROOT,
        help=f"Root folder that contains database folders (default: {DATA_ROOT})",
    )
    parser.add_argument(
        "--database",
        default="",
        help="Optional single database name to process, e.g. ULSAM",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned actions without changing files",
    )
    return parser.parse_args()


def reorganize_database(database_path: str, dry_run: bool = False) -> None:
    database = os.path.basename(database_path)
    print(f"\nProcessing database: {database}")

    for subfolder in os.listdir(database_path):
        subfolder_path = os.path.join(database_path, subfolder)

        if not os.path.isdir(subfolder_path):
            continue

        entities_path = os.path.join(subfolder_path, "entities.xml")
        variables_path = os.path.join(subfolder_path, "variables.xml")
        renamed_path = os.path.join(database_path, f"{subfolder}.xml")

        has_entities = os.path.exists(entities_path)
        has_variables = os.path.exists(variables_path)

        if not has_entities and not has_variables:
            print(
                f"  Skipped {subfolder_path} (no entities.xml/variables.xml found; leaving untouched)"
            )
            continue

        if has_entities:
            if dry_run:
                print(f"  [dry-run] Would delete {entities_path}")
            else:
                os.remove(entities_path)
                print(f"  Deleted {entities_path}")

        if has_variables:
            if dry_run:
                print(f"  [dry-run] Would move {variables_path} to {renamed_path}")
            else:
                shutil.move(variables_path, renamed_path)
                print(f"  Moved and renamed {variables_path} to {renamed_path}")

        try:
            remaining_items = os.listdir(subfolder_path)
            if remaining_items:
                print(
                    f"  Kept folder {subfolder_path} (still contains: {', '.join(remaining_items)})"
                )
            elif dry_run:
                print(f"  [dry-run] Would delete empty folder {subfolder_path}")
            else:
                shutil.rmtree(subfolder_path)
                print(f"  Deleted empty folder {subfolder_path}")
        except Exception as e:
            print(f"  Error checking/deleting folder {subfolder_path}: {e}")


def main() -> None:
    args = parse_args()
    data_root = args.data_root

    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    if args.database:
        database_path = os.path.join(data_root, args.database)
        if not os.path.isdir(database_path):
            raise FileNotFoundError(f"Database folder not found: {database_path}")
        reorganize_database(database_path, dry_run=args.dry_run)
        print(f"\nReorganization complete for database: {args.database}")
        return

    processed = 0
    for database in sorted(os.listdir(data_root)):
        database_path = os.path.join(data_root, database)
        if not os.path.isdir(database_path):
            continue
        reorganize_database(database_path, dry_run=args.dry_run)
        processed += 1

    print(f"\nReorganization complete for {processed} databases.")


if __name__ == "__main__":
    main()