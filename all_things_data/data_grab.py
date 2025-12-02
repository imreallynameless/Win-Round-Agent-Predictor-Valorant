#!/usr/bin/env python3
"""
Run the full data pipeline for a given folder.

Usage:
    python data_grab.py training_data_ascent
    python data_grab.py test_data_haven
"""

import subprocess
import sys
import os

def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n❌ Failed: {description}")
        return False
    
    print(f"✓ Completed: {description}")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python data_grab.py <folder>")
        print("Example: python data_grab.py training_data_ascent")
        sys.exit(1)
    
    folder = sys.argv[1]
    
    # Check if folder exists
    if not os.path.isdir(folder):
        print(f"Error: Folder '{folder}' does not exist")
        sys.exit(1)
    
    print(f"\n🎮 Starting data pipeline for: {folder}")
    print("="*60)
    
    # Step 1: Gather match IDs
    if not run_command(
        ["python", "gather_matches.py", folder],
        "Gather match IDs from rib.gg"
    ):
        sys.exit(1)
    
    # Step 2: Process matches (fetch JSON + create CSV)
    if not run_command(
        ["python", "process_matches.py", folder],
        "Process matches (fetch JSON + create CSV)"
    ):
        sys.exit(1)
    
    # Step 3: Fetch slugs for readable filenames
    if not run_command(
        ["python", "fetch_slugs.py", folder],
        "Fetch slugs for readable filenames"
    ):
        sys.exit(1)
    
    # Step 4: Rename JSON files with slugs
    if not run_command(
        ["python", "rename_json.py", folder],
        "Rename JSON files with slugs"
    ):
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"✅ All steps completed for {folder}!")
    print('='*60)


if __name__ == "__main__":
    main()

