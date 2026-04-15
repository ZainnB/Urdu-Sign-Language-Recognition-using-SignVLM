#!/usr/bin/env python3
"""
Delete specific numbered folders from Train_full dataset.
Deletes entire folders (both video files and frame folders).

Default: dry-run (no deletions).
Use --apply to actually delete.

Usage:
    # dry-run (default)
    python rm_aug_vids.py

    # actually delete
    python rm_aug_vids.py --apply
"""

from pathlib import Path
import argparse
import sys
import shutil

# ============================================================================
# FOLDERS TO DELETE (label/number format)
# ============================================================================
FOLDERS_TO_DELETE = [
    ("Afraid", 12), ("Afraid", 14),
    ("Angry", 10), ("Angry", 12), ("Angry", 14),
    ("Autumn", 10), ("Autumn", 12), ("Autumn", 14),
    ("best", 10), ("best", 12), ("best", 14),
    ("body", 10), ("body", 12), ("body", 14),
    ("brother_or_sister", 10), ("brother_or_sister", 12), ("brother_or_sister", 14),
    ("Come", 10), ("Come", 12), ("Come", 14),
    ("Down", 10), ("Down", 12), ("Down", 14),
    ("Drink", 10), ("Drink", 12), ("Drink", 14),
    ("Eat", 10), ("Eat", 12), ("Eat", 14),
    ("Father", 10), ("Father", 12), ("Father", 14),
    ("Four", 10), ("Four", 12), ("Four", 14),
    ("Friday", 8),
    ("Go", 7), ("Go", 10), ("Go", 12), ("Go", 14),
    ("Happy", 10), ("Happy", 12), ("Happy", 14),
    ("Healthy", 8), ("Healthy", 10),
    ("Hear", 10), ("Hear", 12), ("Hear", 14),
    ("He_or_she", 9),
    ("hi", 10), ("hi", 12), ("hi", 14),
    ("Home", 11),
    ("I", 10), ("I", 12), ("I", 14),
    ("Medicine", 10), ("Medicine", 12), ("Medicine", 14),
    ("Mobile_phone", 6),
    ("Mother", 10), ("Mother", 12), ("Mother", 14),
    ("One", 10), ("One", 12), ("One", 14),
    ("other", 9), ("other", 10), ("other", 12), ("other", 14),
    ("Read", 6),
    ("Sad", 2), ("Sad", 10), ("Sad", 12), ("Sad", 14),
    ("See", 6),
    ("sorry", 10), ("sorry", 12), ("sorry", 14),
    ("Speak", 10), ("Speak", 12), ("Speak", 14),
    ("Spring", 10), ("Spring", 12), ("Spring", 14),
    ("Summer", 10), ("Summer", 12), ("Summer", 14),
    ("Tea", 10), ("Tea", 12), ("Tea", 14),
    ("thank_you", 10), ("thank_you", 12), ("thank_you", 14),
    ("Three", 10), ("Three", 12), ("Three", 14),
    ("Thursday", 10), ("Thursday", 12), ("Thursday", 14),
    ("TV", 10), ("TV", 12), ("TV", 14),
    ("Two", 10), ("Two", 12), ("Two", 14),
    ("Up", 10), ("Up", 12), ("Up", 14),
    ("Water", 10), ("Water", 12), ("Water", 14),
    ("We", 10), ("We", 12), ("We", 14),
    ("What", 10), ("What", 12), ("What", 14),
    ("When", 10), ("When", 12), ("When", 14),
    ("Where", 10), ("Where", 12), ("Where", 14),
    ("Who", 10), ("Who", 12), ("Who", 14),
    ("Why", 10), ("Why", 12), ("Why", 14),
    ("You", 10), ("You", 12), ("You", 14),
    ("ء", 8), ("ء", 10), ("ء", 12),
    ("ا", 10), ("ا", 12), ("ا", 14),
    ("ب", 10), ("ب", 12), ("ب", 14),
    ("پ", 10), ("پ", 12), ("پ", 14),
    ("ت", 10), ("ت", 12), ("ت", 14),
    ("ث", 10), ("ث", 12), ("ث", 14),
    ("ج", 10), ("ج", 12), ("ج", 14),
    ("چ", 10), ("چ", 12), ("چ", 14),
    ("ح", 10), ("ح", 12), ("ح", 14),
    ("خ", 10), ("خ", 12), ("خ", 14),
    ("د", 10), ("د", 12), ("د", 14),
    ("ڈ", 10), ("ڈ", 12), ("ڈ", 14),
    ("ر", 10), ("ر", 12), ("ر", 14),
    ("ڑ", 10), ("ڑ", 12), ("ڑ", 14),
    ("ز", 10), ("ز", 12), ("ز", 14),
    ("س", 10), ("س", 12), ("س", 14),
    ("ش", 1), ("ش", 3), ("ش", 5), ("ش", 10), ("ش", 12), ("ش", 14),
    ("ض", 10), ("ض", 12), ("ض", 14),
    ("ط", 10), ("ط", 12), ("ط", 14),
    ("ع", 10), ("ع", 12), ("ع", 14),
    ("غ", 10), ("غ", 12), ("غ", 14),
    ("ف", 10), ("ف", 12), ("ف", 14),
    ("ک", 10), ("ک", 12), ("ک", 14),
    ("گ", 10), ("گ", 12), ("گ", 14),
    ("ل", 10), ("ل", 12), ("ل", 14),
    ("م", 10), ("م", 12), ("م", 14),
    ("ن", 10), ("ن", 12), ("ن", 14),
    ("ہ", 10), ("ہ", 12), ("ہ", 14),
    ("ھ", 10), ("ھ", 12), ("ھ", 14),
    ("و", 10), ("و", 12), ("و", 14),
    ("ے", 10), ("ے", 12), ("ے", 14),
]

# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Delete specific numbered folders from Train_full dataset.")
    ap.add_argument("--base", "-b", 
                    default=r"C:\Users\Dell\Documents\Github-Repos\Urdu_Sign_Language_Recognition_using_SignVLM\data\Train_full",
                    help="Base directory (Train_full)")
    ap.add_argument("--apply", action="store_true", help="Actually delete folders. Default is dry-run.")
    args = ap.parse_args()

    base = Path(args.base).expanduser().resolve()

    if not base.exists() or not base.is_dir():
        print(f"❌ Base directory not found: {base}")
        sys.exit(1)

    print("=" * 80)
    print("DELETE SPECIFIED LANDSCAPE FOLDERS")
    print("=" * 80)
    print(f"Base directory: {base}")
    print(f"Folders to delete: {len(FOLDERS_TO_DELETE)}")
    print(f"Dry run: {not args.apply}")
    print("=" * 80)
    print()

    folders_to_remove = []
    not_found = []

    # Verify all folders exist
    for label, number in FOLDERS_TO_DELETE:
        folder_path = base / label / str(number)
        
        if folder_path.exists() and folder_path.is_dir():
            folders_to_remove.append((label, number, folder_path))
        else:
            not_found.append((label, number))

    # Show folders that will be deleted
    print(f"Folders found: {len(folders_to_remove)}/{len(FOLDERS_TO_DELETE)}")
    print()
    
    if folders_to_remove:
        print("Folders to DELETE:")
        print("-" * 80)
        for label, number, path in folders_to_remove:
            print(f"  {label}/{number}/  ({path})")
        print("-" * 80)
        print()

    if not_found:
        print(f"⚠️  Folders NOT found ({len(not_found)}):")
        print("-" * 80)
        for label, number in not_found:
            print(f"  {label}/{number}/")
        print("-" * 80)
        print()

    if not args.apply:
        print("🔍 DRY RUN: No folders were deleted. Rerun with --apply to delete.")
        return

    # Apply deletion
    print()
    print("Applying deletion...")
    deleted = 0
    failed = []

    for label, number, path in folders_to_remove:
        try:
            shutil.rmtree(path)
            print(f"✓ Deleted: {label}/{number}/")
            deleted += 1
        except Exception as e:
            print(f"❌ Failed to delete {label}/{number}/: {e}")
            failed.append((label, number, str(e)))

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Successfully deleted: {deleted}/{len(folders_to_remove)}")
    if failed:
        print(f"Failed to delete: {len(failed)}")
        for label, number, err in failed:
            print(f"  {label}/{number}/  ->  {err}")
    print("=" * 80)

if __name__ == "__main__":
    main()

