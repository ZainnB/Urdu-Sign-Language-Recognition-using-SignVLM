#!/usr/bin/env python3
"""
delete_aug_strict.py

Scan only the immediate subdirectories of the base directory (expected 104 folders)
and delete files whose filename contains '_aug_' (case-insensitive).

Default: dry-run (no deletions).
Use --apply to actually delete.

Usage:
    # dry-run (default)
    python delete_aug_strict.py

    # actually delete
    python delete_aug_strict.py --apply

    # custom base dir
    python delete_aug_strict.py --base "/path/to/FYP_PSL/data/virkha+zain+ark" --apply
"""

from pathlib import Path
import argparse
import sys

def find_aug_files_in_immediate_subdirs(base: Path):
    """Return list of files (Path) under each immediate subdirectory of base that contain '_aug_' (case-insensitive)."""
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Base directory not found or not a directory: {base}")

    aug_files = []
    # iterate only immediate children that are directories
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        # recursively find files inside this child directory
        for f in child.rglob("*"):
            if f.is_file() and "_aug_" in f.name.lower():
                aug_files.append(f)
    return aug_files

def main():
    ap = argparse.ArgumentParser(description="Delete files containing '_aug_' inside the 104 immediate subfolders of a base directory.")
    ap.add_argument("--base", "-b", default="FYP_PSL/data/virkha+zain+ark", help="Base directory containing the 104 subfolders (default shown).")
    ap.add_argument("--apply", action="store_true", help="Actually delete files. Default is dry-run.")
    args = ap.parse_args()

    base = Path(args.base).expanduser().resolve()

    try:
        aug_files = find_aug_files_in_immediate_subdirs(base)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    if not aug_files:
        print("No files with '_aug_' found in immediate subfolders.")
        return

    print(f"Found {len(aug_files)} file(s) containing '_aug_' inside immediate subfolders of: {base}")
    for p in aug_files:
        print("  " + str(p))

    if not args.apply:
        print("\nDry-run: no files deleted. Rerun with --apply to delete these files.")
        return

    # apply deletion
    deleted = 0
    failed = []
    for p in aug_files:
        try:
            p.unlink()
            deleted += 1
        except Exception as e:
            failed.append((p, str(e)))

    print(f"\nDeleted: {deleted} file(s).")
    if failed:
        print(f"Failed to delete {len(failed)} file(s):")
        for p, err in failed:
            print(f"  {p}  ->  {err}")

if __name__ == "__main__":
    main()
