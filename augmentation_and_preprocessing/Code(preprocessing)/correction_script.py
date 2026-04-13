#!/usr/bin/env python3
"""
fix_label_names.py

Ensure that each video in each label folder has the folder's label as the prefix
(before the final underscore). If not, rename to <foldername>_<suffix><ext>.

Defaults:
 - ROOT_FOLDER = ~/Documents/PSL_Dataset/merged_labels
 - dry_run = True (preview only)
 - ignore_case = True (compare case-insensitively)
"""

from pathlib import Path
import shutil

# ------------- CONFIG -------------
ROOT_FOLDER = Path.home() / "Documents" / "PSL_Dataset" / "zain"
VIDEO_EXTS = {".mp4"}
dry_run =   False
ignore_case = True
# ----------------------------------

def make_unique(path: Path) -> Path:
    """If path exists, return a unique path by appending _1, _2, ... before ext."""
    if not path.exists():
        return path
    parent = path.parent
    stem = path.stem
    suf = path.suffix
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suf}"
        if not candidate.exists():
            return candidate
        i += 1

def needs_fixing(folder_name: str, file_stem: str) -> (bool, str, str):
    """
    Returns (needs_fix, prefix, suffix)
    prefix = part before last underscore (or entire stem if no underscore)
    suffix = part from last underscore (including underscore), e.g. '_10' or '_abc_2' etc.
    """
    if '_' in file_stem:
        prefix, rest = file_stem.rsplit('_', 1)
        suffix = "_" + rest
    else:
        prefix = file_stem
        suffix = ""  # no suffix present

    if ignore_case:
        same = (prefix.lower() == folder_name.lower())
    else:
        same = (prefix == folder_name)

    return (not same), prefix, suffix

def process_root(root: Path):
    if not root.exists() or not root.is_dir():
        print("Root folder not found or not a directory:", root)
        return

    label_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if not label_dirs:
        print("No subfolders (labels) found in:", root)
        return

    total_checked = 0
    total_fixed = 0

    for label_dir in label_dirs:
        folder_label = label_dir.name
        files = [p for p in sorted(label_dir.iterdir()) if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
        if not files:
            # optionally skip or print
            print(f"[SKIP] no video files in: {label_dir}")
            continue

        for f in files:
            total_checked += 1
            stem = f.stem  # filename without extension
            need_fix, prefix, suffix = needs_fixing(folder_label, stem)

            if not need_fix:
                # OK
                continue

            # Build new name: folderLabel + suffix + ext
            if suffix:
                new_stem = f"{folder_label}{suffix}"
            else:
                # If file had no underscore/suffix, append original stem as suffix to keep info:
                new_stem = f"{folder_label}_{stem}"

            new_name = new_stem + f.suffix
            dest = label_dir / new_name
            dest = make_unique(dest)

            if dry_run:
                print(f"[DRY] Would rename: {f.name} -> {dest.name}  (prefix was: '{prefix}')")
                total_fixed += 1
            else:
                try:
                    f.rename(dest)
                    print(f"Renamed: {f.name} -> {dest.name}")
                    total_fixed += 1
                except Exception as e:
                    print(f"Failed to rename {f} -> {dest}: {e}")

    print(f"\nChecked {total_checked} files. Fixes planned/applied: {total_fixed}.")

if __name__ == "__main__":
    process_root(ROOT_FOLDER)
