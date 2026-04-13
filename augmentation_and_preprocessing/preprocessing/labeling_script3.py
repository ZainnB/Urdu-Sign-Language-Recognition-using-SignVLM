#!/usr/bin/env python3
"""
label_and_rename_friend.py

- Creates folders for each label in LABELS under TARGET_ROOT.
- For each label, picks files named label_1.*, label_2.*, label_3.* (first match)
  from INPUT_FOLDER (case-insensitive match for label and _1/_2/_3).
- Moves each matched file into the label folder and renames:
    *_1.ext -> <Label>_13.ext
    *_2.ext -> <Label>_14.ext
    *_3.ext -> <Label>_15.ext
- dry_run=True prints actions only. Set dry_run=False to perform moves.
"""

from pathlib import Path
import re
import shutil

# ----------------- USER CONFIG -----------------
INPUT_FOLDER = Path.home() / "Documents" / "PSL_Dataset" / "AbdurRehman"
TARGET_ROOT = INPUT_FOLDER / "by_label"
dry_run = False

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".wmv", ".flv", ".mpg", ".mpeg", ".3gp"}
# Your labels list (104)
LABELS = [
    "Afraid","Angry","Autumn","best","body","brother_or_sister","Car","Come","Down","Drink",
    "Eat","Father","Five","food","Forward","Four","free","Friday","Go","Happy","Healthy",
    "Hear","He_or_she","hi","Home","Hospital","I","Internet","Left","Medicine","meet",
    "Mobile_phone","Monday","Mother","One","other","please","Read","Right","Sad","Saturday",
    "See","Sick","Sleep","son_or_daughter","sorry","Speak","Spring","Summer","Sunday",
    "Surprised","Tea","teacher","thank_you","Three","Thursday","Tuesday","TV","Two",
    "Up","Water","We","Wednesday","welcome","What","When","Where","Who","Why","Winter",
    "Write","You",
    "ء","ا","ب","پ","ت","ث","ج","چ","ح","خ","د","ڈ","ر","ڑ","ز","س","ش","ض","ط",
    "ع","غ","ف","ک","گ","ل","م","ن","ہ","ھ","و","ے","ی"
]
# version -> new suffix mapping (as you requested)
VERSION_MAP = {1: 13, 2: 14, 3: 15}
# ------------------------------------------------

def find_first_matching_file(folder: Path, base_label: str, version:int):
    """
    Find the first file in folder matching pattern:
      ^label_version(\.[A-Za-z0-9]+)$
    case-insensitive for label.
    Returns Path or None.
    """
    pattern = re.compile(rf'^{re.escape(base_label)}[_\-\s]?{version}(\.[^.]+)$', re.IGNORECASE)
    # Try exact ext filter for speed, but walk all files
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        name = p.name
        m = pattern.match(name)
        if m:
            return p
    return None

def make_unique(path: Path):
    """Return a non-colliding Path by appending _1,_2... if needed."""
    if not path.exists():
        return path
    parent = path.parent
    stem = path.stem
    suf = path.suffix
    i = 1
    while True:
        cand = parent / f"{stem}_{i}{suf}"
        if not cand.exists():
            return cand
        i += 1

def ensure_label_dirs(root: Path, labels):
    root.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for lab in labels:
        p = root / lab
        p.mkdir(exist_ok=True)
        mapping[lab] = p
    return mapping

def pretty_path_for_print(p: Path, base: Path):
    """
    Try to return path relative to 'base'; fall back to absolute string if not possible.
    """
    try:
        return str(p.relative_to(base))
    except Exception:
        return str(p)

def process():
    print("INPUT_FOLDER:", INPUT_FOLDER)
    print("TARGET_ROOT:", TARGET_ROOT)
    print("dry_run:", dry_run)
    print()

    if not INPUT_FOLDER.exists():
        print("Input folder doesn't exist:", INPUT_FOLDER)
        return

    label_dirs = ensure_label_dirs(TARGET_ROOT, LABELS)

    # For each label, look for version 1,2,3
    total_moved = 0
    for label in LABELS:
        label_dir = label_dirs[label]
        for version in (1,2,3):
            src = find_first_matching_file(INPUT_FOLDER, label, version)
            if src is None:
                print(f"Skip: no file for {label}_{version} in input folder")
                continue

            new_suffix = VERSION_MAP.get(version)
            if new_suffix is None:
                print(f"Skipping version {version} for {label} (no mapping)")
                continue

            # preserve original extension
            out_name = f"{label}_{new_suffix}{src.suffix}"
            dest_path = label_dir / out_name
            dest_path = make_unique(dest_path)

            if dry_run:
                # print source and destination relative to TARGET_ROOT if possible, else absolute
                print(f"[DRY] {src.name} -> {pretty_path_for_print(dest_path, TARGET_ROOT)}")
            else:
                try:
                    shutil.move(str(src), str(dest_path))
                    print(f"Moved: {src.name} -> {dest_path}")
                    total_moved += 1
                except Exception as e:
                    print(f"Failed moving {src} -> {dest_path}: {e}")

    if not dry_run:
        print(f"\nDone. Total moved: {total_moved}")
    else:
        print("\nDry run complete. No files changed.")

if __name__ == "__main__":
    process()
