#!/usr/bin/env python3
"""
distribute_fixed_suffix.py

For each label (in LABELS):
  - take i-th video (0-based) from each source folder (zain_v1, zain_v2, zain_v3) if present
  - copy it into the label-named folder and rename to: <Label>_10.ext (for zain_v1),
                                                    <Label>_11.ext (for zain_v2),
                                                    <Label>_12.ext (for zain_v3)

Dry-run by default (set dry_run = False to perform copies).
"""

from pathlib import Path
import re
import shutil

# --------------- CONFIG ----------------
HOME = Path.home()
INPUT_ROOT = HOME / "Documents" / "PSL_Dataset"   # contains zain_v1, zain_v2, zain_v3
SOURCES = ["zain_v1", "zain_v2", "zain_v3"]      # order matters: s_idx 0->suffix 10, 1->11, 2->12
TARGET_ROOT = INPUT_ROOT / "zain"
dry_run = False

VIDEO_EXTS = {".mp4"}

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
# --------------- end config --------------

def extract_first_number(s: str):
    m = re.search(r'(\d+)', s)
    return int(m.group(1)) if m else None

def sorted_videos(folder: Path):
    if not folder.exists() or not folder.is_dir():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    def key(p: Path):
        num = extract_first_number(p.name)
        return (0, num, p.name.lower()) if num is not None else (1, p.name.lower())
    return sorted(files, key=key)

def make_unique(path: Path):
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

def main():
    print("INPUT_ROOT:", INPUT_ROOT)
    print("SOURCES:", SOURCES)
    print("TARGET_ROOT:", TARGET_ROOT)
    print("Dry run:", dry_run)
    print("Labels count:", len(LABELS))
    print()

    source_paths = [INPUT_ROOT / s for s in SOURCES]
    source_files = [sorted_videos(p) for p in source_paths]
    for idx, p in enumerate(source_paths):
        print(f"Source {idx} ({p}): {len(source_files[idx])} videos")

    label_dirs = ensure_label_dirs(TARGET_ROOT, LABELS)

    # fixed suffix mapping per source index:
    base_suffix = 10
    for i, label in enumerate(LABELS):
        label_dir = label_dirs[label]
        for s_idx, files in enumerate(source_files):
            file_index = i  # take i-th file from this source
            if file_index >= len(files):
                print(f"Skip: label '{label}' source '{SOURCES[s_idx]}' (no file index {file_index})")
                continue

            src = files[file_index]
            suffix_num = base_suffix + s_idx   # 10 for zain_v1, 11 for zain_v2, 12 for zain_v3
            new_name = f"{label}_{suffix_num}{src.suffix}"
            dst = label_dir / new_name
            dst = make_unique(dst)

            if dry_run:
                print(f"[DRY] {src} -> {dst}")
            else:
                try:
                    shutil.copy2(src, dst)
                    print(f"Copied: {src.name} -> {dst.name}")
                except Exception as e:
                    print(f"Failed: {src} -> {dst}: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()
