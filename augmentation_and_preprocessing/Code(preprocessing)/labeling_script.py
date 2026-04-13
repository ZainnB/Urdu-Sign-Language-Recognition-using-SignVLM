from pathlib import Path
import re

# -------------------- USER CONFIG --------------------
FOLDER = Path.home() / "Documents" / "PSL_Dataset" / "zain_a" / "v3"
dry_run = False

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".mpg", ".mpeg", ".3gp"}

LABELS = [
    "Afraid",
    "Angry",
    "Autumn",
    "best",
    "body",
    "brother_or_sister",
    "Car",
    "Come",
    "Down",
    "Drink",
    "Eat",
    "Father",
    "Five",
    "food",
    "Forward",
    "Four",
    "free",
    "Friday",
    "Go",
    "Happy",
    "Healthy",
    "Hear",
    "He_or_she",
    "hi",
    "Home",
    "Hospital",
    "I",
    "Internet",
    "Left",
    "Medicine",
    "meet",
    "Mobile_phone",
    "Monday",
    "Mother",
    "One",
    "other",
    "please",
    "Read",
    "Right",
    "Sad",
    "Saturday",
    "See",
    "Sick",
    "Sleep",
    "son_or_daughter",
    "sorry",
    "Speak",
    "Spring",
    "Summer",
    "Sunday",
    "Surprised",
    "Tea",
    "teacher",
    "thank_you",
    "Three",
    "Thursday",
    "Tuesday",
    "TV",
    "Two",
    "Up",
    "Water",
    "We",
    "Wednesday",
    "welcome",
    "What",
    "When",
    "Where",
    "Who",
    "Why",
    "Winter",
    "Write",
    "You",
    "ء", "ا", "ب", "پ", "ت", "ث",
    "ج", "چ", "ح", "خ", "د", "ڈ", "ر",
    "ڑ", "ز", "س", "ش", "ض", "ط",
    "ع", "غ", "ف", "ک", "گ", "ل",
    "م", "ن", "ہ", "ھ", "و", "ے", "ی"
]

# -----------------------------------------------------

def extract_first_number(s: str):
    m = re.search(r'(\d+)', s)
    return int(m.group(1)) if m else None

def sanitize_label_keep_unicode(label: str):
    """
    Keep Unicode letters and numbers. Replace whitespace with underscore.
    Remove characters illegal in Windows filenames: <>:"/\\|?* and control chars.
    Trim leading/trailing underscores/dots.
    """
    if label is None:
        return "label"

    # remove content inside parentheses (including parentheses)
    label = re.sub(r'\s*\(.*?\)\s*', '', label)

    # collapse whitespace to single underscore
    label = re.sub(r'\s+', '_', label.strip())

    # remove characters illegal in Windows filenames and control chars
    illegal = r'[<>:"/\\|?\*\x00-\x1f]'
    label = re.sub(illegal, '', label)

    # prevent names like '.' or empty
    label = label.strip(' ._')  # trim dots/underscores/spaces at ends

    if label == "":
        # fallback: create a readable fallback using unicode codepoints
        fallback = "_".join(f"U{ord(ch):04X}" for ch in label_original_to_codepoints(label))
        return fallback or "label"

    return label

def label_original_to_codepoints(s: str):
    # helper to build fallback from original input; called only if sanitization empties,
    # but we need the original label - so this function is used in the fallback path above.
    # Because label above is already modified, return an empty list; we will instead construct
    # a fallback in caller if needed. Keep simple: return [].
    return []

# Better fallback that uses original label input
def sanitize_label(label: str):
    original = label or ""
    # perform same sanitization but keep original for fallback
    # remove parentheses content first
    l = re.sub(r'\s*\(.*?\)\s*', '', original)
    l = re.sub(r'\s+', '_', l.strip())
    illegal = r'[<>:"/\\|?\*\x00-\x1f]'
    l = re.sub(illegal, '', l)
    l = l.strip(' ._')
    if l:
        return l
    # fallback: encode each char to UXXXX token
    if original:
        tokens = []
        for ch in original:
            if ch.strip() == "":
                continue
            tokens.append(f"U{ord(ch):04X}")
        return "_".join(tokens) if tokens else "label"
    return "label"

def find_video_files(folder: Path):
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    return files

def sort_files_ascending(files):
    def sort_key(p: Path):
        num = extract_first_number(p.name)
        return (0, num, p.name.lower()) if num is not None else (1, p.name.lower())
    return sorted(files, key=sort_key)

def make_unique_target(target: Path):
    if not target.exists():
        return target
    parent = target.parent
    stem = target.stem
    suffix = target.suffix
    counter = 1
    while True:
        new_name = f"{stem}_{counter}{suffix}"
        new_path = parent / new_name
        if not new_path.exists():
            return new_path
        counter += 1

def main():
    print("Folder:", FOLDER)
    files = find_video_files(FOLDER)
    if not files:
        print("No video files found. Exiting.")
        return

    files_sorted = sort_files_ascending(files)
    print(f"Found {len(files_sorted)} video files. Sorting them by numeric order...")

    max_map = min(len(files_sorted), len(LABELS))
    if len(files_sorted) > len(LABELS):
        print(f"Warning: {len(files_sorted)} files but only {len(LABELS)} labels. Only first {len(LABELS)} files will be renamed.")
    elif len(files_sorted) < len(LABELS):
        print(f"Note: {len(files_sorted)} files but {len(LABELS)} labels. Only {len(files_sorted)} labels will be used.")

    proposed = []
    for i in range(max_map):
        src = files_sorted[i]
        label = LABELS[i]
        sanitized = sanitize_label(label)
        target_name = f"{sanitized}{src.suffix}"
        target = src.with_name(target_name)
        target = make_unique_target(target)
        proposed.append((src, target))

    print("\nProposed renames (first up to 50):")
    for src, dst in proposed[:50]:
        print(f"{src.name}  -->  {dst.name}")

    if dry_run:
        print("\nDry run enabled. No files were changed. Set dry_run = False to perform renames.")
        return

    print("\nPerforming renames...")
    for src, dst in proposed:
        try:
            src.rename(dst)
            print(f"Renamed: {src.name} -> {dst.name}")
        except Exception as e:
            print(f"Failed to rename {src.name} -> {dst.name}: {e}")

    print("Done.")

if __name__ == "__main__":
    main()
