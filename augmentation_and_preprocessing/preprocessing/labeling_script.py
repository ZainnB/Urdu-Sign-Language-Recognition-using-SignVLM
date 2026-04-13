from pathlib import Path
import re

# -------------------- USER CONFIG --------------------
DATASET_ROOT = Path.home() / "Documents" / "Github-Repos" / "Urdu_Sign_Language_Recognition_using_SignVLM" / "data" / "Final Dataset (with roi augmentation)"
MAX_VIDEOS_PER_LABEL = 30
dry_run = False

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".mpg", ".mpeg", ".3gp"}

# Labels define the folder names that should be processed
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
    """Extract first number from string for sorting."""
    m = re.search(r'(\d+)', s)
    return int(m.group(1)) if m else None

def sort_files_ascending(files):
    """Sort files by first numeric value in filename."""
    def sort_key(p: Path):
        num = extract_first_number(p.name)
        return (0, num, p.name.lower()) if num is not None else (1, p.name.lower())
    return sorted(files, key=sort_key)

def find_video_files(folder: Path):
    """Find all video files in a folder."""
    if not folder.exists():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    return files

def main():
    """
    Rename all video files in each label folder to sequential numbering: 1.mp4, 2.mp4, etc.
    Label is determined by folder name, not filename.
    """
    dataset_root = Path(DATASET_ROOT)
    
    if not dataset_root.exists():
        print(f"❌ Dataset root not found: {DATASET_ROOT}")
        return
    
    print(f"Dataset root: {dataset_root}")
    print(f"Max videos per label: {MAX_VIDEOS_PER_LABEL}")
    print(f"Dry run: {dry_run}\n")
    
    total_renamed = 0
    total_skipped = 0
    total_folders_processed = 0
    
    for label in LABELS:
        label_folder = dataset_root / label
        
        if not label_folder.exists():
            print(f"⚠️  Label folder not found: {label}")
            continue
        
        if not label_folder.is_dir():
            print(f"⚠️  Not a directory: {label}")
            continue
        
        # Find and sort video files
        video_files = find_video_files(label_folder)
        
        if not video_files:
            print(f"⚠️  No video files in: {label}")
            continue
        
        video_files = sort_files_ascending(video_files)
        
        # Limit to MAX_VIDEOS_PER_LABEL
        if len(video_files) > MAX_VIDEOS_PER_LABEL:
            print(f"\n📁 {label}: {len(video_files)} videos (limitting to {MAX_VIDEOS_PER_LABEL})")
            video_files = video_files[:MAX_VIDEOS_PER_LABEL]
        else:
            print(f"\n📁 {label}: {len(video_files)} videos")
        
        total_folders_processed += 1
        renamed_count = 0
        skipped_count = 0
        
        # Rename files to sequential numbering
        for idx, video_file in enumerate(video_files, 1):
            new_name = f"{idx}.mp4"
            new_path = label_folder / new_name
            
            # Skip if source and target are the same
            if video_file == new_path:
                print(f"  ⏭️  {video_file.name} → {new_name} (already named)")
                skipped_count += 1
                continue
            
            # Skip if target already exists (don't overwrite)
            if new_path.exists():
                print(f"  ⚠️  {video_file.name} → {new_name} (target exists, skipping)")
                skipped_count += 1
                continue
            
            print(f"  {video_file.name} → {new_name}")
            
            if not dry_run:
                try:
                    video_file.rename(new_path)
                    renamed_count += 1
                except Exception as e:
                    print(f"    ❌ Error: {e}")
                    skipped_count += 1
            else:
                renamed_count += 1
        
        total_renamed += renamed_count
        total_skipped += skipped_count
        print(f"  ✓ {renamed_count} renamed, {skipped_count} skipped")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Folders processed: {total_folders_processed}")
    print(f"Total files renamed: {total_renamed}")
    print(f"Total files skipped: {total_skipped}")
    if dry_run:
        print(f"🔍 Dry run mode - no files were actually changed")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
