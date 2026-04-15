#!/usr/bin/env python3
"""
Move ALL videos to new dataset, preserving label structure.
- Reads from: Train_full (each label folder contains 1.mp4, 2.mp4, ... numbered videos)
- Moves ALL video files (1.mp4, 2.mp4, 3.mp4, etc.)
- Destination: Train_videos (with same label structure, only videos, no frame folders)
- Frame folders are NOT moved, only videos
"""

from pathlib import Path
import shutil

# ============================================================================
# USER CONFIG
# ============================================================================
SOURCE_ROOT = Path(r"C:\Users\Dell\Documents\Github-Repos\Urdu_Sign_Language_Recognition_using_SignVLM\data\Train_ROI")
TARGET_ROOT = Path(r"C:\Users\Dell\Documents\Github-Repos\Urdu_Sign_Language_Recognition_using_SignVLM\data\Train_roi_vids")

dry_run = False

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".wmv", ".flv", ".mpg", ".mpeg", ".3gp"}

LABELS = [
    "Afraid", "Angry", "Autumn", "best", "body", "brother_or_sister", "Car", "Come", "Down", "Drink",
    "Eat", "Father", "Five", "food", "Forward", "Four", "free", "Friday", "Go", "Happy", "Healthy",
    "Hear", "He_or_she", "hi", "Home", "Hospital", "I", "Internet", "Left", "Medicine", "meet",
    "Mobile_phone", "Monday", "Mother", "One", "other", "please", "Read", "Right", "Sad", "Saturday",
    "See", "Sick", "Sleep", "son_or_daughter", "sorry", "Speak", "Spring", "Summer", "Sunday",
    "Surprised", "Tea", "teacher", "thank_you", "Three", "Thursday", "Tuesday", "TV", "Two",
    "Up", "Water", "We", "Wednesday", "welcome", "What", "When", "Where", "Who", "Why", "Winter",
    "Write", "You",
    "ء", "ا", "ب", "پ", "ت", "ث", "ج", "چ", "ح", "خ", "د", "ڈ", "ر", "ڑ", "ز", "س", "ش", "ض", "ط",
    "ع", "غ", "ف", "ک", "گ", "ل", "م", "ن", "ہ", "ھ", "و", "ے", "ی"
]

# ============================================================================

def extract_number(name: str):
    """Extract numeric part from filename or folder name (e.g., '1.mp4' -> 1, '3' -> 3)."""
    stem = Path(name).stem
    try:
        return int(stem)
    except ValueError:
        return None

def ensure_label_folder(root: Path, label: str):
    """Create label folder in target root if it doesn't exist."""
    label_folder = root / label
    label_folder.mkdir(parents=True, exist_ok=True)
    return label_folder

def process_dataset():
    """Main processing function."""
    
    print("=" * 80)
    print("ALL VIDEOS MOVER (PRESERVING LABEL STRUCTURE)")
    print("=" * 80)
    print(f"Source root: {SOURCE_ROOT}")
    print(f"Target root: {TARGET_ROOT}")
    print(f"Moving: ALL video files (1.mp4, 2.mp4, 3.mp4, etc.)")
    print(f"Note: Frame folders are NOT moved, only videos")
    print(f"Dry run: {dry_run}")
    print("=" * 80)
    print()
    
    # Verify source exists
    if not SOURCE_ROOT.exists():
        print(f"❌ Source root not found: {SOURCE_ROOT}")
        return
    
    total_moved_videos = 0
    total_labels_processed = 0
    
    for label in LABELS:
        label_source = SOURCE_ROOT / label
        
        # Check if label folder exists in source
        if not label_source.exists() or not label_source.is_dir():
            print(f"⚠️  Label folder not found: {label}")
            continue
        
        # Get all items in label folder
        all_items = list(label_source.iterdir())
        
        # Collect ALL video files (no filtering for odd/even)
        videos = []
        
        for item in all_items:
            if item.is_file() and item.suffix.lower() in VIDEO_EXTS:
                num = extract_number(item.name)
                videos.append((num if num is not None else 0, item))
        
        if not videos:
            continue
        
        total_labels_processed += 1
        
        # Sort by number for cleaner output
        videos.sort(key=lambda x: x[0])
        
        print(f"📁 {label}: {len(videos)} videos to move")
        
        # Ensure target label folder exists
        label_target = ensure_label_folder(TARGET_ROOT, label)
        
        # Move ALL videos
        for num, video_path in videos:
            dest_path = label_target / video_path.name
            
            if dry_run:
                print(f"  [DRY-VIDEO] {video_path.name} -> {label}/{video_path.name}")
            else:
                try:
                    shutil.move(str(video_path), str(dest_path))
                    print(f"  ✓ Moved video: {video_path.name}")
                    total_moved_videos += 1
                except Exception as e:
                    print(f"  ❌ Failed to move video {video_path.name}: {e}")
    
    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Labels processed: {total_labels_processed}")
    print(f"Video files moved: {total_moved_videos}")
    if dry_run:
        print("🔍 DRY RUN - no files were actually moved")
    else:
        print("✅ All videos successfully moved!")
    print("=" * 80)

if __name__ == "__main__":
    process_dataset()
