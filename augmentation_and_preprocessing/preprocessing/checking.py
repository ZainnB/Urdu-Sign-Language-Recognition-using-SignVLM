#!/usr/bin/env python3
"""
Check image orientations in Train_full dataset.
- Iterates through each label folder
- Checks every numbered subfolder
- Finds folders with LANDSCAPE images (width > height)
- Reports which folders contain landscape images
"""

from pathlib import Path
from PIL import Image

# ============================================================================
# CONFIG
# ============================================================================
DATASET_ROOT = Path(r"C:\Users\Dell\Documents\Github-Repos\Urdu_Sign_Language_Recognition_using_SignVLM\data\Test")

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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

# ============================================================================

def extract_number(name: str):
    """Extract numeric part from folder name."""
    try:
        return int(name)
    except ValueError:
        return None

def get_image_orientation(image_path: Path):
    """
    Get image orientation.
    Returns: 'landscape' if width > height, 'portrait' if height > width, 'square' if equal
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            if width > height:
                return 'landscape', width, height
            elif height > width:
                return 'portrait', width, height
            else:
                return 'square', width, height
    except Exception as e:
        return 'error', None, None

def get_first_jpg(folder: Path):
    """Get first JPG image in folder."""
    for item in sorted(folder.iterdir()):
        if item.is_file() and item.suffix.lower() in IMAGE_EXTS:
            return item
    return None

def main():
    print("=" * 80)
    print("IMAGE ORIENTATION CHECKER - TRAIN_FULL DATASET")
    print("=" * 80)
    print(f"Dataset root: {DATASET_ROOT}")
    print("Looking for: LANDSCAPE images (width > height)")
    print("=" * 80)
    print()
    
    # Verify dataset root exists
    if not DATASET_ROOT.exists():
        print(f"❌ Dataset root not found: {DATASET_ROOT}")
        return
    
    landscape_folders = []
    total_labels_checked = 0
    total_numbered_folders = 0
    
    for label in LABELS:
        label_folder = DATASET_ROOT / label
        
        # Check if label folder exists
        if not label_folder.exists() or not label_folder.is_dir():
            print(f"⚠️  Label folder not found: {label}")
            continue
        
        total_labels_checked += 1
        
        # Get all subfolders with numeric names
        numbered_folders = []
        for item in label_folder.iterdir():
            if item.is_dir():
                num = extract_number(item.name)
                if num is not None:
                    numbered_folders.append((num, item))
        
        # Sort by number
        numbered_folders.sort(key=lambda x: x[0])
        
        if not numbered_folders:
            print(f"ℹ️  {label}: No numbered folders found")
            continue
        
        print(f"📁 {label}: {len(numbered_folders)} numbered folders")
        
        # Check each numbered folder
        label_landscape_count = 0
        for num, folder_path in numbered_folders:
            total_numbered_folders += 1
            
            # Get first JPG in folder
            first_jpg = get_first_jpg(folder_path)
            
            if first_jpg is None:
                print(f"  ⚠️  [{num}] No images found")
                continue
            
            orientation, width, height = get_image_orientation(first_jpg)
            
            if orientation == 'landscape':
                label_landscape_count += 1
                landscape_folders.append((label, num, folder_path, width, height))
                print(f"  🏞️  [{num}] LANDSCAPE ({width}x{height})")
            elif orientation == 'portrait':
                print(f"  ✓ [{num}] Portrait ({width}x{height})")
            elif orientation == 'square':
                print(f"  ▢ [{num}] Square ({width}x{height})")
            else:
                print(f"  ❌ [{num}] Error reading image")
        
        if label_landscape_count > 0:
            print(f"\n  🏞️  {label_landscape_count} LANDSCAPE folders in '{label}'\n")
    
    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Labels checked: {total_labels_checked}")
    print(f"Total numbered folders checked: {total_numbered_folders}")
    print(f"Landscape folders found: {len(landscape_folders)}")
    print()
    
    if landscape_folders:
        print("LANDSCAPE FOLDERS LIST:")
        print("-" * 80)
        for label, num, path, width, height in landscape_folders:
            print(f"  {label}/{num}/ ({width}x{height})")
        print("-" * 80)
    else:
        print("✅ No landscape folders found!")
    
    print("=" * 80)

if __name__ == "__main__":
    main()


