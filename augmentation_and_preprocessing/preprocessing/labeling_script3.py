#!/usr/bin/env python3
"""
Unwrap all Chunk folders by moving their contents to parent directory.
- Reads from: data_only_frames/Train_full
- Finds all Chunk_<num> folders
- Moves all files from inside chunks to parent directory
- Removes empty chunk folders
"""

from pathlib import Path
import shutil

# ============================================================================
# USER CONFIG
# ============================================================================
ROOT_PATH = Path(r"C:\Users\Dell\Documents\Github-Repos\Urdu_Sign_Language_Recognition_using_SignVLM\data\data_only_frames\Train_full")

dry_run = False

# ============================================================================

def process_chunks():
    """Main processing function to unwrap chunk folders."""
    
    print("=" * 80)
    print("CHUNK FOLDER UNWRAPPER")
    print("=" * 80)
    print(f"Working directory: {ROOT_PATH}")
    print(f"Operation: Move all files from Chunk_* folders to parent, then delete chunks")
    print(f"Dry run: {dry_run}")
    print("=" * 80)
    print()
    
    # Verify root exists
    if not ROOT_PATH.exists():
        print(f"❌ Root path not found: {ROOT_PATH}")
        return
    
    # Find all Chunk_* folders
    chunk_folders = sorted([d for d in ROOT_PATH.iterdir() if d.is_dir() and d.name.startswith("Chunk_")])
    
    if not chunk_folders:
        print(f"⚠️  No Chunk_* folders found in {ROOT_PATH}")
        return
    
    print(f"Found {len(chunk_folders)} chunk folder(s):")
    for folder in chunk_folders:
        print(f"  - {folder.name}")
    print()
    
    total_moved_files = 0
    total_moved_dirs = 0
    total_removed_chunks = 0
    
    for chunk_folder in chunk_folders:
        print(f"📦 Processing: {chunk_folder.name}")
        
        # Get all items inside the chunk folder
        items = list(chunk_folder.iterdir())
        
        if not items:
            print(f"  ⚠️  Chunk folder is empty")
            if not dry_run:
                try:
                    chunk_folder.rmdir()
                    print(f"  ✓ Removed empty chunk: {chunk_folder.name}")
                    total_removed_chunks += 1
                except Exception as e:
                    print(f"  ❌ Failed to remove chunk: {e}")
            continue
        
        print(f"  Found {len(items)} item(s) to move")
        
        # Move all items from chunk to parent
        for item in items:
            dest_path = ROOT_PATH / item.name
            
            if dry_run:
                if item.is_file():
                    print(f"    [DRY-FILE] {item.name}")
                else:
                    print(f"    [DRY-DIR] {item.name}/")
            else:
                try:
                    # Handle case where destination exists
                    if dest_path.exists():
                        print(f"    ⚠️  Destination exists: {item.name}, skipping")
                        continue
                    
                    shutil.move(str(item), str(dest_path))
                    
                    if item.is_file():
                        print(f"    ✓ Moved file: {item.name}")
                        total_moved_files += 1
                    else:
                        print(f"    ✓ Moved dir: {item.name}/")
                        total_moved_dirs += 1
                        
                except Exception as e:
                    print(f"    ❌ Failed to move {item.name}: {e}")
        
        # Remove the now-empty chunk folder
        if not dry_run:
            try:
                chunk_folder.rmdir()
                print(f"  ✓ Removed chunk folder: {chunk_folder.name}")
                total_removed_chunks += 1
            except Exception as e:
                print(f"  ❌ Failed to remove chunk folder: {e}")
        
        print()
    
    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Chunk folders processed: {len(chunk_folders)}")
    print(f"Files moved: {total_moved_files}")
    print(f"Directories moved: {total_moved_dirs}")
    print(f"Chunk folders removed: {total_removed_chunks}")
    if dry_run:
        print("🔍 DRY RUN - no files were actually moved")
    else:
        print("✅ All chunks successfully unwrapped!")
    print("=" * 80)

if __name__ == "__main__":
    process_chunks()
