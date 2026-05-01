#!/usr/bin/env python3
"""
crop_long_videos.py

Crop videos longer than 5 seconds in SRC in place, organized by label.
Cropping: remove first 0.5s and last 1s, replace original videos.

- For each LABEL in LABELS:
    scan SRC/LABEL for video files
    check duration using ffprobe
    crop videos >5 seconds in place
- Output a list of cropped videos grouped by label.
"""

from pathlib import Path
import subprocess
import json
import shlex
import shutil

# ---------------- USER CONFIG ----------------
SRC = Path(r'C:\Users\Dell\Documents\Github-Repos\Urdu_Sign_Language_Recognition_using_SignVLM\data\Random_test')  # Base directory containing videos directly

VIDEO_EXTS = {".mov", ".mp4", ".mkv", ".avi", ".wmv", ".flv", ".mpg", ".mpeg", ".3gp"}
START_TRIM = 0.0  # seconds to trim from start
END_TRIM = 0.5    # seconds to trim from end
# ------------------------------------------------

# Processing single-folder of videos; no LABELS list needed.

def ffprobe_meta(path: Path):
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=10)
        return json.loads(out)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None

def get_duration(path: Path):
    meta = ffprobe_meta(path)
    if not meta:
        return None
    try:
        return float(meta.get("format", {}).get("duration"))
    except Exception:
        return None

def run_cmd(cmd):
    print(">", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)

def crop_video(input_path: Path, output_path: Path, start_trim: float, end_trim: float):
    duration = get_duration(input_path)
    if duration is None:
        print(f"Skipping {input_path.name} (no duration)")
        return False
    if duration <= start_trim + end_trim:
        print(f"Skipping {input_path.name} (too short after trim: {duration:.2f}s)")
        return False

    start = start_trim
    trimmed_duration = duration - start_trim - end_trim

    # Use a temporary file to avoid in-place editing issue
    temp_path = input_path.with_suffix('.tmp' + input_path.suffix)
    output_path = temp_path

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(input_path),
        "-t", f"{trimmed_duration:.3f}",
        "-c", "copy",
        str(output_path)
    ]

    try:
        run_cmd(cmd)
        # Move temp to original
        shutil.move(str(temp_path), str(input_path))
        print(f"✅ Cropped: {input_path.name} → replaced in place")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed: {input_path.name}")
        # Clean up temp file if exists
        if temp_path.exists():
            temp_path.unlink()
        return False

def list_files_in(folder: Path):
    """Return list of file paths in folder (non-recursive). Empty list if folder missing."""
    if not folder.exists() or not folder.is_dir():
        return []
    return [p for p in sorted(folder.iterdir()) if p.is_file()]

def main():
    print("SRC:", SRC)
    print("Cropping .mp4 videos in SRC (non-recursive)...")
    print()

    long_videos = []
    cropped_videos = []

    files = list_files_in(SRC)
    for file in files:
        if file.suffix.lower() == ".mp4":
            duration = get_duration(file)
            if duration and duration > 3.5:
                long_videos.append((file.name, duration))
                if crop_video(file, file, START_TRIM, END_TRIM):
                    new_duration = duration - START_TRIM - END_TRIM
                    cropped_videos.append((file.name, new_duration))

    # Output the list of long videos
    print("Original videos longer than 5 seconds:")
    for name, dur in long_videos:
        print(f"  {name} ({dur:.2f}s)")

    total_long = len(long_videos)
    print(f"\nTotal long videos found: {total_long}")

    # Output the list of cropped videos
    print("\nCropped videos (in place):")
    for name, dur in cropped_videos:
        print(f"  {name} ({dur:.2f}s)")

    total_cropped = len(cropped_videos)
    print(f"\nTotal videos cropped: {total_cropped}")

    print("\n✅ Done! Videos cropped in place.")

if __name__ == "__main__":
    main()
