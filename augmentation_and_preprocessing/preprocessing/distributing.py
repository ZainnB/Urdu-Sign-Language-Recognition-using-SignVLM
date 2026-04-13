#!/usr/bin/env python3
"""
list_long_videos.py

List all videos longer than 5 seconds in the directory SRC, organized by label.

- For each LABEL in LABELS:
    scan SRC/LABEL for video files
    check duration using ffprobe
    collect videos >5 seconds
- Output a list of all such videos grouped by label.
"""

from pathlib import Path
import subprocess
import json

# ---------------- USER CONFIG ----------------
SRC = Path(r'C:\Users\Dell\Documents\FYP_PSL\data\Final Dataset (without augmented data)')

VIDEO_EXTS = {".mp4"}
# ------------------------------------------------

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

def list_files_in(folder: Path):
    """Return list of file paths in folder (non-recursive). Empty list if folder missing."""
    if not folder.exists() or not folder.is_dir():
        return []
    return [p for p in sorted(folder.iterdir()) if p.is_file()]

def main():
    print("SRC:", SRC)
    print("Scanning for videos longer than 5 seconds...")
    print()

    long_videos = {}

    for label in LABELS:
        src_label = SRC / label
        files = list_files_in(src_label)
        label_long = []
        for file in files:
            if file.suffix.lower() in VIDEO_EXTS:
                duration = get_duration(file)
                if duration and duration > 4:
                    label_long.append((file.name, duration))
        if label_long:
            long_videos[label] = label_long

    # Output the list
    print("Videos longer than 5 seconds:")
    for label, videos in long_videos.items():
        print(f"\n{label}:")
        for name, dur in videos:
            print(f"  {name} ({dur:.2f}s)")

    total = sum(len(v) for v in long_videos.values())
    print(f"\nTotal long videos found: {total}")

if __name__ == "__main__":
    main()
