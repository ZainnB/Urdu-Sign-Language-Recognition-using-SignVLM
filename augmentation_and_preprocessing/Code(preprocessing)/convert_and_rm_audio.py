#!/usr/bin/env python3
"""
convert_mov_to_mp4_noaudio.py

✅ Converts all .mov videos in INPUT_FOLDER → .mp4
✅ Removes audio track
✅ Preserves resolution (e.g., 720x1280), fps, and visual quality
✅ Uses ffmpeg (re-encodes video with libx264, CRF=18 high quality)

Requires: ffmpeg installed and on PATH
"""

from pathlib import Path
import subprocess
import shlex

# ---------- USER SETTINGS ----------
INPUT_FOLDER = Path.home() / "Documents" / "FYP_PSL" / "data" / "new_Unseen_data"
OUTPUT_FOLDER = INPUT_FOLDER / "mp4_noaudio"
VIDEO_EXTS = {".mp4"}
# -----------------------------------

def run_ffmpeg_convert(input_path: Path, output_path: Path):
    """
    Convert .mov to .mp4, remove audio, preserve video size/fps/quality.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-an",                   # ❌ remove audio
        "-c:v", "libx264",       # re-encode with high quality h264
        "-crf", "18",            # visual quality (lower = higher quality)
        "-preset", "veryfast",   # fast encoding
        "-pix_fmt", "yuv420p",   # ensures compatibility
        str(output_path)
    ]

    print(">", " ".join(shlex.quote(c) for c in cmd))
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"✅ Converted: {input_path.name} → {output_path.name}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to convert {input_path.name}: {e}")

def main():
    print(f"Input folder: {INPUT_FOLDER}")
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    files = [f for f in INPUT_FOLDER.iterdir() if f.suffix.lower() in VIDEO_EXTS and f.is_file()]
    if not files:
        print("No .mov videos found.")
        return

    for file in files:
        out_file = OUTPUT_FOLDER / (file.stem + ".mp4")
        # Avoid overwriting if already exists
        if out_file.exists():
            i = 1
            while (OUTPUT_FOLDER / f"{file.stem}_{i}.mp4").exists():
                i += 1
            out_file = OUTPUT_FOLDER / f"{file.stem}_{i}.mp4"

        run_ffmpeg_convert(file, out_file)

    print("\n🎬 All conversions done!")
    print(f"Converted .mp4 videos saved in: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()
