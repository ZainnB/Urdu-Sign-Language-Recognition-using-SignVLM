from pathlib import Path
import subprocess
import shlex

# ---------- USER SETTINGS ----------
INPUT_FOLDER = Path.home() / "Documents" / "PSL_Dataset" / "AbdurRehman"
OUTPUT_FOLDER = INPUT_FOLDER / "no_audio"
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".wmv", ".flv"}
# -----------------------------------

def remove_audio(input_path: Path, output_path: Path):
    """
    Remove audio track using FFmpeg without re-encoding video.
    """
    cmd = [
        "ffmpeg", "-y",              # overwrite output if exists
        "-i", str(input_path),
        "-c", "copy",                # copy video stream (no re-encode)
        "-an",                       # remove audio
        str(output_path)
    ]

    print(">", " ".join(shlex.quote(c) for c in cmd))
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"✅ Audio removed: {input_path.name} → {output_path.name}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed for {input_path.name}: {e}")

def main():
    print(f"Input folder: {INPUT_FOLDER}")
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    files = [f for f in INPUT_FOLDER.iterdir() if f.suffix.lower() in VIDEO_EXTS and f.is_file()]
    if not files:
        print("No video files found in", INPUT_FOLDER)
        return

    for file in files:
        out_file = OUTPUT_FOLDER / file.name
        if out_file.exists():
            i = 1
            while (OUTPUT_FOLDER / f"{file.stem}_{i}{file.suffix}").exists():
                i += 1
            out_file = OUTPUT_FOLDER / f"{file.stem}_{i}{file.suffix}"

        remove_audio(file, out_file)

    print("\n🎬 All done! Videos without audio saved in:", OUTPUT_FOLDER)

if __name__ == "__main__":
    main()
