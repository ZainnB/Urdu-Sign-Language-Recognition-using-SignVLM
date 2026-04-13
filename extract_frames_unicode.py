"""
Extract video frames for folders with Unicode (Urdu/Arabic) names.

Uses PyAV instead of OpenCV — PyAV handles non-ASCII paths on Windows correctly.

For each video at:
    Data/{split}/{class}/{video}.mp4

Writes frames to:
    Data/{split}/{class}/{video}/frame_000001.jpg
                                frame_000002.jpg
                                ...

This is the exact structure expected by SignVLM --frames_available 1.
Only processes folders whose names contain non-ASCII characters (Urdu classes).
Use --all to also reprocess English/ASCII classes.
"""

import sys
import argparse
from pathlib import Path
from PIL import Image
import io

try:
    import av
except ImportError:
    sys.exit("PyAV not found. Run: pip install av")


DATA_DIRS = [
    r"D:\FAST\zSemesters\7th Semester\FYP-I\Data\train_data",
    r"D:\FAST\zSemesters\7th Semester\FYP-I\Data\validation_data",
    r"D:\FAST\zSemesters\7th Semester\FYP-I\Data\test_data",
]


def is_unicode_path(path: Path) -> bool:
    try:
        path.name.encode('ascii')
        return False
    except UnicodeEncodeError:
        return True


def extract_video_pyav(mp4_path: Path, overwrite: bool = False) -> int:
    frame_dir = mp4_path.parent / mp4_path.stem
    frame_dir.mkdir(parents=True, exist_ok=True)

    if not overwrite:
        existing = list(frame_dir.glob("*.jpg"))
        if existing:
            return len(existing)

    try:
        container = av.open(str(mp4_path))
    except Exception as e:
        print(f"  [WARN] Cannot open {mp4_path.name}: {e}")
        return 0

    count = 0
    try:
        for i, frame in enumerate(container.decode(video=0), start=1):
            img = frame.to_image()  # PIL Image
            out_path = frame_dir / f"frame_{i:06d}.jpg"
            img.save(str(out_path), quality=95)
            count += 1
    except Exception as e:
        print(f"  [WARN] Error mid-extraction {mp4_path.name}: {e}")
    finally:
        container.close()

    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true',
                        help='Process all classes, not just Unicode ones')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-extract even if frames already exist')
    args = parser.parse_args()

    total_videos = 0
    total_frames = 0
    failed = []

    for data_dir in DATA_DIRS:
        root = Path(data_dir)
        if not root.exists():
            print(f"[SKIP] Not found: {root}")
            continue

        videos = sorted(root.rglob("*.mp4"))
        videos = [v for v in videos if not v.stem.endswith("_roi")]

        if not args.all:
            # Only process videos whose class folder has Unicode (Urdu) name
            videos = [v for v in videos if is_unicode_path(v.parent)]

        if not videos:
            print(f"[SKIP] No matching videos in {root.name}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {len(videos)} videos in: {root.name}")
        print(f"{'='*60}")
        sys.stdout.flush()

        for i, vid in enumerate(videos, 1):
            n = extract_video_pyav(vid, overwrite=args.overwrite)
            if n == 0:
                failed.append(str(vid))
            total_frames += n
            total_videos += 1

            if i % 25 == 0 or i == len(videos):
                pct = i / len(videos) * 100
                print(f"  [{i}/{len(videos)}] {pct:.0f}%  frames saved: {total_frames:,}")
                sys.stdout.flush()

    print(f"\n{'='*60}")
    print(f"Done. {total_videos} videos | {total_frames:,} frames saved.")
    if failed:
        print(f"\n[WARN] {len(failed)} videos failed:")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    main()
