"""
Pre-extract video frames to disk as PNG images.

For each video at:
    Data/{split}/{class}/{video}.mp4

Creates a sibling folder:
    Data/{split}/{class}/{video}/frame_000001.png
                                frame_000002.png
                                ...

This is the structure expected by SignVLM when --frames_available 1 is set.
"""

import cv2
import sys
from pathlib import Path


DATA_DIRS = [
    r"D:\FAST\zSemesters\7th Semester\FYP-I\Data\train_data",
    r"D:\FAST\zSemesters\7th Semester\FYP-I\Data\validation_data",
    r"D:\FAST\zSemesters\7th Semester\FYP-I\Data\test_data",
]


def extract_video(mp4_path: Path, overwrite: bool = False) -> int:
    frame_dir = mp4_path.parent / mp4_path.stem
    if frame_dir.exists() and not overwrite:
        existing = list(frame_dir.glob("*.png"))
        if existing:
            return len(existing)  # already done

    frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open: {mp4_path}")
        return 0

    idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = frame_dir / f"frame_{idx:06d}.png"
        cv2.imwrite(str(out_path), frame)
        idx += 1
    cap.release()
    return idx - 1


def main():
    total_videos = 0
    total_frames = 0
    failed = []

    for data_dir in DATA_DIRS:
        root = Path(data_dir)
        if not root.exists():
            print(f"[SKIP] Directory not found: {root}")
            continue

        videos = sorted(root.rglob("*.mp4"))
        # Exclude any *_roi.mp4 files
        videos = [v for v in videos if not v.stem.endswith("_roi")]
        print(f"\n{'='*60}")
        print(f"Processing {len(videos)} videos in: {root.name}")
        print(f"{'='*60}")

        for i, vid in enumerate(videos, 1):
            frame_count = extract_video(vid)
            if frame_count == 0:
                failed.append(str(vid))
            else:
                total_frames += frame_count
            total_videos += 1

            if i % 50 == 0 or i == len(videos):
                pct = i / len(videos) * 100
                print(f"  [{i}/{len(videos)}] {pct:.0f}%  |  frames so far: {total_frames:,}")
                sys.stdout.flush()

    print(f"\n{'='*60}")
    print(f"Done. {total_videos} videos processed, {total_frames:,} frames saved.")
    if failed:
        print(f"\nFailed to open {len(failed)} videos:")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    main()
