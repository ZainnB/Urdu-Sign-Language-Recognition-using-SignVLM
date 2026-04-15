#!/usr/bin/env python3
"""
Extract frames for SignVLM training (Windows-friendly, Unicode-safe).

`video_dataset/dataset.py` expects, for each video:

  <split>/<Label>/<video>.mp4

to have frames available at:

  <split>/<Label>/<video>/frame_000001.jpg (or .png)
                        frame_000002.jpg
                        ...

This script builds exactly that structure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def is_unicode_path(path: Path) -> bool:
    """
    Match the behavior of `extract_frames_unicode.py`.
    Returns True when the *folder name* contains non-ASCII characters.
    """
    try:
        path.name.encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def iter_videos(root: Path, splits: list[str], *, unicode_only: bool) -> list[Path]:
    videos: list[Path] = []
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            raise SystemExit(f"Missing split folder: {split_dir}")
        # Expect: split/Label/*.mp4 (but we just rglob for robustness)
        videos.extend(sorted(split_dir.rglob("*.mp4")))
    # Filter out already-generated ROI videos if present
    videos = [v for v in videos if not v.stem.endswith("_roi")]
    if unicode_only:
        # Only keep videos whose *label folder* is Unicode-named (Urdu/Arabic)
        videos = [v for v in videos if is_unicode_path(v.parent)]
    return videos


def extract_one(video_path: Path, *, out_ext: str, overwrite: bool) -> int:
    frame_dir = video_path.parent / video_path.stem
    frame_dir.mkdir(parents=True, exist_ok=True)

    if not overwrite:
        existing = list(frame_dir.glob(f"*.{out_ext}"))
        if existing:
            return len(existing)

    try:
        import av  # PyAV
    except Exception as e:
        raise SystemExit(f"PyAV not installed or import failed: {e}\nInstall with: pip install av") from e

    count = 0
    try:
        # Use str(Path) (Unicode-safe on Windows). This mirrors `extract_frames_unicode.py`.
        container = av.open(str(video_path))
    except Exception as e:
        print(f"[WARN] Cannot open {video_path}: {e}")
        return 0

    try:
        for i, frame in enumerate(container.decode(video=0), start=1):
            img = frame.to_image()  # PIL Image
            out_path = frame_dir / f"frame_{i:06d}.{out_ext}"
            if out_ext.lower() == "jpg" or out_ext.lower() == "jpeg":
                img.save(str(out_path), quality=95)
            else:
                img.save(str(out_path))
            count += 1
    except Exception as e:
        print(f"[WARN] Error decoding {video_path}: {e}")
    finally:
        try:
            container.close()
        except Exception:
            pass

    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to the folder containing Train/Val/Test (or your split folders).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["Train_full", "Val_aarij", "Test"],
        help='Split folder names under --dataset-root (default: "Train_full Val_aarij Test")',
    )
    parser.add_argument(
        "--unicode-only",
        action="store_true",
        help="Only process label folders with non-ASCII names (Urdu/Arabic).",
    )
    parser.add_argument("--ext", choices=["jpg", "png"], default="jpg", help="Output frame format.")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract even if frames already exist.")
    parser.add_argument("--limit", type=int, default=0, help="If >0, only process first N videos (debug).")
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"--dataset-root not found: {root}")

    videos = iter_videos(root, args.splits, unicode_only=args.unicode_only)
    if args.limit and args.limit > 0:
        videos = videos[: args.limit]

    print(f"dataset_root: {root}")
    print(f"splits:       {args.splits}")
    print(f"videos:       {len(videos)}")
    print(f"unicode_only: {args.unicode_only}")
    print(f"ext:          {args.ext}")
    print(f"overwrite:    {args.overwrite}")
    if args.limit:
        print(f"limit:        {args.limit}")
    print()

    total_frames = 0
    failed: list[Path] = []
    for idx, vid in enumerate(videos, start=1):
        n = extract_one(vid, out_ext=args.ext, overwrite=args.overwrite)
        if n == 0:
            failed.append(vid)
        total_frames += n
        if idx % 25 == 0 or idx == len(videos):
            print(f"[{idx}/{len(videos)}] frames saved so far: {total_frames:,}")
            sys.stdout.flush()

    print()
    print(f"Done. videos: {len(videos)} | total frames: {total_frames:,} | failed: {len(failed)}")
    if failed:
        print("Failed videos (first 20):")
        for v in failed[:20]:
            print(f"  {v}")


if __name__ == "__main__":
    main()

