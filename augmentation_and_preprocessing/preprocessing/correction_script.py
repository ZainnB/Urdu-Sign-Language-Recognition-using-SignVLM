#!/usr/bin/env python3
"""
Re-extract frames for selected videos where saved frames were rotated wrong.

Layout (matches SignVLM frame extraction + your dataset):

  <dataset_root>/<Split>/<Label>/<stem>.mp4
  <dataset_root>/<Split>/<Label>/<stem>/frame_000001.jpg

This replaces the old "rotate every PNG in-place" workflow with a fullre-decode from the source mp4 using PyAV (Unicode-safe on Windows), same as
`tools/extract_frames_signvlm.py`.

Default rotation heuristic (same idea as the previous script):
  If saved frame would be wider than tall (w > h), rotate 90° counter-clockwise
  after decode. Override with --rotate or --no-auto-rotate for specific labels.

Usage (from repo root):

  python augmentation_and_preprocessing/preprocessing/correction_script.py \\
    --dataset-root data \\
    --splits Train_full Val_aarij Test \\
    --ext jpg

Edit TARGETS below or pass --targets-file path/to/list.txt (one "Label/stem" per line).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Videos to re-extract: "LabelName" and numeric video stem (folder name == stem)
# ---------------------------------------------------------------------------
TARGETS: list[tuple[str, str]] = [
    ("body", "10"),
    ("body", "12"),
    ("body", "14"),
]

def load_targets_file(path: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace("\\", "/").strip("/").split("/")
        if len(parts) != 2:
            raise SystemExit(f"Bad targets line (want Label/stem): {raw!r}")
        out.append((parts[0], parts[1]))
    return out


def find_mp4(dataset_root: Path, splits: list[str], label: str, stem: str) -> Path | None:
    name = f"{stem}.mp4"
    for sp in splits:
        p = dataset_root / sp / label / name
        if p.is_file():
            return p
    return None


def clear_frame_dir(frame_dir: Path, *, exts: tuple[str, ...]) -> None:
    if not frame_dir.is_dir():
        return
    for ext in exts:
        for f in frame_dir.glob(f"*.{ext}"):
            try:
                f.unlink()
            except OSError:
                pass


def extract_frames(
    video_path: Path,
    frame_dir: Path,
    *,
    out_ext: str,
    rotate_ccw_deg: int,
) -> int:
    try:
        import av
    except Exception as e:
        raise SystemExit(f"PyAV required: pip install av\n{e}") from e

    frame_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        container = av.open(str(video_path))
    except Exception as e:
        print(f"[WARN] Cannot open {video_path}: {e}")
        return 0

    try:
        for i, frame in enumerate(container.decode(video=0), start=1):
            img = frame.to_image()
            if rotate_ccw_deg:
                img = img.rotate(rotate_ccw_deg, expand=True)
            out_path = frame_dir / f"frame_{i:06d}.{out_ext}"
            if out_ext.lower() in ("jpg", "jpeg"):
                img.save(str(out_path), quality=95)
            else:
                img.save(str(out_path))
            count += 1
    except Exception as e:
        print(f"[WARN] Decode error {video_path}: {e}")
    finally:
        try:
            container.close()
        except Exception:
            pass
    return count


def should_auto_rotate(label: str, frame_w: int, frame_h: int) -> bool:
    return frame_w > frame_h


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-extract frames for mis-rotated videos.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["Train_full", "Val_aarij", "Test"],
    )
    parser.add_argument("--ext", choices=["jpg", "png"], default="jpg")
    parser.add_argument(
        "--targets-file",
        type=Path,
        default=None,
        help="Optional file: one Label/stem per line (overrides embedded TARGETS).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done.",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        default=None,
        choices=(90, 180, 270),
        help="Force rotation (degrees CCW) for every target. Overrides auto heuristic.",
    )
    parser.add_argument(
        "--no-auto-rotate",
        action="store_true",
        help="Never apply w>h auto-rotation (unless --rotate is set).",
    )
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"--dataset-root not found: {root}")

    targets = load_targets_file(args.targets_file) if args.targets_file else TARGETS

    try:
        import av
    except Exception as e:
        raise SystemExit(f"PyAV required: pip install av\n{e}") from e

    print(f"dataset_root: {root}")
    print(f"splits:       {args.splits}")
    print(f"targets:      {len(targets)}")
    print(f"ext:          {args.ext}")
    print(f"dry_run:      {args.dry_run}")
    print()

    done = 0
    missing_mp4 = 0
    for label, stem in targets:
        mp4 = find_mp4(root, args.splits, label, stem)
        if mp4 is None:
            print(f"[MISS] {label}/{stem}.mp4 not found under splits")
            missing_mp4 += 1
            continue

        frame_dir = mp4.parent / mp4.stem
        rotate_deg: int
        if args.rotate is not None:
            rotate_deg = args.rotate
        elif args.no_auto_rotate:
            rotate_deg = 0
        else:
            try:
                container = av.open(str(mp4))
                try:
                    frame = next(container.decode(video=0))
                    pil = frame.to_image()
                    w, h = pil.size
                finally:
                    container.close()
            except Exception as e:
                print(f"[WARN] Probe failed {mp4}: {e} — skipping")
                continue
            rotate_deg = 270 if should_auto_rotate(label, w, h) else 0

        if args.dry_run:
            print(f"[DRY] {mp4} -> {frame_dir}/  rotate_ccw={rotate_deg}")
            done += 1
            continue

        clear_frame_dir(frame_dir, exts=("jpg", "jpeg", "png"))
        n = extract_frames(
            mp4,
            frame_dir,
            out_ext=args.ext,
            rotate_ccw_deg=rotate_deg,
        )
        print(f"[OK] {label}/{stem}  frames={n}  -> {frame_dir}")
        done += 1

    print()
    print(f"Processed: {done}  |  missing mp4: {missing_mp4}")


if __name__ == "__main__":
    main()
