#!/usr/bin/env python3
"""
Build SignVLM split list files (TSV: ``path\\tlabel_int``).

This repo's training pipeline (`video_dataset/dataset.py`) expects:

- `--train_list_path` and `--val_list_path` pointing to list files where each
  line is:  <path>\\t<label_int>
- `--data_root` (or `--train_data_root` / `--val_data_root`) is prepended via
  `os.path.join(data_root, path)`.

This script supports two workflows:

1) **Directory-scan workflow (recommended for your current dataset structure)**:

Dataset layout:

  <dataset_root>/
    Train/<LabelName>/*.mp4
    Val/<LabelName>/*.mp4
    Test/<LabelName>/*.mp4

Run:

  python prepare_psl_splits.py --dataset-root "data/Final Dataset (with roi augmentation)"

Outputs (under `dataset_split_text_files/` by default):

  - `train_signer_signvlm.txt`
  - `val_signer_signvlm.txt`
  - `test_signer_signvlm.txt`

Optional (recommended for experiments):

  - `--write-tsv-aliases` → also `train.tsv`, `val.tsv`, `test.tsv` (same content)
  - `--write-label-map-json` → `label_map.json` (``{"0": "Afraid", ...}``)
  - `--write-train-1shot` → `train_1shot.tsv` (one line per class; prefers ``1.mp4``)

Paths in the outputs are written **relative to `--dataset-root`** so training can
use: `--data_root <dataset_root>`.

2) **Legacy convert workflow**:

Converts `train_files.txt` / `eval_files.txt` / `test_files.txt` files of the form
``rel_path*LabelName`` into SignVLM TSV list files.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Counter as CounterType
from typing import Dict, Iterable, List, Tuple


def load_label_map(path: Path) -> Dict[str, int]:
    """Load class name -> integer id.

    Supports:
      - ClassName:index  (e.g. PSL_recognition_label_map.txt)
      - index: ClassName
    """
    name_to_id: Dict[str, int] = {}
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or ":" not in line:
                continue
            left, right = line.split(":", 1)
            left, right = left.strip(), right.strip()
            if left.isdigit():
                name_to_id[right] = int(left)
            elif right.isdigit():
                name_to_id[left] = int(right)
            else:
                raise ValueError(f"Unrecognized label map line: {line!r}")
    return name_to_id


def _video_sort_key(p: Path) -> tuple:
    stem = p.stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem.casefold())


def _iter_split_videos(split_dir: Path) -> Iterable[Path]:
    """
    Yield all videos under:
      split_dir/<LabelName>/*.mp4

    Deterministic ordering: label folder (casefold) then numeric mp4 stem.
    """
    label_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()], key=lambda p: p.name.casefold())
    for label_dir in label_dirs:
        vids = sorted(label_dir.glob("*.mp4"), key=_video_sort_key)
        for v in vids:
            yield v


def _iter_split_stems_from_frames(split_dir: Path) -> Iterable[tuple[str, str]]:
    """
    Frames-only scan.

    Expected layout:
      split_dir/<LabelName>/<stem>/frame_*.jpg|png

    Yields (label_name, stem) pairs.
    """
    label_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()], key=lambda p: p.name.casefold())
    for label_dir in label_dirs:
        video_dirs = sorted([p for p in label_dir.iterdir() if p.is_dir()], key=lambda p: _video_sort_key(p))
        for vd in video_dirs:
            has_frames = any(vd.glob("*.jpg")) or any(vd.glob("*.png"))
            if not has_frames:
                continue
            yield (label_dir.name, vd.name)


def _write_split_from_scan(
    *,
    dataset_root: Path,
    split_name: str,
    split_dir: Path,
    out_path: Path,
    name_to_id: Dict[str, int],
) -> Tuple[int, CounterType[int]]:
    counts: CounterType[int] = Counter()
    written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fout:
        for vid in _iter_split_videos(split_dir):
            label_name = vid.parent.name
            if label_name not in name_to_id:
                raise SystemExit(
                    f"[{split_name}] label folder {label_name!r} not present in label map. "
                    f"Fix your --label-map or rename the folder."
                )
            label_int = name_to_id[label_name]
            rel = vid.relative_to(dataset_root).as_posix()
            fout.write(f"{rel}\t{label_int}\n")
            written += 1
            counts[label_int] += 1

    return written, counts


def _write_split_from_frames_scan(
    *,
    dataset_root: Path,
    split_name: str,
    split_dir: Path,
    out_path: Path,
    name_to_id: Dict[str, int],
) -> Tuple[int, CounterType[int]]:
    """
    Like `_write_split_from_scan`, but scans `.../<Label>/<stem>/frame_*.jpg|png`
    and writes the *virtual* mp4 path `.../<Label>/<stem>.mp4` to match
    `video_dataset/dataset.py` logic (it uses stem to locate the frame folder).
    """
    counts: CounterType[int] = Counter()
    written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fout:
        for label_name, stem in _iter_split_stems_from_frames(split_dir):
            if label_name not in name_to_id:
                raise SystemExit(
                    f"[{split_name}] label folder {label_name!r} not present in label map. "
                    f"Fix your --label-map or rename the folder."
                )
            label_int = name_to_id[label_name]
            # Write a path that `dataset.py` can join with data_root and then derive
            # the frame folder from: Path(path).parent / Path(path).stem
            virtual_mp4 = (split_dir.relative_to(dataset_root) / label_name / f"{stem}.mp4").as_posix()
            fout.write(f"{virtual_mp4}\t{label_int}\n")
            written += 1
            counts[label_int] += 1

    return written, counts


def id_to_name_map(name_to_id: Dict[str, int]) -> Dict[int, str]:
    """Invert label map; requires unique contiguous ids 0..N-1 for full coverage."""
    out: Dict[int, str] = {}
    for name, i in name_to_id.items():
        if i in out and out[i] != name:
            raise SystemExit(f"Duplicate id {i} for labels {out[i]!r} and {name!r}")
        out[i] = name
    return out


def write_label_map_json(name_to_id: Dict[str, int], out_path: Path) -> None:
    """Write {\"0\": \"Afraid\", ...} sorted by integer id."""
    inv = id_to_name_map(name_to_id)
    n = len(name_to_id)
    missing = set(range(n)) - set(inv.keys())
    if missing:
        print(f"Warning: label_map.json — missing ids: {sorted(missing)[:20]}...", file=sys.stderr)
    obj = {str(i): inv[i] for i in sorted(inv.keys())}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"wrote label_map.json ({len(obj)} entries) -> {out_path}")


def pick_one_shot_video(train_label_dir: Path, stem_prefer: str) -> Path | None:
    """Prefer ``{stem_prefer}.mp4``; else smallest numeric stem, else lexicographic."""
    prefer = train_label_dir / f"{stem_prefer}.mp4"
    if prefer.is_file():
        return prefer
    vids = sorted(train_label_dir.glob("*.mp4"), key=_video_sort_key)
    return vids[0] if vids else None


def pick_one_shot_frame_dir(train_label_dir: Path, stem_prefer: str) -> Path | None:
    """Frames-only layout: prefer <stem_prefer>/ subdir; else lowest-sorted subdir with frames."""
    prefer = train_label_dir / stem_prefer
    if prefer.is_dir() and (any(prefer.glob("*.jpg")) or any(prefer.glob("*.png"))):
        return prefer
    subdirs = sorted([p for p in train_label_dir.iterdir() if p.is_dir()], key=_video_sort_key)
    for sd in subdirs:
        if any(sd.glob("*.jpg")) or any(sd.glob("*.png")):
            return sd
    return None


def write_train_1shot(
    *,
    dataset_root: Path,
    train_dir: Path,
    out_path: Path,
    name_to_id: Dict[str, int],
    stem_prefer: str,
    frames_only: bool = False,
) -> int:
    """Exactly one line per class id in the label map from train split (deterministic).

    When ``frames_only=True``, scans ``<label>/<stem>/`` subdirectories instead of
    ``.mp4`` files and writes a virtual mp4 path so SignVLM can locate the frame folder.
    """
    inv = id_to_name_map(name_to_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for i in sorted(inv.keys()):
            label_name = inv[i]
            label_dir = train_dir / label_name
            if not label_dir.is_dir():
                raise SystemExit(f"train_1shot: missing label folder: {label_dir}")

            if frames_only:
                frame_dir = pick_one_shot_frame_dir(label_dir, stem_prefer)
                if frame_dir is None:
                    raise SystemExit(f"train_1shot: no frame subdir with images in {label_dir}")
                # Same virtual-mp4 path convention as _write_split_from_frames_scan
                virtual_mp4 = (train_dir.relative_to(dataset_root) / label_name / f"{frame_dir.name}.mp4").as_posix()
                fout.write(f"{virtual_mp4}\t{i}\n")
            else:
                vid = pick_one_shot_video(label_dir, stem_prefer)
                if vid is None:
                    raise SystemExit(f"train_1shot: no .mp4 in {label_dir}")
                rel = vid.relative_to(dataset_root).as_posix()
                fout.write(f"{rel}\t{i}\n")
            written += 1
    print(f"wrote train_1shot ({written} lines, 1 per class id) -> {out_path}")
    return written


def copy_alias(src: Path, dst: Path) -> None:
    if not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"alias: {src.name} -> {dst.name}")


def parse_split_line(line: str) -> Tuple[str, str]:
    line = line.strip()
    if not line:
        raise ValueError("empty line")
    if "*" not in line:
        raise ValueError(f"expected '*' separator: {line!r}")
    rel_path, label_str = line.split("*", 1)
    return rel_path.strip(), label_str.strip()


def to_signvlm_line(rel_path: str, label_str: str, project_root: Path, name_to_id: Dict[str, int]) -> str:
    rel_norm = rel_path.replace("\\", "/")
    abs_path = (project_root / rel_norm).resolve()
    if label_str not in name_to_id:
        raise KeyError(f"Label {label_str!r} not in label map")
    label_int = name_to_id[label_str]
    return f"{abs_path.as_posix()}\t{label_int}"


def convert_split(
    in_path: Path,
    out_path: Path,
    project_root: Path,
    name_to_id: Dict[str, int],
) -> Tuple[int, Counter]:
    counts: Counter = Counter()
    written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for lineno, raw in enumerate(fin, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rel_path, label_str = parse_split_line(raw)
                line = to_signvlm_line(rel_path, label_str, project_root, name_to_id)
            except (ValueError, KeyError) as e:
                raise SystemExit(f"{in_path}:{lineno}: {e}") from e
            fout.write(line + "\n")
            written += 1
            counts[name_to_id[label_str]] += 1
    return written, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SignVLM TSV splits for PSL.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "If provided, scan <dataset-root>/{Train,Val,Test}/<LabelName>/*.mp4 "
            "and write SignVLM TSV list files relative to dataset-root."
        ),
    )
    parser.add_argument("--train-dirname", type=str, default="Train", help="Name of training split folder under --dataset-root")
    parser.add_argument("--val-dirname", type=str, default="Val", help="Name of validation split folder under --dataset-root")
    parser.add_argument("--test-dirname", type=str, default="Test", help="Name of test split folder under --dataset-root")
    parser.add_argument(
        "--scan-frames-only",
        action="store_true",
        help=(
            "Scan frames-only layout: <Split>/<Label>/<stem>/frame_*.jpg|png instead of *.mp4. "
            "Writes virtual mp4 paths (<stem>.mp4) so SignVLM can locate the frame folders."
        ),
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("dataset_split_text_files"),
        help="Directory containing train_files.txt, eval_files.txt, test_files.txt",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=Path("trained_models_final/PSL_recognition_label_map.txt"),
        help="Path to PSL class name -> id map",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as --split-dir)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root for resolving relative video paths (default: cwd)",
    )
    parser.add_argument(
        "--write-tsv-aliases",
        action="store_true",
        help="Also write train.tsv, val.tsv, test.tsv (same content as the *_signer_signvlm.txt files).",
    )
    parser.add_argument(
        "--write-label-map-json",
        action="store_true",
        help="Write label_map.json under --out-dir (string keys \"0\"..\"N-1\" -> class names).",
    )
    parser.add_argument(
        "--write-train-1shot",
        action="store_true",
        help="Write train_1shot.tsv: one video per class from the train split (deterministic).",
    )
    parser.add_argument(
        "--one-shot-stem",
        type=str,
        default="1",
        help="Preferred mp4 stem for 1-shot, e.g. 1 -> 1.mp4 (default: 1).",
    )
    args = parser.parse_args()

    out_dir = (args.out_dir if args.out_dir is not None else args.split_dir).resolve()
    project_root = (args.project_root if args.project_root is not None else Path.cwd()).resolve()
    label_map_path = args.label_map
    if not label_map_path.is_absolute():
        label_map_path = (project_root / label_map_path).resolve()

    if not label_map_path.is_file():
        print(f"Error: label map not found: {label_map_path}", file=sys.stderr)
        sys.exit(1)

    name_to_id = load_label_map(label_map_path)
    num_classes = len(name_to_id)
    id_set = set(name_to_id.values())
    if id_set != set(range(num_classes)):
        print(
            f"Warning: label ids are not a contiguous 0..{num_classes - 1} range "
            f"(found {len(id_set)} ids).",
            file=sys.stderr,
        )

    print(f"project_root:   {project_root}")
    print(f"out_dir:        {out_dir}")
    print(f"label_map:      {label_map_path} ({num_classes} classes)")
    print()

    all_seen_labels: Counter = Counter()

    if args.dataset_root is not None:
        dataset_root = args.dataset_root
        if not dataset_root.is_absolute():
            dataset_root = (project_root / dataset_root).resolve()
        if not dataset_root.is_dir():
            print(f"Error: dataset root not found: {dataset_root}", file=sys.stderr)
            sys.exit(1)

        train_dir = dataset_root / args.train_dirname
        val_dir = dataset_root / args.val_dirname
        test_dir = dataset_root / args.test_dirname
        for d in (train_dir, val_dir, test_dir):
            if not d.is_dir():
                print(f"Error: missing split folder: {d}", file=sys.stderr)
                sys.exit(1)

        print(f"dataset_root:   {dataset_root}")
        print(f"train_dir:      {train_dir}")
        print(f"val_dir:        {val_dir}")
        print(f"test_dir:       {test_dir}")
        print()

        outputs = {
            "train": out_dir / "train_signer_signvlm.txt",
            "val": out_dir / "val_signer_signvlm.txt",
            "test": out_dir / "test_signer_signvlm.txt",
        }

        for split_name, split_path in (("train", train_dir), ("val", val_dir), ("test", test_dir)):
            dst = outputs[split_name]
            if args.scan_frames_only:
                n, cov = _write_split_from_frames_scan(
                    dataset_root=dataset_root,
                    split_name=split_name,
                    split_dir=split_path,
                    out_path=dst,
                    name_to_id=name_to_id,
                )
            else:
                n, cov = _write_split_from_scan(
                    dataset_root=dataset_root,
                    split_name=split_name,
                    split_dir=split_path,
                    out_path=dst,
                    name_to_id=name_to_id,
                )
            all_seen_labels.update(cov)
            unique_in_split = len(cov)
            print(f"{split_name}: wrote {n} lines -> {dst}")
            print(f"  unique labels in split: {unique_in_split} / {num_classes}")
            if n == 0:
                print(f"  warning: no .mp4 files found under {split_path}", file=sys.stderr)

        if args.write_tsv_aliases:
            copy_alias(out_dir / "train_signer_signvlm.txt", out_dir / "train.tsv")
            copy_alias(out_dir / "val_signer_signvlm.txt", out_dir / "val.tsv")
            copy_alias(out_dir / "test_signer_signvlm.txt", out_dir / "test.tsv")

        if args.write_label_map_json:
            write_label_map_json(name_to_id, out_dir / "label_map.json")

        if args.write_train_1shot:
            write_train_1shot(
                dataset_root=dataset_root,
                train_dir=train_dir,
                out_path=out_dir / "train_1shot.tsv",
                name_to_id=name_to_id,
                stem_prefer=args.one_shot_stem,
                frames_only=args.scan_frames_only,
            )
    else:
        split_dir = args.split_dir.resolve()
        if not split_dir.is_dir():
            print(f"Error: split dir not found: {split_dir}", file=sys.stderr)
            sys.exit(1)

        print(f"split_dir:      {split_dir}")
        print()

        mappings: List[Tuple[str, str, str]] = [
            ("train_files.txt", "train_signvlm.txt", "train"),
            ("eval_files.txt", "val_signvlm.txt", "val"),
            ("test_files.txt", "test_signvlm.txt", "test"),
        ]

        for src_name, dst_name, split_name in mappings:
            src = split_dir / src_name
            dst = out_dir / dst_name
            if not src.is_file():
                print(f"Error: missing input file: {src}", file=sys.stderr)
                sys.exit(1)
            n, cov = convert_split(src, dst, project_root, name_to_id)
            all_seen_labels.update(cov)
            unique_in_split = len(cov)
            print(f"{split_name}: wrote {n} lines -> {dst}")
            print(f"  unique labels in split: {unique_in_split} / {num_classes}")

    missing = sorted(set(range(num_classes)) - set(all_seen_labels.keys()))
    if missing:
        print()
        print(f"Label coverage (all splits combined): {len(all_seen_labels)} / {num_classes} ids appear")
        print(f"  ids with zero samples across train+val+test: {missing[:20]}{' ...' if len(missing) > 20 else ''}")
    else:
        print()
        print(f"Label coverage (all splits combined): all {num_classes} class ids appear at least once.")


if __name__ == "__main__":
    main()