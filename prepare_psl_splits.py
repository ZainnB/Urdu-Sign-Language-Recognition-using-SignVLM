#!/usr/bin/env python3
"""
Convert PSL split lists (path*label) into tab-separated SignVLM list files
(path\\tlabel_int), compatible with signVLM/video_dataset/dataset.py.

Run from the FYP-I project root, e.g.:
  python signVLM/prepare_psl_splits.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


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
    args = parser.parse_args()

    split_dir = args.split_dir.resolve()
    out_dir = (args.out_dir if args.out_dir is not None else args.split_dir).resolve()
    project_root = (args.project_root if args.project_root is not None else Path.cwd()).resolve()
    label_map_path = args.label_map
    if not label_map_path.is_absolute():
        label_map_path = (project_root / label_map_path).resolve()

    if not split_dir.is_dir():
        print(f"Error: split dir not found: {split_dir}", file=sys.stderr)
        sys.exit(1)
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

    mappings: List[Tuple[str, str, str]] = [
        ("train_files.txt", "train_signvlm.txt", "train"),
        ("eval_files.txt", "val_signvlm.txt", "val"),
        ("test_files.txt", "test_signvlm.txt", "test"),
    ]

    print(f"project_root:   {project_root}")
    print(f"split_dir:      {split_dir}")
    print(f"out_dir:        {out_dir}")
    print(f"label_map:      {label_map_path} ({num_classes} classes)")
    print()

    all_seen_labels: Counter = Counter()

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
