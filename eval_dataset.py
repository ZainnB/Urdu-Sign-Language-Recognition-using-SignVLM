"""
eval_dataset.py
───────────────
Batch accuracy evaluation of SignVLM on a *structured* dataset split folder.

Expected folder hierarchy
─────────────────────────
    <data_dir>/
        <ClassName>/
            <stem>.mp4           ← raw video  (used when no frame subfolder exists)
            <stem>/              ← pre-extracted frames  (preferred — no re-extraction)
                frame_0001.jpg   ← or *.png — any sorted *.jpg / *.png files work
                frame_0002.jpg
                ...

Frame-folder detection
───────────────────────
For every video <ClassName>/<stem>.mp4 the script checks whether
<ClassName>/<stem>/ exists and contains image files.
  • Frames found  →  load from disk  (fast, labelled [f])
  • Frames absent →  decode .mp4 with PyAV  (fallback, labelled [v])

Frame-only entries (a subfolder with no matching .mp4 sibling) are also
detected and processed using the pre-extracted frames.

Usage
─────
    # Evaluate the Test split (class subfolders, pre-extracted frames where available)
    python eval_dataset.py --data_dir Data/Test

    # Override paths or hyper-parameters
    python eval_dataset.py \\
        --data_dir Data/Val \\
        --checkpoint trained_models_final/models/fullshot_16frames.pth \\
        --backbone_path CLIP_weights/ViT-L/ViT-L-14.pt \\
        --label_map trained_models_final/PSL_recognition_label_map_run0.txt \\
        --num_frames 16 --sampling_rate 4

Key:  ✓ = top-1 correct   ~ = in top-5   ✗ = missed
      [f] = loaded from pre-extracted frames   [v] = decoded from video
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from inference import (
    CLIP_MEAN, CLIP_STD,
    PSL_NUM_FRAMES, PSL_SAMPLING_RATE, PSL_SPATIAL_SIZE,
    _resample_indices,
    decode_video,
    frames_to_tensor,
    load_label_map,
    load_model,
    sample_indices_center,
)

# ── Default paths (relative to project root) ───────────────────────────────
_ROOT      = _DIR
CHECKPOINT = _ROOT / "trained_models_final/models/augmentation_16frames_run5.pth"
BACKBONE   = _ROOT / "CLIP_weights/ViT-L/ViT-L-14.pt"
LABEL_MAP  = _ROOT / "trained_models_final/PSL_recognition_label_map_run1-5.txt"


# ── Frame / tensor helpers ─────────────────────────────────────────────────

def _load_frames_from_dir(
    frame_dir: Path,
    num_frames: int,
    sampling_rate: int,
    spatial_size: int,
) -> Optional[torch.Tensor]:
    """
    Load pre-extracted frames (*.png preferred, *.jpg fallback) from a directory,
    apply center-temporal sampling, resize/crop, and normalise.

    Returns a (1, 3, num_frames, spatial_size, spatial_size) tensor or None if
    the directory contains no image files.
    """
    image_files = sorted(frame_dir.glob("*.png"))
    if not image_files:
        image_files = sorted(frame_dir.glob("*.jpg"))
    if not image_files:
        return None

    # Center-temporal sampling on the available frames
    seg_len = (num_frames - 1) * sampling_rate + 1
    total = len(image_files)
    if total < seg_len:
        indices = _resample_indices(total, num_frames)
    else:
        mid_start = (total - seg_len) // 2
        indices = list(range(mid_start, mid_start + num_frames * sampling_rate, sampling_rate))

    raw: List[np.ndarray] = []
    for i in indices:
        idx = min(i, total - 1)
        raw.append(np.array(Image.open(str(image_files[idx])).convert("RGB")))

    # frames_to_tensor expects (raw_frames, indices_to_pick); pass identity indices
    return frames_to_tensor(raw, list(range(len(raw))), spatial_size)


def _get_tensor(
    video_path: Path,
    num_frames: int,
    sampling_rate: int,
    spatial_size: int,
) -> Tuple[torch.Tensor, str]:
    """
    Return (tensor, source_tag) for a video entry.

    Checks for a pre-extracted frame subfolder first:
        <video_path.parent> / <video_path.stem> /  (e.g. Afraid/1/)
    Falls back to PyAV video decoding if frames are absent.

    source_tag is 'frames' or 'video'.
    """
    frame_dir = video_path.parent / video_path.stem

    if frame_dir.is_dir():
        tensor = _load_frames_from_dir(frame_dir, num_frames, sampling_rate, spatial_size)
        if tensor is not None:
            return tensor, "frames"

    # Fall back: PyAV decode
    if not video_path.is_file():
        raise FileNotFoundError(
            f"No frame subfolder and no video file found for stem '{video_path.stem}' "
            f"in {video_path.parent}"
        )
    raw_frames = decode_video(video_path)
    if not raw_frames:
        raise ValueError(f"No frames decoded from {video_path}")
    indices = sample_indices_center(len(raw_frames), num_frames, sampling_rate)
    return frames_to_tensor(raw_frames, indices, spatial_size), "video"


# ── Dataset scanning ────────────────────────────────────────────────────────

def _build_name_to_id(id_to_name: Dict[int, str]) -> Dict[str, int]:
    return {name.lower(): cid for cid, name in id_to_name.items()}


def _scan_class_dir(
    class_dir: Path,
    class_id: int,
) -> List[Tuple[Path, int, str]]:
    """
    Return list of (virtual_or_real_mp4_path, class_id, class_name) for
    all processable items in one class folder.

    Logic:
      1. Collect all *.mp4 stems.
      2. Add each *.mp4 as an item (frame subfolder detection happens in _get_tensor).
      3. If a subfolder exists with frames but has NO matching *.mp4, add it as
         a frame-only item (virtual path — _get_tensor will load from the dir).
    """
    items: List[Tuple[Path, int, str]] = []
    mp4_stems = {v.stem for v in class_dir.glob("*.mp4")}

    # Real video files
    for mp4 in sorted(class_dir.glob("*.mp4")):
        items.append((mp4, class_id, class_dir.name))

    # Frame-only subdirs (no matching .mp4 sibling)
    for sub in sorted(class_dir.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name in mp4_stems:
            continue  # already covered by the .mp4 entry above
        has_frames = any(sub.glob("*.png")) or any(sub.glob("*.jpg"))
        if has_frames:
            # Virtual path: _get_tensor will find the frame dir and use it
            virtual = class_dir / f"{sub.name}.mp4"
            items.append((virtual, class_id, class_dir.name))

    return items


def collect_all_items(
    data_dir: Path,
    label_map: Dict[int, str],
) -> Tuple[List[Tuple[Path, int, str]], List[str]]:
    """
    Walk data_dir/<ClassName>/ and collect all (path, class_id, class_name) items.
    Returns (items, skipped_class_names).
    """
    name_to_id = _build_name_to_id(label_map)
    items: List[Tuple[Path, int, str]] = []
    skipped: List[str] = []

    class_dirs = sorted(
        [d for d in data_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name.casefold(),
    )

    for class_dir in class_dirs:
        class_id = name_to_id.get(class_dir.name.lower())
        if class_id is None:
            skipped.append(class_dir.name)
            continue
        items.extend(_scan_class_dir(class_dir, class_id))

    return items, skipped


# ── Evaluation loop ────────────────────────────────────────────────────────

def run_eval(
    data_dir: Path,
    model: torch.nn.Module,
    label_map: Dict[int, str],
    device: torch.device,
    num_frames: int = PSL_NUM_FRAMES,
    sampling_rate: int = PSL_SAMPLING_RATE,
    spatial_size: int = PSL_SPATIAL_SIZE,
) -> dict:

    items, skipped = collect_all_items(data_dir, label_map)

    if skipped:
        print(f"  [skip] Class folders not found in label map: {skipped}")
    if not items:
        print(f"  [warn] No processable items found under {data_dir}")
        return {}

    print(f"  Found {len(items)} items across "
          f"{len({c for _, c, _ in items})} classes.\n")

    num_classes  = max(label_map.keys()) + 1
    conf_matrix  = np.zeros((num_classes, num_classes), dtype=np.int64)
    top1_hits = top5_hits = total = 0
    per_video:  List[tuple] = []
    per_class:  Dict[int, dict] = {}

    for video_path, gt_id, class_name in items:
        try:
            tensor, source = _get_tensor(video_path, num_frames, sampling_rate, spatial_size)
            tensor = tensor.to(device)

            with torch.no_grad(), torch.amp.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                logits = model(tensor)

            probs    = F.softmax(logits, dim=-1).squeeze(0).cpu()
            top5_ids = probs.argsort(descending=True)[:5].tolist()
            top1_id  = top5_ids[0]

            hit1 = int(top1_id == gt_id)
            hit5 = int(gt_id in top5_ids)
            top1_hits += hit1
            top5_hits += hit5
            total += 1

            conf_matrix[gt_id, top1_id] += 1

            if gt_id not in per_class:
                per_class[gt_id] = {"name": class_name, "hit1": 0, "hit5": 0, "total": 0}
            per_class[gt_id]["hit1"]  += hit1
            per_class[gt_id]["hit5"]  += hit5
            per_class[gt_id]["total"] += 1

            mark = "✓" if hit1 else ("~" if hit5 else "✗")
            per_video.append((
                mark, video_path.name, class_name,
                label_map.get(top1_id, f"class_{top1_id}"),
                float(probs[top1_id]) * 100,
                source,
            ))

        except Exception as exc:
            print(f"  [error] {class_name}/{video_path.name}: {exc}")

    return {
        "total":       total,
        "top1":        top1_hits,
        "top5":        top5_hits,
        "top1_pct":    top1_hits / total * 100 if total else 0.0,
        "top5_pct":    top5_hits / total * 100 if total else 0.0,
        "per_video":   per_video,
        "per_class":   per_class,
        "conf_matrix": conf_matrix,
        "num_classes": num_classes,
    }


# ── Classification metrics (one-vs-rest) ──────────────────────────────────

def compute_ovr_metrics(conf_matrix: np.ndarray, seen_ids: List[int]) -> dict:
    """
    Compute per-class TP, TN, FP, FN, Precision, Recall, F1 using the
    one-vs-rest (OvR) interpretation of the N×N confusion matrix.

    Only classes that actually appeared in this split (seen_ids) are included
    so classes with zero support don't distort macro/weighted averages.

        TP[c] = cm[c, c]               (predicted c AND truly c)
        FP[c] = col_sum[c] - cm[c, c]  (predicted c but NOT truly c)
        FN[c] = row_sum[c] - cm[c, c]  (truly c but NOT predicted c)
        TN[c] = total - TP - FP - FN   (neither truly c nor predicted c)
    """
    cm      = conf_matrix.astype(float)
    total   = cm.sum()

    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP   # column sum minus diagonal
    FN = cm.sum(axis=1) - TP   # row sum minus diagonal
    TN = total - TP - FP - FN

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where((TP + FP) > 0, TP / (TP + FP), 0.0)
        recall    = np.where((TP + FN) > 0, TP / (TP + FN), 0.0)
        f1        = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )

    support = cm.sum(axis=1)   # true samples per class

    # Restrict averages to classes that actually appear in this split
    idx = np.array(seen_ids)
    sup_seen = support[idx]
    w = sup_seen / sup_seen.sum() if sup_seen.sum() > 0 else np.ones(len(idx)) / len(idx)

    macro = {
        "precision": precision[idx].mean(),
        "recall":    recall[idx].mean(),
        "f1":        f1[idx].mean(),
    }
    weighted = {
        "precision": (precision[idx] * w).sum(),
        "recall":    (recall[idx]    * w).sum(),
        "f1":        (f1[idx]        * w).sum(),
    }

    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "support":   support,
        "macro":     macro,
        "weighted":  weighted,
    }


def print_metrics_table(
    metrics: dict,
    label_map: Dict[int, str],
    seen_ids: List[int],
) -> None:
    """
    Print per-class 2×2 OvR table  +  precision / recall / F1  + macro/weighted averages.
    """
    TP   = metrics["TP"]
    TN   = metrics["TN"]
    FP   = metrics["FP"]
    FN   = metrics["FN"]
    prec = metrics["precision"]
    rec  = metrics["recall"]
    f1   = metrics["f1"]
    ma   = metrics["macro"]
    wa   = metrics["weighted"]

    W = 100  # table width

    print(f"\n{'═'*W}")
    print("  CLASSIFICATION METRICS  (one-vs-rest per class)")
    print(f"{'═'*W}")
    hdr = (
        f"  {'#':>4}  {'Class':<26}  "
        f"{'TP':>6}  {'FP':>6}  {'FN':>6}  {'TN':>8}  "
        f"{'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'Supp':>5}  F1 bar"
    )
    print(hdr)
    print(f"  {'─'*96}")

    def _f1_bar(v: float, width: int = 16) -> str:
        filled = int(round(v * width))
        return "█" * filled + "░" * (width - filled)

    for cid in sorted(seen_ids):
        name = label_map.get(cid, str(cid))[:26]
        tp   = int(TP[cid]);  fp = int(FP[cid])
        fn   = int(FN[cid]);  tn = int(TN[cid])
        p    = prec[cid];     r  = rec[cid];  f = f1[cid]
        sup  = int(TP[cid] + FN[cid])
        print(
            f"  {cid:>4}  {name:<26}  "
            f"{tp:>6}  {fp:>6}  {fn:>6}  {tn:>8}  "
            f"{p:>7.3f}  {r:>7.3f}  {f:>7.3f}  {sup:>5}  {_f1_bar(f)}"
        )

    print(f"  {'─'*96}")
    print(
        f"  {'Macro avg':<32}  "
        f"{'':>6}  {'':>6}  {'':>6}  {'':>8}  "
        f"{ma['precision']:>7.3f}  {ma['recall']:>7.3f}  {ma['f1']:>7.3f}"
    )
    print(
        f"  {'Weighted avg':<32}  "
        f"{'':>6}  {'':>6}  {'':>6}  {'':>8}  "
        f"{wa['precision']:>7.3f}  {wa['recall']:>7.3f}  {wa['f1']:>7.3f}"
    )
    print()


# ── Reporting ──────────────────────────────────────────────────────────────

def _bar(value: float, width: int = 20) -> str:
    filled = int(value / 100 * width)
    return "█" * filled + "░" * (width - filled)


def print_report(stats: dict, label_map: Dict[int, str]) -> None:
    if not stats:
        return

    # Per-video table
    print(f"\n  {'':4} {'File':<24} {'Truth':<22} {'Top-1 Pred':<22} {'Conf':>6}  Src")
    print(f"  {'':4} {'----':<24} {'-----':<22} {'----------':<22} {'----':>6}  ---")
    for mark, fname, gt, pred, conf, src in stats["per_video"]:
        tag = "[f]" if src == "frames" else "[v]"
        print(f"  {mark:<4} {fname:<24} {gt:<22} {pred:<22} {conf:>5.1f}%  {tag}")

    # Per-class accuracy table
    print(f"\n{'─'*80}")
    print("  PER-CLASS ACCURACY  (top-1 / top-5)")
    print(f"{'─'*80}")
    print(f"  {'Class':<28} {'Top-1':>12}  {'Top-5':>12}  Bar (top-1)")
    print(f"  {'─'*26:<28} {'─'*10:>12}  {'─'*10:>12}")
    for cid in sorted(stats["per_class"].keys()):
        c = stats["per_class"][cid]
        t = c["total"]
        t1_pct = c["hit1"] / t * 100 if t else 0.0
        t5_pct = c["hit5"] / t * 100 if t else 0.0
        print(
            f"  {c['name']:<28} "
            f"{c['hit1']:>2}/{t:<3} {t1_pct:>5.1f}%  "
            f"{c['hit5']:>2}/{t:<3} {t5_pct:>5.1f}%  "
            f"{_bar(t1_pct)}"
        )

    # Overall accuracy summary
    print(f"\n{'='*80}")
    print(
        f"  SUMMARY  "
        f"top-1: {stats['top1']}/{stats['total']} ({stats['top1_pct']:.1f}%)   "
        f"top-5: {stats['top5']}/{stats['total']} ({stats['top5_pct']:.1f}%)"
    )

    # Classification metrics (precision / recall / F1 / 2×2 OvR)
    seen_ids = sorted(stats["per_class"].keys())
    metrics  = compute_ovr_metrics(stats["conf_matrix"], seen_ids)
    print_metrics_table(metrics, label_map, seen_ids)
    print()


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch accuracy evaluation on a structured dataset split.\n"
            "Folder layout: <data_dir>/<ClassName>/<stem>.mp4 "
            "and/or <data_dir>/<ClassName>/<stem>/ (pre-extracted frames)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_dir",      required=True,
                        help="Split folder, e.g. Data/Test or Data/Val")
    parser.add_argument("--checkpoint",    default=str(CHECKPOINT),
                        help="Path to trained SignVLM checkpoint (.pth)")
    parser.add_argument("--backbone_path", default=str(BACKBONE),
                        help="Path to CLIP ViT-L/14 weights (.pt)")
    parser.add_argument("--label_map",     default=str(LABEL_MAP),
                        help="Path to PSL label map .txt")
    parser.add_argument("--num_frames",    type=int, default=PSL_NUM_FRAMES,
                        help=f"Frames per clip (default: {PSL_NUM_FRAMES})")
    parser.add_argument("--sampling_rate", type=int, default=PSL_SAMPLING_RATE,
                        help=f"Temporal stride (default: {PSL_SAMPLING_RATE})")
    parser.add_argument("--spatial_size",  type=int, default=PSL_SPATIAL_SIZE,
                        help=f"Crop size (default: {PSL_SPATIAL_SIZE})")
    parser.add_argument("--device",        default="auto",
                        help="Device: auto | cpu | cuda | cuda:0")
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"Error: data_dir not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    label_map = load_label_map(args.label_map)
    model = load_model(
        args.checkpoint, args.backbone_path, device, num_frames=args.num_frames
    )
    model.eval()

    print(f"\nDevice:          {device}")
    print(f"Data dir:        {data_dir.resolve()}")
    print(f"Checkpoint:      {Path(args.checkpoint).name}")
    print(f"num_frames:      {args.num_frames}   sampling_rate: {args.sampling_rate}")
    print(f"\nKey:  ✓ top-1 correct   ~ in top-5   ✗ missed   [f] frames   [v] video")
    print("=" * 80)

    stats = run_eval(
        data_dir, model, label_map, device,
        num_frames=args.num_frames,
        sampling_rate=args.sampling_rate,
        spatial_size=args.spatial_size,
    )
    print_report(stats, label_map)


if __name__ == "__main__":
    main()
