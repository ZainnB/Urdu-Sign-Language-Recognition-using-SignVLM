"""
Batch accuracy evaluation of SignVLM on the Unseen Data folder.

The model was trained with num_frames=16, sampling_rate=4.
num_frames is baked into the checkpoint (temporal_pos_embed shape), so we
keep it fixed at 16. We vary sampling_rate to change which 16 frames are
pulled from each video:

  sampling_rate=1 -> samples every frame, covers only the first ~15 frames
  sampling_rate=2 -> covers ~31 frames
  sampling_rate=4 -> default (trained setting), covers ~61 frames
  sampling_rate=8 -> covers ~121 frames (wider temporal window)

Usage (run from FYP-I root or signVLM/):
    python signVLM/eval_unseen.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

import torch
import torch.nn.functional as F

from inference import load_label_map, load_model, preprocess_video, PSL_NUM_FRAMES

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "trained_models_final/signVLM_psl/checkpoint-10000.pth"
BACKBONE   = ROOT / "signVLM/CLIP_weights/ViT-L/ViT-L-14.pt"
LABEL_MAP  = ROOT / "trained_models_final/PSL_recognition_label_map.txt"
UNSEEN_DIR = ROOT / "Data/Unseen Data"

# num_frames is FIXED at 16 (matches checkpoint).
# sampling_rate controls the temporal stride (how spread out the 16 frames are).
NUM_FRAMES        = PSL_NUM_FRAMES  # 16 -- must match training
SAMPLING_VARIANTS = [1, 2, 4, 8]   # 4 = trained default


def label_from_filename(stem: str, id_to_name: dict) -> int | None:
    """Case-insensitive match of video stem to a class id."""
    name_to_id = {v.lower(): k for k, v in id_to_name.items()}
    clean = stem.lower().replace(" (2)", "").strip()
    return name_to_id.get(clean)


def run_eval(sampling_rate: int, model, label_map: dict, device: torch.device) -> dict:
    videos = sorted(UNSEEN_DIR.glob("*.mp4"))
    top1_hits = top5_hits = total = 0
    per_video = []

    for video in videos:
        gt_id = label_from_filename(video.stem, label_map)
        if gt_id is None:
            print(f"  [skip] {video.name} -- no matching class in label map")
            continue

        try:
            tensor = preprocess_video(
                video,
                num_frames=NUM_FRAMES,
                sampling_rate=sampling_rate,
            ).to(device)

            with torch.no_grad(), torch.amp.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                logits = model(tensor)

            probs   = F.softmax(logits, dim=-1).squeeze(0).cpu()
            top5_ids = probs.argsort(descending=True)[:5].tolist()
            top1_id  = top5_ids[0]

            hit1 = int(top1_id == gt_id)
            hit5 = int(gt_id in top5_ids)
            top1_hits += hit1
            top5_hits += hit5
            total += 1

            mark = "✓" if hit1 else ("~" if hit5 else "✗")
            per_video.append((
                mark,
                video.name,
                label_map.get(gt_id, f"class_{gt_id}"),
                label_map.get(top1_id, f"class_{top1_id}"),
                float(probs[top1_id]) * 100,
            ))
        except Exception as e:
            print(f"  [error] {video.name}: {e}")

    return {
        "sampling_rate": sampling_rate,
        "total": total,
        "top1": top1_hits,
        "top5": top5_hits,
        "top1_pct": top1_hits / total * 100 if total else 0.0,
        "top5_pct": top5_hits / total * 100 if total else 0.0,
        "per_video": per_video,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map = load_label_map(LABEL_MAP)

    print(f"Device:          {device}")
    print(f"Checkpoint:      {CHECKPOINT.name}")
    print(f"Unseen data dir: {UNSEEN_DIR}")
    print(f"num_frames:      {NUM_FRAMES}  (fixed -- matches training)")
    print(f"sampling_rate variants: {SAMPLING_VARIANTS}  (4 = trained default)")
    print(f"\nKey:  ✓ = top-1 correct   ~ = in top-5   ✗ = missed")
    print("=" * 72)

    # Load model once -- num_frames=16 is fixed across all experiments
    model = load_model(CHECKPOINT, BACKBONE, device, num_frames=NUM_FRAMES)
    model.eval()

    all_stats = []

    for sr in SAMPLING_VARIANTS:
        video_span = (NUM_FRAMES - 1) * sr + 1
        print(f"\n{'─'*72}")
        trained_tag = "  <-- TRAINED WITH THIS" if sr == 4 else ""
        print(f"  sampling_rate = {sr}  (covers ~{video_span} frames from each video){trained_tag}")
        print(f"{'─'*72}")

        stats = run_eval(sr, model, label_map, device)

        col = f"  {'':4} {'File':<24} {'Ground Truth':<22} {'Top-1 Pred':<22} {'Conf':>6}"
        print(col)
        print(f"  {'':4} {'----':<24} {'------------':<22} {'----------':<22} {'----':>6}")
        for mark, fname, gt, pred, conf in stats["per_video"]:
            print(f"  {mark:<4} {fname:<24} {gt:<22} {pred:<22} {conf:>5.1f}%")

        print(
            f"\n  SUMMARY  top-1: {stats['top1']}/{stats['total']} "
            f"({stats['top1_pct']:.1f}%)   "
            f"top-5: {stats['top5']}/{stats['total']} "
            f"({stats['top5_pct']:.1f}%)"
        )
        all_stats.append(stats)

    # Final comparison table
    print(f"\n{'='*72}")
    print("  ACCURACY vs SAMPLING RATE  (num_frames=16 fixed)")
    print(f"{'='*72}")
    print(f"  {'sampling_rate':<16} {'frames covered':<18} {'Top-1':>8} {'Top-5':>8}")
    print(f"  {'─'*13:<16} {'─'*14:<18} {'─'*5:>8} {'─'*5:>8}")
    for s in all_stats:
        span = (NUM_FRAMES - 1) * s["sampling_rate"] + 1
        trained = "  <- trained" if s["sampling_rate"] == 4 else ""
        print(
            f"  {s['sampling_rate']:<16} {span:<18} "
            f"{s['top1_pct']:>7.1f}% {s['top5_pct']:>7.1f}%{trained}"
        )
    print()


if __name__ == "__main__":
    main()
