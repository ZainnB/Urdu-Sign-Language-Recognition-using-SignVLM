"""
Multi-video sentence inference.

Concatenates multiple sign video clips end-to-end in memory (no temp file),
then runs the sliding-window sentence pipeline across the combined frame stream.
"""
from __future__ import annotations

import sys, time
from pathlib import Path
from typing import List, Tuple, Optional, Dict

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

import cv2
import torch
import torch.nn.functional as F
import numpy as np

from inference import (
    PSL_NUM_FRAMES, PSL_SAMPLING_RATE, PSL_SPATIAL_SIZE,
    load_label_map, load_model, decode_video,
    frames_to_tensor, _resample_indices,
)

# Resize all frames to this resolution before concatenating so boundary
# windows (spanning two clips with different resolutions) can be stacked.
COMMON_H, COMMON_W = 256, 256

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = _DIR.parent
CHECKPOINT = ROOT / "trained_models_final/signVLM_psl/checkpoint-10000.pth"
BACKBONE   = ROOT / "signVLM/CLIP_weights/ViT-L/ViT-L-14.pt"
LABEL_MAP  = ROOT / "trained_models_final/PSL_recognition_label_map.txt"
UNSEEN_DIR = ROOT / "Data/Unseen Data"

# Sliding-window config
WINDOW_FRAMES = (PSL_NUM_FRAMES - 1) * PSL_SAMPLING_RATE + 1  # 61 frames ≈ 2s
STRIDE_FRAMES = 30        # advance 1s each step
CONF_THRESH   = 0.35
FPS           = 30.0


def window_indices(window_len: int, n: int = PSL_NUM_FRAMES) -> List[int]:
    if window_len <= n:
        return _resample_indices(window_len, n)
    return [int(window_len / n * i) for i in range(n)]


@torch.no_grad()
def infer_window(frames, model, label_map, device, threshold):
    idx = window_indices(len(frames))
    t = frames_to_tensor(frames, idx, PSL_SPATIAL_SIZE).to(device)
    with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
        logits = model(t)
    probs = F.softmax(logits, dim=-1).squeeze(0).cpu()
    top5_ids = probs.argsort(descending=True)[:5].tolist()
    conf, label_id = probs[top5_ids[0]].item(), top5_ids[0]
    if conf < threshold:
        return None, conf, top5_ids
    return label_map.get(label_id, f"class_{label_id}"), conf, top5_ids


def run(videos: List[Path], n_videos: int = 10, conf_threshold: float = CONF_THRESH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map = load_label_map(LABEL_MAP)

    print(f"Loading model on {device} ...")
    model = load_model(CHECKPOINT, BACKBONE, device, num_frames=PSL_NUM_FRAMES)
    model.eval()
    print("Model ready.\n")

    # ── Decode & concatenate videos ────────────────────────────────────────
    selected = sorted(videos)[:n_videos]
    combined_frames: List[np.ndarray] = []
    boundaries: List[Tuple[int, int, str]] = []   # (start_frame, end_frame, video_name)

    print("=" * 68)
    print(f"  Decoding {len(selected)} videos and concatenating frames...")
    print("=" * 68)
    for v in selected:
        t0 = time.perf_counter()
        raw = decode_video(v)
        # Normalise to common spatial size so cross-clip boundary windows stack cleanly
        raw = [cv2.resize(f, (COMMON_W, COMMON_H)) for f in raw]
        ms = (time.perf_counter() - t0) * 1000
        start = len(combined_frames)
        combined_frames.extend(raw)
        end = len(combined_frames)
        boundaries.append((start, end, v.stem))
        print(f"  {v.name:<24}  {len(raw):>4} frames  ({len(raw)/FPS:.1f}s)  "
              f"decoded in {ms:.0f}ms  [global {start}–{end}]")

    total_frames = len(combined_frames)
    total_dur = total_frames / FPS
    print(f"\n  Combined: {total_frames} frames  ({total_dur:.1f}s total)\n")

    # ── Build a frame-to-video lookup for the timeline printout ───────────
    def video_at(frame_idx: int) -> str:
        for s, e, name in boundaries:
            if s <= frame_idx < e:
                return name
        return "?"

    # ── Sliding window ─────────────────────────────────────────────────────
    positions = list(range(0, max(total_frames - WINDOW_FRAMES + 1, 1), STRIDE_FRAMES))
    if not positions:
        positions = [0]

    sentence: List[str] = []
    last_label: Optional[str] = None

    print(f"  Window: {WINDOW_FRAMES} frames ({WINDOW_FRAMES/FPS:.1f}s)  |  "
          f"Stride: {STRIDE_FRAMES} frames ({STRIDE_FRAMES/FPS:.1f}s)  |  "
          f"Conf ≥ {conf_threshold:.0%}\n")

    hdr = f"  {'Time':>6}  {'Frames':>12}  {'Active clip':<16}  {'Prediction':<22}  {'Conf':>6}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    t_start = time.perf_counter()
    for pos in positions:
        end = min(pos + WINDOW_FRAMES, total_frames)
        window = combined_frames[pos:end]
        ts = pos / FPS
        active_clip = video_at(pos)

        tw = time.perf_counter()
        label, conf, top5 = infer_window(window, model, label_map, device, conf_threshold)
        infer_ms = (time.perf_counter() - tw) * 1000

        # Only emit a new label when it changes
        if label is not None and label != last_label:
            sentence.append(label)
            last_label = label
            flag = " ◄ NEW"
        else:
            flag = ""

        display = label if label else f"(conf {conf:.0%})"
        print(f"  {ts:>5.1f}s  [{pos:>5}–{end:>5}]  {active_clip:<16}  "
              f"{display:<22}  {conf:>5.0%}  {infer_ms:>4.0f}ms{flag}")

    elapsed = time.perf_counter() - t_start
    print(f"\n  {len(positions)} windows in {elapsed:.1f}s  "
          f"({elapsed/len(positions)*1000:.0f}ms/window avg)\n")

    # ── Ground truth vs predicted ──────────────────────────────────────────
    gt_labels = [p.stem for p in selected]
    print("=" * 68)
    print("  GROUND TRUTH  (video order):")
    print("  " + "  |  ".join(gt_labels))
    print()
    print("  MODEL SENTENCE:")
    print("  " + "  |  ".join(sentence) if sentence else "  (nothing above threshold)")
    print("=" * 68)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_videos", type=int, default=10)
    parser.add_argument("--conf_threshold", type=float, default=CONF_THRESH)
    args = parser.parse_args()
    videos = sorted(UNSEEN_DIR.glob("*.mp4"))
    run(videos, n_videos=args.n_videos, conf_threshold=args.conf_threshold)
