"""Benchmark per-stage inference timing for one video."""
from __future__ import annotations
import sys, time
from pathlib import Path

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

import torch
import torch.nn.functional as F
from inference import (
    load_label_map, load_model, decode_video, frames_to_tensor,
    sample_indices_center, PSL_NUM_FRAMES, PSL_SAMPLING_RATE, PSL_SPATIAL_SIZE,
)

ROOT       = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "trained_models_final/signVLM_psl/checkpoint-10000.pth"
BACKBONE   = ROOT / "signVLM/CLIP_weights/ViT-L/ViT-L-14.pt"
LABEL_MAP  = ROOT / "trained_models_final/PSL_recognition_label_map.txt"
VIDEO      = ROOT / "Data/Unseen Data/Afraid.mp4"
RUNS       = 10

def bench():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map = load_label_map(LABEL_MAP)

    print(f"Device: {device}  |  Video: {VIDEO.name}  |  Runs: {RUNS}\n")

    # ── 1. Model load (one-time cost) ────────────────────────────────────
    t0 = time.perf_counter()
    model = load_model(CHECKPOINT, BACKBONE, device, num_frames=PSL_NUM_FRAMES)
    model.eval()
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"Model load (one-time):       {load_ms:8.1f} ms")

    # GPU warm-up
    dummy = torch.zeros(1, 3, PSL_NUM_FRAMES, 224, 224, device=device)
    with torch.no_grad():
        _ = model(dummy)
    torch.cuda.synchronize()

    # ── 2. Repeated timing ───────────────────────────────────────────────
    decode_times, preproc_times, infer_times, total_times = [], [], [], []

    # Decode once so preprocess timing is isolated from disk I/O
    raw_frames = decode_video(VIDEO)
    indices = sample_indices_center(len(raw_frames), PSL_NUM_FRAMES, PSL_SAMPLING_RATE)

    for _ in range(RUNS):
        t_total = time.perf_counter()

        # Decode (file I/O — this is the cost for the FIRST call only in sentence mode)
        t0 = time.perf_counter()
        _ = decode_video(VIDEO)
        decode_ms = (time.perf_counter() - t0) * 1000
        decode_times.append(decode_ms)

        # Preprocess on already-decoded frames (what happens for every window after first decode)
        t0 = time.perf_counter()
        tensor = frames_to_tensor(raw_frames, indices, PSL_SPATIAL_SIZE).to(device)
        preproc_ms = (time.perf_counter() - t0) * 1000
        preproc_times.append(preproc_ms)

        # Model forward
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=device.type=="cuda"):
            logits = model(tensor)
        torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t0) * 1000
        infer_times.append(infer_ms)

        total_times.append(preproc_ms + infer_ms)  # real per-window cost (no re-decode)

    def stats(vals):
        import statistics
        return min(vals), statistics.mean(vals), max(vals)

    print(f"\n{'Stage':<30} {'Min':>8} {'Avg':>8} {'Max':>8}")
    print(f"{'─'*30} {'───':>8} {'───':>8} {'───':>8}")
    for name, vals in [
        ("Video decode (PyAV) [one-time]", decode_times),
        ("Preprocess (sample+norm+crop)", preproc_times),
        ("Model forward pass (GPU)", infer_times),
        ("Per-window cost (preproc+infer)", total_times),
    ]:
        mn, avg, mx = stats(vals)
        print(f"  {name:<28} {mn:>7.1f}ms {avg:>7.1f}ms {mx:>7.1f}ms")

    avg_window = sum(total_times) / len(total_times)
    avg_infer  = sum(infer_times) / len(infer_times)
    avg_decode = sum(decode_times) / len(decode_times)
    window_dur_ms = ((PSL_NUM_FRAMES - 1) * PSL_SAMPLING_RATE + 1) / 30 * 1000

    print(f"\n  ── Real-time analysis (30fps webcam) ──────────────────")
    print(f"  Decode (one-time per video):  {avg_decode:.0f} ms")
    print(f"  Per-window cost:              {avg_window:.0f} ms  ({1000/avg_window:.1f} windows/sec)")
    print(f"  Model-only:                   {avg_infer:.0f} ms  ({1000/avg_infer:.1f} inferences/sec)")
    print(f"  Sign window duration:         {window_dur_ms:.0f} ms  "
          f"({(PSL_NUM_FRAMES-1)*PSL_SAMPLING_RATE+1} frames at 30fps)")
    rt = "YES" if avg_window < window_dur_ms else "NO"
    print(f"  Real-time feasible?           {rt}  "
          f"(model={avg_window:.0f}ms < window={window_dur_ms:.0f}ms)")
    print(f"\n  With stride=30 frames (1s), model has {1000/30*1000:.0f}ms budget per window -> "
          f"{'OK' if avg_window < 1000 else 'TIGHT'}")

if __name__ == "__main__":
    bench()
