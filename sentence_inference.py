"""
SignVLM sentence-forming inference pipeline.

Takes a long video containing multiple consecutive signs and predicts each
sign sequentially, building a sentence.

Strategy — sliding window over the decoded frame buffer:
  • Decode the full video once.
  • Slide a window of `window_frames` (default 61 = training coverage at
    stride 4) across the timeline with a `stride_frames` step (default 30 =
    ~1 second). At each position sample 16 evenly-spaced frames and run the
    model.
  • Apply confidence threshold + label deduplication so the same sign is
    not repeated for consecutive windows.
  • Print the growing sentence in real-time as each window is processed.

Real-time webcam mode (--realtime):
  • Captures frames from a webcam using OpenCV.
  • Maintains a rolling buffer of `window_frames` frames.
  • Every `stride_frames` new frames the buffer is flushed to the model.
  • Predicted label is printed immediately, appended to the sentence.

Usage:
    # Video file
    python signVLM/sentence_inference.py --video path/to/multi_sign.mp4

    # Webcam (real-time)
    python signVLM/sentence_inference.py --realtime

    # Custom paths
    python signVLM/sentence_inference.py \\
        --video multi.mp4 \\
        --checkpoint trained_models_final/signVLM_psl/checkpoint-10000.pth \\
        --backbone_path signVLM/CLIP_weights/ViT-L/ViT-L-14.pt \\
        --label_map trained_models_final/PSL_recognition_label_map.txt \\
        --conf_threshold 0.40 \\
        --window_frames 61 \\
        --stride_frames 30
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from inference import (
    PSL_NUM_FRAMES,
    PSL_SAMPLING_RATE,
    PSL_SPATIAL_SIZE,
    CLIP_MEAN,
    CLIP_STD,
    load_label_map,
    load_model,
    decode_video,
    frames_to_tensor,
    _resample_indices,
)

# ── Defaults ───────────────────────────────────────────────────────────────
ROOT       = _DIR.parent
CHECKPOINT = ROOT / "trained_models_final/signVLM_psl/checkpoint-10000.pth"
BACKBONE   = ROOT / "signVLM/CLIP_weights/ViT-L/ViT-L-14.pt"
LABEL_MAP  = ROOT / "trained_models_final/PSL_recognition_label_map.txt"

# A sign clip covers (16-1)*4+1 = 61 video frames at 30fps ≈ 2 seconds.
# We slide every 30 frames (1 second) to get ~50% overlap between windows.
DEFAULT_WINDOW_FRAMES = (PSL_NUM_FRAMES - 1) * PSL_SAMPLING_RATE + 1  # 61
DEFAULT_STRIDE_FRAMES = 30   # 1 second at 30fps
DEFAULT_CONF_THRESH   = 0.35  # minimum confidence to emit a label
DEFAULT_CAMERA_ID     = 0


# ── Core: sample 16 frames from a window and run the model ─────────────────
def _window_to_indices(window_len: int, num_frames: int = PSL_NUM_FRAMES) -> List[int]:
    """Evenly pick num_frames indices from [0, window_len)."""
    if window_len <= num_frames:
        return _resample_indices(window_len, num_frames)
    return [int(window_len / num_frames * i) for i in range(num_frames)]


@torch.no_grad()
def _infer_window(
    frames: List[np.ndarray],
    model: torch.nn.Module,
    label_map: Dict[int, str],
    device: torch.device,
    conf_threshold: float,
) -> Tuple[Optional[str], float]:
    """Run one forward pass on a list of numpy frames. Returns (label, conf)."""
    indices = _window_to_indices(len(frames))
    tensor = frames_to_tensor(frames, indices, spatial_size=PSL_SPATIAL_SIZE).to(device)
    with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
        logits = model(tensor)
    probs = F.softmax(logits, dim=-1).squeeze(0).cpu()
    conf, idx = probs.max(dim=0)
    conf, idx = conf.item(), idx.item()
    if conf < conf_threshold:
        return None, conf
    return label_map.get(idx, f"class_{idx}"), conf


# ── Video-file sentence pipeline ───────────────────────────────────────────
def infer_sentence_from_video(
    video_path: str | Path,
    model: torch.nn.Module,
    label_map: Dict[int, str],
    device: torch.device,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    stride_frames: int = DEFAULT_STRIDE_FRAMES,
    conf_threshold: float = DEFAULT_CONF_THRESH,
    fps: float = 30.0,
) -> List[Tuple[str, float, float]]:
    """
    Slide a window across the decoded video and build a sentence.

    Returns list of (label, confidence, timestamp_sec) for each unique sign.
    """
    print(f"\nDecoding video: {Path(video_path).name} ...")
    t0 = time.perf_counter()
    raw_frames = decode_video(video_path)
    decode_ms = (time.perf_counter() - t0) * 1000
    total = len(raw_frames)
    if total == 0:
        raise ValueError(f"No frames decoded from {video_path}")

    duration_s = total / fps
    print(f"  {total} frames decoded in {decode_ms:.0f}ms  |  "
          f"duration ~{duration_s:.1f}s  |  {fps:.0f}fps")
    print(f"  Window: {window_frames} frames ({window_frames/fps:.1f}s)  |  "
          f"Stride: {stride_frames} frames ({stride_frames/fps:.1f}s)")
    print(f"  Confidence threshold: {conf_threshold:.0%}\n")

    sentence: List[Tuple[str, float, float]] = []
    last_label: Optional[str] = None
    positions = list(range(0, max(total - window_frames + 1, 1), stride_frames))
    if not positions:
        positions = [0]

    print(f"  {'Time':>6}  {'Window':>18}  {'Label':<22}  {'Conf':>6}  {'Sentence so far'}")
    print(f"  {'----':>6}  {'------':>18}  {'-----':<22}  {'----':>6}  {'---------------'}")

    t_start = time.perf_counter()
    for pos in positions:
        end = min(pos + window_frames, total)
        window = raw_frames[pos:end]
        ts = pos / fps

        t_w = time.perf_counter()
        label, conf = _infer_window(window, model, label_map, device, conf_threshold)
        infer_ms = (time.perf_counter() - t_w) * 1000

        if label is not None and label != last_label:
            sentence.append((label, conf, ts))
            last_label = label

        current_sentence = " | ".join(lbl for lbl, _, _ in sentence)
        display_label = label if label else f"(low conf: {conf:.0%})"
        print(f"  {ts:>5.1f}s  [{pos:>5}–{end:>5}] {infer_ms:>4.0f}ms  "
              f"{display_label:<22}  {conf:>5.0%}  {current_sentence}")

    elapsed = time.perf_counter() - t_start
    print(f"\n  Processed {len(positions)} windows in {elapsed:.1f}s "
          f"({elapsed/len(positions)*1000:.0f}ms/window avg)")

    return sentence


# ── Real-time webcam pipeline ──────────────────────────────────────────────
def infer_sentence_realtime(
    model: torch.nn.Module,
    label_map: Dict[int, str],
    device: torch.device,
    camera_id: int = DEFAULT_CAMERA_ID,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    stride_frames: int = DEFAULT_STRIDE_FRAMES,
    conf_threshold: float = DEFAULT_CONF_THRESH,
    target_fps: float = 30.0,
) -> List[Tuple[str, float]]:
    """
    Capture from webcam in real-time, predict signs, build sentence.

    Press 'q' or Ctrl-C to stop.
    Returns list of (label, confidence) for each detected sign.
    """
    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python not installed. Run: pip install opencv-python")
        sys.exit(1)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")

    actual_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    print(f"\nWebcam opened  |  Camera {camera_id}  |  ~{actual_fps:.0f}fps")
    print(f"  Window: {window_frames} frames ({window_frames/actual_fps:.1f}s)  |  "
          f"Stride: {stride_frames} frames ({stride_frames/actual_fps:.1f}s)")
    print(f"  Confidence threshold: {conf_threshold:.0%}")
    print("  Press Ctrl-C or 'q' in the window to stop.\n")

    buffer: deque[np.ndarray] = deque(maxlen=window_frames)
    sentence: List[Tuple[str, float]] = []
    last_label: Optional[str] = None
    frames_since_last_infer = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buffer.append(rgb)
            frames_since_last_infer += 1

            # Run inference every stride_frames new frames once buffer is full
            if len(buffer) == window_frames and frames_since_last_infer >= stride_frames:
                frames_since_last_infer = 0
                window = list(buffer)

                t0 = time.perf_counter()
                label, conf = _infer_window(window, model, label_map, device, conf_threshold)
                infer_ms = (time.perf_counter() - t0) * 1000

                if label is not None and label != last_label:
                    sentence.append((label, conf))
                    last_label = label
                    current_sentence = " | ".join(lbl for lbl, _ in sentence)
                    print(f"  [{infer_ms:>4.0f}ms]  NEW SIGN: {label:<22} {conf:>5.0%}  "
                          f"->  {current_sentence}")
                else:
                    display = label if label else f"(low conf: {conf:.0%})"
                    print(f"  [{infer_ms:>4.0f}ms]  {display:<30}  (same/below threshold)")

            # Show live feed if display is available
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return sentence


# ── CLI ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SignVLM sentence-forming inference (video file or webcam)"
    )
    parser.add_argument("--video", type=str, help="Path to a video file with multiple signs")
    parser.add_argument("--realtime", action="store_true", help="Use webcam for real-time inference")
    parser.add_argument("--camera_id", type=int, default=DEFAULT_CAMERA_ID)
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT))
    parser.add_argument("--backbone_path", type=str, default=str(BACKBONE))
    parser.add_argument("--label_map", type=str, default=str(LABEL_MAP))
    parser.add_argument("--window_frames", type=int, default=DEFAULT_WINDOW_FRAMES,
                        help=f"Sliding window size in frames (default: {DEFAULT_WINDOW_FRAMES} = ~2s at 30fps)")
    parser.add_argument("--stride_frames", type=int, default=DEFAULT_STRIDE_FRAMES,
                        help=f"Window advance step in frames (default: {DEFAULT_STRIDE_FRAMES} = ~1s at 30fps)")
    parser.add_argument("--conf_threshold", type=float, default=DEFAULT_CONF_THRESH,
                        help=f"Min confidence to emit a label (default: {DEFAULT_CONF_THRESH:.0%})")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if not args.video and not args.realtime:
        parser.error("Provide --video <path> or --realtime")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else torch.device(args.device)

    print(f"Loading model on {device} ...")
    label_map = load_label_map(args.label_map)
    model = load_model(args.checkpoint, args.backbone_path, device, num_frames=PSL_NUM_FRAMES)
    model.eval()
    print("Model ready.\n")

    if args.realtime:
        sentence = infer_sentence_realtime(
            model, label_map, device,
            camera_id=args.camera_id,
            window_frames=args.window_frames,
            stride_frames=args.stride_frames,
            conf_threshold=args.conf_threshold,
        )
        labels = [lbl for lbl, _ in sentence]
    else:
        result = infer_sentence_from_video(
            args.video, model, label_map, device,
            window_frames=args.window_frames,
            stride_frames=args.stride_frames,
            conf_threshold=args.conf_threshold,
        )
        labels = [lbl for lbl, _, _ in result]

    print("\n" + "=" * 60)
    print("  FINAL SENTENCE:")
    print(f"  {' | '.join(labels) if labels else '(no signs detected above threshold)'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
