#!/usr/bin/env python
"""
SignVLM inference pipeline for PSL (Pakistan Sign Language) recognition.

Accepts a raw .mp4 video, runs it through the trained SignVLM model
(CLIP ViT-L/14 backbone + EVL decoder), and reports top-1 / top-5
predictions with confidence scores mapped to Urdu/English labels.

Usage (CLI):
    python signVLM/inference.py \
        --video path/to/video.mp4 \
        --checkpoint trained_models_final/signVLM_psl/checkpoint-10000.pth \
        --backbone_path signVLM/CLIP_weights/ViT-L/ViT-L-14.pt \
        --label_map trained_models_final/PSL_recognition_label_map.txt

Importable:
    from signVLM.inference import predict
    result = predict("video.mp4", "checkpoint.pth", "ViT-L-14.pt", "label_map.txt")
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import av
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Ensure the signVLM package root is importable when running as a script
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from model import EVLTransformer  # noqa: E402

# ── PSL training hyper-parameters (must match the training script) ─────────
PSL_BACKBONE = "ViT-L/14-lnpre"
PSL_BACKBONE_TYPE = "clip"
PSL_NUM_CLASSES = 104
PSL_NUM_FRAMES = 24
PSL_SAMPLING_RATE = 4
PSL_SPATIAL_SIZE = 224
PSL_DECODER_NUM_LAYERS = 4
PSL_DECODER_QKV_DIM = 1024
PSL_DECODER_NUM_HEADS = 16
PSL_DECODER_MLP_FACTOR = 4.0
PSL_CLS_DROPOUT = 0.5
PSL_DECODER_MLP_DROPOUT = 0.5

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


# ── Label map ──────────────────────────────────────────────────────────────
def load_label_map(path: str | Path) -> Dict[int, str]:
    """Load ``ClassName:index`` or ``index: ClassName`` label map."""
    id_to_name: Dict[int, str] = {}
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or ":" not in line:
                continue
            left, right = line.split(":", 1)
            left, right = left.strip(), right.strip()
            if left.isdigit():
                id_to_name[int(left)] = right
            elif right.isdigit():
                id_to_name[int(right)] = left
            else:
                raise ValueError(f"Unrecognized label map line: {line!r}")
    return id_to_name


# ── Video preprocessing (mirrors val/test path in dataset.py) ─────────────
def _resample_indices(total_frames: int, target: int) -> List[int]:
    """Deterministic resampling when the video has too few or too many frames."""
    if total_frames == target:
        return list(range(target))
    fraction = total_frames / target
    return [int(fraction * i) for i in range(target)]


def decode_video(video_path: str | Path) -> List[np.ndarray]:
    """Decode all frames from a video file using PyAV. Returns list of RGB arrays."""
    container = av.open(str(video_path))
    frames = {}
    for frame in container.decode(video=0):
        frames[frame.pts] = frame
    container.close()
    return [frames[k].to_rgb().to_ndarray() for k in sorted(frames.keys())]


def frames_to_tensor(
    raw_frames: List[np.ndarray],
    indices: List[int],
    spatial_size: int = PSL_SPATIAL_SIZE,
) -> torch.Tensor:
    """
    Convert a pre-selected list of frame indices from raw_frames into a
    normalised, spatially-cropped tensor ready for EVLTransformer.

    Args:
        raw_frames: Full list of decoded RGB numpy frames (H, W, 3).
        indices:    Which frame indices to pick (len = num_frames).
        spatial_size: Target spatial resolution (default 224).

    Returns:
        Tensor of shape ``(1, 3, num_frames, spatial_size, spatial_size)``.
    """
    total = len(raw_frames)
    sampled = [raw_frames[min(i, total - 1)] for i in indices]

    # (T, H, W, C) -> float [0,1], normalise
    frames = torch.as_tensor(np.stack(sampled)).float() / 255.0
    mean = CLIP_MEAN.view(1, 1, 1, 3)
    std = CLIP_STD.view(1, 1, 1, 3)
    frames = (frames - mean) / std
    frames = frames.permute(3, 0, 1, 2)  # C, T, H, W

    # Resize shortest side to spatial_size, then center-crop
    h, w = frames.shape[2], frames.shape[3]
    if h < w:
        new_h, new_w = spatial_size, w * spatial_size // h
    else:
        new_h, new_w = h * spatial_size // w, spatial_size
    frames = F.interpolate(frames, size=(new_h, new_w), mode="bilinear", align_corners=False)

    h_st = (new_h - spatial_size) // 2
    w_st = (new_w - spatial_size) // 2
    frames = frames[:, :, h_st : h_st + spatial_size, w_st : w_st + spatial_size]

    return frames.unsqueeze(0)  # B=1, C, T, H, W


def sample_indices_center(
    total: int,
    num_frames: int = PSL_NUM_FRAMES,
    sampling_rate: int = PSL_SAMPLING_RATE,
) -> List[int]:
    """Center-temporal frame index selection (val/test mode)."""
    seg_len = (num_frames - 1) * sampling_rate + 1
    if total < seg_len:
        return _resample_indices(total, num_frames)
    mid_start = (total - seg_len) // 2
    return list(range(mid_start, mid_start + num_frames * sampling_rate, sampling_rate))


def preprocess_video(
    video_path: str | Path,
    num_frames: int = PSL_NUM_FRAMES,
    sampling_rate: int = PSL_SAMPLING_RATE,
    spatial_size: int = PSL_SPATIAL_SIZE,
) -> torch.Tensor:
    """
    Decode and preprocess a video file into a tensor ready for EVLTransformer.

    For repeated inference on the same video (e.g. sliding window), call
    ``decode_video`` once and then ``frames_to_tensor`` directly to avoid
    re-reading the file from disk each time.

    Returns tensor of shape ``(1, 3, num_frames, spatial_size, spatial_size)``.
    """
    raw_frames = decode_video(video_path)
    total = len(raw_frames)
    if total == 0:
        raise ValueError(f"No frames decoded from {video_path}")
    indices = sample_indices_center(total, num_frames, sampling_rate)
    return frames_to_tensor(raw_frames, indices, spatial_size)


# ── Model loading ─────────────────────────────────────────────────────────
def load_model(
    checkpoint_path: str | Path,
    backbone_path: str | Path,
    device: torch.device,
    num_frames: int = PSL_NUM_FRAMES,
) -> EVLTransformer:
    """Build EVLTransformer with PSL hyper-parameters and load checkpoint."""
    model = EVLTransformer(
        num_frames=num_frames,
        backbone_name=PSL_BACKBONE,
        backbone_type=PSL_BACKBONE_TYPE,
        backbone_path=str(backbone_path),
        backbone_mode="freeze_fp16",
        decoder_num_layers=PSL_DECODER_NUM_LAYERS,
        decoder_qkv_dim=PSL_DECODER_QKV_DIM,
        decoder_num_heads=PSL_DECODER_NUM_HEADS,
        decoder_mlp_factor=PSL_DECODER_MLP_FACTOR,
        num_classes=PSL_NUM_CLASSES,
        enable_temporal_conv=True,
        enable_temporal_pos_embed=True,
        enable_temporal_cross_attention=True,
        cls_dropout=PSL_CLS_DROPOUT,
        decoder_mlp_dropout=PSL_DECODER_MLP_DROPOUT,
    )

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt

    # Strip DDP 'module.' prefix when present
    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        key = k.removeprefix("module.")
        cleaned[key] = v

    model.load_state_dict(cleaned, strict=True)
    model.to(device).eval()
    return model


# ── Public predict() API ──────────────────────────────────────────────────
def predict(
    video_path: str,
    checkpoint_path: str,
    backbone_path: str,
    label_map_path: str,
    num_frames: int = PSL_NUM_FRAMES,
    spatial_size: int = PSL_SPATIAL_SIZE,
    device: str = "auto",
) -> dict:
    """
    Run SignVLM inference on a single video.

    Returns::

        {
            "top1_label": str,
            "top1_confidence": float,
            "top5": [(label, confidence), ...],
            "all_probs": np.ndarray,   # shape (num_classes,)
        }
    """
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    label_map = load_label_map(label_map_path)
    model = load_model(checkpoint_path, backbone_path, dev, num_frames=num_frames)
    tensor = preprocess_video(video_path, num_frames=num_frames, spatial_size=spatial_size).to(dev)

    with torch.no_grad(), torch.amp.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
        logits = model(tensor)

    probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    top5_idx = probs.argsort()[::-1][:5]
    top5: List[Tuple[str, float]] = [
        (label_map.get(int(i), f"class_{i}"), float(probs[i])) for i in top5_idx
    ]

    return {
        "top1_label": top5[0][0],
        "top1_confidence": top5[0][1],
        "top5": top5,
        "all_probs": probs,
    }


# ── CLI entry point ───────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="SignVLM PSL inference")
    parser.add_argument("--video", type=str, required=True, help="Path to input .mp4 video")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained SignVLM checkpoint (.pth)")
    parser.add_argument("--backbone_path", type=str, required=True, help="Path to CLIP ViT-L/14 weights (.pt)")
    parser.add_argument("--label_map", type=str, required=True, help="Path to PSL_recognition_label_map.txt")
    parser.add_argument("--num_frames", type=int, default=PSL_NUM_FRAMES)
    parser.add_argument("--spatial_size", type=int, default=PSL_SPATIAL_SIZE)
    parser.add_argument("--top_k", type=int, default=5, help="Number of top predictions to show")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, cuda:0, ...")
    args = parser.parse_args()

    result = predict(
        video_path=args.video,
        checkpoint_path=args.checkpoint,
        backbone_path=args.backbone_path,
        label_map_path=args.label_map,
        num_frames=args.num_frames,
        spatial_size=args.spatial_size,
        device=args.device,
    )

    k = min(args.top_k, len(result["top5"]))
    top1_label, top1_conf = result["top1_label"], result["top1_confidence"]
    print(f"\nTop-1: {top1_label}  (confidence: {top1_conf * 100:.2f}%)\n")
    print(f"Top-{k}:")
    for rank, (label, conf) in enumerate(result["top5"][:k], start=1):
        print(f"  {rank}. {label:<20s} {conf * 100:.2f}%")
    print()


if __name__ == "__main__":
    main()
