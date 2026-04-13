"""
Batch ROI Extraction for PSL videos.
Process multiple videos with MediaPipe hand detection.
Supports parameter tuning and different ROI output sizes.
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import argparse


# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def clamp_xyxy(xyxy, w, h):
    """Clamp bounding box coordinates to frame boundaries."""
    x1, y1, x2, y2 = xyxy
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return np.array([x1, y1, x2, y2], dtype=float)


def pad_xyxy(xyxy, pad_ratio):
    """Add padding around bounding box."""
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    px, py = w * pad_ratio / 2, h * pad_ratio / 2
    return np.array([x1 - px, y1 - py, x2 + px, y2 + py], dtype=float)


def ema(prev, cur, alpha):
    """Exponential moving average for smoothing."""
    return cur if prev is None else alpha * cur + (1 - alpha) * prev


def resize_letterbox(img, size):
    """Resize image to fixed size with letterboxing (no distortion)."""
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))

    out = np.zeros((size, size, 3), dtype=np.uint8)
    x0, y0 = (size - nw) // 2, (size - nh) // 2
    out[y0:y0+nh, x0:x0+nw] = resized
    return out


def union_boxes(boxes):
    """Combine multiple bounding boxes into one union box."""
    if not boxes:
        return None
    xs1 = [b[0] for b in boxes]
    ys1 = [b[1] for b in boxes]
    xs2 = [b[2] for b in boxes]
    ys2 = [b[3] for b in boxes]
    return np.array([min(xs1), min(ys1), max(xs2), max(ys2)], dtype=float)


def landmarks_to_box(landmarks, w, h):
    """Convert MediaPipe landmarks to bounding box."""
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    return np.array([min(xs), min(ys), max(xs), max(ys)], dtype=float)


# ==========================================
# ROI EXTRACTION
# ==========================================

def extract_roi_video(
    input_video,
    output_video,
    roi_size=256,
    pad_ratio=0.4,
    pos_alpha=0.15,
    size_alpha=0.15,
    grace_frames=10,
    verbose=True
):
    """Extract hand ROI from video using MediaPipe."""
    
    if verbose:
        print(f"\n  Processing: {Path(input_video).name}")
        print(f"  → ROI size: {roi_size}x{roi_size}")
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Open input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        if verbose:
            print(f"  ❌ Error: Cannot open video")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output directory and video writer
    output_dir = Path(output_video).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (roi_size, roi_size)
    )
    
    # Processing state
    smooth_center = None
    smooth_size = None
    miss = 0
    frame_idx = 0
    hands_detected = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        
        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(rgb)
        
        boxes = []
        
        # Extract hand bounding boxes
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                box = landmarks_to_box(hand.landmark, w, h)
                boxes.append(box)
            hands_detected += 1
        
        # Union of all hand boxes
        det = union_boxes(boxes)
        
        if det is None:
            miss += 1
            if smooth_center is not None and miss <= grace_frames:
                center, size = smooth_center, smooth_size
            else:
                center, size = None, None
        else:
            miss = 0
            det = clamp_xyxy(pad_xyxy(det, pad_ratio), w, h)
            
            cx = (det[0] + det[2]) / 2
            cy = (det[1] + det[3]) / 2
            side = max(det[2] - det[0], det[3] - det[1])
            
            # Smooth the center and size
            smooth_center = ema(smooth_center, np.array([cx, cy]), pos_alpha)
            smooth_size = ema(smooth_size, np.array([side]), size_alpha)
            
            center, size = smooth_center, smooth_size
        
        # Extract ROI and write to output
        if center is not None:
            half = size[0] / 2
            x1 = int(max(0, center[0] - half))
            y1 = int(max(0, center[1] - half))
            x2 = int(min(w, center[0] + half))
            y2 = int(min(h, center[1] + half))
            
            roi = frame[y1:y2, x1:x2]
            roi_img = resize_letterbox(roi, roi_size)
        else:
            roi_img = np.zeros((roi_size, roi_size, 3), dtype=np.uint8)
        
        writer.write(roi_img)
        frame_idx += 1
    
    # Cleanup
    cap.release()
    writer.release()
    
    # Summary
    if verbose:
        det_rate = (hands_detected / frame_idx * 100) if frame_idx > 0 else 0
        print(f"  ✓ {frame_idx} frames | {hands_detected} hand detections | {det_rate:.1f}%")
        print(f"  ✓ Saved: {Path(output_video).name}")
    
    return True


# ==========================================
# BATCH PROCESSING
# ==========================================

def batch_extract_roi_psl(
    dataset_root,
    roi_size=256,
    pad_ratio=0.4,
    pos_alpha=0.15,
    size_alpha=0.15,
    grace_frames=10,
    test_first_label_only=False,
    resume_from_label=None,
    skip_existing=True
):
    """
    Process PSL dataset with label folders.
    Structure:
        dataset_root/
            Afraid/
                Afraid_1.mp4, Afraid_2.mp4, ..., Afraid_15.mp4
            Happy/
                Happy_1.mp4, Happy_2.mp4, ..., Happy_15.mp4
            ...
    
    Output naming: {LabelName}_aug_{video_number}.mp4
    
    Args:
        dataset_root: Root directory containing label folders
        roi_size: ROI output size
        pad_ratio: Padding around hands
        pos_alpha: Position smoothing
        size_alpha: Size smoothing
        grace_frames: Grace period for hand loss
        test_first_label_only: Process only first label
        resume_from_label: Resume from specific label name (e.g., "Happy")
        skip_existing: Skip videos that already have output files
    """
    
    dataset_path = Path(dataset_root)
    
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_root}")
        return
    
    # Get label folders
    label_folders = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    
    if not label_folders:
        print(f"❌ No label folders found in {dataset_root}")
        return
    
    # Resume from specific label
    if resume_from_label:
        start_idx = None
        for idx, lf in enumerate(label_folders):
            if lf.name.lower() == resume_from_label.lower():
                start_idx = idx
                break
        
        if start_idx is None:
            print(f"❌ Label '{resume_from_label}' not found. Available labels:")
            for lf in label_folders:
                print(f"   - {lf.name}")
            return
        
        label_folders = label_folders[start_idx:]
        print(f"📌 Resuming from label: {resume_from_label}\n")
    
    # For testing, only process first label
    if test_first_label_only:
        label_folders = label_folders[:1]
    
    print(f"\n{'='*70}")
    print(f"BATCH ROI EXTRACTION - PSL DATASET")
    print(f"{'='*70}")
    print(f"Dataset root: {dataset_root}")
    print(f"ROI size: {roi_size}x{roi_size}")
    print(f"Labels to process: {len(label_folders)}")
    print(f"Skip existing: {skip_existing}")
    print(f"Test mode (first label only): {test_first_label_only}")
    print(f"{'='*70}\n")
    
    total_successful = 0
    total_failed = 0
    total_skipped = 0
    
    for label_idx, label_folder in enumerate(label_folders, 1):
        label_name = label_folder.name
        videos = sorted([v for v in label_folder.glob("*.mp4")])
        
        if not videos:
            print(f"\n[{label_idx}/{len(label_folders)}] Label: {label_name} - ❌ No videos found")
            continue
        
        print(f"\n{'='*70}")
        print(f"[{label_idx}/{len(label_folders)}] Label: {label_name} ({len(videos)} videos)")
        print(f"{'='*70}")
        
        successful = 0
        failed = 0
        skipped = 0
        
        for vid_idx, video_path in enumerate(videos, 1):
            # Parse video number from filename
            # e.g., "Afraid_1.mp4" -> 1, "Afraid_2.mp4" -> 2
            try:
                video_stem = video_path.stem
                # Extract number from end of filename
                video_num = ''.join(filter(str.isdigit, video_stem.split('_')[-1]))
                if not video_num:
                    video_num = str(vid_idx)
            except:
                video_num = str(vid_idx)
            
            # Output naming: {LabelName}_aug_{video_number}.mp4
            output_video_name = f"{label_name}_aug_{video_num}.mp4"
            output_video_path = label_folder / output_video_name
            
            # Check if already processed
            if skip_existing and output_video_path.exists():
                print(f"  [{vid_idx}/{len(videos)}] {video_path.name} → {output_video_name} ⏭️  (exists)")
                skipped += 1
                continue
            
            print(f"  [{vid_idx}/{len(videos)}] {video_path.name} → {output_video_name}", end="")
            
            success = extract_roi_video(
                input_video=str(video_path),
                output_video=str(output_video_path),
                roi_size=roi_size,
                pad_ratio=pad_ratio,
                pos_alpha=pos_alpha,
                size_alpha=size_alpha,
                grace_frames=grace_frames,
                verbose=False  # Disable verbose to keep output clean
            )
            
            if success:
                print(" ✓")
                successful += 1
            else:
                print(" ❌")
                failed += 1
        
        # Label summary
        print(f"\n  → {label_name}: {successful} successful, {failed} failed, {skipped} skipped")
        total_successful += successful
        total_failed += failed
        total_skipped += skipped
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"BATCH COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Total successful: {total_successful}")
    print(f"✗ Total failed: {total_failed}")
    print(f"⏭️  Total skipped: {total_skipped}")
    print(f"{'='*70}\n")


def batch_extract_roi(
    video_folder,
    output_folder,
    roi_size=512,
    pad_ratio=0.6,
    pos_alpha=0.05,
    size_alpha=0.05,
    grace_frames=20
):
    """Legacy batch processing (mirrors subdirectory structure)."""
    
    video_root = Path(video_folder)
    videos = list(video_root.rglob("*.mp4"))
    
    if not videos:
        print(f"❌ No MP4 videos found in {video_folder}")
        return
    
    print(f"\n{'='*70}")
    print(f"BATCH ROI EXTRACTION (LEGACY)")
    print(f"{'='*70}")
    print(f"Video folder: {video_folder}")
    print(f"Output folder: {output_folder}")
    print(f"ROI size: {roi_size}x{roi_size}")
    print(f"Found {len(videos)} video(s)")
    print(f"{'='*70}")
    
    output_root = Path(output_folder)
    output_root.mkdir(parents=True, exist_ok=True)
    
    successful = 0
    failed = 0
    
    for i, vid in enumerate(videos, 1):
        rel = vid.relative_to(video_root)
        out_clip = output_root / rel.parent / f"{vid.stem}_roi_{roi_size}.mp4"
        out_clip.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[{i}/{len(videos)}]", end="")
        
        success = extract_roi_video(
            input_video=str(vid),
            output_video=str(out_clip),
            roi_size=roi_size,
            pad_ratio=pad_ratio,
            pos_alpha=pos_alpha,
            size_alpha=size_alpha,
            grace_frames=grace_frames,
            verbose=True
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*70}")
    print(f"BATCH COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"{'='*70}\n")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Batch ROI extraction using MediaPipe")
    ap.add_argument("--dataset", help="PSL dataset root (contains label folders)")
    ap.add_argument("--video-folder", default="../video_data", help="Input video folder (legacy)")
    ap.add_argument("--out-dir", default="../video_data/roi_output", help="Output directory (legacy)")
    ap.add_argument("--roi-size", type=int, default=256, help="Output ROI size (256, 384, 512)")
    ap.add_argument("--pad", type=float, default=0.4, help="Padding ratio around hands")
    ap.add_argument("--pos-alpha", type=float, default=0.15, help="Position smoothing (lower=smoother)")
    ap.add_argument("--size-alpha", type=float, default=0.15, help="Size smoothing (lower=smoother)")
    ap.add_argument("--grace-frames", type=int, default=10, help="Max frames to hold ROI during hand loss")
    ap.add_argument("--single", help="Process single video file")
    ap.add_argument("--test-first-label", action="store_true", help="Test mode: process only first label")
    ap.add_argument("--resume-from", help="Resume from specific label (e.g., 'Happy' to continue from Happy folder)")
    ap.add_argument("--no-skip", action="store_true", help="Reprocess all videos (don't skip existing outputs)")
    args = ap.parse_args()
    
    if args.single:
        # Single video processing
        input_video = args.single
        if not Path(input_video).exists():
            print(f"❌ Video not found: {input_video}")
        else:
            output_video = Path(input_video).parent / f"{Path(input_video).stem}_roi_{args.roi_size}.mp4"
            extract_roi_video(
                input_video=input_video,
                output_video=str(output_video),
                roi_size=args.roi_size,
                pad_ratio=args.pad,
                pos_alpha=args.pos_alpha,
                size_alpha=args.size_alpha,
                grace_frames=args.grace_frames,
                verbose=True
            )
    elif args.dataset:
        # PSL dataset batch processing (NEW)
        batch_extract_roi_psl(
            dataset_root=args.dataset,
            roi_size=args.roi_size,
            pad_ratio=args.pad,
            pos_alpha=args.pos_alpha,
            size_alpha=args.size_alpha,
            grace_frames=args.grace_frames,
            test_first_label_only=args.test_first_label,
            resume_from_label=args.resume_from,
            skip_existing=(not args.no_skip)  # Default True, set False if --no-skip
        )
    else:
        # Legacy batch processing
        batch_extract_roi(
            video_folder=args.video_folder,
            output_folder=args.out_dir,
            roi_size=args.roi_size,
            pad_ratio=args.pad,
            pos_alpha=args.pos_alpha,
            size_alpha=args.size_alpha,
            grace_frames=args.grace_frames
        )
