"""
ROI Extraction using MediaPipe for single video.
Extracts hand region-of-interest and outputs cropped video.
Perfect for preprocessing PSL videos before pose extraction or model training.
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path


# ==========================================
# UTILITY FUNCTIONS (from mediapipe_roi.py)
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
    show_preview=False
):
    """
    Extract hand ROI from video using MediaPipe.
    
    Args:
        input_video: Path to input video
        output_video: Path to output ROI video
        roi_size: Size of output ROI (default 256x256)
        pad_ratio: Padding around detected hands (0.4 = 40%)
        pos_alpha: Smoothing for ROI position (lower = more smooth)
        size_alpha: Smoothing for ROI size (lower = more smooth)
        grace_frames: Frames to wait before losing track of hands
        show_preview: Display live preview
    """
    
    print(f"\n{'='*70}")
    print(f"ROI EXTRACTION CONFIGURATION")
    print(f"{'='*70}")
    print(f"Input video: {input_video}")
    print(f"Output video: {output_video}")
    print(f"ROI size: {roi_size}x{roi_size}")
    print(f"Padding ratio: {pad_ratio}")
    print(f"Position smoothing alpha: {pos_alpha}")
    print(f"Size smoothing alpha: {size_alpha}")
    print(f"Grace frames: {grace_frames}")
    print(f"{'='*70}\n")
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,  # IMPORTANT: 2 hands for sign language
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Open input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {input_video}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✓ Video loaded: {total_frames} frames @ {fps:.1f} FPS")
    
    # Create output video writer
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
    hands_lost = 0
    
    print(f"✓ Output writer initialized")
    print(f"Processing frames...\n")
    
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
            # No hands detected
            miss += 1
            if smooth_center is not None and miss <= grace_frames:
                # Use previous smooth values
                center, size = smooth_center, smooth_size
            else:
                # Lost tracking
                center, size = None, None
                hands_lost += 1
        else:
            # Hands detected
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
        roi_img = None
        if center is not None:
            half = size[0] / 2
            x1 = int(max(0, center[0] - half))
            y1 = int(max(0, center[1] - half))
            x2 = int(min(w, center[0] + half))
            y2 = int(min(h, center[1] + half))
            
            roi = frame[y1:y2, x1:x2]
            roi_img = resize_letterbox(roi, roi_size)
            
            writer.write(roi_img)
        else:
            # No valid ROI, write black frame
            roi_img = np.zeros((roi_size, roi_size, 3), dtype=np.uint8)
            writer.write(roi_img)
        
        # Preview (optional)
        if show_preview:
            disp = frame.copy()
            if center is not None:
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Original", disp)
            if roi_img is not None:
                cv2.imshow("ROI", roi_img)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break
        
        frame_idx += 1
        
        if frame_idx % 50 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...")
    
    # Cleanup
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
    # Summary
    print(f"\n{'='*70}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Total frames: {frame_idx}")
    print(f"✓ Frames with hands detected: {hands_detected}")
    print(f"✓ Frames without tracking: {hands_lost}")
    if frame_idx > 0:
        det_rate = (hands_detected / frame_idx) * 100
        print(f"✓ Detection rate: {det_rate:.1f}%")
    print(f"✓ Output saved: {output_video}")
    print(f"{'='*70}\n")
    
    return True


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    # Configuration for Food_3.mp4
    input_video = r"../video_data/food_3.mp4"
    output_video = r"../video_data/food_3_roi_256.mp4"
    

    # ROI extraction parameters
    roi_size = 512  # Output size: 512x512
    pad_ratio = 0.6  # 50% padding around hands
    pos_alpha = 0.05  # Position smoothing (lower = smoother)
    size_alpha = 0.05  # Size smoothing
    grace_frames = 20  # Frames to maintain tracking after hand loss
    
    # Run extraction
    success = extract_roi_video(
        input_video=input_video,
        output_video=output_video,
        roi_size=roi_size,
        pad_ratio=pad_ratio,
        pos_alpha=pos_alpha,
        size_alpha=size_alpha,
        grace_frames=grace_frames,
        show_preview=False  # Set to True to see live preview
    )
    
    if success:
        print("✅ ROI extraction successful!")
    else:
        print("❌ ROI extraction failed!")
