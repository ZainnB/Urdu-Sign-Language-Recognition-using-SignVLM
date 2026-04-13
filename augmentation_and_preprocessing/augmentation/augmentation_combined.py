import cv2
import numpy as np
from pathlib import Path
import argparse
import random

# ==========================================
# AUGMENTATION TECHNIQUES
# ==========================================

def apply_lower_body_crop(frame, crop_ratio=0.2):
    """
    Crop lower body (20%) to focus on hands/torso.
    Removes legs and lower body cues.
    
    Args:
        frame: Input frame
        crop_ratio: Ratio to crop from bottom (0.2 = crop bottom 20%)
    
    Returns:
        Frame with lower body removed (resized back to original size)
    """
    height, width = frame.shape[:2]
    keep_height = int(height * (1 - crop_ratio))
    
    cropped = frame[:keep_height, :]
    resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
    
    return resized

def apply_scale(frame, scale_factor):
    """
    Scale frame (0.8x or 1.2x zoom).
    
    Args:
        frame: Input frame
        scale_factor: Scale multiplier (0.8 or 1.2)
    
    Returns:
        Scaled frame resized back to original size
    """
    height, width = frame.shape[:2]
    scaled_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
    new_h, new_w = scaled_frame.shape[:2]
    
    if scale_factor > 1:
        # Crop center when zooming in
        start_x = (new_w - width) // 2
        start_y = (new_h - height) // 2
        return scaled_frame[start_y:start_y + height, start_x:start_x + width]
    
    # Pad when zooming out
    pad_x = (width - new_w) // 2
    pad_y = (height - new_h) // 2
    return cv2.copyMakeBorder(
        scaled_frame, 
        pad_y, height - new_h - pad_y, 
        pad_x, width - new_w - pad_x, 
        cv2.BORDER_CONSTANT, 
        value=(0, 0, 0)
    )

def apply_brightness(frame, factor):
    """
    Adjust brightness (random level).
    
    Args:
        frame: Input frame
        factor: Brightness factor (0.5-1.5)
    
    Returns:
        Brightness-adjusted frame
    """
    return cv2.convertScaleAbs(frame, alpha=factor, beta=0)

def apply_rotation(frame, angle):
    """
    Apply random rotation.
    
    Args:
        frame: Input frame
        angle: Rotation angle in degrees
    
    Returns:
        Rotated frame
    """
    height, width = frame.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(frame, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)

def apply_brightness_saturation_hue(frame, brightness, saturation, hue):
    """
    Adjust brightness, saturation, and hue together.
    
    Args:
        frame: Input frame
        brightness: Brightness level (80-110)
        saturation: Saturation level (100-120)
        hue: Hue shift (-10 to 10)
    
    Returns:
        Color-adjusted frame
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Adjust brightness
    v = cv2.convertScaleAbs(v, alpha=brightness / 100.0, beta=0)
    
    # Adjust saturation
    s = cv2.convertScaleAbs(s, alpha=saturation / 100.0, beta=0)
    
    # Adjust hue
    h = ((h.astype(np.int16) + hue) % 180).astype('uint8')
    
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_frame_jitter(frames, jitter_offset=2):
    """
    Apply random frame jitter (skip frames randomly).
    
    Args:
        frames: List of frames
        jitter_offset: Max offset for frame selection
    
    Returns:
        Jittered frame sequence
    """
    jittered_frames = []
    for i in range(len(frames)):
        offset = np.random.randint(-jitter_offset, jitter_offset + 1)
        idx = min(len(frames) - 1, max(0, i + offset))
        jittered_frames.append(frames[idx])
    
    return jittered_frames

def apply_random_frame_drop(frames, drop_rate=0.1):
    """
    Randomly drop frames (simulate missing frames).
    
    Args:
        frames: List of frames
        drop_rate: Fraction of frames to drop (0.1 = 10%)
    
    Returns:
        Frame sequence with some frames removed
    """
    keep_indices = []
    for i in range(len(frames)):
        if np.random.rand() > drop_rate:
            keep_indices.append(i)
    
    # Ensure minimum frames
    if len(keep_indices) < max(1, int(len(frames) * 0.7)):
        keep_indices = list(range(len(frames)))
    
    dropped_frames = [frames[i] for i in keep_indices]
    return dropped_frames if len(dropped_frames) > 0 else frames

# ==========================================
# COMBINED AUGMENTATION PIPELINE
# ==========================================

def load_video(video_path):
    """Load video frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None, None
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames, fps, (width, height)

def save_video(frames, output_path, fps, resolution):
    """Save video frames."""
    if len(frames) == 0:
        return False
    
    width, height = resolution
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    return True

def augment_video_random(input_video, output_video, augmentation_types=None):
    """
    Apply random augmentations to video.
    Can apply 1 or 2 augmentation techniques randomly.
    
    Args:
        input_video: Input video path
        output_video: Output video path
        augmentation_types: List of augmentation types to randomly select from
    
    Returns:
        True if successful, False otherwise
    """
    frames, fps, resolution = load_video(input_video)
    
    if frames is None or len(frames) == 0:
        return False
    
    if augmentation_types is None:
        augmentation_types = [
            'lower_crop',
            'scale_down',
            'scale_up',
            'brightness',
            'rotation',
            'bsh',  # brightness_saturation_hue
            'jitter',
            'drop'
        ]
    
    # Randomly decide: apply 1 or 2 techniques
    num_techniques = random.choice([1, 2])
    selected_techniques = random.sample(augmentation_types, min(num_techniques, len(augmentation_types)))
    
    # Apply selected augmentations
    for technique in selected_techniques:
        if technique == 'lower_crop':
            frames = [apply_lower_body_crop(f, crop_ratio=0.2) for f in frames]
        
        elif technique == 'scale_down':
            frames = [apply_scale(f, 0.8) for f in frames]
        
        elif technique == 'scale_up':
            frames = [apply_scale(f, 1.2) for f in frames]
        
        elif technique == 'brightness':
            factor = np.random.uniform(0.7, 1.4)
            frames = [apply_brightness(f, factor) for f in frames]
        
        elif technique == 'rotation':
            angle = np.random.uniform(-10, 10)
            frames = [apply_rotation(f, angle) for f in frames]
        
        elif technique == 'bsh':
            # Random selection from brightness_saturation_hue options
            brightness, saturation, hue = random.choice([(90, 110, 10), (85, 115, -10)])
            frames = [apply_brightness_saturation_hue(f, brightness, saturation, hue) for f in frames]
        
        elif technique == 'jitter':
            offset = random.randint(1, 3)
            frames = apply_frame_jitter(frames, jitter_offset=offset)
        
        elif technique == 'drop':
            drop_rate = random.uniform(0.05, 0.15)
            frames = apply_random_frame_drop(frames, drop_rate=drop_rate)
    
    # Ensure minimum frame count
    if len(frames) < 30:
        while len(frames) < 30:
            frames.append(frames[-1])
    
    # Save augmented video
    success = save_video(frames, output_video, fps, resolution)
    
    return success

# ==========================================
# BATCH AUGMENTATION FOR PSL DATASET
# ==========================================

def batch_augment_original_videos(
    dataset_root,
    num_augmented_per_label=50,
    test_first_label_only=False,
    resume_from_label=None,
    verbose=True
):
    """
    Augment original videos in each label folder.
    
    Process only <label>_<num>.mp4 files (not <label>_aug_*.mp4).
    Generate 50 augmented videos per label starting from <num+15>.
    
    Args:
        dataset_root: Root directory containing label folders
        num_augmented_per_label: Number of augmented videos to generate per label
        test_first_label_only: Test on first label only
        resume_from_label: Resume from specific label
        verbose: Print detailed output
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
    
    if test_first_label_only:
        label_folders = label_folders[:1]
    
    print(f"\n{'='*70}")
    print(f"BATCH VIDEO AUGMENTATION - PSL DATASET")
    print(f"{'='*70}")
    print(f"Dataset root: {dataset_root}")
    print(f"Augmented videos per label: {num_augmented_per_label}")
    print(f"Labels to process: {len(label_folders)}")
    print(f"Test mode: {test_first_label_only}")
    print(f"{'='*70}\n")
    
    total_successful = 0
    total_failed = 0
    
    for label_idx, label_folder in enumerate(label_folders, 1):
        label_name = label_folder.name
        
        # Get ONLY original videos (not already augmented)
        # Original: <label>_<num>.mp4
        # Skip: <label>_aug_<num>.mp4
        original_videos = sorted([
            v for v in label_folder.glob("*.mp4")
            if "_aug_" not in v.name  # Skip already augmented
        ])
        
        if not original_videos:
            if verbose:
                print(f"[{label_idx}/{len(label_folders)}] {label_name}: ❌ No original videos found")
            continue
        
        if verbose:
            print(f"{'='*70}")
            print(f"[{label_idx}/{len(label_folders)}] Label: {label_name}")
            print(f"Original videos: {len(original_videos)}")
            print(f"Generating: {num_augmented_per_label} augmented versions")
            print(f"{'='*70}")
        
        successful = 0
        failed = 0
        
        # Generate augmented videos
        for aug_idx in range(num_augmented_per_label):
            # Random source video
            source_video = random.choice(original_videos)
            
            # Extract base name and number
            # e.g., "Afraid_1.mp4" -> base="Afraid", num=1
            video_stem = source_video.stem
            parts = video_stem.split('_')
            
            try:
                original_num = int(parts[-1])
            except:
                original_num = 1
            
            # Output numbering: aug_16, aug_17, ... aug_65 (50 total)
            output_num = 15 + aug_idx + 1  # 16, 17, 18, ...
            output_name = f"{label_name}_aug_{output_num}.mp4"
            output_path = label_folder / output_name
            
            # Skip if already exists
            if output_path.exists():
                if verbose:
                    print(f"  [{aug_idx+1}/{num_augmented_per_label}] {output_name} ⏭️  (exists)", end="")
                continue
            
            if verbose:
                print(f"  [{aug_idx+1}/{num_augmented_per_label}] {source_video.name} → {output_name}", end="")
            
            try:
                success = augment_video_random(
                    input_video=str(source_video),
                    output_video=str(output_path)
                )
                
                if success:
                    if verbose:
                        print(" ✓")
                    successful += 1
                else:
                    if verbose:
                        print(" ❌")
                    failed += 1
                    # Remove failed output if partially created
                    if output_path.exists():
                        output_path.unlink()
            
            except Exception as e:
                if verbose:
                    print(f" ❌ ({str(e)[:40]})")
                failed += 1
        
        if verbose:
            print(f"\n  → {label_name}: {successful} successful, {failed} failed\n")
        
        total_successful += successful
        total_failed += failed
    
    # Final summary
    print(f"{'='*70}")
    print(f"BATCH AUGMENTATION COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Total successful: {total_successful}")
    print(f"✗ Total failed: {total_failed}")
    print(f"{'='*70}\n")

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Batch augmentation for PSL original videos")
    ap.add_argument("dataset", help="PSL dataset root (contains label folders)")
    ap.add_argument("--num-aug", type=int, default=50, help="Number of augmented videos per label")
    ap.add_argument("--test-first-label", action="store_true", help="Test mode: process only first label")
    ap.add_argument("--resume-from", help="Resume from specific label")
    ap.add_argument("--quiet", action="store_true", help="Minimize output")
    
    args = ap.parse_args()
    
    batch_augment_original_videos(
        dataset_root=args.dataset,
        num_augmented_per_label=args.num_aug,
        test_first_label_only=args.test_first_label,
        resume_from_label=args.resume_from,
        verbose=not args.quiet
    )
