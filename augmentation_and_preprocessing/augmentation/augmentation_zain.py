import cv2
import numpy as np
import os
from scipy.interpolate import interp1d

# ==========================================
# INTERPOLATION & HELPER FUNCTIONS
# ==========================================

def interpolate_frames(frames, target_indices):
    """
    Interpolate frames at specified indices using linear interpolation.
    Useful for smooth speed perturbation and temporal resampling.
    
    Args:
        frames: List of frames
        target_indices: Indices where frames should be interpolated
    
    Returns:
        List of interpolated frames
    """
    if len(frames) == 0:
        return frames
    
    # For frame indices, we use nearest neighbor if interpolation would be too expensive
    # For more robust interpolation, could use optical flow but keeping it simple for now
    interpolated = []
    for idx in target_indices:
        idx_int = int(np.clip(idx, 0, len(frames) - 1))
        idx_next = min(idx_int + 1, len(frames) - 1)
        alpha = idx - idx_int
        
        if alpha < 0.5:
            interpolated.append(frames[idx_int])
        else:
            # Simple alpha blending between two frames
            frame_blend = cv2.addWeighted(frames[idx_int], 1 - alpha, frames[idx_next], alpha, 0)
            interpolated.append(frame_blend)
    
    return interpolated

def compute_optical_flow_magnitude(frames, step=2):
    """
    Compute optical flow magnitude for motion-aware frame dropping.
    Returns motion magnitude per frame (low=static, high=motion).
    
    Args:
        frames: List of frames
        step: Compute flow every 'step' frames to save computation
    
    Returns:
        Array of motion magnitudes per frame
    """
    motion = np.zeros(len(frames))
    
    if len(frames) < 2:
        return motion
    
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    motion[0] = 0
    
    for i in range(1, len(frames), step):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion[i] = magnitude.mean()
        prev_gray = curr_gray
    
    # Interpolate motion for skipped frames
    for i in range(len(frames)):
        if motion[i] == 0 and i > 0:
            motion[i] = motion[i - 1]
    
    return motion

def ensure_monotonic_indices(indices):
    """
    Ensure indices are monotonically increasing (no reversals).
    Clamp each index to be >= previous index.
    """
    if len(indices) == 0:
        return indices
    
    monotonic = [indices[0]]
    for i in range(1, len(indices)):
        monotonic.append(max(monotonic[-1], indices[i]))
    
    return np.array(monotonic)

# ==========================================
# SPATIAL AUGMENTATIONS
# ==========================================

def apply_zoom(frame, zoom_factor):
    """
    Apply zoom to the frame (1.0-1.4).
    zoom_factor > 1: zoom in (closer view)
    """
    height, width = frame.shape[:2]
    center = (width // 2, height // 2)

    new_width = int(width / zoom_factor)
    new_height = int(height / zoom_factor)

    x1 = max(0, center[0] - new_width // 2)
    y1 = max(0, center[1] - new_height // 2)
    x2 = min(width, center[0] + new_width // 2)
    y2 = min(height, center[1] + new_height // 2)

    cropped = frame[y1:y2, x1:x2]
    zoomed = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)

    return zoomed

def apply_rotation(frame, angle):
    """
    Apply rotation to the frame (±5-8 degrees).
    """
    height, width = frame.shape[:2]
    center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(frame, rotation_matrix, (width, height), borderMode=cv2.BORDER_REFLECT)

    return rotated

def apply_brightness_jitter(frame, factor):
    """
    Apply brightness jitter (0.8-1.5x).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    brightened = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return brightened

def crop_upper_body(frames, center_ratio=0.4, height_ratio=0.7):
    """
    Crop frames to upper body region to mitigate identity leakage.
    Keep: hands, arms, torso, head
    Remove: background, legs, person-specific details
    
    Args:
        frames: List of frames
        center_ratio: Width crop ratio (±center_ratio from center)
        height_ratio: Height crop ratio (top 0 to height_ratio)
    """
    if len(frames) == 0:
        return frames
    
    height, width = frames[0].shape[:2]
    
    # Calculate crop region
    center_x = width // 2
    crop_width = int(width * center_ratio)
    x1 = max(0, center_x - crop_width // 2)
    x2 = min(width, center_x + crop_width // 2)
    
    crop_height = int(height * height_ratio)
    y1 = 0
    y2 = crop_height
    
    cropped_frames = []
    for frame in frames:
        cropped = frame[y1:y2, x1:x2]
        # Resize back to original dimensions
        resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
        cropped_frames.append(resized)
    
    return cropped_frames

# ==========================================
# ADVANCED TEMPORAL AUGMENTATIONS
# ==========================================

def apply_temporal_crop_smart(frames, min_window=0.5, max_window=0.95, random_seed=None):
    """
    Random temporal crop with smart boundaries (keeps gesture intact).
    
    Args:
        frames: List of frames
        min_window: Minimum fraction of video to keep
        max_window: Maximum fraction of video to keep
    
    Returns:
        Cropped frames
    """
    if len(frames) < 2:
        return frames
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Random window size
    window = np.random.uniform(min_window, max_window)
    crop_length = int(len(frames) * window)
    
    # Random start position (ensure we don't go past the end)
    max_start = len(frames) - crop_length
    start_idx = np.random.randint(0, max(1, max_start + 1))
    
    cropped = frames[start_idx:start_idx + crop_length]
    
    return cropped if len(cropped) > 0 else frames

def apply_speed_perturbation_smooth(frames, speed_factor, random_seed=None):
    """
    Speed perturbation with smooth frame interpolation.
    speed_factor < 1: slow down (more frames)
    speed_factor > 1: speed up (fewer frames)
    
    Args:
        frames: List of frames
        speed_factor: Speed multiplier (0.8-1.2)
    
    Returns:
        Resampled frames with interpolation
    """
    if len(frames) < 2 or speed_factor == 1.0:
        return frames
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    total_frames = len(frames)
    new_frame_count = max(1, int(total_frames / speed_factor))
    
    # Generate resampling indices with interpolation
    target_indices = np.linspace(0, total_frames - 1, new_frame_count)
    resampled = interpolate_frames(frames, target_indices)
    
    return resampled

def apply_frame_jitter_bounded(frames, jitter_range=2, random_seed=None):
    """
    Frame jitter with bounded random offsets and monotonic index constraint.
    Prevents frame reversals while adding temporal randomness.
    
    Args:
        frames: List of frames
        jitter_range: Max offset per frame (±jitter_range)
    
    Returns:
        Jittered frames
    """
    if len(frames) < 2:
        return frames
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    total_frames = len(frames)
    
    # Generate jittered indices with random offsets
    indices = np.arange(total_frames, dtype=float)
    jitter_offsets = np.random.randint(-jitter_range, jitter_range + 1, total_frames)
    jittered_indices = indices + jitter_offsets
    
    # Clamp to valid range and ensure monotonic
    jittered_indices = np.clip(jittered_indices, 0, total_frames - 1)
    jittered_indices = ensure_monotonic_indices(jittered_indices)
    
    # Interpolate frames
    jittered = interpolate_frames(frames, jittered_indices)
    
    return jittered

def apply_motion_aware_drop(frames, target_drop_rate=0.1, optical_flow_step=2, random_seed=None):
    """
    Motion-aware frame dropping: preserve high-motion frames (gesture peaks).
    
    Args:
        frames: List of frames
        target_drop_rate: Target percentage to drop (0.1 = 10%)
        optical_flow_step: Compute flow every N frames to save computation
    
    Returns:
        Frames with low-motion frames preferentially dropped
    """
    if len(frames) < 2:
        return frames
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Compute motion magnitude
    motion = compute_optical_flow_magnitude(frames, step=optical_flow_step)
    
    # Normalize motion to [0, 1]
    motion_min = motion.min()
    motion_max = motion.max()
    if motion_max > motion_min:
        motion_norm = (motion - motion_min) / (motion_max - motion_min)
    else:
        motion_norm = np.zeros_like(motion)
    
    # Drop probability inversely proportional to motion (low motion → higher drop chance)
    drop_prob = target_drop_rate * (1 - motion_norm)
    
    keep_indices = []
    for i in range(len(frames)):
        if np.random.rand() > drop_prob[i]:
            keep_indices.append(i)
    
    # Ensure minimum frames retained
    min_kept = max(1, int(len(frames) * (1 - target_drop_rate - 0.05)))
    if len(keep_indices) < min_kept:
        # Keep highest-motion frames
        top_indices = np.argsort(motion_norm)[-min_kept:]
        keep_indices = sorted(top_indices)
    
    dropped_frames = [frames[i] for i in keep_indices]
    
    return dropped_frames if len(dropped_frames) > 0 else frames

def apply_clip_shift(frames, shift_amount=None, random_seed=None):
    """
    Shift clip forward/backward by a few frames.
    Teaches temporal invariance in sequence ordering.
    
    Args:
        frames: List of frames
        shift_amount: Number of frames to shift (auto-random if None)
    
    Returns:
        Shifted frames (circular shift)
    """
    if len(frames) < 2:
        return frames
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if shift_amount is None:
        shift_amount = np.random.randint(1, min(4, len(frames) // 10 + 1))
    
    # Circular rotation
    shifted = frames[shift_amount:] + frames[:shift_amount]
    
    return shifted

def apply_keyframe_emphasis(frames, duplication_prob=0.4, optical_flow_step=2, random_seed=None):
    """
    Identify and optionally duplicate peak frames (highest motion).
    Peak poses carry most linguistic information in signs.
    
    Args:
        frames: List of frames
        duplication_prob: Probability to duplicate peak frame
        optical_flow_step: Compute flow every N frames
    
    Returns:
        Frames with emphasized peak (optionally duplicated)
    """
    if len(frames) < 2:
        return frames
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Compute motion magnitude
    motion = compute_optical_flow_magnitude(frames, step=optical_flow_step)
    
    # Find peak frame
    peak_idx = np.argmax(motion)
    
    # Optionally duplicate peak frame
    if np.random.rand() < duplication_prob and peak_idx > 0:
        emphasized = frames[:peak_idx] + [frames[peak_idx]] + frames[peak_idx:]
        return emphasized
    
    return frames

# ==========================================
# AUGMENTATION PIPELINE
# ==========================================

def load_video(video_path):
    """Load video frames from file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return None, None, None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
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
    """Save frames to video file."""
    if len(frames) == 0:
        print(f"No frames to save for {output_path}")
        return
    
    width, height = resolution
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    print(f"Saved augmented video to {output_path}")

def augment_video_single(video_path, output_path, augmentation_type):
    """Apply single augmentation technique to video."""
    frames, fps, resolution = load_video(video_path)
    
    if frames is None:
        return
    
    print(f"Loaded {len(frames)} frames from {video_path}")
    
    # Apply specific augmentation
    if augmentation_type == 'zoom':
        frames = [apply_zoom(frame, 1.4) for frame in frames]
        print("Applied zoom augmentation (1.4×)")
    
    elif augmentation_type == 'rotation':
        frames = [apply_rotation(frame, 10) for frame in frames]
        print("Applied rotation augmentation (10°)")
    
    elif augmentation_type == 'brightness':
        frames = [apply_brightness_jitter(frame, 1.5) for frame in frames]
        print("Applied brightness augmentation (1.5×)")
    
    elif augmentation_type == 'temporal_crop':
        frames = apply_temporal_crop_smart(frames, min_window=0.55, max_window=0.75)
        print("Applied temporal crop augmentation (smart boundaries)")
    
    elif augmentation_type == 'speed':
        frames = apply_speed_perturbation_smooth(frames, speed_factor=1.3)
        fps = fps * 1.3
        print("Applied speed perturbation with interpolation (1.3×)")
    
    elif augmentation_type == 'jitter':
        frames = apply_frame_jitter_bounded(frames, jitter_range=2)
        print("Applied bounded frame jitter (±2)")
    
    elif augmentation_type == 'drop':
        frames = apply_motion_aware_drop(frames, target_drop_rate=0.12)
        print("Applied motion-aware frame drop (12%)")
    
    elif augmentation_type == 'shift':
        frames = apply_clip_shift(frames, shift_amount=3)
        print("Applied clip shift (+3 frames)")
    
    elif augmentation_type == 'keyframe':
        frames = apply_keyframe_emphasis(frames, duplication_prob=0.5)
        print("Applied keyframe emphasis")
    
    elif augmentation_type == 'upperBody':
        frames = crop_upper_body(frames, center_ratio=0.4, height_ratio=0.7)
        print("Applied upper-body cropping")
    
    # Ensure minimum temporal duration
    if len(frames) < 30:
        print(f"Warning: Video too short ({len(frames)} frames), padding with last frame")
        while len(frames) < 30:
            frames.append(frames[-1])
    
    print(f"After augmentation: {len(frames)} frames")
    
    save_video(frames, output_path, fps, resolution)

if __name__ == "__main__":
    # Test on single video with enhanced augmentations
    video_path = r"video_data\Afraid_12.mp4"
    
    augmentations = [
        ('zoom', 'Afraid_12_phase1_zoom.mp4'),
        ('rotation', 'Afraid_12_phase1_rotation.mp4'),
        ('brightness', 'Afraid_12_phase1_brightness.mp4'),
        ('temporal_crop', 'Afraid_12_phase1_temporal_crop.mp4'),
        ('speed', 'Afraid_12_phase1_speed_smooth.mp4'),
        ('jitter', 'Afraid_12_phase1_jitter_bounded.mp4'),
        ('drop', 'Afraid_12_phase1_drop_motion_aware.mp4'),
        ('shift', 'Afraid_12_phase1_clip_shift.mp4'),
        ('keyframe', 'Afraid_12_phase1_keyframe.mp4'),
        ('upperBody', 'Afraid_12_phase1_upper_body.mp4'),
    ]
    
    for aug_type, output_name in augmentations:
        output_path = rf"video_data\{output_name}"
        print(f"\n{'='*60}")
        print(f"Augmentation: {aug_type}")
        print(f"{'='*60}")
        augment_video_single(video_path, output_path, aug_type)