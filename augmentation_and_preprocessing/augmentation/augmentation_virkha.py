import cv2
import numpy as np
import os
import random
from skimage import exposure, util
from skimage.transform import swirl
import inspect

#-------------------------------------------------AUGMENTATION FUNCTIONS-----------------------------------------------------
def adjust_brightness(frame, factor):
    return cv2.convertScaleAbs(frame, alpha=factor, beta=0)

def add_gaussian_noise(frame, sigma):
    noise = np.random.normal(0, sigma, frame.shape).astype('int16')
    noisy_frame = np.clip(frame.astype('int16') + noise, 0, 255).astype('uint8')
    return noisy_frame

def rotate_frame(frame, angle):
    (h, w) = frame.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(frame, matrix, (w, h))

def scale_frame(frame, scale_factor):
    (h, w) = frame.shape[:2]
    scaled_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
    new_h, new_w = scaled_frame.shape[:2]
    if scale_factor > 1:
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        return scaled_frame[start_y:start_y + h, start_x:start_x + w]
    pad_x = (w - new_w) // 2
    pad_y = (h - new_h) // 2
    return cv2.copyMakeBorder(scaled_frame, pad_y, h - new_h - pad_y, pad_x, w - new_w - pad_x, cv2.BORDER_CONSTANT, value=(0, 0, 0))

def translate_frame(frame, shift_x, shift_y):
    (h, w) = frame.shape[:2]
    matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(frame, matrix, (w, h))

def adjust_contrast(frame, factor):
    return cv2.convertScaleAbs(frame, alpha=factor, beta=0)

def adjust_brightness_saturation_hue(frame, brightness, saturation, hue):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.convertScaleAbs(v, alpha=brightness / 100)
    s = cv2.convertScaleAbs(s, alpha=saturation / 100)
    h = ((h.astype(np.int16) + hue) % 180).astype('uint8')
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_swirl(frame, strength):
    frame_float = frame.astype(np.float32) / 255.0
    swirled = swirl(frame_float, strength=strength, radius=120, center=(frame.shape[1] // 2, frame.shape[0] // 2), mode='wrap')
    return (swirled * 255).astype('uint8')

def normalize_quantize(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    normalized = exposure.equalize_hist(gray)
    quantized = util.img_as_ubyte(normalized)
    return cv2.cvtColor(quantized, cv2.COLOR_GRAY2BGR)  # Convert back to BGR if needed

def apply_gamma_correction(frame, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype('uint8')
    return cv2.LUT(frame, table)

def add_de_speckle(frame):
    return cv2.medianBlur(frame, 5)

#-------------------------------------------------AUGMENT VIDEO--------------------------------------------------------------
def augment_video(videos_path, output_video_path, label):
    os.makedirs(output_video_path, exist_ok=True)
    # Collect all video paths
    video_files = [os.path.join(videos_path, f) for f in os.listdir(videos_path) if f.endswith(('.mp4', '.mov'))]
    if len(video_files) < 10:
        raise ValueError("Expected at least 10 videos in the input path.")
    augmented_video_count = 76

    # Augmentation parameters
    brightness_factors = [0.5, 0.8, 1.2, 1.5]
    noise_sigmas = [2, 3, 3.2]
    rotation_angles = [-10, -5, -2.5, 2.5, 5, 10]
    scale_factors = [0.8, 0.9, 1.1, 1.2]
    translation_shifts = [-40, 40]
    contrast_factors = [0.8, 1.0, 1.2]
    swirl_strengths = [-5, 5, 10]
    brightness_saturation_hue = [(90, 110, 10), (85, 115, -10)]
    gamma_values = [0.5, 1.2, 2.0]

    # Augment videos randomly
    for i in range(augmented_video_count):
        print('Processing:', i + 1)
        video_path = random.choice(video_files)
        
        # Generate output path based on input filename
        input_filename = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{input_filename}_aug_{i+1}.mp4"
        output_path = os.path.join(output_video_path, output_filename)
        
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = cv2.VideoWriter(filename=output_path, fourcc=fourcc, fps=fps, frameSize=(width, height))

        bright_factor = random.choice(brightness_factors)
        noise_sigma = random.choice(noise_sigmas)
        rot_angle = random.choice(rotation_angles)
        scale_factor=random.choice(scale_factors)
        tran_shift=random.choice(translation_shifts)
        contrast_factor=random.choice(contrast_factors)
        swirl_str=random.choice(swirl_strengths)
        gamma_val=random.choice(gamma_values)
        augmentation = random.choices([
                lambda frame: adjust_brightness(frame, bright_factor),
                lambda frame: add_gaussian_noise(frame, noise_sigma),
                lambda frame: rotate_frame(frame, rot_angle),
                lambda frame: scale_frame(frame, scale_factor),
                lambda frame: translate_frame(frame, tran_shift, 0),
                lambda frame: adjust_contrast(frame, contrast_factor),
                lambda frame: adjust_brightness_saturation_hue(frame, *random.choice(brightness_saturation_hue)),
                lambda frame: apply_swirl(frame, swirl_str),
                lambda frame: normalize_quantize(frame),
                lambda frame: apply_gamma_correction(frame, gamma_val),
                lambda frame: add_de_speckle(frame),
            ], 
            weights=[0.0952381 , 0.0952381 , 0.0952381 , 0.0952381 , 0.0952381 , 0.0952381 , 0.0952381 , 0.04761905, 0.0952381 , 0.0952381 , 0.0952381]
        )[0]
        print(f'\t{inspect.getsource(augmentation)}')
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            aug_frame = augmentation(frame)
            writer.write(aug_frame)

        cap.release()
        writer.release()

#---------------------------------------------------USAGE-------------------------------------------------------------------
import concurrent.futures
import time

def process_label(label, base_path):
    """Helper function to process a single label for parallel execution"""
    try:
        print(f"Starting label: {label}")
        label_path = os.path.join(base_path, label)
        augment_video(
            videos_path=label_path,
            output_video_path=label_path,
            label=label
        )
        print(f"Completed label: {label}")
        return f"Success: {label}"
    except Exception as e:
        return f"Error processing {label}: {str(e)}"


if __name__ == '__main__':
    base_path = r"Data/newdata"
    
    if not os.path.exists(base_path):
        print(f"Error: Path not found: {base_path}")
        exit(1)

    labels = [l for l in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, l))]
    print(f"Found {len(labels)} labels to process.")
    
    # Use 75% of available CPU cores to avoid freezing the system
    max_workers = 2
    print(f"Starting parallel processing with {max_workers} workers...")
    
    start_time = time.time()
    
    import functools

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        func = functools.partial(process_label, base_path=base_path)
        results = list(executor.map(func, labels))


    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*50)
    print(f"Processing complete in {duration:.2f} seconds")
    print("="*50)
    
    # Print any errors
    for res in results:
        if "Error" in res:
            print(res)