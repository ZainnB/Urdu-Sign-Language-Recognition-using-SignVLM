import argparse
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp


# ---------------- Utility Functions ---------------- #

def clamp_xyxy(xyxy, w, h):
    x1, y1, x2, y2 = xyxy
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return np.array([x1, y1, x2, y2], dtype=float)


def pad_xyxy(xyxy, pad_ratio):
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    px, py = w * pad_ratio / 2, h * pad_ratio / 2
    return np.array([x1 - px, y1 - py, x2 + px, y2 + py], dtype=float)


def ema(prev, cur, alpha):
    return cur if prev is None else alpha * cur + (1 - alpha) * prev


def resize_letterbox(img, size):
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))

    out = np.zeros((size, size, 3), dtype=np.uint8)
    x0, y0 = (size - nw) // 2, (size - nh) // 2
    out[y0:y0+nh, x0:x0+nw] = resized
    return out


def union_boxes(boxes):
    if not boxes:
        return None
    xs1 = [b[0] for b in boxes]
    ys1 = [b[1] for b in boxes]
    xs2 = [b[2] for b in boxes]
    ys2 = [b[3] for b in boxes]
    return np.array([min(xs1), min(ys1), max(xs2), max(ys2)], dtype=float)


def landmarks_to_box(landmarks, w, h):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    return np.array([min(xs), min(ys), max(xs), max(ys)], dtype=float)


# ---------------- Main ---------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-folder", default="FYP_DATA")
    ap.add_argument("--out-dir", default="outputs/mp_roi")
    ap.add_argument("--roi-size", type=int, default=256)
    ap.add_argument("--pad", type=float, default=0.4)
    ap.add_argument("--pos-alpha", type=float, default=0.15)
    ap.add_argument("--size-alpha", type=float, default=0.15)
    ap.add_argument("--grace-frames", type=int, default=10)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,  # IMPORTANT for your use-case
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    video_root = Path(args.video_folder)
    videos = list(video_root.rglob("*.mp4"))

    for vid in videos:
        cap = cv2.VideoCapture(str(vid))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        # Mirror subdirectory structure (e.g. class/video.mp4) in output dir
        rel = vid.relative_to(video_root)
        out_clip = out_root / rel.parent / f"{vid.stem}_roi.mp4"
        out_clip.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out_clip),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (args.roi_size, args.roi_size)
        )

        smooth_center = None
        smooth_size = None
        last_box = None
        miss = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            # Convert to RGB (REQUIRED)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_hands.process(rgb)

            boxes = []

            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    box = landmarks_to_box(hand.landmark, w, h)
                    boxes.append(box)

            det = union_boxes(boxes)

            if det is None:
                miss += 1
                if smooth_center is not None and miss <= args.grace_frames:
                    center, size = smooth_center, smooth_size
                else:
                    center, size = None, None
            else:
                miss = 0
                det = clamp_xyxy(pad_xyxy(det, args.pad), w, h)
                last_box = det

                cx = (det[0] + det[2]) / 2
                cy = (det[1] + det[3]) / 2
                side = max(det[2] - det[0], det[3] - det[1])

                smooth_center = ema(smooth_center, np.array([cx, cy]), args.pos_alpha)
                smooth_size = ema(smooth_size, np.array([side]), args.size_alpha)

                center, size = smooth_center, smooth_size

            roi_img = None

            if center is not None:
                half = size[0] / 2
                x1 = int(max(0, center[0] - half))
                y1 = int(max(0, center[1] - half))
                x2 = int(min(w, center[0] + half))
                y2 = int(min(h, center[1] + half))

                roi = frame[y1:y2, x1:x2]
                roi_img = resize_letterbox(roi, args.roi_size)

                writer.write(roi_img)

            if args.show:
                disp = frame.copy()
                if center is not None:
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow("Frame", disp)
                if roi_img is not None:
                    cv2.imshow("ROI", roi_img)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

            frame_idx += 1

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        print(f"Done: {vid.name}")


if __name__ == "__main__":
    main()