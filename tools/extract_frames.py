# === PRE-EXTRACT FRAMES with ffmpeg (run ONCE, before training) =============================
# ffmpeg (optimized C) decodes each video to JPEG frames in a sibling folder
# (X/Y.mp4 -> X/Y/00000.jpg ...). Faster than PyAV+PIL. Idempotent + atomic (safe to resume).
# Retries transient Colab overlay-fs I/O errors. After it finishes: set FRAMES_AVAILABLE = 1
# in Cell 4, then re-run Cell 4 -> 6 -> 7. Safe to re-run: done clips are skipped.
import os, subprocess, shutil, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

CACHE        = "/content/signvlm_data_cache"   # videos live here + frames go here
NUM_WORKERS  = 6                                # parallel ffmpeg procs (lower = less overlay-fs contention)
JPEG_QSCALE  = "2"                              # mjpeg quality: 2 = high (~q90); higher num = smaller
RETRIES      = 3                                # transient overlay-fs I/O errors are retried
USE_GPU      = False                            # NVDEC rarely helps for many tiny clips; leave False

_SPLITS = {"train": (f"{LIST_DIR}/train.tsv", TRAIN_DIR),
           "val":   (f"{LIST_DIR}/val.tsv",   VAL_DIR),
           "test":  (f"{LIST_DIR}/test.tsv",  TEST_DIR)}

def _video_source(relpath, src_root):
    local = Path(CACHE) / relpath          # prefer already-cached local copy (fast, no Drive)
    return local if local.is_file() else Path(src_root) / relpath

def extract_one(relpath, src_root):
    vpath = _video_source(relpath, src_root)
    fdir  = Path(CACHE) / relpath
    fdir  = fdir.parent / fdir.stem        # X/Y frames folder next to X/Y.mp4
    if fdir.is_dir() and any(fdir.glob("*.jpg")):
        return ("skip", relpath)
    if not Path(vpath).is_file():
        return (f"ERROR missing source {vpath}", relpath)
    tmp = fdir.parent / (fdir.name + ".tmp_extract")
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y"]
    if USE_GPU:
        cmd += ["-hwaccel", "cuda"]
    # ffmpeg emits frames in display (presentation) order; %05d starting at 0 keeps sort==order.
    cmd += ["-i", str(vpath), "-qscale:v", JPEG_QSCALE, "-start_number", "0",
            str(tmp / "%05d.jpg")]
    last = ""
    for attempt in range(RETRIES):
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        try:
            tmp.mkdir(parents=True, exist_ok=True)
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and any(tmp.glob("*.jpg")):
                os.replace(str(tmp), str(fdir))   # atomic: only a complete folder appears at fdir
                return ("done", relpath)
            last = f"rc={r.returncode} {r.stderr.strip()[:160]}"
        except Exception as e:
            last = str(e)
        time.sleep(0.4 * (attempt + 1))           # backoff before retrying a transient I/O glitch
    shutil.rmtree(tmp, ignore_errors=True)
    return (f"ERROR {last}", relpath)

if shutil.which("ffmpeg") is None:
    raise RuntimeError("ffmpeg not found on PATH")

tasks = []
for split, (tsv, root) in _SPLITS.items():
    if not os.path.isfile(tsv):
        print(f"(skip split '{split}': no {tsv})"); continue
    for line in open(tsv, encoding="utf-8").read().splitlines():
        if line.strip():
            tasks.append((line.split("\t")[0], root))

print(f"Pre-extracting frames for {len(tasks)} videos via ffmpeg x{NUM_WORKERS} (GPU={USE_GPU})...")
done = skip = err = 0
errors = []
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
    futs = [ex.submit(extract_one, rel, root) for rel, root in tasks]
    for i, f in enumerate(as_completed(futs)):
        status, rel = f.result()
        if   status == "done": done += 1
        elif status == "skip": skip += 1
        else:                  err += 1; errors.append((rel, status))
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(tasks)}  done={done} skip={skip} err={err}")
print(f"FRAME EXTRACTION COMPLETE: extracted={done} already-present={skip} errors={err}")
if errors:
    print(f"--- {len(errors)} clips still failed (re-run this cell to retry just these) ---")
    for rel, status in errors[:30]:
        print("  ", rel, "->", status)
else:
    print("NEXT: set FRAMES_AVAILABLE = 1 in Cell 4, then re-run Cell 4 -> Cell 6 -> Cell 7.")
