# Video Dataset Ingestion Pipeline — Context Handoff

This document describes the data ingestion pipeline built for the SignVLM (Urdu Sign Language
Recognition) project, running on Google Colab. It is meant to be handed to a fresh Claude
conversation as full context so it can design an equivalent (or improved) ingestion pipeline for
another video-classification model. It covers the folder layout, Drive linking, on-disk caching,
RAM caching, frame extraction, DataLoader tuning, and every trap that was hit and fixed along the
way — so the next implementation doesn't repeat the same mistakes.

The pipeline trains a video classifier (CLIP-ViT backbone + temporal decoder) on short `.mp4` sign
clips, in a Colab environment where: the dataset lives on Google Drive (slow, network-mounted),
local disk (`/content`) is fast but ephemeral (wiped every runtime restart), and RAM is the fastest
tier but limited and also wiped on restart.

## 1. The three-tier storage model

```
Google Drive (/content/drive/MyDrive/...)   <- slow, persistent, source of truth
        |  (one-time copy per file, on first access)
        v
Local disk cache (/content/signvlm_data_cache)  <- fast, ephemeral (wiped on runtime reset)
        |  (one-time read into memory, on dataset construction)
        v
RAM (per-process dict of raw JPEG bytes)     <- fastest, ephemeral (wiped on runtime reset)
```

Every tier is populated **lazily and idempotently** — nothing is bulk-copied up front. A clip (or
its extracted frames) moves up a tier only the first time it's actually requested, and every later
access (same epoch, later epoch, or even a re-run of the notebook after a runtime restart, as long
as `/content` survived) short-circuits to the cached copy. This matters because Drive datasets can
be tens of thousands of clips, and eagerly mirroring the whole thing before training could start
would burn many minutes doing nothing useful; the on-first-use design lets Epoch 1 start
immediately while the cache fills in as training touches each clip.

## 2. Google Drive linking (mount + folder structure)

Drive is mounted once via `google.colab.drive.mount("/content/drive")`. All paths downstream are
derived from `/content/drive/MyDrive/<project>/...` — never hardcoded absolute paths, since the
same notebook needs to run in Colab and (via env var overrides) outside Colab for local testing.

Drive folder layout (source of truth, one-time human-curated data):

```
MyDrive/FYP/
├── Dataset_Final/
│   ├── train_split/<ClassName>/*.mp4
│   ├── val_split/<ClassName>/*.mp4
│   ├── test_data_diff_signer/<ClassName>/*.mp4      # held-out signer, tests generalization
│   └── unseen_data/                                  # either flat *.mp4 (stem = class name)
│                                                       # or <ClassName>/*.mp4 subfolders
├── Clip_Weights/                                      # CLIP .pt backbone weights, any depth,
│                                                       # auto-discovered by glob (*.pt)
└── Models/
    ├── signVLM_checkpoints/                           # training checkpoints (persisted!)
    ├── signVLM_lists/                                 # generated TSV splits + label map + logs
    └── Urdu-Sign-Language-Recognition-using-SignVLM/  # optional full repo copy (fallback if no git)
```

**Key trap avoided:** checkpoints and TSV lists are written under `MyDrive/FYP/Models/...`, i.e.
**on Drive, not `/content`**. Colab runtimes are ephemeral — anything written only to `/content`
disappears when the runtime recycles. The notebook actively guards this:

```python
if IN_COLAB and not MODELS_DIR.startswith("/content/drive/MyDrive"):
    raise RuntimeError(
        "MODELS_DIR must be under mounted Google Drive (/content/drive/MyDrive/...). "
        "Set SIGNVLM_MODELS_DIR to a path under MyDrive, not /content alone."
    )
```

Only the **dataset cache and extracted frames** live under `/content` (rebuildable from Drive at
any time, so losing them on a runtime reset just costs a re-extraction pass, not real data).

Every path is overridable via environment variables (`SIGNVLM_DATASET_BASE`, `SIGNVLM_MODELS_DIR`,
`SIGNVLM_BACKBONE_PATH`, etc.) with Drive-relative defaults — this is what lets the exact same
notebook run outside Colab (e.g. a local RTX 3060 box) by just setting env vars instead of editing
code.

Training code itself is pulled from GitHub at notebook start (shallow clone, `--depth 1`, pinned
branch, optional `GITHUB_TOKEN` from Colab Secrets for private repos), not copied by hand onto
Drive — this keeps the notebook self-updating and avoids code/Drive-copy drift. A full-repo-on-Drive
path is kept only as a fallback if no git URL is configured.

## 3. TSV manifest generation (Drive scan → flat list)

Before any video I/O happens, the class-labeled folder structure on Drive is scanned once and
flattened into simple TSV files: `<relative_path>\t<label_id>` per line. This is cheap (just a
directory listing, no video reads) and gives every downstream stage (frame extraction, DataLoader,
standalone eval cells) a single flat source of truth to iterate, instead of each one re-deriving
labels from folder names.

- Label IDs are derived **only from the train split's** folder names (`sorted(...).casefold()` for
  determinism), then reused for val/test/unseen — this guarantees consistent label IDs across
  splits even if a class is momentarily missing from val or test.
- Unknown folder names in a later split raise loudly (`ValueError`) rather than silently getting
  skipped or mis-mapped.
- `unseen_data` supports two on-disk shapes (flat `*.mp4` with the class in the filename stem, or
  class subfolders like the other splits) — auto-detected.
- A `REUSE_EXISTING_TSV_LISTS` flag exists so TSVs are **not** silently stale after adding new data
  (default `False`: always rebuild); flip to `True` only when you deliberately want to freeze splits
  for comparable experiment numbers.

## 4. Local disk cache (Drive → `/content`, on first use)

Implemented in `video_dataset/drive_to_local_cache.py`, function `resolve_cached_path(...)`.

- Every `__getitem__` resolves its path through this function instead of reading straight from
  Drive. First access: copy source → local cache (atomic — copy to a `.partial` temp name, then
  `os.replace`, so a crash mid-copy never leaves a half-written file that looks "done"). Every later
  access: the local file already exists, Drive is never touched again for that clip.
- **Cross-process locking** (`filelock`, or a hand-rolled `msvcrt`/`fcntl` fallback) guards the
  copy so that multiple DataLoader worker processes racing on the same never-yet-cached clip don't
  stomp on each other or duplicate the copy.
- Works for both raw video files (`frames_available=0`) and pre-extracted frame directories
  (`frames_available=1`, copies the whole directory via `shutil.copytree` into a `.partial` dir then
  atomically renames it into place).

**Trap hit and fixed — cross-split cache collisions.** Different splits (train/val/test/unseen)
each have their own Drive root, but routinely contain identically-named files — every split's
`ClassA` folder has a `1.mp4`, numbered independently per split. The cache was originally keyed
only by the file's path *relative to its own split root*, so `train_split/ClassA/1.mp4` and
`val_split/ClassA/1.mp4` resolved to the **same destination path** in the shared local cache dir.
Whichever split populated the cache first silently "won" — the other split would then transparently
serve the wrong split's video (or extracted frames) under its own label, with **zero errors or
warnings**. This is a data-corruption bug, not a crash, so it's exactly the kind of thing that can
quietly poison months of experiment results before anyone notices the accuracy numbers look off.

Fix: namespace every path inside the shared cache dir by a hash of the *resolved* source root:

```python
def _split_cache_subdir(data_root: str) -> str:
    resolved = str(Path(data_root).resolve())
    h = hashlib.sha256(resolved.encode("utf-8", errors="surrogateescape")).hexdigest()[:8]
    name = Path(data_root).name or "root"
    return f"{name}_{h}"
```
so the effective cache path becomes `local_cache_dir/<split_name>_<hash>/<relpath>` — different
splits can never alias each other even with identical relative paths, and the folder name stays
human-readable for debugging.

**Lesson for the next pipeline:** any shared cache keyed by "path relative to root" is a collision
trap the moment more than one root can be in play (train/val/test, or multiple datasets). Always
namespace by (a hash of) the *absolute* source root, not just the relative path.

## 5. Frame extraction (video → JPEG, via ffmpeg)

Rather than decoding video containers repeatedly during training (expensive, and PyAV decode
quirks — see §7), the pipeline extracts every clip to a folder of JPEG frames once, then trains
against those JPEGs (`frames_available=1`).

- Runs automatically as part of building the TSVs (idempotent: skips any clip whose frame folder
  already has `.jpg` files, so re-running the notebook after a partial run only extracts what's
  missing).
- Parallelized via a `ThreadPoolExecutor` (default 6 workers — tuned down from higher counts
  because Colab's overlay filesystem chokes under too much concurrent I/O contention).
- `ffmpeg -nostdin -loglevel error -y -i <video> -qscale:v 2 -start_number 0 <tmp>/%05d.jpg` —
  `qscale:v 2` is high-quality mjpeg (~q90); tune upward for smaller files if disk is the
  constraint.
- **Atomic per-clip commit:** frames are written into a `<name>.tmp_extract` directory, and only
  `os.replace()`'d into the final `<name>` directory once ffmpeg exits 0 *and* produced at least one
  `.jpg`. This means a crash/kill mid-extraction never leaves a folder that looks "done" but is
  actually empty or truncated — the idempotency check (`fdir.is_dir() and any(fdir.glob("*.jpg"))`)
  can trust that a non-empty folder really is complete.
- Retried up to 3 times per clip with a short backoff — Colab's overlay-fs throws transient I/O
  errors under load that usually succeed on retry.
- Extraction is namespaced through the **same** `_split_cache_subdir()` helper as the disk cache
  (§4), for the same collision-avoidance reason.

**Trap hit and fixed — corrupt source clips shouldn't fail the whole split.** A handful of source
videos are genuinely corrupted (ffmpeg: "moov atom not found" — truncated/incomplete uploads), not
fixable by retrying. Originally this either hard-failed the entire evaluation/extraction pass, or
(worse) silently fell through to a zero-frame clip that the dataset loader would then convert to a
**zero tensor** and score as a wrong prediction — quietly deflating accuracy numbers without any
visible error. Fix: track which relpaths failed extraction, exclude *only those* from that split's
TSV (writing a `<split>_clean.tsv`), print a clear `WARNING: excluding N corrupt clip(s)` with the
list of filenames, and only hard-fail if **literally zero** clips in the split extracted — because
that pattern (0/N succeeding) indicates a systemic problem (wrong ffmpeg path, wrong source root),
not a few bad files, and should stop the run rather than silently produce a meaningless empty split.

**Trap hit and fixed — variable-ordering bug in a "standalone" cell.** A cell designed to
re-extract/evaluate a single split independently (without re-running the full pipeline) referenced
helper variables before they were defined in that cell's execution order, and the resulting
failure was swallowed several layers downstream as an opaque `"need at least one array to stack"`
error (i.e. a clip with zero decoded frames) instead of surfacing the real cause. Lesson: when
duplicating a data-loading code path into a "run this cell standalone" variant, verify the
duplicate's variable ordering explicitly — don't assume copy-pasted setup code preserves the
original ordering — and make empty-extraction states fail loudly with a specific message, not
silently degrade into a generic downstream stacking error.

## 6. RAM preloading (JPEGs → per-process memory dict)

Once frames are extracted to disk, an optional further step reads every clip's frame bytes into a
Python dict (`relpath -> list[bytes]`) at `Dataset.__init__` time, so **every subsequent epoch reads
zero bytes from disk** — pure RAM decode from then on.

```python
# relpath -> list[bytes] (raw JPEG). Populated once; whole dataset lives in RAM afterward.
self._ram_frames = {}
```

- Gated by `preload_to_ram` (train) / `preload_val_to_ram` (val) flags — off by default; opt-in per
  dataset since it multiplies baseline RAM use.
- Only valid when `frames_available=1` (pre-extracted frames — raises `ValueError` otherwise,
  since there'd be nothing meaningful to preload).
- Preloading val is worth it specifically because `val_loader` is *reused across every epoch* in
  the training loop (Cell 7), so the RAM cost is paid once and every epoch's validation pass
  benefits; a one-off eval run wouldn't get the same payoff.
- **Empty-frame-dir detection is explicit and fails loudly:** `Path.glob()` on a missing/empty
  directory silently returns `[]` rather than raising, so without an explicit check, running
  `preload_to_ram=True` *before* frame extraction has actually happened would "succeed" with zero
  bytes cached per clip and train silently on all-zero fallback tensors — a completely invisible
  failure mode. The fix collects every relpath with zero frame files found and raises a
  `RuntimeError` listing the first few, telling the user to run extraction first.

**Trap: fork vs spawn semantics with `num_workers > 0`.** The RAM preload dict is populated in the
main process during `Dataset.__init__`, *before* the `DataLoader` forks worker processes. On
Linux/Colab, `multiprocessing`'s default start method is `fork`, so worker processes inherit the
already-populated dict via copy-on-write — no duplication, no wasted memory, no re-read. On
**Windows**, the default start method is `spawn` — each worker process re-imports the script from
scratch and re-runs the entire `_preload_to_ram()` pass **independently**, multiplying RAM usage by
`num_workers + 1` and multiplying the one-time preload cost by the same factor. The documented
workaround: use `num_workers=0` on Windows when RAM-preloading, or only enable RAM preload on
Colab/Linux where fork semantics make it actually cheap. **This is the single most important
platform gotcha to carry into a new pipeline** — any "preload once into a shared in-process
structure" optimization needs to be re-examined per OS before assuming it will behave the same way.

## 7. Video decode optimizations and their safety nets

For the (fallback) path where frames are *not* pre-extracted and PyAV decodes the container
directly, two further optimizations were layered in, each with an explicit correctness escape
hatch — the guiding principle throughout this codebase is **"a decode-path optimization must never
be allowed to silently produce a wrong answer; on any doubt, degrade to the slow-but-correct path
instead."**

- **Partial decode for training** (`_decode_sampled_frames_train`): rather than decoding every
  frame of a clip and then discarding all but a handful the sampler wants, this demuxes the
  container once (header parse, no pixel decode) to get a reliable true frame count, then decodes
  and keeps *only* the sampled indices, breaking out of the decode loop as soon as they're all
  collected. This is a big speedup for long clips.
  - The container's own `nb_frames` metadata was found to be **unreliable** for this dataset
    (re-encoded files overcount by 10-30 frames), so the code counts actual demuxed packets instead
    and caches that true count per file (`_FRAME_COUNT_CACHE`), paying the demux cost once per file
    per process rather than every epoch.
  - The fast path is only trusted when decoded frame PTS values are strictly increasing with no
    duplicates *and* the decoded count matches the packet count — the exact assumption the original
    full-decode path silently relied on (`sorted(dict-by-pts)`). Any violation raises immediately,
    the caller falls back to full decode, and the file is marked `_FULL_DECODE_SENTINEL` so future
    epochs skip straight to the fallback instead of re-attempting (and re-failing) the fast path
    every time.
- **Zero-tensor fallback:** if *both* the partial-decode fast path and the full-decode fallback
  fail (genuinely corrupt file, mid-training), `__getitem__` returns a zero tensor with the correct
  shape rather than raising — this keeps a single bad clip from crashing an entire training epoch
  via a DataLoader worker exception. The tradeoff (worth knowing about) is that this masks the
  failure as a normal (if very wrong) training sample rather than an explicit skip; the later "skip
  corrupt clips from the split entirely" pattern in the extraction step (§5) is the more correct
  fix for eval, and is generally the better pattern to carry forward — **prefer excluding known-bad
  samples from the manifest over silently substituting zero tensors**, since the latter pollutes
  metrics without any signal that it happened.

## 8. DataLoader tuning — profiles per GPU

The DataLoader is tuned with a small number of named profiles rather than magic numbers scattered
through the code, because these knobs are extremely GPU/host-dependent and get re-tuned per
machine (A100 in Colab Pro vs a local RTX 3060 vs a Colab T4):

```python
TRAIN_SPEED_PROFILE = "max_speed"  # or "fast_stable"
if TRAIN_SPEED_PROFILE == "max_speed":       # A100 40GB, 12 vCPUs
    PROFILE_BATCH_SIZE = 20        # confirmed ~23.5GB peak VRAM at this size
    PROFILE_NUM_WORKERS = 8        # 8 is the CPU-optimal worker count for 12 vCPUs here
    PROFILE_PREFETCH_FACTOR = 4
elif TRAIN_SPEED_PROFILE == "fast_stable":   # safer default, less VRAM/CPU pressure
    PROFILE_BATCH_SIZE = 12
    PROFILE_NUM_WORKERS = min(4, max(2, (os.cpu_count() // 4) or 2))
    PROFILE_PREFETCH_FACTOR = 2
```

Other tuning notes worth carrying forward:
- `persistent_workers=True` (when `num_workers>0`) so per-clip caches inside each worker
  (`_frame_list_cache`, the demux frame-count cache) survive across epochs instead of being torn
  down and rebuilt every epoch.
- `pin_memory=True` for faster host→GPU transfer, standard practice.
- A separate, more conservative eval-time worker/prefetch profile
  (`_loader_eval_memory_kwargs` in `dataloader.py`) exists because eval loaders are constructed and
  torn down repeatedly (once per split, in `Cell 8`'s train→val→test→unseen sequence) — using the
  same aggressive prefetching as training there would multiply RAM pressure across several
  short-lived loaders back-to-back. Each split's loader (and its underlying dataset object) is
  explicitly `del`'d and garbage-collected (plus `torch.cuda.empty_cache()`) before building the
  next split's loader — evaluate one split fully, release everything, only then start the next.
- `--batch_split` (gradient accumulation) exists in `main.py` as the OOM release valve when a GPU
  can't fit the desired effective batch size directly (RTX 3060 12GB, in particular).

## 9. Checkpointing and resume semantics

- Per-epoch checkpoints (`signvlm_epoch-N.pth`) saved under `CHECKPOINT_DIR` on **Drive** (see §2),
  plus a "best model so far" save keyed on validation loss.
- Auto-resume: on notebook restart, the latest epoch checkpoint under `CHECKPOINT_DIR` is
  auto-detected and training resumes from there — this is what makes the Drive-not-`/content`
  requirement for `MODELS_DIR` load-bearing: without it, a Colab runtime recycle would silently
  lose every checkpoint and force training to restart from scratch.
- The standalone metrics cell (Cell 10, §10) independently rediscovers the latest checkpoint the
  same way, so it can be run in a *fresh* runtime that never ran the training cells at all.

## 10. Designing for "run any cell standalone"

A recurring design constraint in this notebook: any evaluation/metrics cell should be runnable in a
**fresh Colab runtime** without having executed the full training pipeline first — e.g. to check
metrics on `test_data_diff_signer` without re-running training. This is done by having each such
cell:
1. Assert its own minimal prerequisites explicitly (e.g. "requires Cells 1/2/2b/3 to have run") and
   raise a clear `RuntimeError` naming exactly what's missing, rather than failing deep inside
   unrelated code with a confusing stack trace.
2. Rebuild everything it needs from persisted state on Drive (label map JSON, latest checkpoint,
   TSV manifests) rather than assuming in-memory globals from earlier cells are populated.
3. Re-run the *same* frame-extraction logic (§5), scoped only to the splits it actually needs, using
   the same cache namespacing helper, so it never conflicts with a full pipeline run's cache state.

This pattern is worth deliberately designing into a new pipeline from the start (rather than
retrofitting later) if partial/standalone evaluation runs are expected to be a common workflow.

## 11. Summary checklist for a new ingestion pipeline

If briefing a fresh model-training effort from scratch, carry forward these decisions explicitly:

1. **Three-tier cache (Drive → local disk → RAM), all lazy and idempotent**, never a bulk upfront
   copy — lets training start immediately while the cache fills in.
2. **Namespace every cache path by a hash of the absolute source root**, not just the relative
   path — the moment more than one root shares a cache dir, relative-path-only keys silently
   alias/overwrite across roots with no error.
3. **All destructive/multi-step writes go through a temp-name-then-atomic-rename pattern**
   (`.partial` suffix + `os.replace`) — a crash mid-write must never leave something that looks
   complete but isn't, because idempotency checks elsewhere trust "exists and looks non-empty"
   as "fully done."
4. **Checkpoints, TSV manifests, and any other run-defining artifact must live on persistent
   storage (Drive), never only on ephemeral local disk** — add an explicit runtime assertion for
   this, don't rely on convention.
5. **Corrupt/unreadable source files should be excluded from the manifest with a loud warning**,
   never silently converted into a zero-tensor / placeholder sample that pollutes metrics with no
   visible signal that it happened. Only hard-fail the whole split when *none* of its files
   succeed (systemic problem), not when a handful fail (expected data-quality noise).
6. **RAM-preload-into-shared-structure optimizations must be re-verified per OS** — fork (Linux)
   vs spawn (Windows) start-method semantics change whether the optimization is nearly free or
   multiplies cost by worker count.
7. **Any decode/read fast-path must have an explicit correctness precondition check and a fallback
   to the slow-but-known-correct path** — never trust a fast path to "probably be fine"; verify the
   assumption it depends on (frame count, ordering, etc.) every time, and cache the "this file needs
   the fallback" decision so you don't repeatedly pay for a doomed fast-path retry.
8. **Tune DataLoader `num_workers`/`prefetch_factor`/`batch_size` as named profiles per target GPU**,
   not scattered magic numbers — makes moving between Colab tiers (T4/A100) or a local GPU a
   one-line change.
9. **Design standalone-runnable cells/scripts for evaluation from day one** if partial pipeline runs
   (skip training, just re-check one split) are a realistic workflow — explicit prerequisite
   assertions plus rebuild-from-persisted-state, not reliance on in-memory globals from earlier
   steps.

## 12. Where to look in this repo for the actual code

- `notebooks/SignVLM_Colab_Training.ipynb` — Cell 2 (Drive mount + paths), Cell 2b (git clone of
  training code), Cell 4 (TSV manifest build + auto frame extraction + DataLoader construction),
  Cell 7 (training loop), Cell 8 (multi-split eval with explicit release-between-splits), Cell 10
  (standalone metrics/confusion matrices, runnable independent of Cells 4-9).
- `video_dataset/drive_to_local_cache.py` — `resolve_cached_path`, `_split_cache_subdir`, atomic
  copy helpers, cross-process file locking.
- `video_dataset/dataset.py` — `VideoDataset` (`_preload_to_ram`, `_decode_sampled_frames_train`,
  `_random_sample_frame_idx`, zero-tensor fallbacks), `DummyDataset` (all-zero speed-test dataset
  used to isolate model throughput from I/O throughput).
- `video_dataset/dataloader.py` — `create_train_dataset`/`create_val_dataset`/`create_train_loader`/
  `create_val_loader`, per-mode DataLoader kwargs (`_loader_perf_kwargs`, `_loader_eval_memory_kwargs`).
- `docs/EXPERIMENTS_RTX3060.md` — CLI-flag equivalents of the same ingestion options for a
  non-Colab local-GPU run of `main.py` directly (no notebook).
- `docs/signVLMPipeline.md` — architecture-side context (model, not ingestion) if the new model
  also needs to know how CLIP backbone + temporal decoder integration worked here.
