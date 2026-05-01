# SignVLM notebooks — setup and run guide

This folder contains **Google Colab–oriented** Jupyter notebooks for the Urdu Sign Language / SignVLM workflow:

| Notebook | Purpose |
|----------|---------|
| [`SignVLM_Colab_Training.ipynb`](SignVLM_Colab_Training.ipynb) | Install deps, mount Drive, clone or use repo code, build TSV lists, train **EVLTransformer**, validate, and run multi-split evaluation. |
| [`SignVLM_Data_Augmentation_Colab.ipynb`](SignVLM_Data_Augmentation_Colab.ipynb) | Batch-generate augmented **`.mp4`** files for **validation**, **test**, and **unseen** splits only (not training), in place. |

Both notebooks assume **PyTorch with CUDA** on Colab (GPU runtime). For local use, skip Drive mount and set paths via environment variables as described below.

---

## Prerequisites

1. **Google Colab** with **GPU** (recommended), or a local machine with CUDA and a recent PyTorch build.
2. **Dataset on Google Drive** (Colab): video files organized under a single root `BASE` (default: `MyDrive/FYP_DATA`).
3. **CLIP ViT-L weights** for training: a `ViT-L-14.pt` (or path you configure) reachable from the notebook — see `BACKBONE_PATH` / `SIGNVLM_BACKBONE_PATH` below.
4. Optional repo docs in the main project: `SIGNVLM_TRAINING_NOTES.md`, `docs/signVLMPipeline.md` (if present) for deeper pipeline notes.

---

## Shared folder layout (`BASE`)

Under your dataset root (e.g. `.../MyDrive/FYP_DATA`):

- `train_data/<class_name>/*.mp4` — one subfolder per class; class **names** define the label map (sorted order → class id `0 … num_classes-1`).
- `validation_data/<class_name>/*.mp4`
- `test_data/<class_name>/*.mp4`
- `unseen_data` — optional; either flat `*.mp4` files (stem = class name) or same class-folder layout as train.

The training notebook also expects a **repo checkout** (on Drive or from git) for **CLIP weights** default path:  
`{BASE}/{REPO_SUBDIR}/CLIP_weights/ViT-L/ViT-L/ViT-L-14.pt` unless you override `SIGNVLM_BACKBONE_PATH`.

**Artifacts written under `BASE`:**

- `signVLM_checkpoints/checkpoint-*.pth` — training checkpoints.
- `signVLM_lists/train.tsv`, `val.tsv`, `test.tsv`, `unseen.tsv`, `label_map_auto.json` — list files and id→name map.

List file format: each line is `relative_path<TAB>numeric_label_id` (same convention as the rest of the SignVLM tooling).

---

## Environment variables (reference)

Set these **before** the relevant cells (e.g. `os.environ["NAME"] = "value"` in a small cell, or Colab secrets + string build for tokens).

### Paths and Colab

| Variable | Role |
|----------|------|
| `SIGNVLM_DRIVE_BASE` | Full path to dataset root `BASE`. On Colab, should be under `/content/drive/MyDrive/...`. If unset on Colab, defaults to `{MyDrive}/FYP_DATA`. |
| `SIGNVLM_BASE` | Used **when not on Colab** if `SIGNVLM_DRIVE_BASE` is empty: defaults to `./signvlm_data` under the current working directory. |
| `SIGNVLM_DRIVE_MY` | Non-Colab only: override path that Colab would use as `MyDrive` (rare). |
| `SIGNVLM_TRAIN_SUBDIR` | Train split folder name under `BASE` (default `train_data`). |
| `SIGNVLM_VAL_SUBDIR` | Validation split (default `validation_data`). |
| `SIGNVLM_TEST_SUBDIR` | Test split (default `test_data`). |
| `SIGNVLM_UNSEEN_SUBDIR` | Unseen split folder (default `unseen_data`). |
| `SIGNVLM_REPO_SUBDIR` | Subfolder under `BASE` where a full repo copy may live (default `Urdu-Sign-Language-Recognition-using-SignVLM`). Used for default `DRIVE_REPO_ROOT` and CLIP path. |
| `SIGNVLM_BACKBONE_PATH` | Absolute path to CLIP ViT-L weights `.pt`. If unset, training notebook uses `{DRIVE_REPO_ROOT}/CLIP_weights/ViT-L/ViT-L/ViT-L-14.pt`. |

### Git (training code or augmentation code)

| Variable | Role |
|----------|------|
| `SIGNVLM_GIT_URL` | Clone URL for the SignVLM repo. Training notebook default is the public repo URL in Cell 2b; unset + no `main.py` on Drive → error with instructions. |
| `SIGNVLM_GIT_BRANCH` | Branch to checkout (default in notebook: `training_thru_tensors`). |
| `SIGNVLM_GIT_DEPTH` | Shallow clone depth (default `1`). Use `0` to omit `--depth` for a full history (larger clone). |
| `SIGNVLM_GIT_DIR` | Directory for the clone (default under `/content/...` on Colab). |
| `SIGNVLM_GIT_RECLONE` | If `1` / `true` / `yes`, delete existing `GIT_DIR` and clone again. |

**Private repos:** use `https://<token>@github.com/org/repo.git` or Colab **User secrets** to inject the token into the URL.

### Training-only

| Variable | Role |
|----------|------|
| `SIGNVLM_RESUME_PATH` | Absolute path to a specific `checkpoint-NNNN.pth` to load. When set, overrides automatic “latest checkpoint” discovery. |
| `SIGNVLM_NO_AUTO_RESUME` | If `1` / `true` / `yes` / `on`, disables auto-resume from `CHECKPOINT_DIR` (fresh optimizer schedule from step 0 unless you still pass `SIGNVLM_RESUME_PATH`). |

The training notebook markdown also mentions `SIGNVLM_REUSE_LISTS=1` for reusing TSVs; the **current Cell 4 code** uses an in-cell flag `REUSE_EXISTING_TSV_LISTS` (default `False`) to control the same idea — set that variable in Cell 4 to `True` if you want to skip rebuilding lists when data did not change.

### Augmentation-only

| Variable | Role |
|----------|------|
| `AUG_MAX_WORKERS` | Max parallel processes for per-label jobs (default `4`). Lower (e.g. `2`) if you hit RAM/OOM. |
| `SIGNVLM_REPO_ROOT` | Optional absolute path to repo root; if set, used instead of `{BASE}/{REPO_SUBDIR}` for finding `augmentation_and_preprocessing/augmentation` on Drive when git is not used. |

---

# Notebook 1: `SignVLM_Colab_Training.ipynb`

## What it does

- Installs **PyAV** (`av`), **Pillow**, **scikit-learn** (metrics).
- Mounts Google Drive, defines `BASE`, checkpoint dir, list dir, CLIP backbone path, train/val/test/unseen directories.
- Pulls **Python training code** from **git** (preferred) or from Drive if `main.py` exists under `DRIVE_REPO_ROOT`.
- Initializes **single-GPU distributed** (`gloo` + `FileStore`, world size 1) to match `main.py`.
- Builds **TSV lists** and **DataLoaders** via `video_dataset`.
- Builds **EVLTransformer**, runs **AdamW** + **CosineAnnealingLR** + **CrossEntropyLoss** + **AMP** training loop aligned with `main.py`.
- Optional **Cell 8**: load latest (or chosen) checkpoint and evaluate **train → validation → test → unseen** sequentially.

## How to run (order)

Run cells **top to bottom** the first time. After that, you can re-run from **Cell 4** onward if you only changed data/hyperparameters (and from **Cell 6** if only resuming training).

| Step | Cell | Action |
|------|------|--------|
| 1 | Setup | `pip install` av, pillow, scikit-learn; print torch/CUDA. |
| 2 | Drive + paths | Mount Drive; set `BASE`, `CHECKPOINT_DIR`, `LIST_DIR`, `BACKBONE_PATH`, split dirs. **Colab:** `CHECKPOINT_DIR` must live under `/content/drive/MyDrive/...` or the notebook raises an error. |
| 3 | 2b | Git clone/update **or** use Drive repo with `main.py`. Sets `CODE_ROOT` / `REPO_ROOT` for imports. |
| 4 | 3 | `sys.path` + `chdir` to `CODE_ROOT`; init distributed; import `video_dataset`, `model`, etc. |
| 5 | Metrics helpers | Defines `collect_predictions`, `print_confusion_and_f1` (F1 + confusion matrix + optional `classification_report`). |
| 6 | 4 | Build label map, write TSVs, create `train_loader` / `val_loader` / optional `unseen_loader`; define `args` namespace. |
| 7 | 5 | Construct `EVLTransformer` and move to CUDA. |
| 8 | 6 | Optimizer, scheduler, scaler, resume from checkpoint. |
| 9 | 7 | Training loop (long-running). |
| 10 | 8 | Optional full evaluation on all splits. |

## Cell 4 — data and `args` parameters (detailed)

These are the important **in-cell** and **`args`** fields used by `video_dataset` and training:

| Name | Typical value | Meaning |
|------|----------------|---------|
| `FRAMES_AVAILABLE` | `0` | `0` = load **videos** with PyAV. `1` = pre-extracted **frames** layout (see project notes / `tools/extract_frames_signvlm.py`). |
| `IO_SPEED_TEST_DUMMY` | `False` | If `True`, uses a **dummy dataset** (no disk decode) to compare step times vs real I/O. |
| `REUSE_EXISTING_TSV_LISTS` | `False` | If `True`, skips rewriting TSVs when non-empty files already exist. Set `False` after adding/removing videos or augmentation. |
| `LOCAL_DATA_CACHE` | auto on Colab | On Colab, defaults to `/content/signvlm_data_cache`: each file is **copied once** from Drive then reused (faster). Set to `""` in code to read always from Drive (slower, no local cache). |
| `TRAIN_SPEED_PROFILE` | `"max_speed"` or `"fast_stable"` | Selects worker/prefetch/batch **profile** (see below). |
| `PROFILE_BATCH_SIZE` | `20` (max_speed) / `12` (fast_stable) | Intended batch size for the profile. **Note:** the notebook’s `argparse.Namespace` currently sets `batch_size=12` explicitly; for `max_speed` you may want `batch_size` to match `_batch_size` / `PROFILE_BATCH_SIZE` in that cell to avoid mismatch. |
| `PROFILE_NUM_WORKERS` | derived | `DataLoader` worker count (higher for `max_speed`). |
| `PROFILE_PREFETCH_FACTOR` | `4` or `2` | Prefetch batches per worker. |
| `args.frames_available` | same as `FRAMES_AVAILABLE` | Passed to dataset builders. |
| `args.train_list_path` / `val_list_path` | paths under `LIST_DIR` | TSV paths for train/val. |
| `args.train_data_root` / `val_data_root` | `TRAIN_DIR` / `VAL_DIR` | Roots for resolving relative paths in TSVs. |
| `args.local_data_cache` | path or `""` | Staging cache directory. |
| `args.batch_size` | `12` in code | Global batch size for training loader (see profile note above). |
| `args.n_shots` | `-1` | Few-shot configuration flag used by dataset pipeline (`-1` = not few-shot limited). |
| `args.num_spatial_views` | `1` | Spatial crops/views for training. |
| `args.num_temporal_views` | `3` | Temporal clips/views aggregated in the model. |
| `args.num_frames` | `24` | Frames sampled per clip. |
| `args.sampling_rate` | `4` | Temporal stride between sampled frames. |
| `args.tsn_sampling` | `False` | Dense segment sampling style (off by default in notebook). |
| `args.spatial_size` | `224` | Input height/width after resize/crop. |
| `args.mean` / `args.std` | CLIP defaults | Normalization (ImageNet/CLIP statistics in notebook). |
| `args.num_workers` | from profile | Dataloader workers. |
| `args.pin_memory` | `True` | Faster host→GPU transfer when using CUDA. |
| `args.persistent_workers` | `True` if workers > 0 | Keeps worker processes alive between epochs. |
| `args.prefetch_factor` | from profile | See PyTorch `DataLoader`. |
| `args.dummy_dataset` | `IO_SPEED_TEST_DUMMY` | Synthetic data for throughput tests. |
| `args.auto_augment` | RandAugment string | Training-time image augmentation policy (`rand-m7-n4-mstd0.5-inc1`). |
| `args.interpolation` | `"bicubic"` | Resize interpolation. |
| `args.mirror` | `True` | Random horizontal flip for training. |
| `args.num_steps` | `10000` | Total training steps (iterations), not epochs. |
| `args.checkpoint_dir` | `CHECKPOINT_DIR` | Where `checkpoint-*.pth` are saved. |
| `args.auto_resume` | `True` unless `SIGNVLM_NO_AUTO_RESUME` | Resume from latest checkpoint in `checkpoint_dir`. |
| `args.resume_path` | managed in Cell 6 | Explicit path if `SIGNVLM_RESUME_PATH` set; else cleared for discovery. |
| `args.pretrain` | `None` | Optional pretrain path hook (unused in default notebook). |

**Unseen loader (Cell 4):** If `UNSEEN_DIR` exists and `unseen.tsv` is non-empty, an `unseen_loader` is built with **no** RandAugment and **no** mirror (`auto_augment=None`, `mirror=False`), still using 1×3 views for eval-style loading.

## Cell 5 — model parameters

| Parameter | Default in notebook | Meaning |
|-----------|---------------------|---------|
| `finetune_backbone` | `False` | If `False`, backbone uses **freeze_fp16** (when `fp16=True`). If `True`, uses **finetune** mode (full or partial unfreeze per `model.py`). |
| `fp16` | `True` | Mixed precision; pairs with `GradScaler` in Cell 6. |
| `backbone_name` | `"ViT-L/14-lnpre"` | CLIP ViT-L architecture preset. |
| `backbone_type` | `"clip"` | Weight loader family. |
| `backbone_path` | `BACKBONE_PATH` | Filesystem path to `.pt` weights. |
| `backbone_mode` | `freeze_fp16` / `finetune` | Derived from `finetune_backbone` and `fp16`. |
| `decoder_num_layers` | `4` | Transformer decoder depth. |
| `decoder_qkv_dim` | `1024` | Decoder hidden size. |
| `decoder_num_heads` | `16` | Attention heads. |
| `decoder_mlp_factor` | `4.0` | FFN expansion. |
| `num_classes` | `NUM_CLASSES` | Must match label map from train folders. |
| `enable_temporal_conv` | `True` | Temporal convolution in decoder path. |
| `enable_temporal_pos_embed` | `True` | Temporal positional encoding. |
| `enable_temporal_cross_attention` | `True` | Cross-attention over time. |
| `cls_dropout` | `0.5` | Dropout on class head. |
| `decoder_mlp_dropout` | `0.5` | Dropout inside decoder MLP blocks. |
| `num_frames` | `args.num_frames` | Must stay consistent with dataloader. |

## Cell 6 — optimization and checkpointing

| Name | Value | Meaning |
|------|-------|---------|
| `lr` | `4e-5` | AdamW learning rate. |
| `weight_decay` | `0.05` | AdamW weight decay. |
| `batch_split` | `1` (max_speed) or `2` (fast_stable) | Splits each batch into micro-batches for backward passes to save memory. Must divide `args.batch_size`. |
| `num_steps` | `args.num_steps` | Training horizon for cosine schedule. |
| `save_freq` | `2500` | Save checkpoint every N steps. |
| `eval_freq` | `2500` | Run validation `evaluate()` every N steps. |
| `confusion_eval_freq` | same as `eval_freq` | When non-zero and aligned with step, prints confusion + F1 on **validation** during training. Set to `0` in Cell 6 to skip. |
| `print_freq` | `10` | Log training metrics every N steps. |

**Resume behavior:** `checkpoint.resume_from_checkpoint` loads weights/optimizer/scaler state and returns `resume_step`. Cell 7 rebuilds `train_loader` with `resume_step` so the iterator skips completed steps.

## Cell 7 — training loop

- For each step: splits batch by `batch_split`, forward+backward per slice, `GradScaler.step`, `CosineAnnealingLR.step` **per iteration**.
- Periodic: `main.evaluate` on `val_loader`, optional confusion/F1 on validation, checkpoint save.

## Cell 8 — evaluation toggles

| Variable | Default | Meaning |
|----------|---------|---------|
| `CELL8_RUN_TRAIN` | `True` | Run train-split eval. |
| `CELL8_RUN_VAL` | `True` | Run validation eval. |
| `CELL8_RUN_TEST` | `True` | Run test eval. |
| `CELL8_RUN_UNSEEN` | `True` | Run unseen if folder + `unseen.tsv` exist. |
| `CELL8_EVAL_NUM_WORKERS` | `None` | `None` → use `args.num_workers`; else override for eval loaders. |
| `CELL8_TRAIN_SAMPLES_PER_LABEL` | `15` | **Train split only:** cap samples per class for faster train-set metrics. |
| `CELL8_FAST_CONFUSION_PRINT` | `True` | Faster logs: skip full `classification_report` and large confusion print; still saves `.npy` where applicable. |

Checkpoint loaded: latest `checkpoint-*.pth` in `CHECKPOINT_DIR` by numeric step, or override by editing `EVAL_CKPT` in the cell.

**Inference:** batch size **1** per sample with multi-view tensors; softmax averaged over views (same as `main.evaluate`).

---

# Notebook 2: `SignVLM_Data_Augmentation_Colab.ipynb`

## What it does

- Installs **opencv-python-headless** and **numpy**.
- Mounts Drive and sets `BASE` and repo paths for `augmentation_and_preprocessing/augmentation/augmentation_combined.py`.
- Clones repo from git (recommended) or falls back to Drive copy.
- For each **label folder** under `validation_data`, `test_data`, and `unseen_data`:
  - **Skips** the entire class if any `*_aug_*.mp4` already exists.
  - Otherwise generates up to **`AUGMENTED_PER_LABEL`** (default **9**) new videos in place: names `{label}_aug_16` … `{label}_aug_24` (existing names are skipped).

**Training data (`train_data`) is not augmented** by this notebook by design.

## How to run (order)

1. **Cell 1** — pip installs.  
2. **Cell 2** — mount Drive, set `BASE`, `REPO_ROOT`, `AUG_DIR`.  
3. **Cell 2b (markdown) + Cell 4** — git clone or Drive fallback; sets `REPO_ROOT` / `AUG_DIR`.  
4. **Cell 5** — import `augmentation_combined`, define helpers / pick module functions.  
5. **Cell 6** — run parallel jobs (`ProcessPoolExecutor`).

## Parameters

| Name | Default | Meaning |
|------|---------|---------|
| `SPLITS` | `("validation_data", "test_data", "unseen_data")` | Which top-level folders under `BASE` are processed. |
| `AUGMENTED_PER_LABEL` | `9` | New augmented files per label folder (when not skipped). |
| `MAX_WORKERS` | from `AUG_MAX_WORKERS` env, default `4` | Parallel processes; reduce if Colab OOMs. |

Uses `augmentation_combined.augment_video_random` (OpenCV `mp4v` codec per project script).

---

## Troubleshooting (both notebooks)

- **“CLIP backbone not found”** — Place weights at the expected path or set `SIGNVLM_BACKBONE_PATH`.
- **“CHECKPOINT_DIR must be under … MyDrive”** — On Colab, set `SIGNVLM_DRIVE_BASE` to a Drive path, not `/content` only.
- **“No training code / no augmentation_combined.py”** — Set `SIGNVLM_GIT_URL` or copy the full repo to `DRIVE_REPO_ROOT` / `SIGNVLM_REPO_ROOT`.
- **OOM during training** — Use `TRAIN_SPEED_PROFILE="fast_stable"`, increase `batch_split`, or lower `batch_size` in Cell 4.
- **OOM during augmentation** — Lower `AUG_MAX_WORKERS`.
- **Stale TSVs after new videos** — Set `REUSE_EXISTING_TSV_LISTS = False` in training Cell 4 and re-run from Cell 4.

---

## Related commands (outside notebooks)

For non-notebook training, the project may expose `main.py` and shell scripts (e.g. under `scripts/`). See repository root documentation for CLI flags that mirror many of the `args` fields above.
