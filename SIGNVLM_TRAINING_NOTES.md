# SignVLM training notes (this repo)

This document summarizes the SignVLM training/inference pipeline **as implemented in this repository**, and the parameter “variants” used in the provided `.sh` scripts. It also captures recommended ways to run **1-sample-per-label (1-shot)** training on a dataset arranged as:

```
data/Final Dataset (with roi augmentation)/
  Afraid/
    1.mp4
    2.mp4
    ...
  AnotherLabel/
    1.mp4
    ...
```

## Training entrypoint

- **Trainer**: `main.py`
  - Builds `EVLTransformer` (CLIP-ViT backbone + temporal decoder + classifier).
  - Creates train/val loaders via `video_dataset` (`video_dataset/dataloader.py`).
  - Optimizer: **AdamW**
  - Scheduler: **CosineAnnealingLR** with `T_max = --num_steps`
  - Loss: **CrossEntropyLoss**
  - Mixed precision: **AMP fp16** by default (disable via `--disable_fp16`)
  - Supports:
    - `--eval_only` (eval loop over val loader)
    - `--finetune_backbone` (updates CLIP-ViT weights)
    - `--batch_split` (manual gradient accumulation to avoid OOM)

**Colab notebook (`notebooks/SignVLM_Colab_Training.ipynb`) — high-RAM / tuning:** Training uses **`auto_resume=True`** by default: **Cell 6** rescans `CHECKPOINT_DIR` each run and loads the latest `checkpoint-*.pth` (continues from `next_step`). Use **`SIGNVLM_RESUME_PATH`** for a specific file; **`SIGNVLM_NO_AUTO_RESUME=1`** to start from step 0. On **Colab**, **high-RAM DataLoader** is the default (you do not need to set `SIGNVLM_HIGH_RAM=1`); set **`SIGNVLM_HIGH_RAM=0`** or **`SIGNVLM_LOW_RAM=1`** to use a small, conservative loader. Also: `SIGNVLM_NUM_WORKERS`, `SIGNVLM_PREFETCH`, `SIGNVLM_BATCH_SIZE`, `SIGNVLM_LOCAL_HUGE=1` (stronger than default high-RAM on capable hosts; local non-Colab + `SIGNVLM_HIGH_RAM` / `SIGNVLM_LOCAL_HUGE` can default `batch_split=1` in Cell 6), and `SIGNVLM_BATCH_SPLIT` to force micro-batching. On very large GPU VRAM, prefer `batch_split=1` and increase batch size; on OOM, use `batch_split=2` or lower `SIGNVLM_BATCH_SIZE`.

## Model architecture (what is trained)

- `model.py`
  - **Backbone**: `VisionTransformer2D` created from presets in `vision_transformer.py`
    - `ViT-B/16-lnpre` (feature_dim=768)
    - `ViT-L/14-lnpre` (feature_dim=1024)
  - **Weights loader**: `weight_loaders.py`
    - `--backbone_type clip` loads **OpenAI CLIP JIT** `.pt` via `torch.jit.load(...)` and converts weights.
  - **Decoder**: `EVLDecoder`
    - `decoder_num_layers`
    - temporal modules:
      - temporal depthwise conv (toggle via `--no_temporal_conv`)
      - temporal positional embeddings (toggle via `--no_temporal_pos_embed`)
      - temporal cross-attention (toggle via `--no_temporal_cross_attention`)
  - **Head**: LayerNorm + Dropout + Linear to `--num_classes`

## Do you really need dataset split list files?

### Short answer

**For this training code as written: yes.**  
The dataloader expects `--train_list_path` and `--val_list_path`, and `VideoDataset` reads those files as its sample index.

### Why the repo uses list files

List files let you:
- define train/val splits
- do cross-subject splits (KArSL/ISA64 scripts)
- do reproducible experiments
- run few-shot via `--n_shots` on top of a larger list

### Your dataset is already “foldered by label”

That structure is great, but **the current code does not auto-scan directories**. You have two options:

1. **Keep the pipeline unchanged** (recommended):
   - Auto-generate `train.tsv` and `val.tsv` list files once.
   - Point `--train_list_path` / `--val_list_path` to them.

2. **Modify the pipeline**:
   - Add a “scan dataset directory” mode that builds the list in-memory.
   - This is doable, but it’s a code change and you still need a label→id mapping.

## List file format (important)

In `video_dataset/dataset.py`, each line is parsed as:

```
<relative_or_absolute_path>\t<label_int>
```

Then it does:

- `path = os.path.join(data_root, path)`
- If `--frames_available 1`: it expects frames to exist at:
  - `frame_dir = Path(path).parent / Path(path).stem`
  - loads `*.png` else `*.jpg`
- Else (`--frames_available 0`): it decodes the `.mp4` using PyAV (`av.open`).

## Few-shot training in this repo (`--n_shots`)

`video_dataset/dataset.py` implements:
- `--n_shots -1`: use full list file
- `--n_shots N`: **randomly sample N examples per class** from the *train* list file

Notes:
- Sampling uses `random.sample` without a fixed seed inside the dataset, so **the exact chosen 1-shot items can vary between runs**.
- If you want strict reproducibility, prefer **writing a fixed 1-shot list file** yourself and using `--n_shots -1`.

## What your PSL training script uses

File: `scripts/train_psl_vitl14_16f_dec4x1024.sh`

Key settings:
- **Backbone**: `ViT-L/14-lnpre` (CLIP)  
  `--backbone_path ../CLIP_weights/ViT-L/ViT-L-14.pt`
- **Classes**: `--num_classes 104`
- **Clip sampling**: `--num_frames 16`, `--sampling_rate 4`  (span ≈ 61 frames)
- **Decoder**: 4 layers, QKV dim 1024, 16 heads
- **Train budget**: `--num_steps 10000`
- **Optimizer**: `--lr 4e-5`, `--weight_decay 0.05`
- **Batching**: `--batch_size 8`, `--batch_split 2`
- **Aug**: `--auto_augment rand-m7-n4-mstd0.5-inc1`
- **Normalization**: explicit CLIP mean/std
- **Data**: train/val list files in `../dataset_split_text_files/...`
- **Val views defaults** (not explicitly passed):
  - `num_spatial_views=1`
  - `num_temporal_views=3`

## What the authors varied in other scripts (how they got “different forms”)

Common “axes”:
- **Shots**: `--n_shots` set to 1/2/4/8 (KArSL scripts do this heavily)
- **Temporal window**:
  - `--num_frames` (e.g. 24 or 32)
  - `--sampling_rate` (e.g. 4 or 8)
- **Evaluation-only** runs: `--eval_only` (for ablations/splits)
- **Crops at evaluation**:
  - `--num_spatial_views` (often 3)
  - `--num_temporal_views` (often 1 in many `.sh` files; defaults are 3 temporal views if omitted)
- **Augmentation**: `--auto_augment rand-m7-n4-mstd0.5-inc1` (very common)
- **Compute**: `--num_steps`, `--batch_size`, `--batch_split`, `--num_workers`

## Recommended `.sh` variants you can create

### A) 1-shot, frozen backbone (good baseline)

- Use a **fixed** 1-shot list file (recommended) OR `--n_shots 1`.

Key flags to include:
- `--n_shots 1` (few-shot mode)
- keep `--finetune_backbone` OFF

### B) 1-shot, finetune backbone (aggressive; likely to overfit)

Add:
- `--finetune_backbone`
- reduce LR (often 10× smaller than frozen-backbone runs)

### C) 2/4/8-shot

Change:
- `--n_shots 2` / `4` / `8`

### D) Wider temporal window

Change:
- `--num_frames` and/or `--sampling_rate`

Example spans:
- 16 @ 4 → 61 frames
- 24 @ 8 → 185 frames
- 32 @ 4 → 125 frames

### E) Deterministic “1 sample per label”

Best practice for your dataset layout:
- Create `train_1shot.tsv` containing exactly **one** mp4 per label folder.
- Create `val_rest.tsv` containing the remaining mp4s.
- Train with `--n_shots -1`.

This avoids run-to-run randomness from `--n_shots`.

## Word-only inference (single-word classification)

- `inference.py` provides a word-level inference pipeline:
  - decodes a single video
  - samples frames (defaults match PSL: 16 frames, stride 4, 224 crop)
  - runs the model and returns top-k predictions
- `eval_unseen.py` benchmarks accuracy on an “Unseen Data” folder and varies `sampling_rate` while keeping `num_frames` fixed.

## Practical checklist (to run training)

- CLIP weights downloaded and `--backbone_path` correct.
- Label mapping exists:
  - integer ids `0..num_classes-1`
- `train.tsv` and `val.tsv` exist in the expected `path<TAB>label_int` format.
- Decide data loading mode:
  - `--frames_available 0` (decode mp4 via PyAV)
  - `--frames_available 1` (pre-extracted frames in `{video_stem}/frame_*.png|jpg`)

## How to extract frames for your dataset folders (recommended on Windows)

This repo includes `tools/extract_frames_signvlm.py`, which creates the exact
folder structure that `video_dataset/dataset.py` expects when `--frames_available 1`.

Example (your structure):

```
data/
  Train_full/<Label>/1.mp4
  Val_aarij/<Label>/1.mp4
  Test/<Label>/1.mp4
```

Run from repo root:

```bash
python tools/extract_frames_signvlm.py --dataset-root "data" --splits Train_full Val_aarij Test
```

Then train with:

- `--frames_available 1`
- `--data_root "data"`

