# Experiments on RTX 3060 (12 GB): splits, shots, eval

This guide matches your repo layout:

- Data: `data/Train_full/<Label>/*.mp4`, `data/Val_aarij/...`, `data/Test/...`
- Lists: `dataset_split_text_files/*.txt` + `label_map.json` + `train_1shot.tsv`
- Training: `main.py` with `--data_root data` and (if you extracted frames) `--frames_available 1`

## GPU / driver checks

```powershell
nvidia-smi
```

Watch utilization while training:

```powershell
nvidia-smi -l 1
```

Use one GPU explicitly (optional):

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
```

If you hit OOM, lower `--batch_size` and/or raise `--batch_split` (gradient accumulation).

---

## Step 0 — Generate TSV splits (once; keep for all experiments)

From the **repository root**:

```powershell
python prepare_psl_splits.py `
  --dataset-root data `
  --train-dirname Train_full `
  --val-dirname Val_aarij `
  --test-dirname Test `
  --out-dir dataset_split_text_files `
  --write-tsv-aliases `
  --write-label-map-json `
  --write-train-1shot `
  --one-shot-stem 1
```

This writes:

| File | Purpose |
|------|---------|
| `train_signer_signvlm.txt` / `train.tsv` | Full train split |
| `val_signer_signvlm.txt` / `val.tsv` | Val split |
| `test_signer_signvlm.txt` / `test.tsv` | Test split |
| `label_map.json` | `{"0": "Afraid", ...}` for reproducible id ↔ name |
| `train_1shot.tsv` | **Exactly one video per class**; prefers `1.mp4` per label folder |

**Do not regenerate** these files mid-study if you want comparable numbers across experiments.

---

## Shared training flags (adjust paths to your machine)

Assume:

- CLIP weights: `CLIP_weights/ViT-L/ViT-L-14.pt`
- Extracted frames + lists with paths relative to `data`:

```text
--data_root data
--frames_available 1
--train_list_path dataset_split_text_files/train_signer_signvlm.txt
--val_list_path dataset_split_text_files/val_signer_signvlm.txt
```

**1-shot training (deterministic):** use `train_1shot.tsv` and **`--n_shots -1`** (do **not** use `--n_shots 1` here).

```text
--train_list_path dataset_split_text_files/train_1shot.tsv
--n_shots -1
--num_steps 2500
```

`--num_steps` is still required by the training script’s LR scheduler even for short runs; 2000–3000 is reasonable for 1-shot.

---

## Experiment A — “Zero-shot” floor (`--eval_only`)

This codebase does **not** implement CLIP zero-shot text matching. What you *can* do for a **random-decoder floor** is run evaluation **without** loading a finetuned checkpoint: the decoder + classifier are randomly initialized; only the CLIP backbone is loaded from `--backbone_path`. That gives a very low top-1 (near chance), similar in spirit to “no adaptation”.

Example (match your PSL hyperparameters from `scripts/train_psl_vitl14_16f_dec4x1024.sh`):

```powershell
python main.py `
  --eval_only `
  --num_steps 10000 `
  --batch_size 8 `
  --backbone "ViT-L/14-lnpre" `
  --backbone_path CLIP_weights/ViT-L/ViT-L-14.pt `
  --backbone_type clip `
  --num_classes 104 `
  --num_frames 16 `
  --sampling_rate 4 `
  --decoder_num_layers 4 `
  --decoder_qkv_dim 1024 `
  --decoder_num_heads 16 `
  --data_root data `
  --frames_available 1 `
  --val_list_path dataset_split_text_files/val_signer_signvlm.txt `
  --num_workers 4 `
  --num_spatial_views 1 `
  --num_temporal_views 3 `
  --mean 0.48145466 0.4578275 0.40821073 `
  --std 0.26862954 0.26130258 0.27577711
```

Do **not** pass `--resume_path` / `--auto_resume` / `--pretrain` if you want an untrained head (decoder stays random). If you accidentally have checkpoints in `--checkpoint_dir`, omit that flag or use a clean directory.

---

## Experiment B — 1-shot finetuning (deterministic list)

Use `train_1shot.tsv`, `--n_shots -1`, and fewer steps, e.g. `--num_steps 2500`:

```powershell
python main.py `
  --num_steps 2500 `
  --save_freq 500 `
  --eval_freq 500 `
  --batch_size 8 `
  --batch_split 2 `
  --lr 0.00004 `
  --weight_decay 0.05 `
  --backbone "ViT-L/14-lnpre" `
  --backbone_path CLIP_weights/ViT-L/ViT-L-14.pt `
  --backbone_type clip `
  --num_classes 104 `
  --num_frames 16 `
  --sampling_rate 4 `
  --decoder_num_layers 4 `
  --decoder_qkv_dim 1024 `
  --decoder_num_heads 16 `
  --data_root data `
  --frames_available 1 `
  --train_list_path dataset_split_text_files/train_1shot.tsv `
  --val_list_path dataset_split_text_files/val_signer_signvlm.txt `
  --n_shots -1 `
  --checkpoint_dir trained_models_final/exp_1shot `
  --auto_augment rand-m7-n4-mstd0.5-inc1 `
  --spatial_size 224 `
  --num_workers 4 `
  --mean 0.48145466 0.4578275 0.40821073 `
  --std 0.26862954 0.26130258 0.27577711
```

---

## Shot progression (2 / 4 / 8 …)

Keep the **same** `train_signer_signvlm.txt` and set:

```text
--n_shots 2
```

(or `4`, `8`, …). This **randomly** subsamples that many videos per class each run; for strict reproducibility, pre-build fixed list files per shot (same idea as `train_1shot.tsv`).

---

## RTX 3060 tips

- Start with `--batch_size 8` and `--batch_split 2` (effective batch 8, 2 forward passes).
- If OOM: try `--batch_size 4` and `--batch_split 2` or `4`.
- Prefer `--frames_available 1` once extraction is done (faster epoch-equivalent throughput than decoding mp4 every step).
