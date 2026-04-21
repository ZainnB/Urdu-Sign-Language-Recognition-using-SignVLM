# signVLM Repo Context Memory

## Overview
- Project: `signVLM` (SignVLM: A Pre-trained Large Vision Model for Sign Language Recognition).
- Core idea: use CLIP ViT as fixed or fine-tunable video frame encoder + temporal enhancement + transformer decoder + classification head.
- Code base has 5 major areas:
  - `main.py`: training loop and eval loop with DDP, fp16, checkpointing.
  - `model.py`: EVLTransformer + EVLDecoder + TemporalCrossAttention + backbone initialization from CLIP weights.
  - `video_dataset/`: video data loading, augmentation, sampling, spatial/temporal cropping.
  - `vision_transformer.py`: ViT 2D implementation with CLIP-style blocks and preset configs.
  - `weight_loaders.py`: CLIP .pt JIT weight conversion into local ViT state dict.

## Data Source & Preparation
1. Data list format (required by `--train_list_path`, `--val_list_path`):
   - `<path_to_video_or_frames> <label>` (int in `[0, num_classes-1]`).
   - path can be relative if `--data_root` or `--train_data_root` is provided.

2. Supported datasets in repo examples:
   - KArSL, WLASL, LSA64, AUTSL (paper datasets).

3. `VideoDataset` options (in `video_dataset/dataset.py`):
   - `frames_available` (1 if frames already extracted, else videos read through PyAV).
   - `num_frames` (temporal length), `sampling_rate`, `tsn_sampling`.
   - `num_spatial_views`, `num_temporal_views` for multi-view eval.
   - transform pipeline: normalization (mean/std), resize + crop, optional `auto_augment`, `mirror`.
   - fallback safe frame sampling if video shorter than target with `frames_downUpSamples`.
   - N-shot sampling when `--n_shots` set.

4. `create_train_loader` uses a deterministic step-based sampler:
   - constructs a permutation of dataset indices for exactly `num_steps * batch_size` samples.
   - supports resume from checkpoint by slicing with `resume_step`.

## Preprocessing & Augmentation
- Raw frames loaded with `av` + `to_rgb().to_ndarray()`.
- Normalization: ` (frames - mean) / std `, defaults `mean=0.45`, `std=0.225` per channel.
- Spatial crop:
  - single-view center crop to `spatial_size`, or 3-view (left/center/right).
- Temporal crop:
  - sample segments of length `num_frames` with stride `sampling_rate`.
  - if too short, pad by repeating last frame.
- Random sampling mode for training: `--random_sample` with `auto_augment`, `mirror` etc.

## CLIP Backbone Integration
- `weight_loaders.load_weights_clip` converts CLIP JIT module to local ViT state dict:
  - `class_embedding` -> `cls_token`, `positional_embedding` -> `pos_embed`, `conv1` -> patch embedding.
  - maps each block weight into local transformer block norms + attn + mlp.
- `vision_transformer.VisionTransformer2D` can load these weights and set `return_all_features=True`.
- preset sizes include `ViT-B/16-lnpre` and `ViT-L/14-lnpre`.

## Full Architecture (model.py)
1. `EVLTransformer` composed of:
   - backbone `VisionTransformer2D` (CLIP weights). Optionally frozen or finetuned.
   - decoder: `EVLDecoder` with `num_layers`, temporal conv, pos embed, cross-attention.
   - classification head: `LayerNorm`, `Dropout`, `Linear` to `num_classes`.

2. Backbone modes:
   - `finetune` -> trainable ViT.
   - `freeze_fp16` -> frozen, converted to fp16.
   - `freeze_fp32` -> frozen fp32.

3. Temporal modules in EVLDecoder:
   - Per-layer depthwise temporal conv on patch features (local temporal modelling).
   - Temporal positional embeddings across frames.
   - Temporal cross-attention using `TemporalCrossAttention` (spatial-temporal mixing across adjacent frames).
   - Transformer decoder layer(s) fuse class token with frame features.

4. Forward flow:
   - Input `x`: shape `[B, C=3, T, H, W]`.
   - Flatten space-time to 2D frames for backbone: `x.permute(0,2,1,3,4).flatten(0,1)` => patch features.
   - Backbone returns last `num_layers` feature maps (with `cls+patch` tokens) as frame-wise representations.
   - Reshape per-frame and pass to decoder.
   - Classification output from decoder's final class token.

## Training Loop (main.py)
- Distributed training: `torch.distributed.init_process_group('nccl')`, DDP on selected GPU.
- Optimizer: `AdamW(params, lr=4e-5, weight_decay=0.05)`.
- LR schedule: `CosineAnnealingLR(T_max=num_steps)`.
- Mixed precision: `torch.cuda.amp.grad_scaler.GradScaler(enabled=fp16)` with `autocast` context.
- Loss: `CrossEntropyLoss`.
- Batch split: `--batch_split` for memory efficiency requiring backprop through micro-batches.
- Logging freq: `print_freq`, evaluation freq `eval_freq`, checkpoint freq `save_freq`.
- Evaluation: `top1` and `top5` accuracy with reduction over distributed workers.

## Validation / Testing
- `--eval_only` mode runs `evaluate(model, val_loader)` with `model.eval()`.
- Validation loader uses `num_spatial_views`, `num_temporal_views` defaults in parser.
- Eval aggregates `acc1` and `acc5` across world size.

## Checkpointing
- `checkpoint.py` handles:
  - loading resume checkpoint (`--resume_path`, `--pretrain`).
  - `save_checkpoint(model, optimizer, lr_sched, scaler, step, args)` is called regularly.

## Scripts
- `scripts/train_*` set training args for specific datasets: e.g. `train_AUSTL_vitl14_32f_dec4x1024.sh`.
- key CLI args:
  - `--backbone_type clip`, `--backbone_path`, `--num_classes`, `--num_steps`, `--train_list_path`, `--val_list_path`.

## Fine-tune Methodology
- Start from pre-trained CLIP ViT weights (`ViT-L/14` suggested for SOTA).
- Option 1: freeze backbone (`freeze_fp16`) and train only temporal + decoder (faster, less overfit).
- Option 2: `--finetune_backbone` for end-to-end training (heavier but more capacity).
- Best practice: two-stage
  1. Warm-up with frozen backbone and high lr schedule.
  2. Unfreeze with low lr for backbone + high lr for new heads.

## Inference & Real-Time Testing (proposed)
1. Video capture pipeline:
   - read frames from camera or video stream
   - sample `num_frames` using same rule as dataset (`sampling_rate`, `tsn_sampling`).
   - normalize and resize to `spatial_size`.
   - convert to `[1,3,T,H,W]` and send to model.
   - model.forward, softmax + argmax for class.

2. Real-time optimization:
   - use `torch.jit.trace` or `torch.jit.script` on `EVLTransformer` after training.
   - convert to `torch.cuda.amp.autocast` or fp16 inference.
   - reduce `num_temporal_views` to 1 and `num_spatial_views` to 1.

3. Evaluation metrics:
   - frame-level accuracy, top1/top5.
   - confusion matrix on test split.
   - cross-subject error analysis per sign category.

## Next steps for full testable pipeline
- Add `tests/` with unit tests for `VideoDataset` sampling/crop, `EVLTransformer` shapes and deterministic output (DummyDataset).
- Add `inference.py` script for on-demand classification (video or camera input) with commandline interface.
- Add `benchmark.py` measuring throughput (fps) and latency.
- Add `./scripts/eval_*.sh` to run validation-only across datasets.

## Already existing helper files of interest
- `avg_checkpoints.py`: weighted checkpoint averaging for final model.
- `weight_average_tool.py`: exploration of checkpoint ensemble or weight avg.

---

### Full pipeline from data to model
1. create train/val lists (path,label) and set `--data_root`.
2. choose frames extraction strategy: video decode or pre-extracted frames.
3. set args: `--num_frames`, `--sampling_rate`, `--spatial_size`, `--num_steps`, `--batch_size`, `--num_classes`.
4. choose backbone and freeze/fine-tune behavior.
5. run `python main.py --train_list_path ... --val_list_path ... --backbone_path ... --finetune_backbone`.
6. evaluate with `--eval_only`.
7. save best checkpoints. Optionally average.
8. build `inference.py` and test with sample videos + live camera.

### Specific repository hooks required for method
- `main.py` executes full training + eval. Adding `infer` mode here is easy by reusing `create_val_loader` and `evaluate`.
- `model.py` already supports modular temporal attention; currently `TransformerDecoder` and linear head.
- `video_dataset` robust for both frame-level and video-level input; maintain same sampling for train/val.

## Summary
This memory file now captures the signVLM architecture and code-level details needed to build a fully inferable/testable real-time pipeline without consulting the paper again. Further iterations should encode dataset specs (e.g., KArSL classes, frame count) in a dedicated `docs/pipeline.md` next.