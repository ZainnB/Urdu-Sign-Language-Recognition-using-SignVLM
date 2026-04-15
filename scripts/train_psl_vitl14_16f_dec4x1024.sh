#!/usr/bin/env bash
#
# SignVLM training on the PSL (Pakistan Sign Language) 104-class dataset.
#
# Prerequisites:
#   1. Download OpenAI CLIP ViT-L/14 weights and place them at:
#        signVLM/CLIP_weights/ViT-L/ViT-L-14.pt
#      Direct URL:
#        https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd/ViT-L-14.pt
#
# Usage (Git Bash / Linux), from repo anywhere:
#   bash signVLM/scripts/train_psl_vitl14_16f_dec4x1024.sh
#
# Note: This codebase enables RandAugment via --auto_augment with a rand-... string
# (there is no separate --randaug flag). KArSL scripts use rand-m7-n4-mstd0.5-inc1.
#
# GPU selection: if your checkout still hard-codes cuda:1 in main.py, change it to:
#   cuda_device_id = dist.get_rank() % torch.cuda.device_count()
#   torch.cuda.set_device(cuda_device_id)
# (Many forks already include this fix.)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIGNVLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SIGNVLM_ROOT}"

EXP_DIR="../trained_models_final/signVLM_psl"
mkdir -p "${EXP_DIR}"

python -u -m torch.distributed.run --nproc_per_node=1 --standalone \
  main.py \
  --backbone "ViT-L/14-lnpre" \
  --backbone_path ../CLIP_weights/ViT-L/ViT-L-14.pt \
  --backbone_type clip \
  --num_classes 104 \
  --num_frames 16 \
  --sampling_rate 4 \
  --num_steps 10000 \
  --save_freq 2500 \
  --eval_freq 2500 \
  --lr 0.00004 \
  --weight_decay 0.05 \
  --batch_size 8 \
  --batch_split 2 \
  --decoder_num_layers 4 \
  --decoder_qkv_dim 1024 \
  --decoder_num_heads 16 \
  --decoder_mlp_factor 4.0 \
  --cls_dropout 0.5 \
  --decoder_mlp_dropout 0.5 \
  --data_root ../data \
  --train_list_path ../dataset_split_text_files/train_signer_signvlm.txt \
  --val_list_path ../dataset_split_text_files/val_signer_signvlm.txt \
  --num_workers 4 \
  --checkpoint_dir "${EXP_DIR}" \
  --auto_resume \
  --mean 0.48145466 0.4578275 0.40821073 \
  --std 0.26862954 0.26130258 0.27577711 \
  --auto_augment rand-m7-n4-mstd0.5-inc1 \
  --spatial_size 224 \
  2>&1 | tee "${EXP_DIR}/train-$(date +%Y%m%d_%H%M%S).log"
