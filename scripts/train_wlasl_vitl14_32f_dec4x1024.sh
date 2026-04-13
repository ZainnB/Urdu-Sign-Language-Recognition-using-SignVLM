#!/usr/bin/env sh
exp_dir=runs/wlasl_100_vitB16_32f_dec4x1024_FT 

mkdir -p "${exp_dir}"
python -u -m torch.distributed.run --nproc_per_node 1 --master_port=25678  \
      main_copy.py \
    --num_steps 10000 \
    --backbone "ViT-L/14-lnpre" \
    --backbone_type clip \
    --backbone_path /media/hamzah_luqman/CLIP/frozenClip/CLIP_weights/ViT-L/ViT-L-14.pt \
    --decoder_num_layers 4 \
    --decoder_qkv_dim 1024 \
    --decoder_num_heads 16 \
    --num_classes 100 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --frames_available 1 \
    --data_root /data/Hamzah/signLanguages/wlasl/WLASL2000_frames \
    --train_list_path /data/Hamzah/signLanguages/wlasl/WLASL_100_trainVal.txt \
    --val_list_path /data/Hamzah/signLanguages/wlasl/WLASL_100_test.txt \
    --n_shots -1 \
    --batch_size 16 \
    --batch_split 1 \
    --num_workers 16 \
    --num_frames 24 \
    --sampling_rate 4 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"
 