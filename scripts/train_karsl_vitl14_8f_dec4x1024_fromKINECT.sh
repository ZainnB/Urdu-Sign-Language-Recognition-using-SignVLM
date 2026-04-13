
exp_dir=runs/karsl_100_FromKinect_FS_test03 

mkdir -p "${exp_dir}"
python -u -m torch.distributed.run --nproc_per_node 1   \
      main.py \
    --num_steps 10000 \
    --backbone "ViT-L/14-lnpre" \
    --backbone_type clip \
    --backbone_path /media/hamzah_luqman/CLIP/frozenClip/CLIP_weights/ViT-L/ViT-L-14.pt \
    --decoder_num_layers 4 \
    --decoder_qkv_dim 1024 \
    --decoder_num_heads 16 \
    --num_classes 400 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --resume_path /media/hamzah_luqman/CLIP/frozenClip/runs/karsl_100_FromKinect_FS_test03/checkpoint-5000.pth \
    --frames_available 1 \
    --data_root /data/Hamzah/signLanguages/karsl/color_cropped \
    --train_list_path /data/Hamzah/signLanguages/karsl/color_cropped/01-02-train-100.txt \
    --val_list_path /data/Hamzah/signLanguages/karsl/color_cropped/03_100.txt \
    --n_shots -1 \
    --batch_size 8 \
    --batch_split 1 \
    --auto_augment rand-m7-n4-mstd0.5-inc1 \
    --mean 0.48145466 0.4578275 0.40821073 \
    --std 0.26862954 0.26130258 0.27577711 \
    --num_workers 16 \
    --num_frames 16 \
    --sampling_rate 16 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"
