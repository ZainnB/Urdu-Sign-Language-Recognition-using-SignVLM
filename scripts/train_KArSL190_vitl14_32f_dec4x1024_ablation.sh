
#!/usr/bin/env sh
exp_dir=runs/KArSL190_LR0_001_FT

mkdir -p "${exp_dir}"
python -u -m torch.distributed.run --nproc_per_node 1   \
      main_copy.py \
    --num_steps 10001 \
    --lr 0.001 \
    --backbone "ViT-L/14-lnpre" \
    --backbone_type clip \
    --backbone_path /media/hamzah_luqman/CLIP/frozenClip/CLIP_weights/ViT-L/ViT-L-14.pt \
    --decoder_num_layers 4 \
    --decoder_qkv_dim 1024 \
    --decoder_num_heads 16 \
    --num_classes 190 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --frames_available 1 \
    --data_root /data/Hamzah/signLanguages/karsl/color_cropped \
    --train_list_path /data/Hamzah/signLanguages/karsl/color_cropped/01-02-train-190.txt \
    --val_list_path /data/Hamzah/signLanguages/karsl/color_cropped/03_190.txt \
    --n_shots -1 \
    --batch_size 16 \
    --batch_split 1 \
    --num_workers 16 \
    --num_frames 24 \
    --sampling_rate 4 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"


#!/usr/bin/env sh
exp_dir=runs/KArSL190_LR0_0001_FT

mkdir -p "${exp_dir}"
python -u -m torch.distributed.run --nproc_per_node 1   \
      main_copy.py \
    --num_steps 10001 \
    --lr 0.0001 \
    --backbone "ViT-L/14-lnpre" \
    --backbone_type clip \
    --backbone_path /media/hamzah_luqman/CLIP/frozenClip/CLIP_weights/ViT-L/ViT-L-14.pt \
    --decoder_num_layers 4 \
    --decoder_qkv_dim 1024 \
    --decoder_num_heads 16 \
    --num_classes 190 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --frames_available 1 \
    --data_root /data/Hamzah/signLanguages/karsl/color_cropped \
    --train_list_path /data/Hamzah/signLanguages/karsl/color_cropped/01-02-train-190.txt \
    --val_list_path /data/Hamzah/signLanguages/karsl/color_cropped/03_190.txt \
    --n_shots -1 \
    --batch_size 16 \
    --batch_split 1 \
    --num_workers 16 \
    --num_frames 24 \
    --sampling_rate 4 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"
  

#!/usr/bin/env sh
exp_dir=runs/KArSL190_LR0_0004_FT

mkdir -p "${exp_dir}"
python -u -m torch.distributed.run --nproc_per_node 1   \
      main_copy.py \
    --num_steps 10001 \
    --lr 0.0004 \
    --backbone "ViT-L/14-lnpre" \
    --backbone_type clip \
    --backbone_path /media/hamzah_luqman/CLIP/frozenClip/CLIP_weights/ViT-L/ViT-L-14.pt \
    --decoder_num_layers 4 \
    --decoder_qkv_dim 1024 \
    --decoder_num_heads 16 \
    --num_classes 190 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --frames_available 1 \
    --data_root /data/Hamzah/signLanguages/karsl/color_cropped \
    --train_list_path /data/Hamzah/signLanguages/karsl/color_cropped/01-02-train-190.txt \
    --val_list_path /data/Hamzah/signLanguages/karsl/color_cropped/03_190.txt \
    --n_shots -1 \
    --batch_size 16 \
    --batch_split 1 \
    --num_workers 16 \
    --num_frames 24 \
    --sampling_rate 4 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"
    
#!/usr/bin/env sh
exp_dir=runs/KArSL190_LR0_00004_FT

mkdir -p "${exp_dir}"
python -u -m torch.distributed.run --nproc_per_node 1   \
      main_copy.py \
    --num_steps 10001 \
    --lr 0.00004 \
    --backbone "ViT-L/14-lnpre" \
    --backbone_type clip \
    --backbone_path /media/hamzah_luqman/CLIP/frozenClip/CLIP_weights/ViT-L/ViT-L-14.pt \
    --decoder_num_layers 4 \
    --decoder_qkv_dim 1024 \
    --decoder_num_heads 16 \
    --num_classes 190 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --frames_available 1 \
    --data_root /data/Hamzah/signLanguages/karsl/color_cropped \
    --train_list_path /data/Hamzah/signLanguages/karsl/color_cropped/01-02-train-190.txt \
    --val_list_path /data/Hamzah/signLanguages/karsl/color_cropped/03_190.txt \
    --n_shots -1 \
    --batch_size 16 \
    --batch_split 1 \
    --num_workers 16 \
    --num_frames 24 \
    --sampling_rate 4 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"

#!/usr/bin/env sh
exp_dir=runs/KArSL190_LR0_000004_FT

mkdir -p "${exp_dir}"
python -u -m torch.distributed.run --nproc_per_node 1   \
      main_copy.py \
    --num_steps 10001 \
    --lr 0.000004 \
    --backbone "ViT-L/14-lnpre" \
    --backbone_type clip \
    --backbone_path /media/hamzah_luqman/CLIP/frozenClip/CLIP_weights/ViT-L/ViT-L-14.pt \
    --decoder_num_layers 4 \
    --decoder_qkv_dim 1024 \
    --decoder_num_heads 16 \
    --num_classes 190 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --frames_available 1 \
    --data_root /data/Hamzah/signLanguages/karsl/color_cropped \
    --train_list_path /data/Hamzah/signLanguages/karsl/color_cropped/01-02-train-190.txt \
    --val_list_path /data/Hamzah/signLanguages/karsl/color_cropped/03_190.txt \
    --n_shots -1 \
    --batch_size 16 \
    --batch_split 1 \
    --num_workers 16 \
    --num_frames 24 \
    --sampling_rate 4 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"        