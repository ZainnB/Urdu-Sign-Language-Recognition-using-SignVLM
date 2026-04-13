#!/usr/bin/env sh
exp_dir=runs/ISA64_vitl14_32f_dec4x1024_8Shot_signer01

mkdir -p "${exp_dir}"
python -u -m torch.distributed.run --nproc_per_node 1   \
      main.py \
    --num_steps 4000 \
    --backbone "ViT-L/14-lnpre" \
    --backbone_type clip \
    --backbone_path /media/hamzah_luqman/CLIP/frozenClip/CLIP_weights/ViT-L/ViT-L-14.pt \
    --decoder_num_layers 4 \
    --decoder_qkv_dim 1024 \
    --decoder_num_heads 16 \
    --num_classes 64 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --frames_available 1 \
    --data_root /data/Hamzah/signLanguages/ISA64/frames \
    --train_list_path /data/Hamzah/signLanguages/ISA64/train_01.txt \
    --val_list_path /data/Hamzah/signLanguages/ISA64/test_signer_01.txt \
    --n_shots 8 \
    --batch_size 16 \
    --batch_split 1 \
    --num_workers 16 \
    --num_frames 32 \
    --sampling_rate 4 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"

#!/usr/bin/env sh
exp_dir=runs/ISA64_vitl14_32f_dec4x1024_8Shot_signer02

mkdir -p "${exp_dir}"
python -u -m torch.distributed.run --nproc_per_node 1   \
      main.py \
    --num_steps 4000 \
    --backbone "ViT-L/14-lnpre" \
    --backbone_type clip \
    --backbone_path /media/hamzah_luqman/CLIP/frozenClip/CLIP_weights/ViT-L/ViT-L-14.pt \
    --decoder_num_layers 4 \
    --decoder_qkv_dim 1024 \
    --decoder_num_heads 16 \
    --num_classes 64 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --frames_available 1 \
    --data_root /data/Hamzah/signLanguages/ISA64/frames \
    --train_list_path /data/Hamzah/signLanguages/ISA64/train_02.txt \
    --val_list_path /data/Hamzah/signLanguages/ISA64/test_signer_02.txt \
    --n_shots 8 \
    --batch_size 16 \
    --batch_split 1 \
    --num_workers 16 \
    --num_frames 32 \
    --sampling_rate 4 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"

#!/usr/bin/env sh
exp_dir=runs/ISA64_vitl14_32f_dec4x1024_8Shot_signer03

mkdir -p "${exp_dir}"
python -u -m torch.distributed.run --nproc_per_node 1   \
      main.py \
    --num_steps 4000 \
    --backbone "ViT-L/14-lnpre" \
    --backbone_type clip \
    --backbone_path /media/hamzah_luqman/CLIP/frozenClip/CLIP_weights/ViT-L/ViT-L-14.pt \
    --decoder_num_layers 4 \
    --decoder_qkv_dim 1024 \
    --decoder_num_heads 16 \
    --num_classes 64 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --frames_available 1 \
    --data_root /data/Hamzah/signLanguages/ISA64/frames \
    --train_list_path /data/Hamzah/signLanguages/ISA64/train_03.txt \
    --val_list_path /data/Hamzah/signLanguages/ISA64/test_signer_03.txt \
    --n_shots 8 \
    --batch_size 16 \
    --batch_split 1 \
    --num_workers 16 \
    --num_frames 32 \
    --sampling_rate 4 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"

#!/usr/bin/env sh
exp_dir=runs/ISA64_vitl14_32f_dec4x1024_8Shot_signer04

mkdir -p "${exp_dir}"
python -u -m torch.distributed.run --nproc_per_node 1   \
      main.py \
    --num_steps 4000 \
    --backbone "ViT-L/14-lnpre" \
    --backbone_type clip \
    --backbone_path /media/hamzah_luqman/CLIP/frozenClip/CLIP_weights/ViT-L/ViT-L-14.pt \
    --decoder_num_layers 4 \
    --decoder_qkv_dim 1024 \
    --decoder_num_heads 16 \
    --num_classes 64 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --frames_available 1 \
    --data_root /data/Hamzah/signLanguages/ISA64/frames \
    --train_list_path /data/Hamzah/signLanguages/ISA64/train_04.txt \
    --val_list_path /data/Hamzah/signLanguages/ISA64/test_signer_04.txt \
    --n_shots 8 \
    --batch_size 16 \
    --batch_split 1 \
    --num_workers 16 \
    --num_frames 32 \
    --sampling_rate 4 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"

#!/usr/bin/env sh
exp_dir=runs/ISA64_vitl14_32f_dec4x1024_8Shot_signer05

mkdir -p "${exp_dir}"
python -u -m torch.distributed.run --nproc_per_node 1   \
      main.py \
    --num_steps 4000 \
    --backbone "ViT-L/14-lnpre" \
    --backbone_type clip \
    --backbone_path /media/hamzah_luqman/CLIP/frozenClip/CLIP_weights/ViT-L/ViT-L-14.pt \
    --decoder_num_layers 4 \
    --decoder_qkv_dim 1024 \
    --decoder_num_heads 16 \
    --num_classes 64 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --frames_available 1 \
    --data_root /data/Hamzah/signLanguages/ISA64/frames \
    --train_list_path /data/Hamzah/signLanguages/ISA64/train_05.txt \
    --val_list_path /data/Hamzah/signLanguages/ISA64/test_signer_05.txt \
    --n_shots 8 \
    --batch_size 16 \
    --batch_split 1 \
    --num_workers 16 \
    --num_frames 32 \
    --sampling_rate 4 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"

#!/usr/bin/env sh
exp_dir=runs/ISA64_vitl14_32f_dec4x1024_8Shot_signer06

mkdir -p "${exp_dir}"
python -u -m torch.distributed.run --nproc_per_node 1   \
      main.py \
    --num_steps 4000 \
    --backbone "ViT-L/14-lnpre" \
    --backbone_type clip \
    --backbone_path /media/hamzah_luqman/CLIP/frozenClip/CLIP_weights/ViT-L/ViT-L-14.pt \
    --decoder_num_layers 4 \
    --decoder_qkv_dim 1024 \
    --decoder_num_heads 16 \
    --num_classes 64 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --frames_available 1 \
    --data_root /data/Hamzah/signLanguages/ISA64/frames \
    --train_list_path /data/Hamzah/signLanguages/ISA64/train_06.txt \
    --val_list_path /data/Hamzah/signLanguages/ISA64/test_signer_06.txt \
    --n_shots 8 \
    --batch_size 16 \
    --batch_split 1 \
    --num_workers 16 \
    --num_frames 32 \
    --sampling_rate 4 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"

#!/usr/bin/env sh
exp_dir=runs/ISA64_vitl14_32f_dec4x1024_8Shot_signer07

mkdir -p "${exp_dir}"
python -u -m torch.distributed.run --nproc_per_node 1   \
      main.py \
    --num_steps 4000 \
    --backbone "ViT-L/14-lnpre" \
    --backbone_type clip \
    --backbone_path /media/hamzah_luqman/CLIP/frozenClip/CLIP_weights/ViT-L/ViT-L-14.pt \
    --decoder_num_layers 4 \
    --decoder_qkv_dim 1024 \
    --decoder_num_heads 16 \
    --num_classes 64 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --frames_available 1 \
    --data_root /data/Hamzah/signLanguages/ISA64/frames \
    --train_list_path /data/Hamzah/signLanguages/ISA64/train_07.txt \
    --val_list_path /data/Hamzah/signLanguages/ISA64/test_signer_07.txt \
    --n_shots 8 \
    --batch_size 16 \
    --batch_split 1 \
    --num_workers 16 \
    --num_frames 32 \
    --sampling_rate 4 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"

#!/usr/bin/env sh
exp_dir=runs/ISA64_vitl14_32f_dec4x1024_8Shot_signer08

mkdir -p "${exp_dir}"
python -u -m torch.distributed.run --nproc_per_node 1   \
      main.py \
    --num_steps 4000 \
    --backbone "ViT-L/14-lnpre" \
    --backbone_type clip \
    --backbone_path /media/hamzah_luqman/CLIP/frozenClip/CLIP_weights/ViT-L/ViT-L-14.pt \
    --decoder_num_layers 4 \
    --decoder_qkv_dim 1024 \
    --decoder_num_heads 16 \
    --num_classes 64 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --frames_available 1 \
    --data_root /data/Hamzah/signLanguages/ISA64/frames \
    --train_list_path /data/Hamzah/signLanguages/ISA64/train_08.txt \
    --val_list_path /data/Hamzah/signLanguages/ISA64/test_signer_08.txt \
    --n_shots 8 \
    --batch_size 16 \
    --batch_split 1 \
    --num_workers 16 \
    --num_frames 32 \
    --sampling_rate 4 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"

#!/usr/bin/env sh
exp_dir=runs/ISA64_vitl14_32f_dec4x1024_8Shot_signer09

mkdir -p "${exp_dir}"
python -u -m torch.distributed.run --nproc_per_node 1   \
      main.py \
    --num_steps 4000 \
    --backbone "ViT-L/14-lnpre" \
    --backbone_type clip \
    --backbone_path /media/hamzah_luqman/CLIP/frozenClip/CLIP_weights/ViT-L/ViT-L-14.pt \
    --decoder_num_layers 4 \
    --decoder_qkv_dim 1024 \
    --decoder_num_heads 16 \
    --num_classes 64 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --frames_available 1 \
    --data_root /data/Hamzah/signLanguages/ISA64/frames \
    --train_list_path /data/Hamzah/signLanguages/ISA64/train_09.txt \
    --val_list_path /data/Hamzah/signLanguages/ISA64/test_signer_09.txt \
    --n_shots 8 \
    --batch_size 16 \
    --batch_split 1 \
    --num_workers 16 \
    --num_frames 32 \
    --sampling_rate 4 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"

#!/usr/bin/env sh
exp_dir=runs/ISA64_vitl14_32f_dec4x1024_8Shot_signer10

mkdir -p "${exp_dir}"
python -u -m torch.distributed.run --nproc_per_node 1   \
      main.py \
    --num_steps 4000 \
    --backbone "ViT-L/14-lnpre" \
    --backbone_type clip \
    --backbone_path /media/hamzah_luqman/CLIP/frozenClip/CLIP_weights/ViT-L/ViT-L-14.pt \
    --decoder_num_layers 4 \
    --decoder_qkv_dim 1024 \
    --decoder_num_heads 16 \
    --num_classes 64 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --frames_available 1 \
    --data_root /data/Hamzah/signLanguages/ISA64/frames \
    --train_list_path /data/Hamzah/signLanguages/ISA64/train_10.txt \
    --val_list_path /data/Hamzah/signLanguages/ISA64/test_signer_10.txt \
    --n_shots 8 \
    --batch_size 16 \
    --batch_split 1 \
    --num_workers 16 \
    --num_frames 32 \
    --sampling_rate 4 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"

