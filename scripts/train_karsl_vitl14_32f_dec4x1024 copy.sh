#!/usr/bin/env sh
exp_dir=runs/karsl_502_vitl14_32f_dec4x1024_0shot_test01 

mkdir -p "${exp_dir}"
python -u -m torch.distributed.run --nproc_per_node 1   \
      main.py \
    --num_steps 10000 \
    --eval_only \
    --backbone "ViT-L/14-lnpre" \
    --backbone_type clip \
    --backbone_path /media/hamzah_luqman/CLIP/frozenClip/CLIP_weights/ViT-L/ViT-L-14.pt \
    --decoder_num_layers 4 \
    --decoder_qkv_dim 1024 \
    --decoder_num_heads 16 \
    --num_classes 502 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --frames_available 1 \
    --data_root /data/Hamzah/signLanguages/karsl/color_cropped \
    --train_list_path /data/Hamzah/signLanguages/karsl/color_cropped/02-03-train.txt \
    --val_list_path /data/Hamzah/signLanguages/karsl/color_cropped/01.txt \
    --n_shots -1 \
    --batch_size 16 \
    --batch_split 1 \
    --auto_augment rand-m7-n4-mstd0.5-inc1 \
    --mean 0.48145466 0.4578275 0.40821073 \
    --std 0.26862954 0.26130258 0.27577711 \
    --num_workers 16 \
    --num_frames 24 \
    --sampling_rate 8 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"


#!/usr/bin/env sh
exp_dir=runs/karsl_502_vitl14_32f_dec4x1024_0shot_test02 

mkdir -p "${exp_dir}"
python -u -m torch.distributed.run --nproc_per_node 1   \
      main.py \
    --num_steps 10000 \
    --eval_only \
    --backbone "ViT-L/14-lnpre" \
    --backbone_type clip \
    --backbone_path /media/hamzah_luqman/CLIP/frozenClip/CLIP_weights/ViT-L/ViT-L-14.pt \
    --decoder_num_layers 4 \
    --decoder_qkv_dim 1024 \
    --decoder_num_heads 16 \
    --num_classes 502 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --frames_available 1 \
    --data_root /data/Hamzah/signLanguages/karsl/color_cropped \
    --train_list_path /data/Hamzah/signLanguages/karsl/color_cropped/01-03-train.txt \
    --val_list_path /data/Hamzah/signLanguages/karsl/color_cropped/02.txt \
    --n_shots -1 \
    --batch_size 16 \
    --batch_split 1 \
    --auto_augment rand-m7-n4-mstd0.5-inc1 \
    --mean 0.48145466 0.4578275 0.40821073 \
    --std 0.26862954 0.26130258 0.27577711 \
    --num_workers 16 \
    --num_frames 24 \
    --sampling_rate 8 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"


#!/usr/bin/env sh
exp_dir=runs/karsl_502_vitl14_32f_dec4x1024_0Shot_test03 

mkdir -p "${exp_dir}"
python -u -m torch.distributed.run --nproc_per_node 1   \
      main.py \
    --num_steps 10000 \
    --eval_only \
    --backbone "ViT-L/14-lnpre" \
    --backbone_type clip \
    --backbone_path /media/hamzah_luqman/CLIP/frozenClip/CLIP_weights/ViT-L/ViT-L-14.pt \
    --decoder_num_layers 4 \
    --decoder_qkv_dim 1024 \
    --decoder_num_heads 16 \
    --num_classes 502 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --frames_available 1 \
    --data_root /data/Hamzah/signLanguages/karsl/color_cropped \
    --train_list_path /data/Hamzah/signLanguages/karsl/color_cropped/01-02-train.txt \
    --val_list_path /data/Hamzah/signLanguages/karsl/color_cropped/03.txt \
    --n_shots -1 \
    --batch_size 16 \
    --batch_split 1 \
    --auto_augment rand-m7-n4-mstd0.5-inc1 \
    --mean 0.48145466 0.4578275 0.40821073 \
    --std 0.26862954 0.26130258 0.27577711 \
    --num_workers 16 \
    --num_frames 24 \
    --sampling_rate 8 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"

#!/usr/bin/env sh
exp_dir=runs/karsl_502_vitl14_32f_dec4x1024_1shot_test01 

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
    --num_classes 502 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --frames_available 1 \
    --data_root /data/Hamzah/signLanguages/karsl/color_cropped \
    --train_list_path /data/Hamzah/signLanguages/karsl/color_cropped/02-03-train.txt \
    --val_list_path /data/Hamzah/signLanguages/karsl/color_cropped/01.txt \
    --n_shots 1 \
    --batch_size 16 \
    --batch_split 1 \
    --auto_augment rand-m7-n4-mstd0.5-inc1 \
    --mean 0.48145466 0.4578275 0.40821073 \
    --std 0.26862954 0.26130258 0.27577711 \
    --num_workers 16 \
    --num_frames 24 \
    --sampling_rate 8 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"


#!/usr/bin/env sh
exp_dir=runs/karsl_502_vitl14_32f_dec4x1024_1shot_test02 

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
    --num_classes 502 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --frames_available 1 \
    --data_root /data/Hamzah/signLanguages/karsl/color_cropped \
    --train_list_path /data/Hamzah/signLanguages/karsl/color_cropped/01-03-train.txt \
    --val_list_path /data/Hamzah/signLanguages/karsl/color_cropped/02.txt \
    --n_shots 1 \
    --batch_size 16 \
    --batch_split 1 \
    --auto_augment rand-m7-n4-mstd0.5-inc1 \
    --mean 0.48145466 0.4578275 0.40821073 \
    --std 0.26862954 0.26130258 0.27577711 \
    --num_workers 16 \
    --num_frames 24 \
    --sampling_rate 8 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"


#!/usr/bin/env sh
exp_dir=runs/karsl_502_vitl14_32f_dec4x1024_1Shot_test03 

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
    --num_classes 502 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --frames_available 1 \
    --data_root /data/Hamzah/signLanguages/karsl/color_cropped \
    --train_list_path /data/Hamzah/signLanguages/karsl/color_cropped/01-02-train.txt \
    --val_list_path /data/Hamzah/signLanguages/karsl/color_cropped/03.txt \
    --n_shots 1 \
    --batch_size 16 \
    --batch_split 1 \
    --auto_augment rand-m7-n4-mstd0.5-inc1 \
    --mean 0.48145466 0.4578275 0.40821073 \
    --std 0.26862954 0.26130258 0.27577711 \
    --num_workers 16 \
    --num_frames 24 \
    --sampling_rate 8 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"


  #!/usr/bin/env sh
exp_dir=runs/karsl_502_vitl14_32f_dec4x1024_2shot_test01 

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
    --num_classes 502 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --frames_available 1 \
    --data_root /data/Hamzah/signLanguages/karsl/color_cropped \
    --train_list_path /data/Hamzah/signLanguages/karsl/color_cropped/02-03-train.txt \
    --val_list_path /data/Hamzah/signLanguages/karsl/color_cropped/01.txt \
    --n_shots 2 \
    --batch_size 16 \
    --batch_split 1 \
    --auto_augment rand-m7-n4-mstd0.5-inc1 \
    --mean 0.48145466 0.4578275 0.40821073 \
    --std 0.26862954 0.26130258 0.27577711 \
    --num_workers 16 \
    --num_frames 24 \
    --sampling_rate 8 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"


#!/usr/bin/env sh
exp_dir=runs/karsl_502_vitl14_32f_dec4x1024_2shot_test02 

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
    --num_classes 502 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --frames_available 1 \
    --data_root /data/Hamzah/signLanguages/karsl/color_cropped \
    --train_list_path /data/Hamzah/signLanguages/karsl/color_cropped/01-03-train.txt \
    --val_list_path /data/Hamzah/signLanguages/karsl/color_cropped/02.txt \
    --n_shots 2 \
    --batch_size 16 \
    --batch_split 1 \
    --auto_augment rand-m7-n4-mstd0.5-inc1 \
    --mean 0.48145466 0.4578275 0.40821073 \
    --std 0.26862954 0.26130258 0.27577711 \
    --num_workers 16 \
    --num_frames 24 \
    --sampling_rate 8 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"


#!/usr/bin/env sh
exp_dir=runs/karsl_502_vitl14_32f_dec4x1024_2shot_test03 

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
    --num_classes 502 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --frames_available 1 \
    --data_root /data/Hamzah/signLanguages/karsl/color_cropped \
    --train_list_path /data/Hamzah/signLanguages/karsl/color_cropped/01-02-train.txt \
    --val_list_path /data/Hamzah/signLanguages/karsl/color_cropped/03.txt \
    --n_shots 2 \
    --batch_size 16 \
    --batch_split 1 \
    --auto_augment rand-m7-n4-mstd0.5-inc1 \
    --mean 0.48145466 0.4578275 0.40821073 \
    --std 0.26862954 0.26130258 0.27577711 \
    --num_workers 16 \
    --num_frames 24 \
    --sampling_rate 8 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"

#!/usr/bin/env sh
exp_dir=runs/karsl_502_vitl14_32f_dec4x1024_4shot_test01 

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
    --num_classes 502 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --frames_available 1 \
    --data_root /data/Hamzah/signLanguages/karsl/color_cropped \
    --train_list_path /data/Hamzah/signLanguages/karsl/color_cropped/02-03-train.txt \
    --val_list_path /data/Hamzah/signLanguages/karsl/color_cropped/01.txt \
    --n_shots 4 \
    --batch_size 16 \
    --batch_split 1 \
    --auto_augment rand-m7-n4-mstd0.5-inc1 \
    --mean 0.48145466 0.4578275 0.40821073 \
    --std 0.26862954 0.26130258 0.27577711 \
    --num_workers 16 \
    --num_frames 24 \
    --sampling_rate 8 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"


#!/usr/bin/env sh
exp_dir=runs/karsl_502_vitl14_32f_dec4x1024_4shot_test02 

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
    --num_classes 502 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --frames_available 1 \
    --data_root /data/Hamzah/signLanguages/karsl/color_cropped \
    --train_list_path /data/Hamzah/signLanguages/karsl/color_cropped/01-03-train.txt \
    --val_list_path /data/Hamzah/signLanguages/karsl/color_cropped/02.txt \
    --n_shots 4 \
    --batch_size 16 \
    --batch_split 1 \
    --auto_augment rand-m7-n4-mstd0.5-inc1 \
    --mean 0.48145466 0.4578275 0.40821073 \
    --std 0.26862954 0.26130258 0.27577711 \
    --num_workers 16 \
    --num_frames 24 \
    --sampling_rate 8 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"


#!/usr/bin/env sh
exp_dir=runs/karsl_502_vitl14_32f_dec4x1024_4shot_test03 

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
    --num_classes 502 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --frames_available 1 \
    --data_root /data/Hamzah/signLanguages/karsl/color_cropped \
    --train_list_path /data/Hamzah/signLanguages/karsl/color_cropped/01-02-train.txt \
    --val_list_path /data/Hamzah/signLanguages/karsl/color_cropped/03.txt \
    --n_shots 4 \
    --batch_size 16 \
    --batch_split 1 \
    --auto_augment rand-m7-n4-mstd0.5-inc1 \
    --mean 0.48145466 0.4578275 0.40821073 \
    --std 0.26862954 0.26130258 0.27577711 \
    --num_workers 16 \
    --num_frames 24 \
    --sampling_rate 8 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"

#!/usr/bin/env sh
exp_dir=runs/karsl_502_vitl14_32f_dec4x1024_8shot_test01 

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
    --num_classes 502 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --frames_available 1 \
    --data_root /data/Hamzah/signLanguages/karsl/color_cropped \
    --train_list_path /data/Hamzah/signLanguages/karsl/color_cropped/02-03-train.txt \
    --val_list_path /data/Hamzah/signLanguages/karsl/color_cropped/01.txt \
    --n_shots 8 \
    --batch_size 16 \
    --batch_split 1 \
    --auto_augment rand-m7-n4-mstd0.5-inc1 \
    --mean 0.48145466 0.4578275 0.40821073 \
    --std 0.26862954 0.26130258 0.27577711 \
    --num_workers 16 \
    --num_frames 24 \
    --sampling_rate 8 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"


#!/usr/bin/env sh
exp_dir=runs/karsl_502_vitl14_32f_dec4x1024_8shot_test02 

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
    --num_classes 502 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --frames_available 1 \
    --data_root /data/Hamzah/signLanguages/karsl/color_cropped \
    --train_list_path /data/Hamzah/signLanguages/karsl/color_cropped/01-03-train.txt \
    --val_list_path /data/Hamzah/signLanguages/karsl/color_cropped/02.txt \
    --n_shots 8 \
    --batch_size 16 \
    --batch_split 1 \
    --auto_augment rand-m7-n4-mstd0.5-inc1 \
    --mean 0.48145466 0.4578275 0.40821073 \
    --std 0.26862954 0.26130258 0.27577711 \
    --num_workers 16 \
    --num_frames 24 \
    --sampling_rate 8 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"


#!/usr/bin/env sh
exp_dir=runs/karsl_502_vitl14_32f_dec4x1024_8shot_test03 

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
    --num_classes 502 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --frames_available 1 \
    --data_root /data/Hamzah/signLanguages/karsl/color_cropped \
    --train_list_path /data/Hamzah/signLanguages/karsl/color_cropped/01-02-train.txt \
    --val_list_path /data/Hamzah/signLanguages/karsl/color_cropped/03.txt \
    --n_shots 8 \
    --batch_size 16 \
    --batch_split 1 \
    --auto_augment rand-m7-n4-mstd0.5-inc1 \
    --mean 0.48145466 0.4578275 0.40821073 \
    --std 0.26862954 0.26130258 0.27577711 \
    --num_workers 16 \
    --num_frames 24 \
    --sampling_rate 8 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"

