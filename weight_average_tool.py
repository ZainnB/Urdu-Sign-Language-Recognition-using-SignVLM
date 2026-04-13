import torch
import os
import glob

raw_clip = '/media/hamzah_luqman/CLIP/frozenClip/CLIP_weights/ViT-L/ViT-L-14.pt'
# source_dir = '/DDN_ROOT/ytcheng/code/patching_checkpoint/basetraining/temporalclip_vitb16_8x16_interpolation_bugfix_0.5ratio_rand0.0_0.6sample_seed2/checkpoints'
# output_dir = '/DDN_ROOT/ytcheng/code/patching_checkpoint/basetraining/temporalclip_vitb16_8x16_interpolation_bugfix_0.5ratio_rand0.0_0.6sample_seed2/wa_checkpoints'
source_dir = '/media/hamzah_luqman/CLIP/frozenClip/runs/karsl_AUSTL_vitl14_32f_dec4x1024_FT'
output_dir = '/media/hamzah_luqman/CLIP/frozenClip/runs/karsl_AUSTL_vitl14_32f_dec4x1024_FT'

wa_start = 1
wa_end = 18

def average_checkpoint(checkpoint_list):
    # ckpt_list = []
    
    # # raw clip
    # raw_clip_weight = {}
    # clip_ori_state = torch.jit.load(raw_clip, map_location='cpu').state_dict() 
    # _ = [clip_ori_state.pop(i) for i in ['input_resolution', 'context_length', 'vocab_size']]
    # for key in clip_ori_state:
    #     raw_clip_weight['model.' + key] = clip_ori_state[key]

    # ckpt_list.append((0, raw_clip_weight))
    # for name, ckpt_id in checkpoint_list:
    #     ckpt_list.append((ckpt_id, torch.load(name, map_location='cpu')['model']))
    #     #ckpt_list.append((ckpt_id, torch.load(name, map_location='cpu')))
    
    # linear_porj_keys = []
    # for k, v in ckpt_list[-1][1].items():
    #     if 'projector' in k:
    #         linear_porj_keys.append(k)
    #     elif 'adapter' in k:
    #         linear_porj_keys.append(k)
    #     elif 'post_prompt' in k:
    #         linear_porj_keys.append(k)
    # print(linear_porj_keys)

    # # threshold filter
    # new_ckpt_list = []
    # ckpt_id_list = []
    # for i in ckpt_list:
    #     if int(i[0]) >= wa_start and int(i[0]) <= wa_end:
    #         new_ckpt_list.append(i)
    #         ckpt_id_list.append(int(i[0]))
    
    # print("Files with the following paths will participate in the parameter averaging")
    # print(ckpt_id_list)

    # state_dict = {}
    # for key in raw_clip_weight:
    #     state_dict[key] = []
    #     for ckpt in new_ckpt_list:
    #         state_dict[key].append(ckpt[1][key])
    
    # for key in linear_porj_keys:
    #     state_dict[key] = []
    #     for ckpt in new_ckpt_list:
    #         state_dict[key].append(ckpt[1][key])
    
    # for key in state_dict:
    #     try:
    #         state_dict[key] = torch.mean(torch.stack(state_dict[key], 0), 0)
    #     except:
    #         print('Error in : ', key)

    # Load the first checkpoint to initialize the averaged model
    checkpoint = torch.load(checkpoint_list[0][0])
    avg_model_state = checkpoint['model']

    # Initialize the summed state dict with zeros
    for key in avg_model_state.keys():
        avg_model_state[key] = torch.zeros_like(avg_model_state[key])

    # Sum up all model weights
    for path, name in checkpoint_list:
        checkpoint = torch.load(path)
        model_state = checkpoint['model']
        for key in avg_model_state.keys():
            avg_model_state[key] += model_state[key]

    # Average the weights
    num_checkpoints = len(checkpoint_list)
    for key in avg_model_state.keys():
        avg_model_state[key] /= num_checkpoints

    # Save the averaged model
    # averaged_checkpoint = {
    #     'model_state_dict': avg_model_state,
    #     'optimizer_state_dict': checkpoint['optimizer_state_dict'],  # optional
    #     'epoch': checkpoint['epoch'],  # optional
    # }
    # torch.save(averaged_checkpoint, "path/to/averaged_model.pth")

    print("Averaged model saved.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)
    loss_scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=args.fp16)
    criterion = torch.nn.CrossEntropyLoss()

    chk_last = checkpoint_list[-1][0]
    optimizer = checkpoint['optimizer']
    lr_sched = checkpoint['lr_sched']
    loss_scaler = checkpoint['loss_scaler']
    next_step = checkpoint['next_step']

    to_save = {
     'model': avg_model_state,
     'optimizer': optimizer,
     'lr_sched': lr_sched,
     'loss_scaler': loss_scaler,
     'next_step': next_step,
    }




    return to_save


os.makedirs(output_dir, exist_ok=True)
checkpoint_list = os.listdir(source_dir)
checkpoint_list = sorted(glob.glob(source_dir +'/*.pth'))   
checkpoint_list = [(chkName, int(chkName.split('/')[-1].split('.')[0].split('-')[-1])) for chkName in checkpoint_list]

#checkpoint_list = [(os.path.join(source_dir, i), int(i.split('.')[0].split('-')[-1])) for i in checkpoint_list]
# checkpoint_list = sorted(checkpoint_list, key=lambda d: d[1])
print(checkpoint_list)

to_save = average_checkpoint(checkpoint_list)

torch.save(to_save, os.path.join(output_dir, 'swa_%d_%d.pth'%(wa_start, wa_end)))

#torch.save({'model': swa_state_dict}, os.path.join(output_dir, 'swa_%d_%d.pth'%(wa_start, wa_end)))


 