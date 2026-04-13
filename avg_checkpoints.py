#!/usr/bin/env python

import argparse
from datetime import datetime
import builtins
import os
import glob

import torch
import torch.distributed as dist

import video_dataset
import checkpoint
from model import EVLTransformer
from weight_loaders import weight_loader_fn_dict
from vision_transformer import vit_presets

def setup_print(is_master: bool):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            now = datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def main(checkpointsFolder = "runs/karsl_AUSTL_vitl14_32f_dec4x1024_FT",
      num_steps=50005 , backbone="ViT-L/14-lnpre",
    backbone_type="clip",
    backbone_path= "/media/hamzah_luqman/CLIP/frozenClip/CLIP_weights/ViT-L/ViT-L-14.pt",
    decoder_num_layers= 4,
    decoder_qkv_dim= 1024,
    decoder_num_heads= 16,
    num_classes=256,
    checkpoint_dir="uns/karsl_AUSTL_vitl14_32f_dec4x1024_FT",
    auto_resume =True,
    frames_available= 1,
    data_root= "/data/Hamzah/signLanguages/AUST/frames",
    train_list_path = "/data/Hamzah/signLanguages/AUST/frames/trainVal_labels.txt",
    val_list_path= "/data/Hamzah/signLanguages/AUST/frames/test_labels.txt",
    n_shots= -1,
    batch_size=16,
    batch_split=1,
    num_workers=16,
    num_frames=24,
    sampling_rate=4,
    num_spatial_views=3,
    num_temporal_views=1 ):

    parser = argparse.ArgumentParser()
    
    video_dataset.setup_arg_parser(parser)
    checkpoint.setup_arg_parser(parser)

    parser.add_argument('--num_steps', type=int,
                        help='number of training steps')
    parser.add_argument('--eval_only', action='store_true',
                        help='run evaluation only')
    parser.add_argument('--save_freq', type=int, default=2500,
                        help='save a checkpoint every N steps')
    parser.add_argument('--eval_freq', type=int, default=2500,
                        help='evaluate every N steps')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print log message every N steps')
    # parser.add_argument('--frames_available', type=bool, default=False,
    #                     help='True if the frames already extracted.')
    parser.add_argument('--backbone', type=str, choices=vit_presets.keys(), default='ViT-B/16-lnpre',
                        help='the backbone variant used to generate image feature maps')
    parser.add_argument('--backbone_path', type=str,
                        help='path to pretrained backbone weights')
    parser.add_argument('--backbone_type', type=str, default='clip', choices=weight_loader_fn_dict.keys(),
                        help='type of backbone weights (used to determine how to convert state_dict from different pretraining codebase)')
    parser.add_argument('--finetune_backbone', action='store_true',
                        help='finetune backbone weights')
    parser.add_argument('--decoder_num_layers', type=int, default=4,
                        help='number of decoder layers')
    parser.add_argument('--decoder_qkv_dim', type=int, default=768,
                        help='q (k, v) projection output dimensions in decoder attention layers')
    parser.add_argument('--decoder_num_heads', type=int, default=12,
                        help='number of heads in decoder attention layers')
    parser.add_argument('--decoder_mlp_factor', type=float, default=4.0,
                        help='expansion factor of feature dimension in the middle of decoder MLPs')
    parser.add_argument('--num_classes', type=int, default=400,
                        help='number of classes')
    # parser.add_argument('--n_shots', type=int, default=-1,
    #                 help='number of shots')
    parser.add_argument('--cls_dropout', type=float, default=0.5,
                        help='dropout rate applied before the final classification linear projection')
    parser.add_argument('--decoder_mlp_dropout', type=float, default=0.5,
                        help='dropout rate applied in MLP layers in the decoder')
    parser.add_argument('--no_temporal_conv', action='store_false', dest='temporal_conv',
                        help='disable temporal convolution on frame features')
    parser.add_argument('--no_temporal_pos_embed', action='store_false', dest='temporal_pos_embed',
                        help='disable temporal position embeddings added to frame features')
    parser.add_argument('--no_temporal_cross_attention', action='store_false', dest='temporal_cross_attention',
                        help='disable temporal cross attention on frame query and key features')
    parser.set_defaults(temporal_conv=True, temporal_pos_embed=True, temporal_cross_attention=True)

    parser.add_argument('--lr', type=float, default=4e-5,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='optimizer weight decay')
    parser.add_argument('--disable_fp16', action='store_false', dest='fp16',
                        help='disable fp16 during training or inference')
    parser.set_defaults(fp16=True)

    parser.add_argument('--batch_split', type=int, default=1,
                        help='optionally split the batch into smaller shards and forward/backward one shard '
                             'at a time to avoid out-of-memory error.')

    args = parser.parse_args()
    #print('//////////// ', args.frames_available)
    #args.save_freq = int(args.num_steps / 2)
    #cuda_device_id = dist.get_rank() % torch.cuda.device_count()
    #torch.device('cuda:1')
    cuda_device_id = 'cuda:1'
    torch.cuda.set_device(1)
    
    print('>>>>>>>finetune_backbone: ', args.finetune_backbone)
    print('>>>>>>>Backbone name: ', args.backbone)
    print('>>>>>>>backbone_mode: ', 'finetune' if args.finetune_backbone else ('freeze_fp16' if args.fp16 else 'freeze_fp32'))
    print('>>>>>>>Used GPU: ', torch.cuda.current_device())

    model = EVLTransformer(
        backbone_name=backbone,
        backbone_type=backbone_type,
        backbone_path=backbone_path,
        backbone_mode='finetune' if args.finetune_backbone else ('freeze_fp16' if args.fp16 else 'freeze_fp32'),
        decoder_num_layers=decoder_num_layers,
        decoder_qkv_dim=decoder_qkv_dim,
        decoder_num_heads=decoder_num_heads,
        decoder_mlp_factor=args.decoder_mlp_factor,
        num_classes=num_classes,
        enable_temporal_conv=args.temporal_conv,
        enable_temporal_pos_embed=args.temporal_pos_embed,
        enable_temporal_cross_attention=args.temporal_cross_attention,
        cls_dropout=args.cls_dropout,
        decoder_mlp_dropout=args.decoder_mlp_dropout,
        num_frames=num_frames,
        #number_shots = args.num_classes
    )
    #print(model)
    model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(
    #     model, device_ids=[cuda_device_id], output_device=cuda_device_id,
    # )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)
    loss_scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=args.fp16)

    #resume_step = checkpoint.resume_from_checkpoint(model, optimizer, lr_sched, loss_scaler, args)
    getAverageChks(checkpointsFolder, model, optimizer, lr_sched, loss_scaler)

def getAverageChks(checkpointsFolder, model, optimizer, lr_sched, loss_scaler):
    checkpoint_list = sorted(glob.glob(checkpointsFolder +'/*.pth'))   
    checkpoint_list = [(chkName, int(chkName.split('/')[-1].split('.')[0].split('-')[-1])) for chkName in checkpoint_list]

    # ckpt = torch.load(args.resume_path, map_location='cpu')
    # model.load_state_dict(ckpt['model'], strict=True)
    ##
        # Load the first checkpoint to initialize the averaged model
    checkpoint = torch.load(checkpoint_list[0][0], map_location='cpu')
    avg_model_state = checkpoint['model']

    # Initialize the summed state dict with zeros
    for key in avg_model_state.keys():
        avg_model_state[key] = torch.zeros_like(avg_model_state[key])
    print(avg_model_state)
    # Sum up all model weights
    for path, name in checkpoint_list:
        checkpoint = torch.load(path, map_location='cpu')
        model_state = checkpoint['model']
        for key in avg_model_state.keys():
            avg_model_state[key] += model_state[key]

    # Average the weights
    num_checkpoints = len(checkpoint_list)
    for key in avg_model_state.keys():
        avg_model_state[key] /= num_checkpoints



    ##
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_sched.load_state_dict(checkpoint['lr_sched'])
        loss_scaler.load_state_dict(checkpoint['loss_scaler'])
        
        # Save the averaged model
    to_save = {
        'model': avg_model_state,
        'optimizer': optimizer.state_dict(),
        'lr_sched': lr_sched.state_dict(),
        'loss_scaler': loss_scaler.state_dict(),
        'next_step': checkpoint['next_step'],
    }
    torch.save(to_save, os.path.join(checkpointsFolder, 'average.pth'))
    
if __name__ == '__main__': 
    main(checkpointsFolder = "runs/karsl_AUSTL_vitl14_32f_dec4x1024_FT",
        num_steps=50005 , backbone="ViT-L/14-lnpre",
    backbone_type="clip",
    backbone_path= "/media/hamzah_luqman/CLIP/frozenClip/CLIP_weights/ViT-L/ViT-L-14.pt",
    decoder_num_layers= 4,
    decoder_qkv_dim= 1024,
    decoder_num_heads= 16,
    num_classes=256,
    checkpoint_dir="runs/karsl_AUSTL_vitl14_32f_dec4x1024_FT",
    auto_resume =True,
    frames_available= 1,
    data_root= "/data/Hamzah/signLanguages/AUST/frames",
    train_list_path = "/data/Hamzah/signLanguages/AUST/frames/trainVal_labels.txt",
    val_list_path= "/data/Hamzah/signLanguages/AUST/frames/test_labels.txt",
    n_shots= -1,
    batch_size=16,
    batch_split=1,
    num_workers=16,
    num_frames=24,
    sampling_rate=4,
    num_spatial_views=3,
    num_temporal_views=1 
)