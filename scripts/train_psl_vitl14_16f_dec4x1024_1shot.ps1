
$PYTHON = "C:\Users\Dell\Anaconda3\envs\psl_pose\python.exe"

& $PYTHON -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# Stop on errors (equivalent to `set -e`)
$ErrorActionPreference = "Stop"

# Get script directory and move to project root
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Definition
$SIGNVLM_ROOT = Resolve-Path "$SCRIPT_DIR\.."
Set-Location $SIGNVLM_ROOT

# Experiment directory
$EXP_DIR = "..\Urdu_Sign_Language_Recognition_using_SignVLM\trained_models_final\signVLM_psl"

# Create directory if it doesn't exist
if (!(Test-Path $EXP_DIR)) {
    New-Item -ItemType Directory -Path $EXP_DIR | Out-Null
}

# Timestamp (YYYYMMDD_HHMMSS)
$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"

# Log file
$LOG_FILE = "$EXP_DIR\train-$TIMESTAMP.log"

Write-Host "Starting training..."
Write-Host "Logs: $LOG_FILE"

# Build command
# Build argument list (no string joining)
$args = @(
    "-u",
    "main.py",
    "--backbone", "ViT-L/14-lnpre",
    "--backbone_path", "..\Urdu_Sign_Language_Recognition_using_SignVLM\CLIP_weights\ViT-L\ViT-L-14.pt",
    "--backbone_type", "clip",
    "--decoder_num_layers", "4",
    "--decoder_qkv_dim", "1024",
    "--decoder_num_heads", "16",
    "--num_classes", "104",
    "--checkpoint_dir", $EXP_DIR,
    "--frames_available", "1",
    "--data_root", "..\Urdu_Sign_Language_Recognition_using_SignVLM\data\data_only_frames",
    "--train_list_path", "..\Urdu_Sign_Language_Recognition_using_SignVLM\dataset_split_text_files\train_signer_signvlm.txt",
    "--val_list_path", "..\Urdu_Sign_Language_Recognition_using_SignVLM\dataset_split_text_files\val_signer_signvlm.txt",
    "--n_shots", "-1",
    "--num_frames", "16",
    "--sampling_rate", "4",
    "--num_steps", "2500",
    "--save_freq", "500",
    "--eval_freq", "500",
    "--lr", "0.00004",
    "--weight_decay", "0.05",
    "--batch_size", "8",
    "--batch_split", "2",
    "--decoder_mlp_factor", "4.0",
    "--cls_dropout", "0.5",
    "--decoder_mlp_dropout", "0.5",
    "--num_workers", "4",
    "--auto_resume",
    "--mean", "0.48145466", "0.4578275", "0.40821073",
    "--std", "0.26862954", "0.26130258", "0.27577711",
    "--auto_augment", "rand-m7-n4-mstd0.5-inc1",
    "--spatial_size", "224"
)

# Run (proper execution)
$ErrorActionPreference = "Continue"
& $PYTHON @args 2>&1 | Tee-Object -FilePath $LOG_FILE
$ErrorActionPreference = "Stop"

Write-Host "Training finished."