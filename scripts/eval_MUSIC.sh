#!/bin/bash

OPTS=""
OPTS+="--mode eval "
OPTS+="--id Exp3_con2_dum "
OPTS+="--list_train data/train.csv  "
OPTS+="--list_val data/val.csv "
OPTS+="--batch_size_per_gpu 16 "
# OPTS+="--val_without_vis "
OPTS+="--num_vis 100 "
# OPTS+="--rate_dc 1 "


# Models
OPTS+="--arch_sound unet7 "
OPTS+="--arch_synthesizer linear "
OPTS+="--arch_frame resnet18dilated "
OPTS+="--img_pool maxpool "
OPTS+="--num_channels 2 "
OPTS+="--vis_channels 512 "
OPTS+="--img_activation relu "
OPTS+="--fusion_type con2 "

# binary mask, BCE loss, weighted loss
OPTS+="--binary_mask 1 "
OPTS+="--loss bce "
OPTS+="--weighted_loss 1 "
# logscale in frequency
OPTS+="--num_mix 2 "
OPTS+="--log_freq 1 "

# frames-related
OPTS+="--num_frames 3 "
OPTS+="--stride_frames 8 "
OPTS+="--frameRate 30 "
OPTS+="--max_silent 0.83 "

# audio-related
OPTS+="--audLen 65535 "
OPTS+="--audRate 11025 "

python -u main.py $OPTS
