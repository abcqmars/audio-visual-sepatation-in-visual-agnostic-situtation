#!/bin/bash
OPTS=""
OPTS+="--id Exp5_BaseSig "
# OPTS+="--list_train data/train.csv "
OPTS+="--av_list_train data/train.csv "
OPTS+="--ao_list_train data/train.csv "
OPTS+="--list_val data/val.csv "
# OPTS+="--mode eval "
# OPTS+="--load_ckpt 1 "
OPTS+="--start_av_first "
OPTS+="--num_fsteps 0 "
# Models
OPTS+="--arch_sound unet7 "
OPTS+="--arch_synthesizer linear "
OPTS+="--arch_frame resnet18dilated "
OPTS+="--img_pool maxpool "
OPTS+="--num_channels 2 "
OPTS+="--img_activation relu "
OPTS+="--output_activation sigmoid "
OPTS+="--vis_channels 256 "
OPTS+="--fusion_type hidsep "
OPTS+="--not_pool_vis "
OPTS+="--att_type sig "
# OPTS+="--val_without_vis "
# OPTS+="--ckpt Exp1+ "
# binary mask, BCE loss, weighted loss
OPTS2=" "
OPTS2+="--binary_mask 1 "
OPTS2+="--loss bce "
OPTS2+="--weighted_loss 1 "
# logscale in frequency
OPTS2+="--num_mix 2 "
OPTS2+="--log_freq 1 "
# frames-related
OPTS2+="--num_frames 3 "
OPTS2+="--stride_frames 8 "
OPTS2+="--frameRate 30 "
# audio-related
OPTS2+="--audLen 65535 "
OPTS2+="--audRate 11025 "
# learning params
OPTS2+="--num_gpus 2 "
OPTS2+="--workers 4 "
OPTS2+="--batch_size_per_gpu 16 "
OPTS2+="--lr_frame 1e-4 "
OPTS2+="--lr_sound 1e-3 "
OPTS2+="--lr_synthesizer 1e-3 "
OPTS2+="--lr_steps 50000 70000 90000 "
OPTS2+="--num_iters 95001 "
OPTS2+="--iter_per_av 2 "
OPTS2+="--eval_iter 1000 "
OPTS2+="--train_repeat 50 " # 100

# display, viz
OPTS2+="--disp_iter 20 "
OPTS2+="--num_vis 100 "
OPTS2+="--num_val 256 "
OPTS2+="--rate_dc 1 "
OPTS2+="--max_silent 0.87 "
OPTS2+="--mask_thres 0.5 "
OPTS2+="--match_weight 0.1 "
OPTS2+="--one_frame "

python -u main.py $OPTS $OPTS2
