#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
# export CUDA_VISIBLE_DEVICES=0

python3 ./tools/train_v2.py \
  --dataset ycb \
  --dataset_root /media/DATACENTER2/nikolas/dev/data/datasets/ycb_video_dataset/YCB_dataset \
  --outf trained_models_v2/ycb_v2_a0/models \
  --log_dir trained_models_v2/ycb_v2_a0/logs \
  --num_points 1000 \
  --repeat_epoch 2 \
  --workers 0 \
  --batch_size 10