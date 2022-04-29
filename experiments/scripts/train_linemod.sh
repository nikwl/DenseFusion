#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 ./tools/train.py --dataset linemod\
  --dataset_root /media/DATACENTER2/nikolas/dev/data/datasets/ycbv/adv_deep_learning