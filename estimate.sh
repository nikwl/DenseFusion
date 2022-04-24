#!/bin/bash

# run the script with all the correct arguments
python3 ./tools/estimate.py \
    --input "$1" \
    --model trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth \
    --refine_model trained_checkpoints/ycb/pose_refine_model_69_0.009449292959118935.pth