
python3 ./tools/eval_ycb2.py \
    --dataset_root /opt/data/DenseFusion/YCB_dataset \
    --model trained_models/trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth \
    --refine_model trained_models/trained_checkpoints/ycb/pose_refine_model_69_0.009449292959118935.pth \
    --object_idx 10
