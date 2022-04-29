# --dataset_root ./datasets/linemod/Linemod_preprocessed \

python3 ./tools/eval_linemod2.py \
  --dataset_root /opt/ws/DenseFusion/datasets/linemod/Linemod_preprocessed \
  --model trained_models/trained_checkpoints/linemod/pose_model_9_0.01310166542980859.pth \
  --refine_model trained_models/trained_checkpoints/linemod/pose_refine_model_493_0.006761023565178073.pth