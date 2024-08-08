#!/bin/bash
python3 train.py --data-path ~/autodl-tmp/MuReD/label_one --data-set MuReD --num_workers 4 --gp --change_qkv --relative_position \
--mode retrain --model_type 'AUTOFORMER' --cfg './experiments/subnet_autoformer/TF_TAS-T-MuReD.yaml' --output_dir './OUTPUT/sample'


