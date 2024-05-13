#!/bin/bash
python3 train.py --data-set CIFAR10 --num_workers 4 --gp --change_qkv --relative_position \
--mode retrain --model_type 'AUTOFORMER' --cfg './experiments/subnet_autoformer/TF_TAS-T-CIFAR10.yaml' --output_dir './OUTPUT/sample'


