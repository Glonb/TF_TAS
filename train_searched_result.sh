#!/bin/bash
python3 train.py --data-set CIFAR100 --num_workers 4 --gp --change_qkv --relative_position \
--mode retrain --model_type 'AUTOFORMER' --cfg './experiments/subnet_autoformer/TF_TAS-T-CIFAR100.yaml' --output_dir './OUTPUT/sample'


